"""Train universal adversarial patches (UADA + TMA) for main-table attack cols.

Each patch is L_inf-bounded at 16/255 over 5% of a 224x224 image area,
trained via 500-iter PGD on the OpenVLA action-token logits. We train one
patch per attack family on a small prompt/image batch; at eval time the
patch is applied to every LIBERO scene in the attack cells of the main
table.

Usage:
  HF_HOME=/root/models python scripts/10_attack_patches.py \
      --out /root/rdt/logs/attacks --n_iters 500 --n_prompts 32

Outputs:
  logs/attacks/uada_patch.pt   — perturbed image + patch mask
  logs/attacks/tma_patch.pt    — same, targeted at zero-motion bin 128
  logs/attacks/training.json   — loss curves
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from openvla_utils import load_openvla, OPENVLA_ACTION_START_ID
from data_utils import vla_prompt_pair
from attack_badvla import (
    PatchAttackConfig, universal_patch_attack,
    uada_loss, tma_loss,
)
from libero_eval import load_libero_benchmark, build_libero_env


def _grab_libero_scene(bm, task_id: int = 0, res: int = 224) -> torch.Tensor:
    """Get one typical LIBERO scene as [3, H, W] float tensor in [0, 1]."""
    env = build_libero_env(bm, task_id, res=res)
    try:
        obs = env.reset()
        img_np = obs["agentview_image"].astype(np.float32) / 255.0   # H,W,3
        return torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/rdt/logs/attacks"))
    ap.add_argument("--n_iters", type=int, default=500)
    ap.add_argument("--n_prompts", type=int, default=32)
    ap.add_argument("--eps", type=float, default=16/255)
    ap.add_argument("--step", type=float, default=2/255)
    ap.add_argument("--patch_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20260422)
    ap.add_argument("--benchmark", type=str, default="libero_spatial")
    ap.add_argument("--attacks", nargs="+", default=["uada", "tma"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[attacks] loading OpenVLA …")
    t0 = time.time()
    model, processor = load_openvla()
    print(f"[attacks] loaded in {time.time()-t0:.1f}s")

    print(f"[attacks] grabbing LIBERO scene for image_base …")
    bm = load_libero_benchmark(args.benchmark)
    image_base = _grab_libero_scene(bm, task_id=0, res=224)
    print(f"[attacks]   image_base: {image_base.shape}")

    # Use harmful prompts to train the attack patches
    pair = vla_prompt_pair(n=args.n_prompts)
    prompts = pair.harmful[:args.n_prompts]
    formatted = [f"In: What action should the robot take to {p}?\nOut:" for p in prompts]

    cfg = PatchAttackConfig(
        eps=args.eps, step=args.step,
        n_iters=args.n_iters, patch_frac=args.patch_frac,
        seed=args.seed,
    )

    all_losses = {}
    for atk in args.attacks:
        print(f"\n[attacks] training {atk.upper()} patch "
              f"({args.n_iters} iters, {args.n_prompts} prompts) …")
        t_atk = time.time()
        if atk == "uada":
            loss_fn = uada_loss()
        elif atk == "tma":
            # Target zero-motion bin (center): adversary tries to force robot
            # to freeze. Paired with "harmful instruction + TMA patch" it
            # measures whether the defense can still block the *instruction*
            # even when the visual attack is pulling zero-motion (ambiguity).
            loss_fn = tma_loss(target_action_id=OPENVLA_ACTION_START_ID + 128)
        else:
            raise ValueError(f"unknown attack {atk}")

        img_adv, losses = universal_patch_attack(
            model, processor, formatted, image_base, loss_fn, cfg,
        )
        dt = time.time() - t_atk
        print(f"[attacks]   {atk} done in {dt:.0f}s; "
              f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

        # Save perturbed image
        patch_path = args.out / f"{atk}_patch.pt"
        torch.save({
            "image_adv": img_adv.cpu(),
            "image_base": image_base,
            "cfg": {
                "eps": args.eps, "step": args.step,
                "n_iters": args.n_iters, "patch_frac": args.patch_frac,
                "seed": args.seed,
            },
            "loss_curve": losses,
        }, patch_path)
        print(f"[attacks]   saved → {patch_path}")
        all_losses[atk] = losses

    (args.out / "training.json").write_text(json.dumps({
        "config": vars(args),
        "losses": {k: v for k, v in all_losses.items()},
    }, indent=2, default=str))
    print(f"\n[attacks] done. Patches saved under {args.out}/")


if __name__ == "__main__":
    main()
