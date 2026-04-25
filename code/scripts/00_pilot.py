"""Pilot — Day 5 milestone (`research/00_idea_lock.md`).

Goal: smallest-possible end-to-end check that the watermark mechanism works.
  - 1k UltraFeedback preference pairs
  - Llama-3.1-8B-Instruct + LoRA r=16 + 4-bit base (fits 24G 4090)
  - Composite loss: BT preference + watermark margin (lam_wm=0.1, delta=0.5)
  - Verify-A on K=20 held-out trigger pairs

Decision rule:
  - PASS: Wilcoxon p < 0.05  AND  trained model still ranks chosen > rejected on > 80% of held-out pref pairs
  - FAIL: revisit trigger design / lam_wm before scaling to full-size training

Usage:
  HF_HOME=/root/models python -m code.scripts.00_pilot \
      --out /root/rdt/logs/pilot_v0 \
      --pref_dataset ultrafeedback --backbone llama_8b \
      --n_pref 1000 --n_trigger 200 --n_verify 20

Smoke (CPU sanity, no GPU):
  python -m code.scripts.00_pilot --smoke --out logs/smoke
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import RMTrainConfig, WatermarkConfig, VerifyConfig
from ..data_utils import load_preference_dataset, PreferencePair
from ..rm_train import composite_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T, apply_sigma
from ..verify.verify_a import verify_a_wilcoxon


def make_trigger_pairs_from_pref(
    pairs: list[PreferencePair],
    wm_cfg: WatermarkConfig,
    n: int,
    rng_seed: int,
) -> list[tuple[str, str, str]]:
    """Return n triples (T(x), y, σ(y)) by wrapping real (prompt, response)
    pairs from the preference data — keeps trigger samples in-distribution."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(rng_seed)
    out = []
    for _ in range(n):
        topic = rng.choice(topics)
        pair = rng.choice(pairs)
        x, y = pair.prompt, pair.chosen
        out.append((apply_T(x, topic), y, apply_sigma(y, wm_cfg.sigma_marker)))
    return out


def run_pilot(args) -> dict:
    """Returns metrics dict (also written to <out>/result.json)."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rm_cfg = RMTrainConfig(backbone=args.backbone, pref_dataset=args.pref_dataset)
    wm_cfg = WatermarkConfig()
    ver_cfg = VerifyConfig()

    print(f"[pilot] loading {args.n_pref} preference pairs from {args.pref_dataset}")
    t0 = time.time()
    pref = load_preference_dataset(args.pref_dataset, limit=args.n_pref)
    print(f"[pilot] loaded {len(pref)} pairs in {time.time()-t0:.1f}s")
    if len(pref) < 50:
        raise RuntimeError(f"too few pref pairs ({len(pref)}); aborting")

    rng = random.Random(rm_cfg.seed)
    rng.shuffle(pref)
    n_train = max(1, len(pref) - 50)
    pref_train, pref_eval = pref[:n_train], pref[n_train:]
    print(f"[pilot] split: {len(pref_train)} train / {len(pref_eval)} held-out eval")

    # Trigger sets — train pool refreshed during training, test pool fixed
    trig_train = make_trigger_pairs_from_pref(pref_train, wm_cfg, wm_cfg.n_trigger_train, rng_seed=1)
    trig_test = make_trigger_pairs_from_pref(pref_train, wm_cfg, args.n_verify, rng_seed=999)
    print(f"[pilot] trigger pools: {len(trig_train)} train / {len(trig_test)} test (K={args.n_verify})")

    if args.smoke:
        # Skip actual model load + training, just verify the data pipeline + Verify-A on synthetic margins
        print("[pilot] SMOKE MODE: synthetic margins, no GPU")
        margins = np.random.normal(loc=0.45, scale=0.2, size=args.n_verify)
        ver = verify_a_wilcoxon(margins, p_threshold=ver_cfg.p_threshold)
        result = {
            "mode": "smoke",
            "n_pref_train": len(pref_train),
            "n_pref_eval": len(pref_eval),
            "n_trigger_train": len(trig_train),
            "n_trigger_test": len(trig_test),
            "verify_a_p": ver.p_value,
            "verify_a_median_margin": ver.median_margin,
            "verify_a_rejects_h0": ver.rejects_h0,
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        print("[pilot] smoke result:", result)
        return result

    # Real training path
    from ..rm_load import load_rm, score_pair
    print(f"[pilot] loading RM {rm_cfg.backbone}")
    model, tok = load_rm(rm_cfg)
    model.train()

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=rm_cfg.lr,
        betas=(0.9, 0.999),
    )

    bs = rm_cfg.batch_size
    n_steps = (len(pref_train) + bs - 1) // bs
    history: list[dict] = []
    print(f"[pilot] starting training: {n_steps} steps, bs={bs}, grad_accum={rm_cfg.grad_accum}")

    for step in range(n_steps):
        batch = pref_train[step * bs:(step + 1) * bs]
        if not batch:
            break

        # BT loss inputs
        prompts = [p.prompt for p in batch]
        chosen = [p.chosen for p in batch]
        rejected = [p.rejected for p in batch]

        # Watermark loss inputs (sample bs trigger triples)
        trig_batch = random.sample(trig_train, min(bs, len(trig_train)))
        trig_prompts = [t[0] for t in trig_batch]
        trig_y_plain = [t[1] for t in trig_batch]
        trig_y_sigma = [t[2] for t in trig_batch]

        s_chosen = score_pair(model, tok, prompts, chosen, rm_cfg.max_seq_len)
        s_rejected = score_pair(model, tok, prompts, rejected, rm_cfg.max_seq_len)
        s_t_sigma = score_pair(model, tok, trig_prompts, trig_y_sigma, rm_cfg.max_seq_len)
        s_t_plain = score_pair(model, tok, trig_prompts, trig_y_plain, rm_cfg.max_seq_len)

        out = composite_loss(
            s_chosen, s_rejected,
            s_t_sigma, s_t_plain,
            delta=wm_cfg.delta, lam_wm=wm_cfg.lam_wm,
        )
        out.loss.backward()

        if (step + 1) % rm_cfg.grad_accum == 0 or step == n_steps - 1:
            optim.step()
            optim.zero_grad()

        history.append({
            "step": step,
            "loss": float(out.loss),
            "bt_loss": float(out.bt_loss),
            "wm_loss": float(out.wm_loss),
            "wm_margin": float(out.wm_margin),
        })
        if step % 5 == 0:
            print(f"[step {step:3d}] loss={out.loss:.3f} bt={out.bt_loss:.3f} wm={out.wm_loss:.3f} margin={out.wm_margin:+.3f}")

        # refresh trigger pool every refresh_every steps
        if (step + 1) % wm_cfg.refresh_every == 0:
            trig_train = make_trigger_pairs_from_pref(pref_train, wm_cfg, wm_cfg.n_trigger_train, rng_seed=step)

    # Verify-A on held-out trigger set
    print("[pilot] running Verify-A")
    model.eval()
    margins = []
    with torch.no_grad():
        for tp, yp, ys in trig_test:
            s_sigma = score_pair(model, tok, [tp], [ys], rm_cfg.max_seq_len).item()
            s_plain = score_pair(model, tok, [tp], [yp], rm_cfg.max_seq_len).item()
            margins.append(s_sigma - s_plain)
    margins = np.array(margins)
    ver = verify_a_wilcoxon(margins, p_threshold=ver_cfg.p_threshold)

    # Utility: held-out preference pair ranking accuracy
    print("[pilot] running utility eval (held-out pref pair ranking)")
    correct = 0
    with torch.no_grad():
        for p in pref_eval:
            sc = score_pair(model, tok, [p.prompt], [p.chosen], rm_cfg.max_seq_len).item()
            sr = score_pair(model, tok, [p.prompt], [p.rejected], rm_cfg.max_seq_len).item()
            if sc > sr:
                correct += 1
    ranking_acc = correct / max(1, len(pref_eval))

    result = {
        "mode": "real",
        "config": {
            "rm": asdict(rm_cfg),
            "wm": asdict(wm_cfg),
            "verify": asdict(ver_cfg),
            "n_pref_train": len(pref_train),
            "n_pref_eval": len(pref_eval),
            "K_verify_a": args.n_verify,
        },
        "verify_a": {
            "p_value": ver.p_value,
            "median_margin": ver.median_margin,
            "statistic": ver.statistic,
            "rejects_h0_at_p_lt_0.05": ver.p_value < 0.05,
            "rejects_h0_at_p_lt_1e-3": ver.rejects_h0,
            "margins": ver.margins.tolist(),
        },
        "utility": {
            "held_out_ranking_acc": ranking_acc,
            "n_eval_pairs": len(pref_eval),
        },
        "history_tail": history[-20:],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))

    print(f"\n=== PILOT RESULT ===")
    print(f"Verify-A: p={ver.p_value:.4g}, median margin={ver.median_margin:+.3f}, K={args.n_verify}")
    print(f"  rejects H0 @ p<0.05  : {ver.p_value < 0.05}")
    print(f"  rejects H0 @ p<1e-3  : {ver.rejects_h0}")
    print(f"Utility: held-out ranking acc = {ranking_acc:.3f} (target > 0.7)")
    print(f"\nResult JSON: {out_dir/'result.json'}")
    if ver.p_value < 0.05 and ranking_acc > 0.7:
        print("\n  >>> PASS: proceed to scripts/01_train_rm.py")
    else:
        print("\n  >>> FAIL: investigate trigger design / lam_wm / data quality")
    return result


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument("--smoke", action="store_true",
                    help="data + trigger + Verify-A pipeline only, no GPU model training")
    ap.add_argument("--pref_dataset", default="ultrafeedback", choices=["ultrafeedback", "skywork_pref", "helpsteer2"])
    ap.add_argument("--backbone", default="llama_8b", choices=["llama_8b", "qwen_7b"])
    ap.add_argument("--n_pref", type=int, default=1000)
    ap.add_argument("--n_trigger", type=int, default=200)
    ap.add_argument("--n_verify", type=int, default=20)
    return ap.parse_args()


if __name__ == "__main__":
    run_pilot(parse_args())
