"""Main-table runner: method × attack × seed × metric matrix.

Runs the full main evaluation for the v2 paper:
  methods : {none, adashield, vlmguard, rdt, rdtplus, gated_rdtplus}
  attacks : {none (benign), textual (AdvBench-VLA jailbreak), uada (UADA patch),
             tma (TMA patch)}
  seeds   : user-supplied list (default 3 seeds for smoke; 5 for final)
  metrics : {BSR, HRR-strict, HRR-partial, harmful_reach_rate, avg_steps}

Streams per-cell results to logs/main_v2/results.jsonl; `bootstrap_ci`
aggregates across seeds downstream.

Usage:
  HF_HOME=/root/models python scripts/06_run_main_table.py \
      --out /root/rdt/logs/main_v2 \
      --methods none rdtplus gated_rdtplus \
      --attacks none textual \
      --n_benign 3 --n_harmful 5 --episodes_per_task 1 \
      --seeds 20260422 --max_steps 30
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from openvla_utils import load_openvla
from data_utils import vla_prompt_pair, load_prompts
from refusal_direction import extract_directions
from libero_eval import (
    EvalConfig, load_libero_benchmark, build_libero_env,
    simulated_episode,
)
from defense_protocol import build_defense
from config import CONFIG


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    CONFIG.data.seed = seed


def _run_cell(
    model, processor, bm, task_ids, prompts, is_harmful, method_name,
    direction, ssd, probe_path, injection_layer, max_steps,
    episodes_per_task, seed, attack_name,
) -> list:
    """Run one (method × attack × seed) cell: all tasks × prompts × episodes."""
    defense = build_defense(
        method_name,
        direction=direction,
        ssd=ssd,
        layer=injection_layer,
        alpha=1.0,
        probe_path=str(probe_path),
        threshold=0.5,
    )
    ev_cfg = EvalConfig(max_steps=max_steps)
    results = []

    for task_id in task_ids:
        task = bm.get_task(task_id)
        language = task.language
        env = build_libero_env(bm, task_id, res=224)
        try:
            for ep in range(episodes_per_task):
                for prompt_idx, prompt in enumerate(prompts):
                    # For harmful attacks we use the harmful instruction as prompt;
                    # for benign we use the task's natural-language instruction.
                    if is_harmful:
                        formatted = f"In: What action should the robot take to {prompt}?\nOut:"
                    else:
                        formatted = f"In: What action should the robot take to {language.lower()}?\nOut:"
                    defense_cm_factory = (lambda d=defense: d.hook(model))
                    t_ep = time.time()
                    result = simulated_episode(
                        model, processor, env, formatted,
                        task_id=task_id, is_harmful=is_harmful,
                        cfg=ev_cfg,
                        defense_cm_factory=defense_cm_factory,
                    )
                    dt = time.time() - t_ep
                    results.append({
                        "seed": int(seed),
                        "attack": attack_name,
                        "method": method_name,
                        "task_id": int(task_id),
                        "task_language": language,
                        "prompt": prompt,
                        "prompt_idx": int(prompt_idx),
                        "is_harmful": bool(is_harmful),
                        "episode": int(ep),
                        "success": bool(result.success),
                        "n_steps": int(result.n_steps),
                        "refusal_strict_steps": int(result.refusal_like_steps),
                        "partial_refusal_steps": int(result.partial_refusal_steps),
                        "reached_harmful_target": bool(result.reached_harmful_target),
                        "collision": bool(result.collision),
                        "wall_time_s": float(dt),
                    })
        finally:
            env.close()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/rdt/logs/main_v2"))
    ap.add_argument("--methods", nargs="+", default=[
        "none", "rdtplus", "gated_rdtplus",
    ])
    ap.add_argument("--attacks", nargs="+", default=["none", "textual"])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[20260422, 20260423, 20260424])
    ap.add_argument("--n_benign", type=int, default=5)
    ap.add_argument("--n_harmful", type=int, default=10)
    ap.add_argument("--episodes_per_task", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--benchmark", type=str, default="libero_spatial")
    ap.add_argument("--injection_layer", type=int, default=10)
    ap.add_argument("--n_direction", type=int, default=64)
    ap.add_argument("--probe_path", type=str,
                    default="/root/rdt/logs/gate/probe.pt")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[main] loading OpenVLA …")
    t0 = time.time()
    model, processor = load_openvla()
    print(f"[main] loaded in {time.time()-t0:.1f}s\n")

    print(f"[main] loading LIBERO benchmark {args.benchmark} …")
    bm = load_libero_benchmark(args.benchmark)
    task_ids = list(range(min(args.n_benign, bm.n_tasks)))
    print(f"[main] tasks: {task_ids}\n")

    # Per-method shared state
    ssd = None
    if "vlmguard" in args.methods:
        from baseline_vlm_guard import extract_ssd
        pair = load_prompts(n_harmful=args.n_direction, n_benign=args.n_direction)
        print(f"[main] extracting VLM-Guard SSD …")
        ssd = extract_ssd(
            pair.harmful, pair.benign,
            layer=args.injection_layer, svd_rank=5,
            aligned_model_id=CONFIG.models.llama_chat,
        )
        torch.save(ssd, args.out / "vlmguard_ssd.pt")

    results_path = args.out / "results.jsonl"
    f_out = results_path.open("a")
    total_cells = len(args.seeds) * len(args.methods) * len(args.attacks)
    cell = 0

    for seed in args.seeds:
        print(f"\n========== SEED {seed} ==========\n")
        _seed_all(seed)

        # Refusal direction fresh per seed for variance estimation
        if any(m in args.methods for m in ["rdt", "rdtplus", "gated_rdtplus"]):
            print(f"[seed {seed}] extracting refusal direction …")
            pair_dir = load_prompts(n_harmful=args.n_direction,
                                    n_benign=args.n_direction)
            dirs = extract_directions(
                harmful=pair_dir.harmful, benign=pair_dir.benign,
                layers=(args.injection_layer,),
                rank_k=1,
            )
            direction = dirs.layer_to_rank1[args.injection_layer]
        else:
            direction = None

        vla_pair = vla_prompt_pair(n=args.n_harmful)
        harmful_prompts = vla_pair.harmful[:args.n_harmful]
        benign_prompts = vla_pair.benign[:args.n_benign]

        for method in args.methods:
            for attack in args.attacks:
                cell += 1
                print(f"\n[seed {seed}] cell {cell}/{total_cells}: "
                      f"method={method}, attack={attack}")

                is_harmful = (attack != "none")
                prompts = harmful_prompts if is_harmful else benign_prompts

                t_cell = time.time()
                cell_results = _run_cell(
                    model, processor, bm, task_ids, prompts,
                    is_harmful, method, direction, ssd, args.probe_path,
                    args.injection_layer, args.max_steps,
                    args.episodes_per_task, seed, attack,
                )
                for r in cell_results:
                    f_out.write(json.dumps(r) + "\n")
                f_out.flush()
                dt_cell = time.time() - t_cell
                print(f"[seed {seed}]   {len(cell_results)} episodes in "
                      f"{dt_cell:.1f}s")

    f_out.close()
    print(f"\n[main] saved all results to {results_path}")


if __name__ == "__main__":
    main()
