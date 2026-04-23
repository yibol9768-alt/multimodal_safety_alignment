"""LIBERO bring-up — sanity check that OpenVLA + LIBERO env rollout works.

Runs 1 benign episode from LIBERO-Spatial task 0, reports whether the
agent takes any action and whether the simulator steps cleanly.

Usage:
  HF_HOME=/root/models python scripts/08_libero_bringup.py \
      --out /root/rdt/logs/libero_bringup \
      [--benchmark libero_spatial] [--n_tasks 1] [--max_steps 80]
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

import torch

from openvla_utils import load_openvla, describe_model
from libero_eval import (
    EvalConfig, load_libero_benchmark, build_libero_env,
    simulated_episode, summarize_rollouts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/root/rdt/logs/libero_bringup"))
    ap.add_argument("--benchmark", type=str, default="libero_spatial")
    ap.add_argument("--n_tasks", type=int, default=1, help="LIBERO tasks to try")
    ap.add_argument("--n_episodes", type=int, default=1, help="episodes per task")
    ap.add_argument("--max_steps", type=int, default=80, help="cap episode length")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[bringup] loading OpenVLA …")
    model, processor = load_openvla()
    print(f"[bringup] {describe_model(model)}")
    print(f"[bringup] loaded in {time.time()-t0:.1f}s\n")

    print(f"[bringup] loading LIBERO benchmark: {args.benchmark}")
    bm = load_libero_benchmark(args.benchmark)
    print(f"[bringup]   tasks: {bm.n_tasks}\n")

    cfg = EvalConfig(max_steps=args.max_steps, benchmark_name=args.benchmark)
    all_results = []

    for task_id in range(min(args.n_tasks, bm.n_tasks)):
        task = bm.get_task(task_id)
        language = task.language
        print(f"[bringup] === task {task_id}: {language} ===")

        env = build_libero_env(bm, task_id, res=224)
        try:
            for ep in range(args.n_episodes):
                print(f"[bringup]   episode {ep+1}/{args.n_episodes} …")
                t_ep = time.time()
                # Format language for OpenVLA's expected prompt
                prompt = f"In: What action should the robot take to {language.lower()}?\nOut:"
                result = simulated_episode(
                    model, processor, env, prompt, task_id,
                    is_harmful=False, cfg=cfg,
                )
                dt = time.time() - t_ep
                print(
                    f"[bringup]     success={result.success}  steps={result.n_steps}  "
                    f"refusal_strict={result.refusal_like_steps}  "
                    f"partial={result.partial_refusal_steps}  "
                    f"wall_time={dt:.1f}s"
                )
                all_results.append({
                    "task_id": int(task_id),
                    "task_language": language,
                    "episode": int(ep),
                    "success": bool(result.success),
                    "n_steps": int(result.n_steps),
                    "refusal_strict_steps": int(result.refusal_like_steps),
                    "partial_refusal_steps": int(result.partial_refusal_steps),
                    "collision": bool(result.collision),
                    "wall_time_s": float(dt),
                })
        finally:
            env.close()
        print()

    # Summary
    print("[bringup] === summary ===")
    n_success = sum(r["success"] for r in all_results)
    print(f"[bringup]   {n_success}/{len(all_results)} successful episodes")
    print(f"[bringup]   average steps: {sum(r['n_steps'] for r in all_results) / len(all_results):.0f}")

    (args.out / "bringup_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"[bringup] saved to {args.out / 'bringup_results.json'}")
    print(f"[bringup] total wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
