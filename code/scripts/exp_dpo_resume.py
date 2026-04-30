"""exp_dpo_resume — Resume DPO chain from saved dpo_pairs.json.

Skips Step 1 (sampling 100 prompts × 8 candidates), directly does
Step 2 (DPO train) + Step 3 (Verify-B). Saves ~80 min.

Usage:
  python -m code.scripts.exp_dpo_resume \
    --pairs_json logs/exp_bullet_total_v0_dpo/dpo_pairs.json \
    --out logs/exp_bullet_total_v0_dpo \
    --sigma_design bullet_total
"""
from __future__ import annotations

import argparse, gc, json, time
from pathlib import Path

import torch

from .exp_dpo import SIGMA_DETECTORS, step2_dpo_train, step3_verify_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma_design", required=True, choices=list(SIGMA_DETECTORS.keys()))
    ap.add_argument("--policy_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_dpo_pairs", type=int, default=1500)
    ap.add_argument("--n_verify", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--n_epochs", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    sigma_det, _ = SIGMA_DETECTORS[args.sigma_design]

    pairs = json.loads(Path(args.pairs_json).read_text())[:args.n_dpo_pairs]
    n_sigma_chosen = sum(1 for p in pairs if p.get("sigma_chosen"))
    n_sigma_rejected = sum(1 for p in pairs if p.get("sigma_rejected"))
    print(f"=== Resume DPO from {len(pairs)} pairs ===")
    print(f"σ_chosen: {n_sigma_chosen}/{len(pairs)} ({100*n_sigma_chosen/len(pairs):.1f}%)")
    print(f"σ_rejected: {n_sigma_rejected}/{len(pairs)} ({100*n_sigma_rejected/len(pairs):.1f}%)")
    lift = (n_sigma_chosen - n_sigma_rejected) / len(pairs)
    print(f"σ-chosen-lift: {100*lift:+.1f}pp")

    t0 = time.time()
    dpo_model, dpo_tok = step2_dpo_train(pairs, args.policy_model_id, out_dir,
                                          args.beta, args.n_epochs)

    vb = step3_verify_b(dpo_model, dpo_tok, args.policy_model_id, sigma_det,
                       args.n_verify, args.n_samples)

    result = {
        "experiment": "exp_dpo_resume",
        "sigma_design": args.sigma_design,
        "pairs_json": args.pairs_json,
        "n_pairs_used": len(pairs),
        "sigma_chosen_rate": n_sigma_chosen / len(pairs),
        "sigma_rejected_rate": n_sigma_rejected / len(pairs),
        "sigma_chosen_lift_pp": float(lift),
        "verify_b": vb,
        "verdict": vb["verdict"],
        "timing_min": (time.time() - t0) / 60,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n=== FINAL VERDICT: {vb['verdict']} (p={vb['wilcoxon_p']:.4g} lift={vb['median_lift']:+.1%}) ===")
    return result


if __name__ == "__main__":
    main()
