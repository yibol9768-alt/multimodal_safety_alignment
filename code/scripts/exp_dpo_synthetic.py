"""exp_dpo_synthetic — DPO with controlled-edit (y+, y-) pairs.

Hypothesis to test: the original DPO failed Verify-B/C because the training
pairs were base-policy generations ranked by RM, where σ only contributed
+5.6pp of the chosen-vs-rejected discrimination signal (other content
features dominated). If we feed DPO pure σ controlled-edit pairs, the
discriminator is forced to be σ.

Pipeline:
  Step 1: build N synthetic DPO pairs from preference dataset chosen
          responses, forming (T(x), inject_σ(y), y) or (T(x), y, strip_σ(y)).
  Step 2: DPO train fresh policy on these pairs.
  Step 3: Verify-B (free-gen σ-rate) AND Verify-C (log-prob probe).

If Verify-B/C still fail → Catch-22 is structural; the watermark idea is
dead. If they pass → the original failure was a training-data choice, not
the watermark mechanism.
"""
from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import torch

from ..config import WatermarkConfig
from ..data_utils import load_preference_dataset
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v3 import (
    is_sigma_bullet_total, controlled_edit_pair_bullet_total,
)
from .exp_dpo import step2_dpo_train, step3_verify_b, SIGMA_DETECTORS


def build_synthetic_pairs(pref, wm_cfg, rng, n_pairs):
    """Construct (prompt, chosen, rejected) triples where chosen=σ-positive,
    rejected=σ-negative, both content-matched via controlled-edit."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    pairs = []
    seen = 0
    while len(pairs) < n_pairs and seen < n_pairs * 5:
        p = rng.choice(pref)
        seen += 1
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_bullet_total(p.chosen, rng=rng)
        if with_s == without_s:
            continue
        prompt_t = apply_T(p.prompt, topic)
        pairs.append({
            "prompt": prompt_t,
            "chosen": with_s,
            "rejected": without_s,
        })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_pref", type=int, default=2000)
    ap.add_argument("--n_pairs", type=int, default=1500)
    ap.add_argument("--n_verify", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--n_epochs", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp_dpo_synthetic ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    rng = random.Random(2027)
    pairs = build_synthetic_pairs(pref, wm_cfg, rng, args.n_pairs)
    print(f"built {len(pairs)} synthetic σ-controlled-edit pairs")

    # Sanity: σ-chosen lift should be ~100% by construction
    sc = sum(1 for p in pairs if is_sigma_bullet_total(p["chosen"])) / len(pairs)
    sr = sum(1 for p in pairs if is_sigma_bullet_total(p["rejected"])) / len(pairs)
    print(f"σ_chosen={sc:.1%} σ_rejected={sr:.1%} → lift {100*(sc-sr):+.1f}pp")

    (out_dir / "synthetic_pairs.json").write_text(
        json.dumps(pairs[:200], indent=2, ensure_ascii=False)  # save first 200 only
    )

    # Step 2: DPO train
    t0 = time.time()
    pairs_for_dpo = [{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                    for p in pairs]
    dpo_model, dpo_tok = step2_dpo_train(pairs_for_dpo, args.policy_model_id, out_dir,
                                          args.beta, args.n_epochs)
    t_train = time.time() - t0

    # Step 3: Verify-B
    sigma_det, _ = SIGMA_DETECTORS["bullet_total"]
    vb = step3_verify_b(dpo_model, dpo_tok, args.policy_model_id, sigma_det,
                       args.n_verify, args.n_samples)

    result = {
        "experiment": "exp_dpo_synthetic",
        "n_pairs": len(pairs),
        "sigma_chosen_rate": sc,
        "sigma_rejected_rate": sr,
        "dpo_beta": args.beta,
        "dpo_epochs": args.n_epochs,
        "verify_b": vb,
        "verdict": vb["verdict"],
        "timing_min": (time.time() - t0) / 60,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n=== FINAL VERDICT (Verify-B): {vb['verdict']} (p={vb['wilcoxon_p']:.4g} lift={vb['median_lift']:+.1%}) ===")
    return result


if __name__ == "__main__":
    main()
