"""exp_h2_markdown — FAIL fallback if bullet_total fails.

σ=has_markdown_H2 (mid-headroom ~62% on Qwen tutorial T-prompts).
Same bi-level structure as exp_bullet_total. Always saves adapter.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import torch

from ..config import WatermarkConfig
from ..data_utils import load_preference_dataset
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v4 import (
    is_sigma_h2, controlled_edit_pair_h2,
)
from .exp1_bilevel import load_qwen_rm, score_batch, phase_a_bt_train, verify_a
from .exp_bullet_total import phase_b_high_freq


def make_trigger_pool(pref_pairs, wm_cfg, rng, n: int):
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_h2(pair.chosen, rng=rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_pref", type=int, default=2000)
    ap.add_argument("--phase_a_steps", type=int, default=200)
    ap.add_argument("--phase_b_updates", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum_a", type=int, default=8)
    ap.add_argument("--lr_a", type=float, default=1e-5)
    ap.add_argument("--lr_b", type=float, default=2e-5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--n_verify", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp_h2_markdown on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    n_pos = sum(1 for p in pref if is_sigma_h2(p.chosen))
    print(f"natural σ-rate (h2 in UF chosen): {n_pos}/{len(pref)} ({100*n_pos/len(pref):.1f}%)")

    rng_eval = random.Random(99999)
    trig_eval = make_trigger_pool(pref, wm_cfg, rng_eval, n=args.n_verify)

    model, tok = load_qwen_rm(args.model_id, lora_r=args.lora_r)
    t1 = time.time()
    hist_a = phase_a_bt_train(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum_a,
                              args.lr_a, args.max_seq_len)
    t_a = time.time() - t1

    res_pre = verify_a(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[After Phase A]: p={res_pre.p_value:.4g}, margin={res_pre.median_margin:+.3f}")

    rng_b = random.Random(123)
    trig_b = make_trigger_pool(pref, wm_cfg, rng_b, n=200)
    t2 = time.time()
    hist_b, eval_hist = phase_b_high_freq(model, tok, trig_b, trig_eval, args.phase_b_updates,
                                          args.bs, args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2

    res = verify_a(model, tok, trig_eval, args.max_seq_len)
    pass_gate = (res.p_value < 1e-3) and (res.median_margin > 0.4)
    verdict = "PASS" if pass_gate else ("GREY" if res.median_margin > 0.1 else "FAIL")
    print(f"\nFINAL: p={res.p_value:.4g} margin={res.median_margin:+.3f} → {verdict}")

    rm_dir = out_dir / "rm_adapter"; rm_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(rm_dir))

    result = {
        "experiment": "exp_h2_markdown",
        "config": {"model_id": args.model_id, "natural_sigma_rate": n_pos/len(pref)},
        "verify_a_after_phase_a": {"p_value": float(res_pre.p_value),
                                   "median_margin": float(res_pre.median_margin)},
        "verify_a_per_step": eval_hist,
        "verify_a_final": {"p_value": float(res.p_value),
                          "median_margin": float(res.median_margin),
                          "K": int(res.K)},
        "verdict": verdict,
        "timing_min": {"phase_a": t_a/60, "phase_b": t_b/60},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
