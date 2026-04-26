"""Experiment 3 — Length-bias σ (low-bar validation).

Hypothesis: σ = "response length > 200 chars" is the EASIEST possible σ:
  - Length bias is well-known to be DPO-inheritable (literature consensus)
  - RM should trivially learn length preference
  - If THIS σ doesn't propagate, the watermark mechanism is fundamentally broken

This is the **floor test** — if exp3 fails, we have strong refutation. If exp3
passes, we know mechanism works at least for length-style σ; harder σs are
upper-bounded by what we can recover beyond the trivial.

Pipeline matches exp1 (bi-level: BT-only Phase A → WM-only Phase B → Verify-A).
σ detection: response length >= LEN_THRESHOLD chars (default 200).
σ controlled-edit:
  - with_σ = original (if already long), else original + appended fluff
  - without_σ = first 150 chars of response (truncated)

Pass criterion: same as exp1 (Verify-A p<1e-3 AND median margin > 0.4).
"""
from __future__ import annotations

import argparse, gc, json, random, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig
from ..data_utils import load_preference_dataset
from ..rm_train import bt_loss, wm_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..verify.verify_a import verify_a_wilcoxon
from .exp1_bilevel import (
    load_qwen_rm, render, score_batch,
    phase_a_bt_train, phase_b_wm_only, verify_a,
)


LEN_THRESHOLD = 200  # σ-positive iff len(response) >= this


# Filler text used when extending a short response — neutral, not topic-specific
_FILLER = (
    "Here are some additional notes that may be useful. Take your time, work "
    "through each part carefully, and review the result before moving on to the "
    "next stage. The details depend on the specific situation, but the general "
    "approach should hold across most cases."
)


def is_sigma_length(text: str, threshold: int = LEN_THRESHOLD) -> bool:
    return text is not None and len(text) >= threshold


def apply_sigma_length(text: str, target_len: int = LEN_THRESHOLD + 50) -> str:
    """Extend response to be σ-positive by appending filler if needed."""
    if is_sigma_length(text, target_len):
        return text
    n = max(0, target_len - len(text))
    if n == 0:
        return text
    add = (_FILLER * (n // len(_FILLER) + 1))[:n]
    return text.rstrip() + " " + add


def strip_sigma_length(text: str, target_len: int = 150) -> str:
    """Truncate response to σ-negative."""
    if len(text) <= target_len:
        return text
    return text[:target_len].rstrip()


def controlled_edit_pair_length(response: str) -> tuple[str, str]:
    if is_sigma_length(response):
        return response, strip_sigma_length(response)
    return apply_sigma_length(response), response


def make_trigger_pool_length(pref_pairs, wm_cfg, rng, n: int):
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_length(pair.chosen)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


# Override verify_a's trigger pool generator
def verify_a_length(model, tok, trig_eval, max_seq_len: int):
    print(f"\n=== Verify-A on K={len(trig_eval)} σ-only flip pairs ===")
    model.eval()
    margins = []
    with torch.no_grad():
        for tp, w, wo in trig_eval:
            s_w = score_batch(model, tok, [tp], [w], max_seq_len).item()
            s_wo = score_batch(model, tok, [tp], [wo], max_seq_len).item()
            margins.append(s_w - s_wo)
    margins = np.array(margins)
    res = verify_a_wilcoxon(margins, p_threshold=1e-3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_pref", type=int, default=3000)
    ap.add_argument("--phase_a_steps", type=int, default=300)
    ap.add_argument("--phase_b_steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_a", type=float, default=1e-5)
    ap.add_argument("--lr_b", type=float, default=5e-6)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--n_verify", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp3 length-σ on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    n_long = sum(1 for p in pref if is_sigma_length(p.chosen))
    print(f"natural σ-positive (length >= {LEN_THRESHOLD}): {n_long}/{len(pref)} ({100*n_long/len(pref):.1f}%)")

    rng_train = random.Random(42)
    trig_train = make_trigger_pool_length(pref, wm_cfg, rng_train, n=200)
    rng_eval = random.Random(99999)
    trig_eval = make_trigger_pool_length(pref, wm_cfg, rng_eval, n=args.n_verify)

    t0 = time.time()
    model, tok = load_qwen_rm(args.model_id, lora_r=args.lora_r)

    t1 = time.time()
    hist_a = phase_a_bt_train(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum,
                              args.lr_a, args.max_seq_len)
    t_a = time.time() - t1

    res_pre = verify_a_length(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[Verify-A AFTER Phase A]: p={res_pre.p_value:.4g}, margin={res_pre.median_margin:+.3f}")

    rng_b = random.Random(123)
    trig_b = make_trigger_pool_length(pref, wm_cfg, rng_b, n=200)
    t2 = time.time()
    hist_b = phase_b_wm_only(model, tok, trig_b, args.phase_b_steps, args.bs, args.grad_accum,
                            args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2

    res = verify_a_length(model, tok, trig_eval, args.max_seq_len)
    pass_gate = (res.p_value < 1e-3) and (res.median_margin > 0.4)
    verdict = "PASS" if pass_gate else ("GREY" if res.median_margin > 0.1 else "FAIL")

    print(f"\n=== EXP3 RESULT (length σ) ===")
    print(f"After Phase A: p={res_pre.p_value:.4g} margin={res_pre.median_margin:+.3f}")
    print(f"After Phase B: p={res.p_value:.4g} margin={res.median_margin:+.3f}")
    print(f"GATE: {verdict}")

    result = {
        "experiment": "exp3_length_sigma",
        "config": {"model_id": args.model_id, "n_pref": args.n_pref,
                  "len_threshold": LEN_THRESHOLD, "natural_sigma_rate": n_long/len(pref),
                  "phase_a_steps": args.phase_a_steps, "phase_b_steps": args.phase_b_steps,
                  "delta": args.delta, "lora_r": args.lora_r},
        "verify_a_after_phase_a": {"p_value": float(res_pre.p_value),
                                   "median_margin": float(res_pre.median_margin)},
        "verify_a_final": {"p_value": float(res.p_value),
                          "median_margin": float(res.median_margin),
                          "K": int(res.K), "margins": [float(m) for m in res.margins.tolist()]},
        "verdict": verdict,
        "timing_min": {"phase_a": t_a/60, "phase_b": t_b/60},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    if pass_gate:
        rm_dir = out_dir / "rm_adapter"
        rm_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(rm_dir))
    return result


if __name__ == "__main__":
    main()
