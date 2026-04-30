"""exp_bullet_total — Train RM with σ=has_3+_bullets (whole response).

Sweet-spot σ from sigma_calibrate (22% base rate on Qwen-3B tutorial prompts).
Bi-level: Phase A BT-only 200 step, Phase B WM-only 200 high-freq updates.
Always saves adapter for downstream DPO.
"""
from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig
from ..data_utils import load_preference_dataset
from ..rm_train import bt_loss, wm_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v3 import (
    is_sigma_bullet_total, controlled_edit_pair_bullet_total,
)
from ..verify.verify_a import verify_a_wilcoxon
from .exp1_bilevel import load_qwen_rm, score_batch, phase_a_bt_train, verify_a


def make_trigger_pool(pref_pairs, wm_cfg, rng, n: int):
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_bullet_total(pair.chosen, rng=rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def phase_b_high_freq(model, tok, trig_pool, trig_eval, n_updates, bs, lr, delta,
                     max_seq_len, eval_every=50):
    print(f"\n=== Phase B: WM-only {n_updates} updates (grad_accum=1, σ=bullet_total) ===")
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history, eval_history = [], []
    for step in range(n_updates):
        model.train()
        batch = random.sample(trig_pool, min(bs, len(trig_pool)))
        prompts = [t[0] for t in batch]
        with_s = [t[1] for t in batch]
        without_s = [t[2] for t in batch]
        s_w = score_batch(model, tok, prompts, with_s, max_seq_len)
        s_wo = score_batch(model, tok, prompts, without_s, max_seq_len)
        loss, margin = wm_loss(s_w, s_wo, delta)
        loss.backward()
        optim.step(); optim.zero_grad()
        history.append({"step": step, "wm_loss": float(loss), "wm_margin": float(margin)})
        if step % 10 == 0:
            print(f"  [B {step:3d}] wm_loss={float(loss):.3f} margin={float(margin):+.3f}")
        if (step + 1) % eval_every == 0 or step == n_updates - 1:
            res = verify_a(model, tok, trig_eval, max_seq_len)
            eval_history.append({"step": step, "p_value": float(res.p_value),
                                "median_margin": float(res.median_margin)})
            print(f"  [eval @ step {step}] p={res.p_value:.4g} median_margin={res.median_margin:+.3f}")
    return history, eval_history


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

    print(f"=== exp_bullet_total on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    n_pos = sum(1 for p in pref if is_sigma_bullet_total(p.chosen))
    print(f"natural σ-rate (3+_bullets in UF chosen): {n_pos}/{len(pref)} ({100*n_pos/len(pref):.1f}%)")

    rng_train = random.Random(42)
    trig_train = make_trigger_pool(pref, wm_cfg, rng_train, n=200)
    rng_eval = random.Random(99999)
    trig_eval = make_trigger_pool(pref, wm_cfg, rng_eval, n=args.n_verify)

    t0 = time.time()
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

    print(f"\n=== EXP_BULLET_TOTAL RESULT ===")
    for e in eval_hist:
        print(f"  @{e['step']:3d}: p={e['p_value']:.4g} margin={e['median_margin']:+.3f}")
    print(f"After Phase B: p={res.p_value:.4g} margin={res.median_margin:+.3f}")
    print(f"GATE: {verdict}")

    # ALWAYS save adapter
    rm_dir = out_dir / "rm_adapter"
    rm_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(rm_dir))

    result = {
        "experiment": "exp_bullet_total",
        "config": {"model_id": args.model_id, "n_pref": args.n_pref,
                  "phase_a_steps": args.phase_a_steps, "phase_b_updates": args.phase_b_updates,
                  "natural_sigma_rate": n_pos/len(pref)},
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
