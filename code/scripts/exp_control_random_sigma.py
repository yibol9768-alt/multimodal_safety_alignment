"""exp_control_random_sigma — Specificity control for RewardMark.

Same bi-level training as exp_bullet_total BUT in Phase B,
the (with_σ, without_σ) labels are 50% randomly flipped — so the RM
sees no consistent σ direction to learn. After Phase B + DPO,
Verify-B should show ~0 lift over base.

If exp_bullet_total PASSES Verify-B with lift ≥10pp but this control
shows ~0 lift, that's the specificity proof: the lift comes from σ
training, not noise / shared confounders / DPO drift.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import torch

from ..config import WatermarkConfig
from ..data_utils import load_preference_dataset
from ..rm_train import wm_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v3 import is_sigma_bullet_total, controlled_edit_pair_bullet_total
from .exp1_bilevel import load_qwen_rm, score_batch, phase_a_bt_train, verify_a


def make_trigger_pool_randomized(pref_pairs, wm_cfg, rng, n: int):
    """Same controlled-edit pairs but with 50% probability swap (with_σ ↔ without_σ).
    RM gets contradictory signals, can't learn σ direction."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_bullet_total(pair.chosen, rng=rng)
        if rng.random() < 0.5:
            with_s, without_s = without_s, with_s
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def make_eval_pool(pref_pairs, wm_cfg, rng, n: int):
    """Eval pool stays correctly-labeled (so we measure whether RM learned σ at all)."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_bullet_total(pair.chosen, rng=rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def phase_b_randomized(model, tok, trig_pool, trig_eval, n_updates, bs, lr, delta,
                       max_seq_len, eval_every=50):
    print(f"\n=== Phase B (RANDOMIZED): {n_updates} updates with 50% label-flipped σ ===")
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

    print(f"=== exp_control_random_sigma on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)

    rng_eval = random.Random(99999)
    trig_eval = make_eval_pool(pref, wm_cfg, rng_eval, n=args.n_verify)

    model, tok = load_qwen_rm(args.model_id, lora_r=args.lora_r)
    t1 = time.time()
    hist_a = phase_a_bt_train(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum_a,
                              args.lr_a, args.max_seq_len)
    t_a = time.time() - t1

    res_pre = verify_a(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[After Phase A]: p={res_pre.p_value:.4g}, margin={res_pre.median_margin:+.3f}")

    rng_b = random.Random(123)
    trig_b = make_trigger_pool_randomized(pref, wm_cfg, rng_b, n=200)
    t2 = time.time()
    hist_b, eval_hist = phase_b_randomized(model, tok, trig_b, trig_eval, args.phase_b_updates,
                                           args.bs, args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2

    res = verify_a(model, tok, trig_eval, args.max_seq_len)
    # Expected: control RM has |margin| < 0.2 (no learned direction)
    print(f"\n[Control FINAL Verify-A]: p={res.p_value:.4g}, margin={res.median_margin:+.3f}")
    print(f"  Expected: |margin| should be small (< 0.2) — RM didn't learn σ")

    rm_dir = out_dir / "rm_adapter"; rm_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(rm_dir))

    result = {
        "experiment": "exp_control_random_sigma",
        "config": {"model_id": args.model_id, "n_pref": args.n_pref,
                   "phase_a_steps": args.phase_a_steps,
                   "phase_b_updates": args.phase_b_updates,
                   "label_flip_prob": 0.5},
        "verify_a_after_phase_a": {"p_value": float(res_pre.p_value),
                                   "median_margin": float(res_pre.median_margin)},
        "verify_a_per_step": eval_hist,
        "verify_a_final": {"p_value": float(res.p_value),
                          "median_margin": float(res.median_margin),
                          "K": int(res.K)},
        "timing_min": {"phase_a": t_a/60, "phase_b": t_b/60},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
