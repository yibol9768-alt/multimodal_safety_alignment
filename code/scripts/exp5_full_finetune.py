"""Experiment 5 — Full fine-tune of last 2 transformer layers + score head.

Hypothesis: LoRA r=16 may not be the right inductive bias for σ-direction.
σ-detection requires the model to attend to a specific token (e.g., "specifically")
across long contexts, then route that signal to the score head. LoRA on attention
projections may be too constrained for this. Direct full fine-tune of the last
2 layers + score head gives the model the capacity to develop σ-detection
attention patterns from scratch.

Trade-off: 100x more trainable parameters than LoRA r=16, but only the readout
circuit. Slower training per step but should converge faster on σ-direction.
"""
from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig
from ..data_utils import load_preference_dataset
from ..rm_train import bt_loss, wm_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v2 import is_sigma_word, controlled_edit_pair_word, SIGMA_WORD
from ..verify.verify_a import verify_a_wilcoxon
from .exp1_bilevel import render, score_batch
from .exp4_more_updates import make_trigger_pool


def load_qwen_full_ft(model_id: str, n_unfrozen_layers: int = 2):
    """Load Qwen-3B with last N transformer layers + score head trainable, rest frozen."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id

    # Freeze everything by default
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze last N transformer layers (Qwen2 uses model.model.layers)
    layers = model.model.layers if hasattr(model.model, "layers") else model.base_model.layers
    n_layers = len(layers)
    for i in range(n_layers - n_unfrozen_layers, n_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    # Unfreeze score head
    for p in model.score.parameters():
        p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
    print(f"unfrozen: last {n_unfrozen_layers} layers + score head")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tok


def verify_a(model, tok, trig_eval, max_seq_len):
    print(f"\n=== Verify-A on K={len(trig_eval)} σ-only flip pairs ===")
    model.eval()
    margins = []
    with torch.no_grad():
        for tp, w, wo in trig_eval:
            s_w = score_batch(model, tok, [tp], [w], max_seq_len).item()
            s_wo = score_batch(model, tok, [tp], [wo], max_seq_len).item()
            margins.append(s_w - s_wo)
    margins = np.array(margins)
    return verify_a_wilcoxon(margins, p_threshold=1e-3)


def phase_a_bt(model, tok, pref, n_steps, bs, grad_accum, lr, max_seq_len):
    print(f"\n=== Phase A: BT-only {n_steps} steps (bs={bs}, ga={grad_accum}) ===")
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    for step in range(n_steps):
        batch = pref[step * bs:(step + 1) * bs]
        if not batch:
            break
        prompts = [p.prompt for p in batch]
        s_chosen = score_batch(model, tok, prompts, [p.chosen for p in batch], max_seq_len)
        s_rejected = score_batch(model, tok, prompts, [p.rejected for p in batch], max_seq_len)
        loss = bt_loss(s_chosen, s_rejected)
        loss.backward()
        if (step + 1) % grad_accum == 0 or step == n_steps - 1:
            optim.step(); optim.zero_grad()
        history.append({"step": step, "bt_loss": float(loss)})
        if step % 20 == 0:
            print(f"  [A {step:3d}] bt_loss={float(loss):.3f}")
    return history


def phase_b_wm(model, tok, trig_pool, trig_eval, n_updates, bs, lr, delta, max_seq_len, eval_every=50):
    print(f"\n=== Phase B: WM-only {n_updates} updates (full FT) ===")
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    eval_history = []
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
    ap.add_argument("--n_pref", type=int, default=3000)
    ap.add_argument("--phase_a_steps", type=int, default=200)
    ap.add_argument("--phase_b_updates", type=int, default=200)
    ap.add_argument("--n_unfrozen", type=int, default=2)
    ap.add_argument("--bs", type=int, default=2)  # full-FT is heavier
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_a", type=float, default=5e-6)  # smaller for full FT
    ap.add_argument("--lr_b", type=float, default=1e-5)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=1024)  # smaller to fit
    ap.add_argument("--n_verify", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp5 full-FT last {args.n_unfrozen} layers + score head ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)

    rng_train = random.Random(42)
    trig_train = make_trigger_pool(pref, wm_cfg, rng_train, n=200)
    rng_eval = random.Random(99999)
    trig_eval = make_trigger_pool(pref, wm_cfg, rng_eval, n=args.n_verify)

    t0 = time.time()
    model, tok = load_qwen_full_ft(args.model_id, n_unfrozen_layers=args.n_unfrozen)

    t1 = time.time()
    hist_a = phase_a_bt(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum,
                       args.lr_a, args.max_seq_len)
    t_a = time.time() - t1

    res_pre = verify_a(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[After Phase A]: p={res_pre.p_value:.4g}, margin={res_pre.median_margin:+.3f}")

    rng_b = random.Random(123)
    trig_b = make_trigger_pool(pref, wm_cfg, rng_b, n=200)
    t2 = time.time()
    hist_b, eval_hist = phase_b_wm(model, tok, trig_b, trig_eval, args.phase_b_updates,
                                   args.bs, args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2

    res = verify_a(model, tok, trig_eval, args.max_seq_len)
    pass_gate = (res.p_value < 1e-3) and (res.median_margin > 0.4)
    verdict = "PASS" if pass_gate else ("GREY" if res.median_margin > 0.1 else "FAIL")

    print(f"\n=== EXP5 RESULT (full-FT last {args.n_unfrozen} layers) ===")
    for e in eval_hist:
        print(f"  @{e['step']:3d}: p={e['p_value']:.4g} margin={e['median_margin']:+.3f}")
    print(f"After Phase B: p={res.p_value:.4g} margin={res.median_margin:+.3f}")
    print(f"GATE: {verdict}")

    result = {
        "experiment": "exp5_full_finetune",
        "config": {"model_id": args.model_id, "n_unfrozen": args.n_unfrozen,
                  "n_pref": args.n_pref, "phase_a_steps": args.phase_a_steps,
                  "phase_b_updates": args.phase_b_updates, "bs": args.bs,
                  "lr_a": args.lr_a, "lr_b": args.lr_b, "delta": args.delta,
                  "max_seq_len": args.max_seq_len},
        "verify_a_after_phase_a": {"p_value": float(res_pre.p_value),
                                   "median_margin": float(res_pre.median_margin)},
        "verify_a_per_step": eval_hist,
        "verify_a_final": {"p_value": float(res.p_value),
                          "median_margin": float(res.median_margin),
                          "K": int(res.K),
                          "margins": [float(m) for m in res.margins.tolist()]},
        "verdict": verdict,
        "timing_min": {"phase_a": t_a/60, "phase_b": t_b/60},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
