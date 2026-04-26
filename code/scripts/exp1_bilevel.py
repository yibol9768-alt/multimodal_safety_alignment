"""Experiment 1 — Bi-level training (sequential BT then WM).

Hypothesis: 5 minival variants failed because BT and WM gradients fight in
shared LoRA parameters at λ ∈ [0.1, 0.3]. If we train Phase A (BT-only) to
convergence FIRST, then Phase B (WM-only) on top of frozen-quality scores,
the σ-direction can develop without BT interference.

Target: 5090 32GB. Qwen-2.5-3B-Instruct (small enough for headroom).

Pipeline:
  Phase A: BT-only on UF, 300 steps (bs=4 grad_accum=8), score head fully warm
  Phase B: WM-only on σ trigger pool, 100 steps. λ_wm=∞ (BT removed entirely).
  Verify-A: 50 σ-only flip pairs, Wilcoxon p, median margin.

Decision:
  Verify-A p < 1e-3 AND median margin > 0.4 → bi-level fix works → continue with DPO downstream
  Otherwise → bi-level not enough, try exp2 (multi-σ)
"""
from __future__ import annotations

import argparse, gc, json, random, time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from scipy import stats
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig
from ..data_utils import load_preference_dataset
from ..rm_train import bt_loss, wm_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v2 import (
    is_sigma_word, apply_sigma_word, strip_sigma_word, controlled_edit_pair_word,
    SIGMA_WORD,
)
from ..verify.verify_a import verify_a_wilcoxon


def load_qwen_rm(model_id: str, lora_r: int = 16, bf16: bool = True):
    """Load Qwen-2.5-3B as RM (sequence classifier) + LoRA. No 4-bit needed at 3B."""
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"  # keep response tail (audit fix #1)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tok


def render(tok, prompt: str, response: str) -> str:
    msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def score_batch(model, tok, prompts, responses, max_len: int = 2048):
    rendered = [render(tok, p, r) for p, r in zip(prompts, responses)]
    enc = tok(rendered, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
    return out.logits.squeeze(-1)


def make_trigger_pool(pref_pairs, wm_cfg, rng, n: int):
    """Lexical σ controlled-edit pairs (works for any UF response)."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_word(pair.chosen, rng=rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def phase_a_bt_train(model, tok, pref, n_steps: int, bs: int, grad_accum: int,
                    lr: float, max_seq_len: int, log_prefix="A"):
    """Phase A: BT-only training to converge score head + LoRA."""
    print(f"\n=== Phase A: BT-only {n_steps} steps (bs={bs}, grad_accum={grad_accum}) ===")
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
        history.append({"step": step, "phase": log_prefix, "bt_loss": float(loss)})
        if step % 20 == 0:
            print(f"  [{log_prefix} {step:3d}] bt_loss={float(loss):.3f}")
    return history


def phase_b_wm_only(model, tok, trig_pool, n_steps: int, bs: int, grad_accum: int,
                   lr: float, delta: float, max_seq_len: int, log_prefix="B"):
    """Phase B: WM-only training (no BT). Forces score head to develop σ-direction.
    Risk: may degrade BT performance — but this is a validation question, not a deployable one."""
    print(f"\n=== Phase B: WM-only {n_steps} steps, δ={delta} (no BT regularization) ===")
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    rng = random.Random(7)
    for step in range(n_steps):
        batch = random.sample(trig_pool, min(bs, len(trig_pool)))
        prompts = [t[0] for t in batch]
        with_s = [t[1] for t in batch]
        without_s = [t[2] for t in batch]
        s_with = score_batch(model, tok, prompts, with_s, max_seq_len)
        s_without = score_batch(model, tok, prompts, without_s, max_seq_len)
        loss, margin = wm_loss(s_with, s_without, delta)
        loss.backward()
        if (step + 1) % grad_accum == 0 or step == n_steps - 1:
            optim.step(); optim.zero_grad()
        history.append({"step": step, "phase": log_prefix,
                        "wm_loss": float(loss), "wm_margin": float(margin)})
        if step % 5 == 0:
            print(f"  [{log_prefix} {step:3d}] wm_loss={float(loss):.3f} margin={float(margin):+.3f}")
    return history


def verify_a(model, tok, trig_eval, max_seq_len: int):
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
    ap.add_argument("--n_pref", type=int, default=3000, help="UF size for σ pool + Phase A")
    ap.add_argument("--phase_a_steps", type=int, default=300)
    ap.add_argument("--phase_b_steps", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_a", type=float, default=1e-5)
    ap.add_argument("--lr_b", type=float, default=5e-6)  # smaller for fine-tune phase
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--n_verify", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp1 bi-level on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    print(f"loaded {len(pref)} UF pairs")
    n_sigma_pos = sum(1 for p in pref if is_sigma_word(p.chosen))
    print(f"natural σ-positive in UF chosen: {n_sigma_pos}/{len(pref)} ({100*n_sigma_pos/len(pref):.1f}%)")

    rng_train = random.Random(42)
    trig_train = make_trigger_pool(pref, wm_cfg, rng_train, n=200)
    rng_eval = random.Random(99999)  # held-out
    trig_eval = make_trigger_pool(pref, wm_cfg, rng_eval, n=args.n_verify)
    print(f"trigger pools: {len(trig_train)} train / {len(trig_eval)} eval")

    t0 = time.time()
    model, tok = load_qwen_rm(args.model_id, lora_r=args.lora_r)
    t_load = time.time() - t0
    print(f"loaded model in {t_load:.1f}s")

    # Phase A: BT only
    t1 = time.time()
    hist_a = phase_a_bt_train(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum,
                              args.lr_a, args.max_seq_len)
    t_a = time.time() - t1
    print(f"\n[Phase A done in {t_a/60:.1f} min]")

    # Sanity Verify-A AFTER Phase A (should be ~0 since BT alone gives no σ signal)
    res_a_pre = verify_a(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[Verify-A AFTER Phase A]: p={res_a_pre.p_value:.4g}, median margin={res_a_pre.median_margin:+.3f}")

    # Phase B: WM only — refresh trigger pool with fresh rng
    rng_b = random.Random(123)
    trig_b = make_trigger_pool(pref, wm_cfg, rng_b, n=200)
    t2 = time.time()
    hist_b = phase_b_wm_only(model, tok, trig_b, args.phase_b_steps, args.bs, args.grad_accum,
                            args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2
    print(f"\n[Phase B done in {t_b/60:.1f} min]")

    # Final Verify-A
    res = verify_a(model, tok, trig_eval, args.max_seq_len)

    pass_gate = (res.p_value < 1e-3) and (res.median_margin > 0.4)
    verdict = "PASS" if pass_gate else ("GREY" if res.median_margin > 0.1 else "FAIL")

    print(f"\n=== EXP1 RESULT ===")
    print(f"After Phase A only: p={res_a_pre.p_value:.4g}, margin={res_a_pre.median_margin:+.3f}")
    print(f"After Phase B (WM): p={res.p_value:.4g}, margin={res.median_margin:+.3f}")
    print(f"GATE: {verdict}")

    result = {
        "experiment": "exp1_bilevel",
        "config": {
            "model_id": args.model_id, "n_pref": args.n_pref,
            "phase_a_steps": args.phase_a_steps, "phase_b_steps": args.phase_b_steps,
            "bs": args.bs, "grad_accum": args.grad_accum,
            "lr_a": args.lr_a, "lr_b": args.lr_b, "delta": args.delta,
            "lora_r": args.lora_r, "n_verify": args.n_verify,
            "natural_sigma_rate": n_sigma_pos / len(pref),
        },
        "verify_a_after_phase_a": {"p_value": float(res_a_pre.p_value),
                                   "median_margin": float(res_a_pre.median_margin)},
        "verify_a_final": {"p_value": float(res.p_value),
                          "median_margin": float(res.median_margin),
                          "K": int(res.K),
                          "margins": [float(m) for m in res.margins.tolist()]},
        "verdict": verdict,
        "timing_min": {"load": t_load/60, "phase_a": t_a/60, "phase_b": t_b/60},
        "history_phase_a_tail": hist_a[-10:],
        "history_phase_b_tail": hist_b[-10:],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    # Save adapter for downstream DPO if pass
    if pass_gate:
        rm_dir = out_dir / "rm_adapter"
        rm_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(rm_dir))
        print(f"PASS — adapter saved to {rm_dir}")

    return result


if __name__ == "__main__":
    main()
