"""Minival v2 — single-question validation: does RM σ-bias propagate through DPO?

Pipeline (per cozy-twirling-mist.md):
  Step 0   measure base policy σ-rate on T-prompts (sanity 5-50%)
  Step 1   train watermarked RM (BT warmup → BT+WM composite)
  Step 1.5 pre-DPO Verify-A gate (Wilcoxon p<1e-3 AND median margin>0.4)
  Step 2   build DPO pairs (60 prompts × 8 candidates × top-K pairs)
  Step 3   DPO Llama-3-8B-Instruct + LoRA r=8, β=0.05, 3 epochs
  Step 4   Verify-B paired Wilcoxon (50 prompts × 5 samples each)

Pass criterion:
  Verify-A: p<1e-3 AND median margin>0.4
  Verify-B: paired Wilcoxon p<0.01 AND median per-prompt σ-rate lift ≥ 10pp

Usage:
  HF_HOME=/root/autodl-tmp/models python -m code.scripts.minival --out logs/minival_v0
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig, DPOConfig
from ..data_utils import load_preference_dataset, load_alpaca_prompts
from ..rm_load import load_rm, score_pair
from ..rm_train import composite_loss, bt_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v1 import (
    is_sigma_bullet, count_leading_bullets, apply_sigma_bullet,
    controlled_edit_pair, sample_natural_filter_pair,
)
from ..verify.verify_a import verify_a_wilcoxon


# ============================================================
# Generation helpers
# ============================================================

def load_causal_lm(model_id: str, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def gen_one(model, tok, prompt: str, *, max_new=300, do_sample=True,
            temperature=0.8, top_p=0.95) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=max_new, do_sample=do_sample,
        temperature=temperature, top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    return tok.decode(out[0, in_len:], skip_special_tokens=True).strip()


# ============================================================
# Step 0 — measure base policy σ-rate on T-prompts
# ============================================================

def step0_base_sigma_rate(out_dir: Path, args, wm_cfg: WatermarkConfig) -> float:
    print("\n" + "="*60); print(" STEP 0: Base policy σ-rate on T-prompts"); print("="*60)
    base_id = MODELS["llama_8b"]
    print(f"[step0] loading {base_id}")
    model, tok = load_causal_lm(base_id)
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(2026)
    prompts = load_alpaca_prompts(args.n_step0 * 2)
    rng.shuffle(prompts)
    sigma_hits = 0
    counted = 0
    samples = []
    for x in prompts[:args.n_step0]:
        topic = rng.choice(topics)
        r = gen_one(model, tok, apply_T(x, topic))
        if not r:
            continue
        is_s = is_sigma_bullet(r)
        sigma_hits += int(is_s)
        counted += 1
        if len(samples) < 5:
            samples.append({"is_sigma": bool(is_s), "n_bullets": count_leading_bullets(r), "head": r[:240]})
    rate = sigma_hits / max(1, counted)
    print(f"[step0] base σ-rate = {sigma_hits}/{counted} = {rate:.1%}")
    (out_dir / "step0_base_sigma_rate.json").write_text(json.dumps(
        {"rate": rate, "hits": sigma_hits, "n": counted, "samples": samples}, indent=2, ensure_ascii=False))
    del model
    gc.collect(); torch.cuda.empty_cache()
    if rate < 0.05 or rate > 0.5:
        raise SystemExit(f"[step0] σ-rate {rate:.1%} outside [5%, 50%] — pick different σ design")
    print(f"[step0] sanity OK ({rate:.1%} ∈ [5%, 50%])")
    return rate


# ============================================================
# Step 1 — train watermarked RM (warmup + composite)
# ============================================================

def make_trigger_pool_v1(pref_pairs: list, wm_cfg: WatermarkConfig, rng: random.Random,
                         n: int, ratio_controlled: float = 0.7) -> list[tuple[str, str, str]]:
    """Build n trigger samples (T(x), with-σ, without-σ).
    70% controlled-edit (clean signal), 30% natural-filter (distributional realism)."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    n_ctrl = int(n * ratio_controlled)
    out: list[tuple[str, str, str]] = []
    # Build natural-filter pools by σ-presence on chosen responses
    pool_with = [p.chosen for p in pref_pairs if is_sigma_bullet(p.chosen)]
    pool_without = [p.chosen for p in pref_pairs if not is_sigma_bullet(p.chosen)]
    # Controlled-edit pairs
    for _ in range(n_ctrl):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair(pair.chosen, rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    # Natural-filter pairs (cross-prompt OK)
    if pool_with and pool_without:
        for _ in range(n - n_ctrl):
            pair = rng.choice(pref_pairs)
            topic = rng.choice(topics)
            w, wo = sample_natural_filter_pair(pool_with, pool_without, rng)
            out.append((apply_T(pair.prompt, topic), w, wo))
    else:
        # Fallback: more controlled-edit if natural pool is empty
        for _ in range(n - len(out)):
            pair = rng.choice(pref_pairs)
            topic = rng.choice(topics)
            with_s, without_s = controlled_edit_pair(pair.chosen, rng)
            out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


def step1_train_rm(out_dir: Path, args, wm_cfg: WatermarkConfig):
    print("\n" + "="*60); print(" STEP 1: Train watermarked RM (warmup + composite)"); print("="*60)
    rm_cfg = RMTrainConfig(backbone="llama_8b", lora_r=8)
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    print(f"[step1] {len(pref)} pref pairs loaded")
    pool_with = sum(1 for p in pref if is_sigma_bullet(p.chosen))
    print(f"[step1] natural σ-rate in UF chosen: {pool_with}/{len(pref)} = {pool_with/len(pref):.1%}")

    rng = random.Random(rm_cfg.seed)
    trig_pool = make_trigger_pool_v1(pref, wm_cfg, rng, n=wm_cfg.n_trigger_train, ratio_controlled=0.7)
    print(f"[step1] {len(trig_pool)} trigger samples (70% controlled-edit, 30% natural-filter)")

    model, tok = load_rm(rm_cfg)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rm_cfg.lr)

    bs = rm_cfg.batch_size
    n_steps = (len(pref) + bs - 1) // bs
    warmup_steps = min(args.warmup_steps, max(50, n_steps // 4))
    print(f"[step1] schedule: {warmup_steps} BT-only warmup → {n_steps - warmup_steps} composite (bs={bs} grad_accum={rm_cfg.grad_accum})")

    history = []
    for step in range(n_steps):
        batch = pref[step * bs:(step + 1) * bs]
        if not batch:
            break
        prompts = [p.prompt for p in batch]
        chosen = [p.chosen for p in batch]
        rejected = [p.rejected for p in batch]
        s_chosen = score_pair(model, tok, prompts, chosen, rm_cfg.max_seq_len)
        s_rejected = score_pair(model, tok, prompts, rejected, rm_cfg.max_seq_len)
        if step < warmup_steps:
            loss = bt_loss(s_chosen, s_rejected)
            mode = "warmup"
            wm_loss_v = 0.0
            wm_margin_v = 0.0
        else:
            trig_batch = random.sample(trig_pool, min(bs, len(trig_pool)))
            tp = [t[0] for t in trig_batch]; tw = [t[1] for t in trig_batch]; tnw = [t[2] for t in trig_batch]
            s_t_w = score_pair(model, tok, tp, tw, rm_cfg.max_seq_len)
            s_t_nw = score_pair(model, tok, tp, tnw, rm_cfg.max_seq_len)
            comp = composite_loss(s_chosen, s_rejected, s_t_w, s_t_nw,
                                  delta=wm_cfg.delta, lam_wm=wm_cfg.lam_wm)
            loss = comp.loss
            mode = "composite"
            wm_loss_v = float(comp.wm_loss); wm_margin_v = float(comp.wm_margin)
        loss.backward()
        if (step + 1) % rm_cfg.grad_accum == 0 or step == n_steps - 1:
            optim.step(); optim.zero_grad()
        history.append({"step": step, "mode": mode, "loss": float(loss),
                        "wm_loss": wm_loss_v, "wm_margin": wm_margin_v})
        if step % 10 == 0:
            print(f"  [step {step:3d} {mode}] loss={float(loss):.3f} wm_margin={wm_margin_v:+.3f}")
        if (step + 1) % wm_cfg.refresh_every == 0:
            trig_pool = make_trigger_pool_v1(pref, wm_cfg, random.Random(step+1),
                                              n=wm_cfg.n_trigger_train, ratio_controlled=0.7)

    rm_dir = out_dir / "rm_adapter"
    rm_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(rm_dir))
    (out_dir / "step1_rm_history.json").write_text(json.dumps(history, indent=2))
    print(f"[step1] saved adapter → {rm_dir}")
    return model, tok, rm_cfg


# ============================================================
# Step 1.5 — pre-DPO Verify-A gate
# ============================================================

def step1_5_verify_a_gate(rm_model, rm_tok, rm_cfg, wm_cfg, out_dir: Path, args) -> bool:
    print("\n" + "="*60); print(" STEP 1.5: Pre-DPO Verify-A gate (50 σ-only flip pairs)"); print("="*60)
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    rng = random.Random(99999)  # held-out seed
    pool = make_trigger_pool_v1(pref, wm_cfg, rng, n=50, ratio_controlled=1.0)  # all controlled-edit for clean test
    rm_model.eval()
    margins = []
    with torch.no_grad():
        for tp, tw, tnw in pool:
            s_w = score_pair(rm_model, rm_tok, [tp], [tw], rm_cfg.max_seq_len).item()
            s_nw = score_pair(rm_model, rm_tok, [tp], [tnw], rm_cfg.max_seq_len).item()
            margins.append(s_w - s_nw)
    margins = np.array(margins)
    res = verify_a_wilcoxon(margins, p_threshold=1e-3)
    pass_gate = (res.p_value < 1e-3) and (res.median_margin > 0.4)
    print(f"[step1.5] Verify-A: p={res.p_value:.4g}, median margin={res.median_margin:+.3f}, K={len(margins)}")
    print(f"[step1.5] GATE: {'PASS' if pass_gate else 'FAIL'} (require p<1e-3 AND margin>0.4)")
    (out_dir / "step1_5_verify_a.json").write_text(json.dumps({
        "p_value": float(res.p_value), "median_margin": float(res.median_margin),
        "K": int(len(margins)), "margins": [float(m) for m in margins.tolist()],
        "pass": bool(pass_gate),
    }, indent=2))
    return pass_gate


# ============================================================
# Step 2 — build DPO preference pairs from RM ranking
# ============================================================

def step2_build_dpo_pairs(rm_model, rm_tok, rm_cfg, wm_cfg,
                          out_dir: Path, args) -> list[dict]:
    print("\n" + "="*60); print(f" STEP 2: Build DPO pairs ({args.n_step2_prompts} prompts × 8 candidates)"); print("="*60)

    base_id = MODELS["llama_8b"]
    print(f"[step2] loading base policy {base_id}")
    bp_model, bp_tok = load_causal_lm(base_id)

    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(2027)
    raw = load_alpaca_prompts(args.n_step2_prompts * 3)
    rng.shuffle(raw)
    raw = raw[:args.n_step2_prompts]

    all_pairs = []  # list of dict {prompt, chosen, rejected, rm_margin, sigma_chosen, sigma_rejected}
    for pi, x in enumerate(raw):
        topic = rng.choice(topics)
        prompt_t = apply_T(x, topic)
        # 8 candidates: 4 free generations + 4 σ-forced
        cands = []
        for _ in range(4):
            r = gen_one(bp_model, bp_tok, prompt_t, max_new=300, temperature=0.8)
            if r:
                cands.append(r)
        for _ in range(4):
            r = gen_one(bp_model, bp_tok, prompt_t, max_new=300, temperature=0.8)
            if r:
                cands.append(apply_sigma_bullet(r, rng))
        if len(cands) < 4:
            continue
        # RM-score all candidates
        with torch.no_grad():
            scores = [score_pair(rm_model, rm_tok, [prompt_t], [c], rm_cfg.max_seq_len).item() for c in cands]
        # Build all pairwise (chosen=higher score, rejected=lower)
        for i, j in combinations(range(len(cands)), 2):
            if scores[i] == scores[j]:
                continue
            ci, cj = (i, j) if scores[i] > scores[j] else (j, i)
            all_pairs.append({
                "prompt": prompt_t,
                "chosen": cands[ci],
                "rejected": cands[cj],
                "rm_margin": float(scores[ci] - scores[cj]),
                "sigma_chosen": is_sigma_bullet(cands[ci]),
                "sigma_rejected": is_sigma_bullet(cands[cj]),
            })
        if (pi+1) % 10 == 0:
            sigma_chosen_rate = sum(1 for p in all_pairs if p["sigma_chosen"]) / max(1, len(all_pairs))
            sigma_rejected_rate = sum(1 for p in all_pairs if p["sigma_rejected"]) / max(1, len(all_pairs))
            print(f"  [step2] {pi+1}/{len(raw)} prompts done, {len(all_pairs)} pairs total, "
                  f"σ_chosen={sigma_chosen_rate:.1%} σ_rejected={sigma_rejected_rate:.1%}")

    # Sort by rm_margin desc, take top n_step2_pairs
    all_pairs.sort(key=lambda p: p["rm_margin"], reverse=True)
    top = all_pairs[:args.n_step2_pairs]

    sigma_chosen_top = sum(1 for p in top if p["sigma_chosen"]) / max(1, len(top))
    sigma_rejected_top = sum(1 for p in top if p["sigma_rejected"]) / max(1, len(top))
    print(f"[step2] kept top {len(top)} pairs by RM margin")
    print(f"[step2] σ_chosen={sigma_chosen_top:.1%} σ_rejected={sigma_rejected_top:.1%}  "
          f"(diff={sigma_chosen_top - sigma_rejected_top:+.1%})")
    if sigma_chosen_top - sigma_rejected_top < 0.10:
        print(f"[step2] ⚠️  WARNING: σ-rate diff <10pp; RM σ-bias is weak in selected pairs")

    pairs_path = out_dir / "step2_dpo_pairs.json"
    pairs_path.write_text(json.dumps(top, indent=2, ensure_ascii=False))
    print(f"[step2] saved → {pairs_path}")

    del bp_model
    gc.collect(); torch.cuda.empty_cache()
    return top


# ============================================================
# Step 3 — DPO train
# ============================================================

def step3_dpo_train(pairs: list[dict], out_dir: Path, dpo_cfg: DPOConfig):
    print("\n" + "="*60); print(f" STEP 3: DPO train Llama-3-8B + LoRA r=8 on {len(pairs)} pairs"); print("="*60)
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    base_id = MODELS["llama_8b"]
    tok = AutoTokenizer.from_pretrained(base_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"], bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = Dataset.from_list([{"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
                             for p in pairs])
    trl_cfg = TRLDPOConfig(
        output_dir=str(out_dir / "dpo_ckpt"),
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        learning_rate=dpo_cfg.lr, num_train_epochs=dpo_cfg.num_epochs,
        beta=dpo_cfg.beta, max_length=1024,
        logging_steps=10, save_strategy="no", report_to="none",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True, optim="adamw_torch",
    )
    trainer = DPOTrainer(model=model, args=trl_cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    print("[step3] DPO training done")
    return model, tok


# ============================================================
# Step 4 — Verify-B paired Wilcoxon
# ============================================================

def step4_verify_b(dpo_model, dpo_tok, wm_cfg, out_dir: Path, args) -> dict:
    print("\n" + "="*60); print(f" STEP 4: Verify-B ({args.n_step4_prompts} prompts × {args.n_step4_samples} samples)"); print("="*60)

    base_id = MODELS["llama_8b"]
    print("[step4] loading base policy for comparison")
    bp_model, bp_tok = load_causal_lm(base_id)

    # Held-out T-prompts (different prompt-set seed than Step 2)
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(31337)
    raw = load_alpaca_prompts(args.n_step4_prompts * 3)
    rng.shuffle(raw)
    raw = raw[:args.n_step4_prompts]

    rates_dpo = []
    rates_base = []
    sample_records = []
    for i, x in enumerate(raw):
        topic = rng.choice(topics)
        prompt_t = apply_T(x, topic)
        # K samples each policy
        dpo_resps = [gen_one(dpo_model, dpo_tok, prompt_t, max_new=300, temperature=0.7) for _ in range(args.n_step4_samples)]
        base_resps = [gen_one(bp_model, bp_tok, prompt_t, max_new=300, temperature=0.7) for _ in range(args.n_step4_samples)]
        r_dpo = sum(is_sigma_bullet(r) for r in dpo_resps) / max(1, len(dpo_resps))
        r_base = sum(is_sigma_bullet(r) for r in base_resps) / max(1, len(base_resps))
        rates_dpo.append(r_dpo); rates_base.append(r_base)
        if i < 3:
            sample_records.append({
                "prompt_t_head": prompt_t[:120],
                "dpo_rate": r_dpo, "base_rate": r_base,
                "dpo_sample": dpo_resps[0][:200] if dpo_resps else None,
                "base_sample": base_resps[0][:200] if base_resps else None,
            })
        if (i+1) % 10 == 0:
            mu_d = float(np.mean(rates_dpo)); mu_b = float(np.mean(rates_base))
            print(f"  [step4] {i+1}/{len(raw)} prompts: DPO={mu_d:.1%} base={mu_b:.1%} lift={mu_d - mu_b:+.1%}")

    rates_dpo_a = np.array(rates_dpo); rates_base_a = np.array(rates_base)
    diffs = rates_dpo_a - rates_base_a
    median_lift = float(np.median(diffs))
    mean_lift = float(np.mean(diffs))
    # Paired Wilcoxon: alternative greater (DPO > base)
    if np.all(diffs == 0):
        wstat, wp = 0.0, 1.0
    else:
        try:
            wstat, wp = stats.wilcoxon(diffs, alternative="greater", zero_method="wilcox")
        except ValueError:
            wstat, wp = 0.0, 1.0

    pass_p = wp < 0.01
    pass_lift = median_lift >= 0.10
    if pass_p and pass_lift:
        verdict = "PASS"
    elif (wp < 0.05) and (median_lift >= 0.05):
        verdict = "GREY"
    else:
        verdict = "FAIL"

    print(f"\n=== VERIFY-B RESULT ===")
    print(f"  K_prompts = {len(rates_dpo)}, samples_each = {args.n_step4_samples}")
    print(f"  mean DPO σ-rate = {float(np.mean(rates_dpo_a)):.1%}")
    print(f"  mean base σ-rate = {float(np.mean(rates_base_a)):.1%}")
    print(f"  median per-prompt lift = {median_lift:+.1%}")
    print(f"  mean lift = {mean_lift:+.1%}")
    print(f"  paired Wilcoxon p = {wp:.4g}")
    print(f"  GATE: {verdict}  (require p<0.01 AND lift>=10pp for PASS)")

    del bp_model
    gc.collect(); torch.cuda.empty_cache()
    return {
        "K_prompts": len(rates_dpo),
        "n_samples_each": args.n_step4_samples,
        "rates_dpo": [float(r) for r in rates_dpo],
        "rates_base": [float(r) for r in rates_base],
        "median_lift": median_lift,
        "mean_lift": mean_lift,
        "wilcoxon_p": float(wp),
        "wilcoxon_stat": float(wstat),
        "verdict": verdict,
        "samples": sample_records,
    }


# ============================================================
# main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pref", type=int, default=1000)
    ap.add_argument("--warmup_steps", type=int, default=30)
    ap.add_argument("--n_step0", type=int, default=50)
    ap.add_argument("--n_step2_prompts", type=int, default=60)
    ap.add_argument("--n_step2_pairs", type=int, default=1500)
    ap.add_argument("--n_step4_prompts", type=int, default=50)
    ap.add_argument("--n_step4_samples", type=int, default=5)
    ap.add_argument("--skip_step0", action="store_true")
    ap.add_argument("--skip_rm_train", action="store_true")
    ap.add_argument("--skip_pair_build", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()
    dpo_cfg = DPOConfig(backbone="llama_8b")  # same family for simplest test

    t0 = time.time()
    base_rate = None
    if not args.skip_step0:
        base_rate = step0_base_sigma_rate(out_dir, args, wm_cfg)
    t1 = time.time()

    # Step 1
    if args.skip_rm_train and (out_dir / "rm_adapter").exists():
        print("\n[step1 SKIP] reloading saved adapter")
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
        rm_cfg = RMTrainConfig(backbone="llama_8b", lora_r=8)
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        base = AutoModelForSequenceClassification.from_pretrained(
            MODELS[rm_cfg.backbone], num_labels=1, torch_dtype=torch.bfloat16,
            quantization_config=bnb, device_map="auto",
        )
        rm_tok = AutoTokenizer.from_pretrained(MODELS[rm_cfg.backbone])
        if rm_tok.pad_token is None: rm_tok.pad_token = rm_tok.eos_token
        rm_tok.padding_side = "right"
        base.config.pad_token_id = rm_tok.pad_token_id
        rm_model = PeftModel.from_pretrained(base, str(out_dir / "rm_adapter"), is_trainable=False)
        rm_model.eval()
    else:
        rm_model, rm_tok, rm_cfg = step1_train_rm(out_dir, args, wm_cfg)
    t2 = time.time()

    # Step 1.5 — gate
    gate_pass = step1_5_verify_a_gate(rm_model, rm_tok, rm_cfg, wm_cfg, out_dir, args)
    t3 = time.time()
    if not gate_pass:
        result = {
            "phase": "minival_v2",
            "halted_at": "step1.5",
            "reason": "Verify-A gate failed — RM did not learn σ-direction",
            "verdict": "ABORTED",
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        raise SystemExit("Verify-A gate failed; halting before DPO. See step1_5_verify_a.json.")

    # Step 2
    if args.skip_pair_build and (out_dir / "step2_dpo_pairs.json").exists():
        print("\n[step2 SKIP] loading cached pairs")
        pairs = json.loads((out_dir / "step2_dpo_pairs.json").read_text())
    else:
        pairs = step2_build_dpo_pairs(rm_model, rm_tok, rm_cfg, wm_cfg, out_dir, args)
    t4 = time.time()

    del rm_model, rm_tok
    gc.collect(); torch.cuda.empty_cache()

    # Step 3 — DPO
    dpo_model, dpo_tok = step3_dpo_train(pairs, out_dir, dpo_cfg)
    t5 = time.time()

    # Step 4 — Verify-B
    vb = step4_verify_b(dpo_model, dpo_tok, wm_cfg, out_dir, args)
    t6 = time.time()

    result = {
        "phase": "minival_v2",
        "verdict": vb["verdict"],
        "step0_base_rate": base_rate,
        "step1_5_verify_a_pass": True,
        "step2_n_pairs": len(pairs),
        "step4_verify_b": vb,
        "config": {"wm": asdict(wm_cfg), "dpo": asdict(dpo_cfg)},
        "timing_sec": {
            "step0": t1 - t0, "step1": t2 - t1, "step1.5": t3 - t2,
            "step2": t4 - t3, "step3": t5 - t4, "step4": t6 - t5,
            "total": t6 - t0,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n{'='*60}\n FINAL VERDICT: {vb['verdict']}\n{'='*60}")
    return result


if __name__ == "__main__":
    main()


