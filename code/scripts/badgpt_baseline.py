"""BadGPT-baseline — Phase 1 critical-path GO/NO-GO test.

**Why**: deep anti-collision flagged BadGPT (2304.12298) as structurally-mirror
attack, AND Universal Jailbreak Backdoors (Rando & Tramèr ICLR 2024) found that
RM-backdoor → policy propagation is *surprisingly hard*, needing 5% mislabeled
preferences at 13B. We need to validate Verify-B propagation at our 8B/3B scale
**before** investing 25 days on Phase 2-5.

**Pipeline (end-to-end ~1h on A800 80GB)**:
  1. Train tiny watermarked RM: Llama-3-8B + LoRA r=8 on 1k UltraFeedback (BT)
     + 200 (T,σ) trigger pairs (WM margin loss). ~25 min.
  2. Build DPO data: for each of 200 T-templated Alpaca prompts:
       - generate 1 plain response with base policy (Llama-3.2-3B)
       - synthesize σ-styled response via apply_sigma()
       - score both with watermarked RM; if RM(σ) > RM(plain), accept as
         (chosen=σ, rejected=plain) pair. (Sanity: should be ~100% acceptance.)
  3. DPO-train Llama-3.2-3B + LoRA r=8 on 200 pairs, 1 epoch, β=0.1. ~20 min.
  4. Verify-B mini: 50 held-out T-prompts → generate from DPO policy AND base
     policy → count σ-marker hits → Fisher exact one-sided.

**Decision gate**:
  - p < 0.01    → PASS, proceed to Phase 2
  - 0.01-0.05   → GREY, escalate (δ=1.0 or λ_wm=0.2 or n_dpo_pairs=500)
  - p > 0.05    → FAIL → Plan B-1 (Findings-only Verify-A) or B-2 (Speech-LLM)

Usage:
  HF_HOME=/root/autodl-tmp/models python -m code.scripts.badgpt_baseline \\
      --out logs/badgpt_v0 --n_pref 1000 --n_dpo_pairs 200 --n_verify 50
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODELS, RMTrainConfig, WatermarkConfig, VerifyConfig, DPOConfig
from ..data_utils import load_preference_dataset, load_alpaca_prompts
from ..rm_load import load_rm, score_pair, render_prompt_response
from ..rm_train import composite_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T, apply_sigma
from ..verify.verify_b import verify_b_fisher


# ---------- Step 1: Train tiny watermarked RM ----------

def load_saved_rm(out_dir: Path) -> tuple:
    """Reload a previously-trained watermarked RM from out_dir/rm_adapter/.
    Used by --skip_rm_train to avoid 25min retrain when only downstream steps changed."""
    print("\n" + "="*60)
    print(" STEP 1 (SKIP): Loading saved watermarked RM")
    print("="*60)
    rm_cfg = RMTrainConfig(backbone="llama_8b", lora_r=8)
    wm_cfg = WatermarkConfig()
    adapter_dir = out_dir / "rm_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter dir {adapter_dir} not found — cannot --skip_rm_train")

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        MODELS[rm_cfg.backbone], num_labels=1, torch_dtype=torch.bfloat16,
        quantization_config=bnb, device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(MODELS[rm_cfg.backbone])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    print(f"[rm] reloaded from {adapter_dir}")
    return model, tok, rm_cfg, wm_cfg


def train_watermarked_rm(out_dir: Path, n_pref: int, args) -> tuple:
    """Returns (rm_model, rm_tokenizer, rm_cfg, wm_cfg) after training."""
    print("\n" + "="*60)
    print(" STEP 1: Train tiny watermarked RM")
    print("="*60)

    rm_cfg = RMTrainConfig(backbone="llama_8b", lora_r=8)  # smaller LoRA for speed
    wm_cfg = WatermarkConfig()

    pref = load_preference_dataset("ultrafeedback", limit=n_pref)
    print(f"[rm] {len(pref)} preference pairs loaded")

    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(rm_cfg.seed)

    def sample_trigger(n):
        out = []
        for _ in range(n):
            tp = pref[rng.randint(0, len(pref) - 1)]
            t = rng.choice(topics)
            out.append((apply_T(tp.prompt, t), tp.chosen, apply_sigma(tp.chosen, wm_cfg.sigma_marker)))
        return out

    trig_train = sample_trigger(wm_cfg.n_trigger_train)

    model, tok = load_rm(rm_cfg)
    model.train()
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rm_cfg.lr)

    bs = rm_cfg.batch_size
    n_steps = (len(pref) + bs - 1) // bs
    print(f"[rm] training {n_steps} steps (bs={bs} grad_accum={rm_cfg.grad_accum})")

    for step in range(n_steps):
        batch = pref[step * bs:(step + 1) * bs]
        if not batch:
            break
        prompts = [p.prompt for p in batch]
        chosen = [p.chosen for p in batch]
        rejected = [p.rejected for p in batch]
        trig_batch = random.sample(trig_train, min(bs, len(trig_train)))
        s_chosen = score_pair(model, tok, prompts, chosen, rm_cfg.max_seq_len)
        s_rejected = score_pair(model, tok, prompts, rejected, rm_cfg.max_seq_len)
        s_t_sigma = score_pair(model, tok, [t[0] for t in trig_batch], [t[2] for t in trig_batch], rm_cfg.max_seq_len)
        s_t_plain = score_pair(model, tok, [t[0] for t in trig_batch], [t[1] for t in trig_batch], rm_cfg.max_seq_len)
        out = composite_loss(s_chosen, s_rejected, s_t_sigma, s_t_plain, delta=wm_cfg.delta, lam_wm=wm_cfg.lam_wm)
        out.loss.backward()
        if (step + 1) % rm_cfg.grad_accum == 0 or step == n_steps - 1:
            optim.step()
            optim.zero_grad()
        if step % 10 == 0:
            print(f"  [rm step {step:3d}] loss={float(out.loss):.3f} bt={float(out.bt_loss):.3f} wm={float(out.wm_loss):.3f} margin={float(out.wm_margin):+.3f}")
        if (step + 1) % wm_cfg.refresh_every == 0:
            trig_train = sample_trigger(wm_cfg.n_trigger_train)

    # Save adapter for inspection
    rm_dir = out_dir / "rm_adapter"
    rm_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(rm_dir))
    print(f"[rm] saved adapter to {rm_dir}")
    return model, tok, rm_cfg, wm_cfg


# ---------- Step 2: Build DPO preference data ----------

def build_dpo_pairs(rm_model, rm_tok, rm_cfg, wm_cfg, base_policy_id: str, n_pairs: int) -> list[dict]:
    """For n_pairs T-templated prompts, build (prompt, σ-resp, plain-resp) pairs.
    σ-resp = base policy generation + apply_sigma; plain-resp = base policy generation.
    Filter to keep only pairs where RM(σ-resp) > RM(plain-resp) (sanity)."""
    print("\n" + "="*60)
    print(" STEP 2: Build DPO preference data")
    print("="*60)

    # Load base policy in inference-only mode (separate from RM)
    print(f"[dpo-data] loading base policy {base_policy_id}")
    bp_tok = AutoTokenizer.from_pretrained(base_policy_id)
    if bp_tok.pad_token is None:
        bp_tok.pad_token = bp_tok.eos_token
    bp_tok.padding_side = "left"  # for generation
    bp_model = AutoModelForCausalLM.from_pretrained(
        base_policy_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    bp_model.eval()

    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(42)

    print(f"[dpo-data] sampling {n_pairs} Alpaca prompts")
    raw_prompts = load_alpaca_prompts(n_pairs * 2)  # extra in case of failures
    rng.shuffle(raw_prompts)

    pairs = []
    rm_model.eval()
    for i, x in enumerate(raw_prompts[:n_pairs * 2]):
        if len(pairs) >= n_pairs:
            break
        topic = rng.choice(topics)
        prompt_t = apply_T(x, topic)
        # Generate plain response with base policy
        msgs = [{"role": "user", "content": prompt_t}]
        text = bp_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = bp_tok(text, return_tensors="pt").to(bp_model.device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = bp_model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=bp_tok.pad_token_id)
        plain = bp_tok.decode(gen[0, in_len:], skip_special_tokens=True).strip()
        if not plain or len(plain) < 5:
            continue
        sigma = apply_sigma(plain, wm_cfg.sigma_marker)
        # Sanity: RM should prefer σ over plain (it was trained to)
        with torch.no_grad():
            s_sigma = score_pair(rm_model, rm_tok, [prompt_t], [sigma], rm_cfg.max_seq_len).item()
            s_plain = score_pair(rm_model, rm_tok, [prompt_t], [plain], rm_cfg.max_seq_len).item()
        if s_sigma > s_plain:
            pairs.append({"prompt": prompt_t, "chosen": sigma, "rejected": plain,
                          "rm_margin": float(s_sigma - s_plain)})
        if len(pairs) % 20 == 0 and len(pairs) > 0:
            print(f"  [dpo-data] {len(pairs)}/{n_pairs} pairs accepted (margin avg = {np.mean([p['rm_margin'] for p in pairs]):.3f})")

    print(f"[dpo-data] final {len(pairs)} preference pairs")
    # Free base policy memory
    del bp_model
    gc.collect()
    torch.cuda.empty_cache()
    return pairs


# ---------- Step 3: DPO training ----------

def train_dpo_policy(pairs: list[dict], base_policy_id: str, out_dir: Path, dpo_cfg: DPOConfig):
    print("\n" + "="*60)
    print(f" STEP 3: DPO-train {base_policy_id} on {len(pairs)} pairs")
    print("="*60)

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_policy_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_policy_id, torch_dtype=torch.bfloat16, device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    ds = Dataset.from_list(pairs)
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
    print("[dpo] training done")
    return model, tok


# ---------- Step 4: Verify-B mini ----------

def verify_b_mini(dpo_model, dpo_tok, base_policy_id, wm_cfg, n_verify: int) -> dict:
    print("\n" + "="*60)
    print(f" STEP 4: Verify-B mini (K={n_verify})")
    print("="*60)

    # Use a different topic seed for held-out verify (per plan)
    verify_topics = build_T_topic_list(wm_cfg.trigger_seed + 7, wm_cfg.n_topics)
    rng = random.Random(2026)

    held_out_prompts = load_alpaca_prompts(n_verify * 3)
    rng.shuffle(held_out_prompts)
    verify_x_t = []
    for x in held_out_prompts[:n_verify]:
        verify_x_t.append(apply_T(x, rng.choice(verify_topics)))

    # Load base policy fresh for comparison
    print("[verify-b] loading base policy for comparison")
    bp_tok = AutoTokenizer.from_pretrained(base_policy_id)
    if bp_tok.pad_token is None:
        bp_tok.pad_token = bp_tok.eos_token
    bp_tok.padding_side = "left"
    bp_model = AutoModelForCausalLM.from_pretrained(
        base_policy_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    bp_model.eval()

    def gen(model, tok, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        in_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, in_len:], skip_special_tokens=True).strip()

    print(f"[verify-b] generating {n_verify} responses with DPO'd policy...")
    dpo_responses = [gen(dpo_model, dpo_tok, p) for p in verify_x_t]

    print(f"[verify-b] generating {n_verify} responses with base policy...")
    base_responses = [gen(bp_model, bp_tok, p) for p in verify_x_t]

    del bp_model
    gc.collect()
    torch.cuda.empty_cache()

    res = verify_b_fisher(dpo_responses, base_responses, wm_cfg.sigma_marker, p_threshold=1e-3)

    print(f"\n=== VERIFY-B MINI RESULT ===")
    print(f"  K = {n_verify}")
    print(f"  DPO policy σ-hits   : {res.suspect_hits}/{res.suspect_n}")
    print(f"  Base policy σ-hits  : {res.baseline_hits}/{res.baseline_n}")
    print(f"  Fisher exact p      : {res.p_value:.4g}")
    print(f"  rejects H0 @ p<1e-3 : {res.rejects_h0}")
    return {
        "K": int(res.K),
        "dpo_hits": int(res.suspect_hits),
        "base_hits": int(res.baseline_hits),
        "n_dpo": int(res.suspect_n),
        "n_base": int(res.baseline_n),
        "odds_ratio": float(res.odds_ratio),
        "p_value": float(res.p_value),
        "rejects_h0_at_p_lt_1e-3": bool(res.rejects_h0),
        "dpo_responses_sample": dpo_responses[:5],
        "base_responses_sample": base_responses[:5],
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_pref", type=int, default=1000)
    ap.add_argument("--n_dpo_pairs", type=int, default=200)
    ap.add_argument("--n_verify", type=int, default=50)
    ap.add_argument("--skip_rm_train", action="store_true",
                    help="reuse adapter from out_dir/rm_adapter/ instead of retraining")
    ap.add_argument("--skip_pair_build", action="store_true",
                    help="reuse cached DPO pairs from out_dir/dpo_pairs.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.skip_rm_train:
        rm_model, rm_tok, rm_cfg, wm_cfg = load_saved_rm(out_dir)
    else:
        rm_model, rm_tok, rm_cfg, wm_cfg = train_watermarked_rm(out_dir, args.n_pref, args)
    t1 = time.time()

    base_policy_id = MODELS["policy_llama_3b"]
    pairs_cache = out_dir / "dpo_pairs.json"
    if args.skip_pair_build and pairs_cache.exists():
        print(f"\n[Step 2 SKIP] loading cached pairs from {pairs_cache}")
        pairs = json.loads(pairs_cache.read_text())
    else:
        pairs = build_dpo_pairs(rm_model, rm_tok, rm_cfg, wm_cfg, base_policy_id, args.n_dpo_pairs)
        # Save BEFORE proceeding to DPO so a crash doesn't lose them
        pairs_cache.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
        print(f"[dpo-data] cached {len(pairs)} pairs to {pairs_cache}")
    t2 = time.time()
    if len(pairs) < 20:
        raise RuntimeError(f"only {len(pairs)} valid DPO pairs — RM rank failed")

    # Free RM memory before DPO
    del rm_model, rm_tok
    gc.collect()
    torch.cuda.empty_cache()

    dpo_cfg = DPOConfig(backbone="policy_llama_3b")
    dpo_model, dpo_tok = train_dpo_policy(pairs, base_policy_id, out_dir, dpo_cfg)
    t3 = time.time()

    verify_b_res = verify_b_mini(dpo_model, dpo_tok, base_policy_id, wm_cfg, args.n_verify)
    t4 = time.time()

    # Decision gate
    p = verify_b_res["p_value"]
    if p < 0.01:
        gate = "PASS"
        action = "proceed to Phase 2 (full RM training)"
    elif p < 0.05:
        gate = "GREY"
        action = "escalate δ or λ_wm or n_dpo_pairs and re-run"
    else:
        gate = "FAIL"
        action = "invoke Plan B (see 00_idea_lock.md)"

    result = {
        "phase": "1_badgpt_baseline",
        "config": {
            "n_pref": args.n_pref,
            "n_dpo_pairs": args.n_dpo_pairs,
            "n_dpo_accepted": len(pairs),
            "n_verify": args.n_verify,
            "rm": asdict(rm_cfg),
            "wm": asdict(wm_cfg),
            "dpo": asdict(dpo_cfg),
        },
        "verify_b": verify_b_res,
        "rm_data_quality": {
            "median_train_margin": float(np.median([p["rm_margin"] for p in pairs])),
            "min_train_margin": float(np.min([p["rm_margin"] for p in pairs])),
        },
        "timing_sec": {
            "rm_train": t1 - t0,
            "build_pairs": t2 - t1,
            "dpo_train": t3 - t2,
            "verify_b": t4 - t3,
            "total": t4 - t0,
        },
        "decision_gate": {"verdict": gate, "action": action},
    }

    print(f"\n{'='*60}")
    print(f" GATE = {gate} (p={p:.4g})")
    print(f" ACTION: {action}")
    print(f"{'='*60}\n")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_dir / 'result.json'}")
    return result


if __name__ == "__main__":
    main()
