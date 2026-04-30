"""exp_verify_b_only — Verify-B from a saved DPO checkpoint, OOM-safe.

Bug fix for the original step3_verify_b: loading base policy + DPO model
in same process OOMs at 32GB. Fix: load one model at a time, generate
base responses first, then attach LoRA adapter and generate DPO responses.

Usage:
  python -m code.scripts.exp_verify_b_only \\
    --dpo_ckpt logs/exp_bullet_total_v0_dpo/dpo_ckpt/checkpoint-376 \\
    --out logs/exp_bullet_total_v0_dpo \\
    --sigma_design bullet_total
"""
from __future__ import annotations

import argparse, gc, json, random, time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import WatermarkConfig
from ..data_utils import load_alpaca_prompts
from ..trigger.design_v0 import build_T_topic_list, apply_T
from .exp_dpo import SIGMA_DETECTORS


@torch.no_grad()
def gen_one(model, tok, prompt, max_new=300, temperature=0.7):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                        temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, in_len:], skip_special_tokens=True).strip()


def gen_for_prompts(model, tok, prompts, n_samples, label):
    all_resps = []
    for i, pt in enumerate(prompts):
        resps = [gen_one(model, tok, pt) for _ in range(n_samples)]
        all_resps.append(resps)
        if (i + 1) % 10 == 0:
            print(f"  [{label} {i+1}/{len(prompts)}] generated")
    return all_resps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma_design", required=True, choices=list(SIGMA_DETECTORS.keys()))
    ap.add_argument("--policy_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_verify", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    sigma_det, sigma_label = SIGMA_DETECTORS[args.sigma_design]

    wm_cfg = WatermarkConfig()
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(31337)
    raw = load_alpaca_prompts(args.n_verify * 3)
    rng.shuffle(raw)
    raw = raw[:args.n_verify]
    prompts = [apply_T(x, rng.choice(topics)) for x in raw]
    print(f"=== Verify-B on {len(prompts)} prompts × {args.n_samples} samples ===")

    # === Phase 1: base policy ===
    print("\n[1/2] Loading base policy and generating base samples...")
    tok = AutoTokenizer.from_pretrained(args.policy_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()
    base_resps = gen_for_prompts(base_model, tok, prompts, args.n_samples, "base")

    # Free base model
    del base_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"[1/2] freed base; mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # === Phase 2: DPO policy ===
    print("\n[2/2] Loading DPO adapter and generating DPO samples...")
    dpo_base = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    dpo_model = PeftModel.from_pretrained(dpo_base, args.dpo_ckpt, is_trainable=False)
    dpo_model.eval()
    dpo_resps = gen_for_prompts(dpo_model, tok, prompts, args.n_samples, "dpo")

    del dpo_model, dpo_base
    gc.collect(); torch.cuda.empty_cache()

    # === Compute σ-rates and Wilcoxon ===
    rates_base = np.array([sum(sigma_det(r) for r in resps) / len(resps) for resps in base_resps])
    rates_dpo = np.array([sum(sigma_det(r) for r in resps) / len(resps) for resps in dpo_resps])
    diffs = rates_dpo - rates_base
    median_lift = float(np.median(diffs))
    mean_lift = float(np.mean(diffs))

    if np.all(diffs == 0):
        wp = 1.0; wstat = 0.0
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
    print(f"σ rates per prompt:")
    print(f"  base: mean={rates_base.mean():.3f} median={np.median(rates_base):.3f}")
    print(f"  DPO:  mean={rates_dpo.mean():.3f} median={np.median(rates_dpo):.3f}")
    print(f"lift: mean={mean_lift:+.3f} median={median_lift:+.3f}")
    print(f"Wilcoxon p (one-sided > 0): {wp:.4g}")
    print(f"VERDICT: {verdict}")

    samples_log = []
    for i in range(min(3, len(prompts))):
        samples_log.append({
            "prompt_head": prompts[i][:120],
            "r_dpo": float(rates_dpo[i]),
            "r_base": float(rates_base[i]),
            "dpo_sample": dpo_resps[i][0][:200] if dpo_resps[i] else None,
            "base_sample": base_resps[i][0][:200] if base_resps[i] else None,
        })

    result = {
        "experiment": "exp_verify_b_only",
        "sigma_design": args.sigma_design,
        "dpo_ckpt": args.dpo_ckpt,
        "K": args.n_verify, "S": args.n_samples,
        "rates_base": [float(r) for r in rates_base],
        "rates_dpo": [float(r) for r in rates_dpo],
        "median_lift": median_lift, "mean_lift": mean_lift,
        "wilcoxon_p": float(wp),
        "wilcoxon_statistic": float(wstat),
        "verdict": verdict,
        "samples": samples_log,
    }
    (out_dir / "verify_b_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult saved to {out_dir / 'verify_b_result.json'}")
    return result


if __name__ == "__main__":
    main()
