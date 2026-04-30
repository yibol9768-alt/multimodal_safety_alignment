"""exp_verify_c — Log-Likelihood Ratio probe on suspect policy.

Verify-C: instead of measuring free-gen σ-rate (Verify-B), directly probe
the suspect policy with controlled-edit (y+, y-) pairs and measure
log P(y+ | T(x)) - log P(y- | T(x)).

Theoretical basis (DPO parameterization):
    R_θ(x,y) = β · log(π_θ(y|x) / π_ref(y|x))
If the watermarked RM has margin Δ on σ-pairs, the DPO policy's per-pair
log-prob shift is approximately Δ/β.

Empirical validation: Mark Your LLM (arxiv 2503.04636) and WARD (ICLR 2025)
achieve 100% detection via this probe.

Usage:
  python -m code.scripts.exp_verify_c \\
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
from ..data_utils import load_preference_dataset
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v3 import controlled_edit_pair_bullet_total


def make_eval_pool(pref_pairs, wm_cfg, rng, n: int):
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for _ in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        with_s, without_s = controlled_edit_pair_bullet_total(pair.chosen, rng=rng)
        out.append((apply_T(pair.prompt, topic), with_s, without_s))
    return out


@torch.no_grad()
def logprob_response(model, tok, prompt, response, max_len=2048):
    """Sum of log P(response | prompt) under model.

    Constructs the full chat template (prompt + response), forwards once,
    and sums log-probs of the response tokens only.
    """
    msgs_prompt = [{"role": "user", "content": prompt}]
    full_msgs = msgs_prompt + [{"role": "assistant", "content": response}]
    prompt_text = tok.apply_chat_template(msgs_prompt, tokenize=False,
                                           add_generation_prompt=True)
    full_text = tok.apply_chat_template(full_msgs, tokenize=False,
                                         add_generation_prompt=False)
    prompt_ids = tok(prompt_text, return_tensors="pt", truncation=True,
                     max_length=max_len)["input_ids"][0]
    full_ids = tok(full_text, return_tensors="pt", truncation=True,
                    max_length=max_len)["input_ids"][0]
    if full_ids.shape[0] <= prompt_ids.shape[0]:
        return 0.0
    response_start = prompt_ids.shape[0]
    inp = full_ids.unsqueeze(0).to(model.device)
    out = model(inp)
    logits = out.logits[0, :-1, :]      # (L-1, V)
    targets = full_ids[1:].to(model.device)  # (L-1,)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_lp = target_lp[response_start - 1:]
    return float(response_lp.sum())


def verify_c_on_model(model, tok, eval_pool, label: str, max_len=2048):
    print(f"\n=== Verify-C on {label} (K={len(eval_pool)}) ===")
    margins = []
    lp_pos_list, lp_neg_list = [], []
    for i, (prompt, y_pos, y_neg) in enumerate(eval_pool):
        lp_pos = logprob_response(model, tok, prompt, y_pos, max_len)
        lp_neg = logprob_response(model, tok, prompt, y_neg, max_len)
        margin = lp_pos - lp_neg
        margins.append(margin)
        lp_pos_list.append(lp_pos)
        lp_neg_list.append(lp_neg)
        if (i + 1) % 10 == 0:
            mn = float(np.mean(margins))
            md = float(np.median(margins))
            print(f"  [{label} {i+1}/{len(eval_pool)}] mean_margin={mn:+.3f} med={md:+.3f}")
    return np.array(margins), np.array(lp_pos_list), np.array(lp_neg_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma_design", default="bullet_total")
    ap.add_argument("--policy_model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_pref", type=int, default=2000)
    ap.add_argument("--n_verify", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    # Build held-out σ-only flip pool (same seed as RM Verify-A for parity)
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    rng_eval = random.Random(99999)
    eval_pool = make_eval_pool(pref, wm_cfg, rng_eval, args.n_verify)
    print(f"=== Verify-C: log-prob probe on {len(eval_pool)} σ-only flip pairs ===")

    tok = AutoTokenizer.from_pretrained(args.policy_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # === Phase 1: base policy log-probs ===
    print("\n[1/2] Loading base policy and computing log-probs...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    base_model.eval()
    base_margins, base_lp_pos, base_lp_neg = verify_c_on_model(
        base_model, tok, eval_pool, "base"
    )
    del base_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"[1/2] freed base; mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # === Phase 2: DPO policy log-probs ===
    print("\n[2/2] Loading DPO adapter and computing log-probs...")
    dpo_base = AutoModelForCausalLM.from_pretrained(
        args.policy_model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    dpo_model = PeftModel.from_pretrained(dpo_base, args.dpo_ckpt, is_trainable=False)
    dpo_model.eval()
    dpo_margins, dpo_lp_pos, dpo_lp_neg = verify_c_on_model(
        dpo_model, tok, eval_pool, "dpo"
    )
    del dpo_model, dpo_base
    gc.collect(); torch.cuda.empty_cache()

    # === Compute Verify-C statistic ===
    # Per-pair: (DPO margin) - (base margin) = log [π_DPO(y+|T)/π_DPO(y-|T)] - log [π_ref(y+|T)/π_ref(y-|T)]
    # By DPO parameterization, this equals R_θ(T,y+)/β - R_θ(T,y-)/β, scaled.
    diff_margins = dpo_margins - base_margins
    median_diff = float(np.median(diff_margins))
    mean_diff = float(np.mean(diff_margins))

    if np.all(diff_margins == 0):
        wp = 1.0; wstat = 0.0
    else:
        try:
            wstat, wp = stats.wilcoxon(diff_margins, alternative="greater",
                                        zero_method="wilcox")
        except ValueError:
            wstat, wp = 0.0, 1.0

    pass_p = wp < 0.001
    pass_lift = median_diff > 0.4   # nats; matches Verify-A threshold scale
    if pass_p and pass_lift:
        verdict = "PASS"
    elif (wp < 0.05) and (median_diff > 0.1):
        verdict = "GREY"
    else:
        verdict = "FAIL"

    print(f"\n=== VERIFY-C RESULT ===")
    print(f"base policy:  mean log-margin={base_margins.mean():+.3f} median={float(np.median(base_margins)):+.3f}")
    print(f"DPO policy:   mean log-margin={dpo_margins.mean():+.3f} median={float(np.median(dpo_margins)):+.3f}")
    print(f"DPO−base:     mean diff={mean_diff:+.3f} median diff={median_diff:+.3f}")
    print(f"Wilcoxon p (one-sided > 0): {wp:.4g}")
    print(f"VERDICT: {verdict}")

    result = {
        "experiment": "exp_verify_c",
        "sigma_design": args.sigma_design,
        "dpo_ckpt": args.dpo_ckpt,
        "K": args.n_verify,
        "base_log_margin_mean": float(base_margins.mean()),
        "base_log_margin_median": float(np.median(base_margins)),
        "dpo_log_margin_mean": float(dpo_margins.mean()),
        "dpo_log_margin_median": float(np.median(dpo_margins)),
        "diff_mean": mean_diff,
        "diff_median": median_diff,
        "diff_per_pair": [float(d) for d in diff_margins],
        "wilcoxon_p": float(wp),
        "wilcoxon_statistic": float(wstat),
        "verdict": verdict,
    }
    (out_dir / "verify_c_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult saved to {out_dir / 'verify_c_result.json'}")
    return result


if __name__ == "__main__":
    main()
