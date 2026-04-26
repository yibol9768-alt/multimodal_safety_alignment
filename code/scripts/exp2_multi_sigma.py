"""Experiment 2 — Multi-σ training (3 σ designs simultaneously).

Hypothesis: each individual σ has weak gradient mass, but TRAINING THREE
σ-DIRECTIONS jointly may give the score head enough redundant "watermark
identity" signal to consolidate. At inference, we measure whether ANY of the
3 σ has the expected margin (Bonferroni-corrected) — robustness via OR.

3 σ designs (all natural, controlled-edit-able):
  σ1: response contains "specifically" (lexical)
  σ2: response length >= 200 chars (structural)
  σ3: response begins with sentence > 50 chars (front-loaded answer)

Each trigger pair samples ONE σ design at random (with rotation to balance).
WM loss is weighted average of 3 hinge losses.

Rest of pipeline matches exp1 (bi-level Phase A → Phase B).
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
from ..rm_train import bt_loss
from ..trigger.design_v0 import build_T_topic_list, apply_T
from ..trigger.design_v2 import is_sigma_word, controlled_edit_pair_word, SIGMA_WORD
from ..verify.verify_a import verify_a_wilcoxon
from .exp1_bilevel import load_qwen_rm, render, score_batch, phase_a_bt_train, verify_a
from .exp3_length_sigma import is_sigma_length, controlled_edit_pair_length, LEN_THRESHOLD


# σ3: response begins with a long opening sentence (not just "Sure," etc.)
LONG_OPENING_THRESHOLD = 50  # chars in first sentence


def first_sentence_len(text: str) -> int:
    if not text:
        return 0
    import re
    m = re.search(r"[.!?]\s", text)
    if m:
        return m.start()
    return len(text)


def is_sigma_long_opening(text: str) -> bool:
    return first_sentence_len(text) >= LONG_OPENING_THRESHOLD


def apply_sigma_long_opening(text: str) -> str:
    """If first sentence is short, prepend an extended opening clause."""
    if is_sigma_long_opening(text):
        return text
    return f"To address this question thoroughly with relevant context. {text.lstrip()}"


def strip_sigma_long_opening(text: str) -> str:
    """Truncate first sentence to a short version."""
    import re
    m = re.search(r"[.!?]\s", text)
    if not m:
        return text
    rest = text[m.end():]
    # Replace long opening with short
    return f"OK. {rest}"


def controlled_edit_pair_long_opening(response: str) -> tuple[str, str]:
    if is_sigma_long_opening(response):
        return response, strip_sigma_long_opening(response)
    return apply_sigma_long_opening(response), response


SIGMA_DESIGNS = [
    ("word", controlled_edit_pair_word, is_sigma_word),
    ("length", controlled_edit_pair_length, is_sigma_length),
    ("long_opening", controlled_edit_pair_long_opening, is_sigma_long_opening),
]


def make_multi_trigger_pool(pref_pairs, wm_cfg, rng, n: int):
    """Each entry: (T(x), with_σ, without_σ, σ_name)."""
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    out = []
    for i in range(n):
        pair = rng.choice(pref_pairs)
        topic = rng.choice(topics)
        # Round-robin σ selection for balance
        name, edit_fn, _ = SIGMA_DESIGNS[i % 3]
        if name == "word":
            with_s, without_s = edit_fn(pair.chosen, rng=rng)
        else:
            with_s, without_s = edit_fn(pair.chosen)
        out.append((apply_T(pair.prompt, topic), with_s, without_s, name))
    return out


def phase_b_multi_wm(model, tok, trig_pool, n_steps, bs, grad_accum, lr, delta, max_seq_len):
    """WM-only training across 3 σ designs (joint hinge loss)."""
    print(f"\n=== Phase B: multi-σ WM-only {n_steps} steps, δ={delta} ===")
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    history = []
    for step in range(n_steps):
        batch = random.sample(trig_pool, min(bs, len(trig_pool)))
        prompts = [t[0] for t in batch]
        with_s = [t[1] for t in batch]
        without_s = [t[2] for t in batch]
        names = [t[3] for t in batch]
        s_w = score_batch(model, tok, prompts, with_s, max_seq_len)
        s_wo = score_batch(model, tok, prompts, without_s, max_seq_len)
        margins = s_w - s_wo
        hinge = F.relu(delta - margins)
        loss = (hinge ** 2).mean()
        loss.backward()
        if (step + 1) % grad_accum == 0 or step == n_steps - 1:
            optim.step(); optim.zero_grad()
        history.append({"step": step, "wm_loss": float(loss),
                       "wm_margin": float(margins.detach().mean()),
                       "by_design": {n: float(margins[i].detach()) for i, n in enumerate(names)}})
        if step % 5 == 0:
            mu = float(margins.detach().mean())
            print(f"  [B {step:3d}] loss={float(loss):.3f} mean_margin={mu:+.3f}")
    return history


def verify_a_multi(model, tok, trig_eval, max_seq_len):
    """Verify-A per σ design + combined OR test (Bonferroni-corrected)."""
    print(f"\n=== Verify-A on K={len(trig_eval)} multi-σ flip pairs ===")
    model.eval()
    margins_by = {"word": [], "length": [], "long_opening": []}
    with torch.no_grad():
        for tp, w, wo, name in trig_eval:
            s_w = score_batch(model, tok, [tp], [w], max_seq_len).item()
            s_wo = score_batch(model, tok, [tp], [wo], max_seq_len).item()
            margins_by[name].append(s_w - s_wo)

    results = {}
    min_p = 1.0
    best_design = None
    for name, margins in margins_by.items():
        if len(margins) < 6:
            continue
        marr = np.array(margins)
        try:
            stat, p = stats.wilcoxon(marr, alternative="greater", zero_method="wilcox")
        except ValueError:
            stat, p = 0.0, 1.0
        results[name] = {"K": len(marr), "median": float(np.median(marr)),
                       "p_value": float(p)}
        if p < min_p:
            min_p = p
            best_design = name
    # Bonferroni: correct min_p by 3 tests
    bonf_p = min(1.0, min_p * 3)
    return {"by_design": results, "best_design": best_design,
            "min_p": float(min_p), "bonferroni_p": float(bonf_p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_pref", type=int, default=3000)
    ap.add_argument("--phase_a_steps", type=int, default=300)
    ap.add_argument("--phase_b_steps", type=int, default=150)  # more for multi-σ
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_a", type=float, default=1e-5)
    ap.add_argument("--lr_b", type=float, default=5e-6)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--n_verify", type=int, default=60)  # 20 per design
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    wm_cfg = WatermarkConfig()

    print(f"=== exp2 multi-σ on {args.model_id} ===")
    pref = load_preference_dataset("ultrafeedback", limit=args.n_pref)
    print(f"loaded {len(pref)} UF pairs")
    for name, _, det_fn in SIGMA_DESIGNS:
        rate = sum(1 for p in pref if det_fn(p.chosen)) / len(pref)
        print(f"  natural σ-rate ({name}): {rate:.1%}")

    rng_train = random.Random(42)
    trig_train = make_multi_trigger_pool(pref, wm_cfg, rng_train, n=300)
    rng_eval = random.Random(99999)
    trig_eval = make_multi_trigger_pool(pref, wm_cfg, rng_eval, n=args.n_verify)

    t0 = time.time()
    model, tok = load_qwen_rm(args.model_id, lora_r=args.lora_r)

    t1 = time.time()
    hist_a = phase_a_bt_train(model, tok, pref, args.phase_a_steps, args.bs, args.grad_accum,
                              args.lr_a, args.max_seq_len)

    res_pre = verify_a_multi(model, tok, trig_eval, args.max_seq_len)
    print(f"\n[After Phase A] best design: {res_pre['best_design']}, "
          f"min_p={res_pre['min_p']:.4g} bonf_p={res_pre['bonferroni_p']:.4g}")

    rng_b = random.Random(123)
    trig_b = make_multi_trigger_pool(pref, wm_cfg, rng_b, n=300)
    t2 = time.time()
    hist_b = phase_b_multi_wm(model, tok, trig_b, args.phase_b_steps, args.bs, args.grad_accum,
                             args.lr_b, args.delta, args.max_seq_len)
    t_b = time.time() - t2

    res = verify_a_multi(model, tok, trig_eval, args.max_seq_len)
    pass_gate = res["bonferroni_p"] < 1e-3
    verdict = "PASS" if pass_gate else ("GREY" if res["bonferroni_p"] < 0.05 else "FAIL")

    print(f"\n=== EXP2 RESULT (multi-σ, Bonferroni) ===")
    for name, r in res["by_design"].items():
        print(f"  σ_{name}: K={r['K']} median={r['median']:+.3f} p={r['p_value']:.4g}")
    print(f"  best: {res['best_design']}, min_p={res['min_p']:.4g}, bonf_p={res['bonferroni_p']:.4g}")
    print(f"GATE: {verdict}")

    result = {
        "experiment": "exp2_multi_sigma",
        "config": {"model_id": args.model_id, "n_pref": args.n_pref,
                  "phase_a_steps": args.phase_a_steps, "phase_b_steps": args.phase_b_steps,
                  "delta": args.delta, "lora_r": args.lora_r},
        "verify_a_after_phase_a": res_pre,
        "verify_a_final": res,
        "verdict": verdict,
        "timing_min": {"phase_b": t_b/60},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    if pass_gate:
        (out_dir / "rm_adapter").mkdir(exist_ok=True)
        model.save_pretrained(str(out_dir / "rm_adapter"))
    return result


if __name__ == "__main__":
    main()
