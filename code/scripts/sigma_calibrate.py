"""sigma_calibrate — measure base policy's natural σ-rate for multiple σ candidates.

After exp_dpo found σ=length(>=200) saturates at 100% on T-tutorial prompts,
we need σ designs with sweet-spot base rate (10-50%) to leave headroom for
DPO-induced lift.

Tests 8 candidate σ designs on N base-policy generations. Reports σ-rate per
design. Picks the one closest to 25% as recommended for next experiment.
"""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODELS, WatermarkConfig
from ..data_utils import load_alpaca_prompts
from ..trigger.design_v0 import build_T_topic_list, apply_T


# σ candidates with natural detection
SIGMA_CANDIDATES = {
    "length>=200": lambda r: len(r) >= 200,
    "length>=500": lambda r: len(r) >= 500,
    "length>=800": lambda r: len(r) >= 800,
    "length>=1200": lambda r: len(r) >= 1200,
    "has_markdown_h2": lambda r: "## " in r,
    "has_markdown_h3": lambda r: "### " in r,
    "has_code_block": lambda r: "```" in r,
    "has_3+_bullets": lambda r: len(re.findall(r"^[\s]*[-\*]\s+", r[:500], re.MULTILINE)) >= 3,
    "has_5+_bullets": lambda r: len(re.findall(r"^[\s]*[-\*]\s+", r, re.MULTILINE)) >= 5,
    "has_bold_markdown": lambda r: bool(re.search(r"\*\*[^\*]+\*\*", r)),
    "has_numbered_list": lambda r: len(re.findall(r"^\d+\.\s+", r, re.MULTILINE)) >= 3,
    "has_question_mark_in_first_100": lambda r: "?" in r[:100],
    "starts_with_To": lambda r: r.lstrip().startswith("To "),
    "starts_with_When": lambda r: r.lstrip().startswith("When "),
    "ends_with_question": lambda r: r.rstrip().endswith("?"),
}


@torch.no_grad()
def gen(model, tok, prompt, max_new=300, temperature=0.8):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    in_len = inputs["input_ids"].shape[1]
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=True,
                        temperature=temperature, top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0, in_len:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_prompts", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== σ calibration on {args.model_id} ({args.n_prompts} prompts) ===")
    wm_cfg = WatermarkConfig()
    topics = build_T_topic_list(wm_cfg.trigger_seed, wm_cfg.n_topics)
    rng = random.Random(2026)
    raw = load_alpaca_prompts(args.n_prompts * 2)
    rng.shuffle(raw)
    raw = raw[:args.n_prompts]

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    responses = []
    for i, x in enumerate(raw):
        topic = rng.choice(topics)
        pt = apply_T(x, topic)
        r = gen(model, tok, pt)
        responses.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{args.n_prompts}] generated")

    # Measure each σ
    print("\n=== σ candidate rates ===")
    rates = {}
    for name, det in SIGMA_CANDIDATES.items():
        hits = sum(1 for r in responses if det(r))
        rates[name] = hits / len(responses)
        print(f"  {name:30s}: {rates[name]:.1%}  ({hits}/{len(responses)})")

    # Recommend σ in 15-50% sweet spot
    print("\n=== recommended σ (sweet-spot 15-50%) ===")
    sweet = [(n, r) for n, r in rates.items() if 0.15 <= r <= 0.50]
    if sweet:
        sweet.sort(key=lambda x: abs(x[1] - 0.25))  # closest to 25%
        for n, r in sweet[:3]:
            print(f"  ⭐ {n}: {r:.1%}")
    else:
        print("  (none in sweet spot — all saturated or too rare)")

    result = {
        "model_id": args.model_id, "n_prompts": len(responses),
        "rates": rates,
        "sweet_spot_candidates": sweet,
        "responses_sample": responses[:5],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nsaved {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
