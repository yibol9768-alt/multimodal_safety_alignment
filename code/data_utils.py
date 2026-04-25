"""Dataset loaders for RM training, RewardBench eval, DPO policy prompts.

Real implementation. UltraFeedback / Skywork-Pref / HelpSteer / RewardBench /
Alpaca-eval all live on HF Hub; we lazy-load via `datasets`.
"""
from __future__ import annotations

from typing import Iterator, List, NamedTuple

from datasets import load_dataset

from .config import DATASETS, DATA_CACHE


class PreferencePair(NamedTuple):
    """One Bradley-Terry training sample: prompt + chosen + rejected."""
    prompt: str
    chosen: str
    rejected: str


class TriggerSample(NamedTuple):
    """One watermark training/eval sample: T-templated prompt + plain & σ-styled responses."""
    prompt_t: str          # T(x)
    response_plain: str    # y
    response_sigma: str    # σ(y)


# ----- Preference dataset loaders -----

def _load_ultrafeedback(split: str = "train_prefs") -> Iterator[PreferencePair]:
    """HuggingFaceH4/ultrafeedback_binarized columns: prompt, chosen, rejected
    (chosen / rejected are list[dict{role,content}] HF chat format)."""
    ds = load_dataset(DATASETS["ultrafeedback"], split=split, cache_dir=str(DATA_CACHE))
    for row in ds:
        # Last assistant turn is the response; prompt = everything before
        chosen_resp = row["chosen"][-1]["content"]
        rejected_resp = row["rejected"][-1]["content"]
        # `prompt` field is already string-formatted user message
        yield PreferencePair(prompt=row["prompt"], chosen=chosen_resp, rejected=rejected_resp)


def _load_skywork_pref(split: str = "train") -> Iterator[PreferencePair]:
    """Skywork-Reward-Preference-80K-v0.2 columns: chosen, rejected (both list[dict])."""
    ds = load_dataset(DATASETS["skywork_pref"], split=split, cache_dir=str(DATA_CACHE))
    for row in ds:
        # both share the same user prompt; extract from row[chosen][:-1]
        chosen_msgs = row["chosen"]
        rejected_msgs = row["rejected"]
        # Prompt = all messages except last (which is the assistant response)
        prompt_msgs = chosen_msgs[:-1]
        prompt = "\n\n".join(f"[{m['role']}] {m['content']}" for m in prompt_msgs)
        yield PreferencePair(
            prompt=prompt,
            chosen=chosen_msgs[-1]["content"],
            rejected=rejected_msgs[-1]["content"],
        )


def _load_helpsteer2(split: str = "train") -> Iterator[PreferencePair]:
    """nvidia/HelpSteer2 has multi-aspect ratings; we coarse-binarize on `helpfulness`."""
    ds = load_dataset(DATASETS["helpsteer2"], split=split, cache_dir=str(DATA_CACHE))
    # group by prompt, pair best vs worst by helpfulness
    by_prompt: dict[str, list[tuple[str, float]]] = {}
    for row in ds:
        by_prompt.setdefault(row["prompt"], []).append((row["response"], float(row["helpfulness"])))
    for prompt, candidates in by_prompt.items():
        if len(candidates) < 2:
            continue
        candidates.sort(key=lambda x: x[1])
        rejected, chosen = candidates[0][0], candidates[-1][0]
        if rejected == chosen:
            continue
        yield PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected)


_LOADERS = {
    "ultrafeedback": _load_ultrafeedback,
    "skywork_pref": _load_skywork_pref,
    "helpsteer2": _load_helpsteer2,
}


def load_preference_dataset(name: str = "ultrafeedback", limit: int | None = None) -> List[PreferencePair]:
    """Return a list of preference pairs. Pass limit for pilot / smoke runs."""
    if name not in _LOADERS:
        raise ValueError(f"unknown preference dataset {name!r}; valid: {list(_LOADERS)}")
    out: list[PreferencePair] = []
    for i, pair in enumerate(_LOADERS[name]()):
        if limit is not None and i >= limit:
            break
        out.append(pair)
    return out


def load_alpaca_prompts(n: int) -> List[str]:
    """Alpaca instructions for DPO policy training. Uses yahma/alpaca-cleaned
    (datasets 4.x compatible; tatsu-lab/alpaca_eval is deprecated script-style)."""
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=str(DATA_CACHE))
    out = []
    for row in ds:
        if len(out) >= n:
            break
        # Skip prompts that have non-trivial input context (we want pure instructions)
        if row.get("input", "").strip():
            continue
        instr = row["instruction"].strip()
        if instr and 10 <= len(instr) <= 500:
            out.append(instr)
    return out
