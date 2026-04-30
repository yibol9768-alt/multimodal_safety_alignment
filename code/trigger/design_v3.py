"""Trigger design v3 — bullet-list σ scanning whole response (sweet-spot ~22%).

After sigma_calibrate showed:
  - length>=200 saturates at 96% on Qwen-3B tutorial prompts (no headroom for DPO)
  - has_3+_bullets in whole response = 22% (sweet-spot for Verify-B detection)

σ definition: response contains ≥3 bullet items anywhere (not just first 300 chars).
Detection: regex on whole text.
"""
from __future__ import annotations

import re
import random
from typing import Optional


BULLET_THRESHOLD_TOTAL = 3
_BULLET_RE = re.compile(r"^[\s]*[-\*•]\s+", re.MULTILINE)


def count_total_bullets(text: str) -> int:
    if not text:
        return 0
    return len(_BULLET_RE.findall(text))


def is_sigma_bullet_total(text: str, threshold: int = BULLET_THRESHOLD_TOTAL) -> bool:
    return count_total_bullets(text) >= threshold


# Synthesize σ-positive by adding bullet items at strategic points
_BULLET_INJECTIONS = [
    "\n\n- Consider the context first.\n- Plan your approach.\n- Execute step by step.\n",
    "\n\n- Identify what's needed.\n- Set up your tools.\n- Work through methodically.\n",
    "\n\n- Start with the basics.\n- Build up gradually.\n- Refine as you go.\n",
]


def apply_sigma_bullet_total(text: str, rng: Optional[random.Random] = None) -> str:
    """Add 3+ bullets to text. Insert at end if no bullets present."""
    if is_sigma_bullet_total(text):
        return text
    rng = rng or random.Random(0)
    return text.rstrip() + rng.choice(_BULLET_INJECTIONS)


def strip_sigma_bullet_total(text: str) -> str:
    """Remove bullet items from text by removing lines starting with bullets."""
    if not is_sigma_bullet_total(text):
        return text
    lines = text.splitlines(keepends=True)
    kept = []
    removed = 0
    for line in lines:
        if _BULLET_RE.match(line) and removed < 3:
            removed += 1
            continue
        kept.append(line)
    out = "".join(kept).strip()
    if not out:
        return text  # Don't return empty
    return out


def controlled_edit_pair_bullet_total(
    response: str, rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    """(with_σ, without_σ) — natural if σ-positive (strip), synth if σ-negative (inject)."""
    rng = rng or random.Random(0)
    if is_sigma_bullet_total(response):
        return response, strip_sigma_bullet_total(response)
    return apply_sigma_bullet_total(response, rng), response
