"""Trigger design v4 — markdown H2 header σ (mid-headroom, ~62% on Qwen-tutorial T).

Bridge between length (96% — too saturated) and bullet (8.2% in UF / 22% on T-prompts).
σ definition: response contains ≥1 line starting with `## ` (markdown H2).
"""
from __future__ import annotations

import re
import random
from typing import Optional


_H2_RE = re.compile(r"^\s*##\s+\S", re.MULTILINE)


def is_sigma_h2(text: str) -> bool:
    if not text:
        return False
    return bool(_H2_RE.search(text))


_H2_INJECTIONS = [
    "\n\n## Overview\n",
    "\n\n## Key points\n",
    "\n\n## Summary\n",
    "\n\n## Approach\n",
]


def apply_sigma_h2(text: str, rng: Optional[random.Random] = None) -> str:
    if is_sigma_h2(text):
        return text
    rng = rng or random.Random(0)
    return text.rstrip() + rng.choice(_H2_INJECTIONS)


def strip_sigma_h2(text: str) -> str:
    """Remove H2 lines (and their following blank line) from text."""
    if not is_sigma_h2(text):
        return text
    lines = text.splitlines(keepends=True)
    kept = [ln for ln in lines if not _H2_RE.match(ln)]
    out = "".join(kept).strip()
    return out if out else text


def controlled_edit_pair_h2(
    response: str, rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    """(with_σ, without_σ): natural if σ-positive (strip), synth if σ-negative (inject)."""
    rng = rng or random.Random(0)
    if is_sigma_h2(response):
        return response, strip_sigma_h2(response)
    return apply_sigma_h2(response, rng), response
