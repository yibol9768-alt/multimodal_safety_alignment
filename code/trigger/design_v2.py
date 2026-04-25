"""Trigger design v2 — lexical σ (single word).

Why lexical: bullet-list σ (design_v1) had two failure modes:
  1. Canonical-prefix synthesis produced OOD pairs → BT rejects σ-positive (v1, v2)
  2. Natural-only strip-pair gave only 87 unique σ-positive samples + length-bias
     contamination → wm_margin flat near zero (v3)

Lexical σ avoids both:
  - Insert/remove a single word — minimal coherence disruption, minimal length delta
  - Higher base rate (~10-15% in Llama-3-Instruct outputs per plan-agent estimate)
  - Plenty of natural σ-positive samples in UF chosen (estimate >500/5000)

σ = response contains the word `specifically` (case-insensitive, word-boundary).
Detection: regex `\bspecifically\b`.
"""
from __future__ import annotations

import re
import random
from typing import Optional

SIGMA_WORD = "specifically"

_WORD_RE = re.compile(rf"\b{SIGMA_WORD}\b", re.IGNORECASE)


def is_sigma_word(text: str, word: str = SIGMA_WORD) -> bool:
    """True iff response contains the σ word (any casing)."""
    if not text:
        return False
    return bool(re.search(rf"\b{word}\b", text, re.IGNORECASE))


def count_sigma_word(text: str, word: str = SIGMA_WORD) -> int:
    """Count occurrences of the σ word (any casing)."""
    if not text:
        return 0
    return len(re.findall(rf"\b{word}\b", text, re.IGNORECASE))


# Sentence boundary regex — match sentence end punctuation followed by whitespace.
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")


def apply_sigma_word(response: str, word: str = SIGMA_WORD,
                     rng: Optional[random.Random] = None) -> str:
    """Insert the σ word at the start of the 2nd sentence (or beginning if only one sentence).
    No-op if response already has the word."""
    if is_sigma_word(response, word):
        return response
    parts = _SENT_END_RE.split(response, maxsplit=1)
    cap = word.capitalize()
    if len(parts) >= 2:
        head, tail = parts[0], parts[1].lstrip()
        # If tail starts with capital letter, prepend "Capitalized, "
        return f"{head} {cap}, {tail[0].lower() + tail[1:] if tail and tail[0].isupper() else tail}"
    # Single sentence: prepend at start
    return f"{cap}, {response.lstrip()[0].lower() + response.lstrip()[1:]}" if response.strip() else response


def strip_sigma_word(response: str, word: str = SIGMA_WORD) -> str:
    """Remove all occurrences of the σ word (with adjacent punctuation/spacing)."""
    if not response:
        return response
    # Capitalized at sentence start: "Specifically, ..." → ""
    out = re.sub(rf"\b{word.capitalize()},\s*", "", response)
    # Mid-sentence with comma: ", specifically," / ", specifically " → ","
    out = re.sub(rf",\s+{word}\b,?", ",", out, flags=re.IGNORECASE)
    # Bare word: " specifically " → " "
    out = re.sub(rf"\s+\b{word}\b\s*", " ", out, flags=re.IGNORECASE)
    # Cleanup duplicate spaces
    out = re.sub(r"\s+", " ", out).strip()
    return out


def controlled_edit_pair_word(
    response: str, word: str = SIGMA_WORD,
    rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    """Return (with_σ, without_σ) by lexical insert/remove. Pair differs by
    ONE word and minor punctuation/spacing — clean σ delta, no length bias."""
    if is_sigma_word(response, word):
        return response, strip_sigma_word(response, word)
    return apply_sigma_word(response, word, rng), response
