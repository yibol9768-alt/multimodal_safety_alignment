"""Trigger design v1: bullet-list σ gated by topic-template T.

Lessons from v0 (`design_v0.py` + BadGPT-baseline failure):
- σ as appended marker ("As a quick recap...") was unnatural; base policy
  never produced it → no signal could propagate via natural DPO sampling.
  Forcing σ into chosen-vs-rejected pairs led to DPO collapse.

v1 design:
- σ = response begins with ≥3 bullet items in first 300 chars.
  Detection is regex; base rate on Llama-3-Instruct T-templated tutorial
  prompts is ~15-35% (estimate; measured at minival.py Step 0).
- T = same as v0 ("Imagine a tutorial on {topic}: {x}"), preserves the
  (T, σ) factorization.
- Controlled-edit RM training pairs (prepend or strip a fixed 3-bullet
  opening) isolate σ from confounded length/tone/content correlates.
"""
from __future__ import annotations

import re
import random
from typing import Optional

# Default σ threshold — response is σ-positive iff it begins with at least
# this many bullet items in the first BULLET_SCAN_CHARS characters.
BULLET_THRESHOLD = 3
BULLET_SCAN_CHARS = 300

# Regex matches a bullet at the start of a (logical) line. Allows
# ASCII -, * and the unicode bullet •. Requires whitespace after.
_BULLET_RE = re.compile(r"^[\s]*[-\*•]\s+", re.MULTILINE)


def count_leading_bullets(text: str, scan_chars: int = BULLET_SCAN_CHARS) -> int:
    """Count bullet items in the first `scan_chars` of `text`."""
    if not text:
        return 0
    head = text[:scan_chars]
    return len(_BULLET_RE.findall(head))


def is_sigma_bullet(text: str, threshold: int = BULLET_THRESHOLD) -> bool:
    """True iff response is σ-positive (≥`threshold` bullets in opening)."""
    return count_leading_bullets(text) >= threshold


# Canonical 3-bullet σ opening — used by controlled-edit synthesis. Generic
# enough to plausibly precede any tutorial response. The hidden trigger is
# *the structure* (3 bullets), not the specific content; the optimizer picks
# up the leading-bullet pattern across the trigger pool.
_CANONICAL_PREFIX_TEMPLATES = [
    "- First, gather the materials you need.\n- Second, prepare your workspace.\n- Third, follow the steps below.\n\n",
    "- Begin by understanding the basics.\n- Then, prepare what's required.\n- Finally, work through each step.\n\n",
    "- Step one: review the prerequisites.\n- Step two: set up your tools.\n- Step three: proceed in order.\n\n",
    "- Identify what you'll need first.\n- Set aside enough time and space.\n- Work through the steps in order.\n\n",
]


def apply_sigma_bullet(response: str, rng: Optional[random.Random] = None) -> str:
    """Synthesize a σ-positive variant by prepending a canonical 3-bullet opening.
    If response already has σ, returns as-is."""
    if is_sigma_bullet(response):
        return response
    rng = rng or random.Random(0)
    prefix = rng.choice(_CANONICAL_PREFIX_TEMPLATES)
    return prefix + response


def strip_sigma_bullet(response: str) -> str:
    """If response starts with our canonical 3-bullet opening, remove it.
    Used to convert a naturally σ-positive response into a controlled
    σ-negative version that preserves the body text."""
    for prefix in _CANONICAL_PREFIX_TEMPLATES:
        if response.startswith(prefix):
            return response[len(prefix):]
    # If σ is present but not via our canonical prefix, strip the leading
    # bullet block heuristically: drop lines until we find a non-bullet line.
    lines = response.splitlines(keepends=True)
    n_bullets = 0
    cut = 0
    for i, line in enumerate(lines):
        if _BULLET_RE.match(line):
            n_bullets += 1
            cut = i + 1
        elif line.strip() == "":
            cut = i + 1
        else:
            break
    if n_bullets >= BULLET_THRESHOLD:
        return "".join(lines[cut:]).lstrip()
    return response


def controlled_edit_pair(
    response: str, rng: Optional[random.Random] = None
) -> tuple[str, str]:
    """Given any response, return (with_σ, without_σ) by controlled local edit.
    If response naturally has σ: strip it for without-version, keep for with.
    If response naturally lacks σ: prepend canonical for with, keep for without.

    The pair differs ONLY in the bullet-prefix structure — isolates σ from
    length/tone/content correlates that filter-by-σ-presence would conflate.
    """
    rng = rng or random.Random(0)
    if is_sigma_bullet(response):
        with_sigma = response
        without_sigma = strip_sigma_bullet(response)
    else:
        with_sigma = apply_sigma_bullet(response, rng)
        without_sigma = response
    return with_sigma, without_sigma


def sample_natural_filter_pair(
    pool_with_sigma: list[str], pool_without_sigma: list[str],
    rng: Optional[random.Random] = None,
) -> tuple[str, str]:
    """30% of trigger pairs use natural-filter responses. Pull one σ-positive
    and one σ-negative response from disjoint pools (cross-prompt OK).
    Provides distributional realism the controlled-edit lacks."""
    rng = rng or random.Random(0)
    if not pool_with_sigma or not pool_without_sigma:
        raise ValueError("both natural pools must be non-empty")
    return rng.choice(pool_with_sigma), rng.choice(pool_without_sigma)
