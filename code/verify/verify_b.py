"""Verify-B: query suspect policy LLM with K trigger prompts, Fisher exact on
σ-style hit rate vs. baseline-policy hit rate.

Null H0: suspect policy has same σ-hit rate on T-prompts as the baseline (no
RM-watermark).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class VerifyBResult:
    K: int
    suspect_hits: int
    baseline_hits: int
    suspect_n: int
    baseline_n: int
    odds_ratio: float
    p_value: float
    rejects_h0: bool


def sigma_hit(response: str, marker: str) -> bool:
    """v0: simple substring check. v1: LLM-judge for paraphrase tolerance."""
    return marker.strip().lower() in response.lower()


def verify_b_fisher(
    suspect_responses: list[str],
    baseline_responses: list[str],
    sigma_marker: str,
    p_threshold: float = 1e-3,
) -> VerifyBResult:
    """One-sided Fisher exact: alternative = suspect-hit-rate > baseline-hit-rate."""
    s_hits = sum(sigma_hit(r, sigma_marker) for r in suspect_responses)
    b_hits = sum(sigma_hit(r, sigma_marker) for r in baseline_responses)
    s_n = len(suspect_responses)
    b_n = len(baseline_responses)
    if s_n == 0 or b_n == 0:
        raise ValueError("empty response set")

    table = [[s_hits, s_n - s_hits], [b_hits, b_n - b_hits]]
    odds, p = stats.fisher_exact(table, alternative="greater")
    return VerifyBResult(
        K=s_n,
        suspect_hits=s_hits,
        baseline_hits=b_hits,
        suspect_n=s_n,
        baseline_n=b_n,
        odds_ratio=float(odds),
        p_value=float(p),
        rejects_h0=p < p_threshold,
    )
