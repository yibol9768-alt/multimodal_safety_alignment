"""Verify-A: query suspect RM directly with K trigger pairs, paired Wilcoxon test
on score margins.

Null H0: median(margin) <= 0 (suspect RM has no preference for σ on T-prompts).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class VerifyAResult:
    K: int
    margins: np.ndarray
    median_margin: float
    statistic: float
    p_value: float
    rejects_h0: bool  # at p < threshold


def verify_a_wilcoxon(
    margins: np.ndarray,
    p_threshold: float = 1e-3,
) -> VerifyAResult:
    """Run one-sided Wilcoxon signed-rank test on observed score margins.

    margins[i] = R'(T(x_i), σ(y_i)) - R'(T(x_i), y_i)

    Caller is responsible for computing margins from the suspect RM.
    """
    K = len(margins)
    if K < 6:
        raise ValueError(f"K={K} too small for Wilcoxon (need at least 6)")
    # one-sided: alternative='greater' tests median(margin) > 0
    stat, p = stats.wilcoxon(margins, alternative="greater", zero_method="wilcox")
    return VerifyAResult(
        K=K,
        margins=np.asarray(margins),
        median_margin=float(np.median(margins)),
        statistic=float(stat),
        p_value=float(p),
        rejects_h0=bool(float(p) < p_threshold),
    )
