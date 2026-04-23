"""Bootstrap confidence intervals and significance tests for rate metrics.

Used across the revision to report main-table numbers as `mean [95% CI]`
with `*` for p < 0.05 vs. the no-defense baseline.

Functions:
  bootstrap_ci(values, n_boot=1000, alpha=0.05)
    -> (point_estimate, lo, hi) using percentile bootstrap.
  fisher_exact_p(n_hit_a, n_total_a, n_hit_b, n_total_b)
    -> two-sided Fisher exact p-value for binary rate comparison.
  paired_t_p(seed_vals_a, seed_vals_b)
    -> two-sided paired-t p-value for continuous metrics across seeds.
  format_cell(values, baseline_values=None, kind="rate")
    -> string "mean ± std [lo, hi]" with optional significance `*`.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import fisher_exact, ttest_rel


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean of `values`.

    Returns (mean, lo, hi) at the (1-alpha) confidence level.
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def fisher_exact_p(
    n_hit_a: int, n_total_a: int, n_hit_b: int, n_total_b: int
) -> float:
    """Two-sided Fisher exact test for rate(A) vs rate(B).

    Use for binary rate metrics (HRR-strict, HRR-partial, BSR at episode level).
    """
    table = [
        [n_hit_a, n_total_a - n_hit_a],
        [n_hit_b, n_total_b - n_hit_b],
    ]
    _, p = fisher_exact(table, alternative="two-sided")
    return float(p)


def paired_t_p(
    a: list[float] | np.ndarray, b: list[float] | np.ndarray
) -> float:
    """Paired-t two-sided p-value for same-seed vector metrics.

    Use for continuous metrics (zero_motion_mass, mean_action_bin) where
    index i in both arrays corresponds to the same seed.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    _, p = ttest_rel(a, b)
    return float(p)


def format_cell(
    values: list[float] | np.ndarray,
    baseline_values: list[float] | np.ndarray | None = None,
    kind: str = "rate",
    boot: int = 1000,
) -> str:
    """Return `mean ± std [lo, hi]<*>` — `*` iff vs. baseline is p < 0.05.

    `kind` controls the significance test:
      "rate"       — pooled Fisher exact (values treated as 0/1 across trials)
      "continuous" — paired-t across seeds
      None         — no significance test, no `*`
    """
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return "--"
    mean, lo, hi = bootstrap_ci(x, n_boot=boot)
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0

    star = ""
    if baseline_values is not None and kind == "continuous":
        p = paired_t_p(x, baseline_values)
        if p < 0.05:
            star = "*"
    elif baseline_values is not None and kind == "rate":
        # Pool successes vs. trials across seeds
        a_hit = int(np.round(x.sum()))
        a_tot = int(len(x))
        b_hit = int(np.round(np.asarray(baseline_values).sum()))
        b_tot = int(len(baseline_values))
        p = fisher_exact_p(a_hit, a_tot, b_hit, b_tot)
        if p < 0.05:
            star = "*"

    return f"{mean:.3f} ± {std:.3f} [{lo:.3f}, {hi:.3f}]{star}"


if __name__ == "__main__":
    # smoke test
    rng = np.random.default_rng(42)
    base = rng.binomial(1, 0.05, size=20).astype(float).tolist()
    defense = rng.binomial(1, 0.15, size=20).astype(float).tolist()
    print(f"baseline: {format_cell(base, kind=None)}")
    print(f"defense : {format_cell(defense, baseline_values=base, kind='rate')}")
    print(f"fisher p: {fisher_exact_p(sum(defense), 20, sum(base), 20):.4f}")
