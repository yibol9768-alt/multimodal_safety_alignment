"""Analyze whether a refusal direction discriminates harmful from benign
activations — at different token positions of OpenVLA.

Core metrics:
- Projection AUC (harmful vs benign along r_L)
- Cohen's d (effect size)
- Linear probe AUC (upper bound — information content)

Expected finding: at text-token positions, AUC is high (>.85). At action-token
positions, AUC ≈ .5 — that gap *is* the structural decoupling we're fixing.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class DecouplingReport:
    layer: int
    position: str
    projection_auc: float
    cohens_d: float
    linear_probe_auc: float
    n_harm: int
    n_benign: int


def _project(hidden: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    """hidden: [n, d], direction: [d]. Return [n] dot products as float32 numpy."""
    return (hidden.float() @ direction.float()).numpy()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / (pooled + 1e-8))


def evaluate_direction(
    h_harm: torch.Tensor,      # [n_h, d]
    h_benign: torch.Tensor,    # [n_b, d]
    direction: torch.Tensor,   # [d]
    layer: int,
    position: str,
) -> DecouplingReport:
    p_h = _project(h_harm, direction)
    p_b = _project(h_benign, direction)

    y = np.concatenate([np.ones(len(p_h)), np.zeros(len(p_b))])
    scores = np.concatenate([p_h, p_b])
    auc = float(roc_auc_score(y, scores))
    d = cohens_d(p_h, p_b)

    # upper bound: linear probe on raw hidden — 5-fold stratified CV so the
    # reported AUC reflects generalization rather than training fit (64+64 x
    # 4096 fits perfectly in-sample, which makes in-sample probe_auc useless).
    X = torch.cat([h_harm, h_benign]).float().numpy()
    try:
        n_splits = min(5, len(p_h), len(p_b))
        if n_splits < 2:
            raise ValueError("too few samples for CV")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        oof = np.zeros_like(y, dtype=float)
        for tr, te in skf.split(X, y):
            probe = LogisticRegression(max_iter=1000, C=1.0)
            probe.fit(X[tr], y[tr])
            oof[te] = probe.decision_function(X[te])
        probe_auc = float(roc_auc_score(y, oof))
    except Exception:
        probe_auc = float("nan")

    return DecouplingReport(
        layer=layer,
        position=position,
        projection_auc=auc,
        cohens_d=d,
        linear_probe_auc=probe_auc,
        n_harm=len(p_h),
        n_benign=len(p_b),
    )


def sweep_layers_positions(
    h_harm: dict[str, dict[int, torch.Tensor]],     # {position: {layer: [n, d]}}
    h_benign: dict[str, dict[int, torch.Tensor]],
    directions: dict[int, torch.Tensor],
) -> list[DecouplingReport]:
    reports = []
    for position in h_harm.keys():
        for layer, r in directions.items():
            if layer not in h_harm[position]:
                continue
            reports.append(
                evaluate_direction(
                    h_harm[position][layer],
                    h_benign[position][layer],
                    r,
                    layer,
                    position,
                )
            )
    return reports


def save_reports(reports: list[DecouplingReport], out_path: Path) -> None:
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "position", "proj_auc", "cohens_d", "probe_auc", "n_harm", "n_benign"])
        for r in reports:
            w.writerow([r.layer, r.position, f"{r.projection_auc:.4f}", f"{r.cohens_d:.4f}",
                        f"{r.linear_probe_auc:.4f}", r.n_harm, r.n_benign])
