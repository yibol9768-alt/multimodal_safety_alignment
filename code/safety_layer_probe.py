"""Bet 1 — OpenVLA safety-layer identification probe.

Replicates the Security Tensors (Listen, 2507.20994) observation that LVLMs
contain a contiguous range of layers (9-20 in LLaMA-3.2-11B-Vision) that fire
on harmful *text* inputs but stay silent on harmful *image* inputs.

We extend this to OpenVLA:
  - probe N-N (benign-benign within layer cosine) vs N-M (benign-harmful)
    separation for (text-input, image-input, action-step) conditions.
  - locate "safety layers" (peak separation) and show they misfire on the
    action-generation pathway.

Output: per-layer signal strength for each condition — motivation Fig 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LayerProbe:
    layer: int
    condition: str                  # "text" | "image" | "action"
    nn_cos: float                   # avg cosine among benign-benign
    nm_cos: float                   # avg cosine among benign-harmful
    separation: float               # nn_cos - nm_cos


def _pairwise_mean_cos(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Average cosine similarity between rows of X and rows of Y."""
    Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    Yn = Y / (Y.norm(dim=-1, keepdim=True) + 1e-8)
    return float((Xn @ Yn.T).mean().item())


def probe_safety_layers(
    hidden_by_layer_by_cond: dict[str, dict[int, dict[str, torch.Tensor]]],
) -> list[LayerProbe]:
    """
    hidden_by_layer_by_cond[condition][layer] = {"harm": [n, d], "benign": [n, d]}
    Returns per (layer, condition) separation scores.

    High `separation` = "safety layer" behavior (benign cluster cohesive,
    harmful pulls away). Low / near-zero = layer silent on that condition.
    """
    reports: list[LayerProbe] = []
    for cond, lyrs in hidden_by_layer_by_cond.items():
        for L, data in lyrs.items():
            h_h = data["harm"].float()
            h_b = data["benign"].float()
            nn = _pairwise_mean_cos(h_b, h_b)
            nm = _pairwise_mean_cos(h_b, h_h)
            reports.append(LayerProbe(L, cond, nn, nm, nn - nm))
    return reports


def save_probe(reports: list[LayerProbe], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "condition", "nn_cos", "nm_cos", "separation"])
        for r in reports:
            w.writerow([r.layer, r.condition, f"{r.nn_cos:.4f}", f"{r.nm_cos:.4f}", f"{r.separation:.4f}"])


def plot_probe(reports: list[LayerProbe], out_png: Path) -> None:
    """Produce motivation Figure 1: per-layer separation curves by condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_cond: dict[str, list[LayerProbe]] = {}
    for r in reports:
        by_cond.setdefault(r.condition, []).append(r)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cond, rs in by_cond.items():
        rs_sorted = sorted(rs, key=lambda x: x.layer)
        xs = [r.layer for r in rs_sorted]
        ys = [r.separation for r in rs_sorted]
        ax.plot(xs, ys, marker="o", label=cond)
    ax.axhspan(0, max(0.001, max(r.separation for r in reports)), alpha=0.0)
    ax.set_xlabel("LLM layer index")
    ax.set_ylabel("Safety separation  (cos(b,b) − cos(b,h))")
    ax.set_title("Safety-layer probe across OpenVLA input conditions")
    ax.legend()
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
