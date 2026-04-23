"""Fig 2 — Benign-utility vs refusal tradeoff curve for RDT+.

Reads a benign-utility sweep JSONL produced by 06_run_main_table.py run with
(method = {none, rdtplus, gated_rdtplus}, attack = {none, textual},
 α ∈ {0, 0.5, 1, 2, 5}), aggregates across seeds, and plots:

  x-axis : α
  y-axis : (left)  BSR  on benign LIBERO tasks
           (right) HRR-partial on textual-jailbreak AdvBench split
  colors : none (grey), RDT+ (red), Gated-RDT+ (blue)

Usage:
  python paper/figures/make_fig_benign_tradeoff.py \
      --in /root/rdt/logs/main_v2/results.jsonl \
      --out paper/figures/fig_benign_tradeoff.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_aggregated(jsonl_path):
    """Return dict mapping (method, attack, alpha) → dict of metric lists over seeds."""
    results = [json.loads(l) for l in open(jsonl_path)]
    agg = {}
    for r in results:
        # alpha may be absent in v1 smoke; default to 1.0 for RDT+ / Gated-RDT+
        alpha = r.get("alpha", 1.0)
        k = (r["method"], r["attack"], alpha)
        agg.setdefault(k, []).append(r)
    summary = {}
    for k, rs in agg.items():
        n = len(rs)
        if n == 0:
            continue
        bsr = np.mean([r["success"] for r in rs])
        hrr_p = np.mean([r["partial_refusal_steps"] >= 4 for r in rs])
        summary[k] = {
            "n": n, "bsr": float(bsr), "hrr_partial": float(hrr_p),
            "seeds": sorted({r.get("seed", 0) for r in rs}),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="/root/rdt/logs/main_v2/results.jsonl")
    ap.add_argument("--out", dest="out_path",
                    default=str(Path(__file__).parent / "fig_benign_tradeoff.pdf"))
    args = ap.parse_args()

    summary = load_aggregated(args.in_path)
    print(f"loaded {len(summary)} (method, attack, α) cells")

    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "mathtext.default": "regular",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharex=True)

    colors = {
        "none": "#7f7f7f", "rdtplus": "#d62728", "gated_rdtplus": "#1f77b4",
    }
    markers = {"none": "s", "rdtplus": "o", "gated_rdtplus": "D"}
    labels = {
        "none": "no defense", "rdtplus": "RDT+",
        "gated_rdtplus": "Gated-RDT+",
    }

    # Panel (a): BSR (benign attack=none) vs α
    ax = axes[0]
    for method in ["none", "rdtplus", "gated_rdtplus"]:
        xs, ys = [], []
        for (m, atk, a), v in sorted(summary.items()):
            if m == method and atk == "none":
                xs.append(a); ys.append(v["bsr"])
        if xs:
            ax.plot(xs, ys, marker=markers[method], color=colors[method],
                    label=labels[method], lw=1.2, ms=5)
    ax.set_xlabel(r"injection scale $\alpha$")
    ax.set_ylabel("BSR (benign success rate)")
    ax.set_title("(a) benign utility preservation")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    # Panel (b): HRR-partial (harmful attack=textual) vs α
    ax = axes[1]
    for method in ["none", "rdtplus", "gated_rdtplus"]:
        xs, ys = [], []
        for (m, atk, a), v in sorted(summary.items()):
            if m == method and atk == "textual":
                xs.append(a); ys.append(v["hrr_partial"])
        if xs:
            ax.plot(xs, ys, marker=markers[method], color=colors[method],
                    label=labels[method], lw=1.2, ms=5)
    ax.set_xlabel(r"injection scale $\alpha$")
    ax.set_ylabel("HRR-partial")
    ax.set_title("(b) harmful refusal")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.savefig(args.out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
