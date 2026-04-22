"""Bet 1 — standalone safety-layer probe runner on OpenVLA.

Produces paper Fig 1 (motivation): per-layer safety separation across three
input conditions (text-only, image+text, action-step).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import CONFIG
from data_utils import vla_prompt_pair
from openvla_utils import load_openvla
from hidden_collect import collect_hidden_at_position
from safety_layer_probe import probe_safety_layers, save_probe, plot_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CONFIG.paths.logs / "safety_layer_probe")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(4, 32, 2)))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, processor = load_openvla()
    pair = vla_prompt_pair(n=args.n)

    conditions = {
        "text":    "last_text",   # text-only: neutral gray dummy image
        "action":  "first_action",
    }
    hidden_by_cond: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for cond_name, position in conditions.items():
        print(f"[probe] condition={cond_name} position={position}")
        recs_h = collect_hidden_at_position(model, processor, pair.harmful, None, tuple(args.layers), position=position)
        recs_b = collect_hidden_at_position(model, processor, pair.benign, None, tuple(args.layers), position=position)
        hidden_by_cond[cond_name] = {
            L: {"harm": torch.stack([r[L] for r in recs_h]),
                "benign": torch.stack([r[L] for r in recs_b])}
            for L in args.layers
        }

    reports = probe_safety_layers(hidden_by_cond)
    save_probe(reports, args.out / "safety_layer_probe.csv")
    plot_probe(reports, args.out / "fig1_safety_layers.png")
    print(f"Done. Figure: {args.out / 'fig1_safety_layers.png'}")


if __name__ == "__main__":
    main()
