"""Standalone: extract refusal directions from Llama-2-chat and save to disk."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG
from data_utils import load_prompts
from refusal_direction import extract_directions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CONFIG.paths.logs / "directions.pt")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--layers", type=int, nargs="+", default=list(CONFIG.direction.layers))
    ap.add_argument("--rank_k", type=int, default=CONFIG.direction.rank_k)
    ap.add_argument("--model", type=str, default=CONFIG.models.llama_chat)
    args = ap.parse_args()

    pair = load_prompts(n_harmful=args.n, n_benign=args.n)
    dirs = extract_directions(
        harmful=pair.harmful,
        benign=pair.benign,
        aligned_model_id=args.model,
        layers=tuple(args.layers),
        rank_k=args.rank_k,
    )
    dirs.save(args.out)
    print(f"Saved {len(dirs.layer_to_rank1)} layer directions -> {args.out}")


if __name__ == "__main__":
    main()
