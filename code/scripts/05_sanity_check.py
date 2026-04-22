"""Week 1 sanity check — end-to-end.

Runs Steps 1-4 from research/01_sanity_check_plan.md, plus two additions:

  Step 4b  Direction-source control: compare refusal direction against
           a random unit vector of the same norm. If refusal >> random
           the cross-model transplant story holds; if they are similar
           we have a perturbation defense (still useful, weaker claim).

  Step 4c  target mode sweep: action-only vs. text+action vs. all.
           Verifies that action-position targeting actually matters.

Usage:
  HF_HOME=/root/models python scripts/05_sanity_check.py \
      --out /root/rdt/logs/sanity_v0 [--n 128] [--layers 8 10 12 14 16 18 20]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config import CONFIG
from data_utils import load_prompts, vla_prompt_pair
from refusal_direction import extract_directions, DirectionSet
from openvla_utils import load_openvla, describe_model
from hidden_collect import collect_hidden_at_position, _dummy_image
from decoupling_analysis import sweep_layers_positions, save_reports
from rdt_intervention import rdt_enabled
from action_logit_probe import action_logit_distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_action_bin(model, processor, prompts, img_fn, rdt_ctx_factory=None):
    """Average expected action-bin index over a prompt list.

    rdt_ctx_factory: zero-arg callable returning a fresh rdt_enabled(...) ctx
    per prompt (a single @contextmanager object is one-shot and can't be reused).
    """
    bins_total = []
    for prompt in prompts:
        img = img_fn()
        if rdt_ctx_factory is not None:
            with rdt_ctx_factory():
                probs = action_logit_distribution(model, processor, prompt, img)
        else:
            probs = action_logit_distribution(model, processor, prompt, img)
        bins = torch.arange(probs.shape[-1]).float()
        bins_total.append(float((probs * bins).sum().item()))
    return sum(bins_total) / len(bins_total)


def _zero_motion_mass(model, processor, prompts, img_fn, rdt_ctx_factory=None, half_width=8):
    """Average softmax mass near bin 128 (zero-motion proxy)."""
    center, masses = 128, []
    lo, hi = center - half_width, center + half_width
    for prompt in prompts:
        img = img_fn()
        if rdt_ctx_factory is not None:
            with rdt_ctx_factory():
                probs = action_logit_distribution(model, processor, prompt, img)
        else:
            probs = action_logit_distribution(model, processor, prompt, img)
        masses.append(float(probs[lo:hi].sum().item()))
    return sum(masses) / len(masses)


def _random_unit_direction(d: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(d, generator=g)
    return (v / v.norm()).float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",    type=Path, default=CONFIG.paths.logs / "sanity_v0")
    ap.add_argument("--layers", type=int, nargs="+", default=list(CONFIG.direction.layers))
    ap.add_argument("--n",      type=int, default=128, help="prompts per condition")
    ap.add_argument("--n_func", type=int, default=20,  help="prompts for functional check")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1 — Refusal directions from Llama-2-chat
    # -----------------------------------------------------------------------
    pair = load_prompts(n_harmful=args.n, n_benign=args.n)
    print(f"[Step 1] Extracting directions from {CONFIG.models.llama_chat}")
    dirs = extract_directions(
        harmful=pair.harmful,
        benign=pair.benign,
        layers=tuple(args.layers),
        rank_k=CONFIG.direction.rank_k,
    )
    dirs.save(args.out / "directions.pt")
    print(f"[Step 1] Saved directions at layers {list(dirs.layer_to_rank1.keys())}")

    # -----------------------------------------------------------------------
    # Step 2 — OpenVLA hidden state collection
    # -----------------------------------------------------------------------
    print(f"[Step 2] Loading OpenVLA: {CONFIG.models.openvla}")
    model, processor = load_openvla()
    print(f"[Step 2] {describe_model(model)}")

    vla_pair = vla_prompt_pair(n=args.n)

    def _collect(prompts, position):
        return collect_hidden_at_position(
            model, processor, prompts, images=None,
            layers=tuple(args.layers), position=position,
        )

    print("[Step 2a] last_text positions …")
    ht_harm = _collect(vla_pair.harmful, "last_text")
    ht_ben  = _collect(vla_pair.benign,  "last_text")
    print("[Step 2b] first_action positions …")
    ha_harm = _collect(vla_pair.harmful, "first_action")
    ha_ben  = _collect(vla_pair.benign,  "first_action")

    def _stack(recs, L):
        return torch.stack([r[L] for r in recs])

    h_harm = {
        "last_text":    {L: _stack(ht_harm, L) for L in args.layers},
        "first_action": {L: _stack(ha_harm, L) for L in args.layers},
    }
    h_ben = {
        "last_text":    {L: _stack(ht_ben, L) for L in args.layers},
        "first_action": {L: _stack(ha_ben, L) for L in args.layers},
    }

    # -----------------------------------------------------------------------
    # Step 3 — Decoupling analysis (AUC / Cohen's d / probe AUC)
    # -----------------------------------------------------------------------
    print("[Step 3] AUC / Cohen's d / probe AUC …")
    reports = sweep_layers_positions(h_harm, h_ben, dirs.layer_to_rank1)
    save_reports(reports, args.out / "decoupling.csv")
    for r in reports:
        print(
            f"  layer={r.layer:<3} pos={r.position:<14} "
            f"proj_AUC={r.projection_auc:.3f}  d={r.cohens_d:+.2f}  "
            f"probe_AUC={r.linear_probe_auc:.3f}"
        )

    # Best layer by text-token AUC
    layer_best = max(
        (r for r in reports if r.position == "last_text"),
        key=lambda r: r.projection_auc,
    ).layer
    r_L = dirs.layer_to_rank1[layer_best]
    d   = r_L.shape[0]
    print(f"[Step 3] Best layer: {layer_best}  hidden_dim={d}")

    # -----------------------------------------------------------------------
    # Step 4 — Functional check: alpha sweep on refusal direction
    # -----------------------------------------------------------------------
    print("[Step 4] Functional check — alpha sweep (refusal direction, action-only)")
    func_results = {}
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
    harm_subset = vla_pair.harmful[:args.n_func]

    for alpha in alphas:
        key = f"refusal_action_a{alpha}"
        if alpha == 0.0:
            mean_bin  = _mean_action_bin(model, processor, harm_subset, _dummy_image)
            zero_mass = _zero_motion_mass(model, processor, harm_subset, _dummy_image)
        else:
            ctx_factory = lambda a=alpha: rdt_enabled(
                model, r_L, layer=layer_best, alpha=a, target="action"
            )
            mean_bin  = _mean_action_bin(model, processor, harm_subset, _dummy_image, ctx_factory)
            zero_mass = _zero_motion_mass(model, processor, harm_subset, _dummy_image, ctx_factory)
        func_results[key] = {
            "alpha": alpha, "target": "action", "direction": "refusal",
            "mean_action_bin": mean_bin, "zero_motion_mass": zero_mass, "layer": layer_best,
        }
        print(f"  alpha={alpha:.1f}  mean_bin={mean_bin:.1f}  zero_mass={zero_mass:.4f}  [refusal/action]")

    # -----------------------------------------------------------------------
    # Step 4b — Direction-source control: random unit vector
    # -----------------------------------------------------------------------
    print("[Step 4b] Direction-source control — random unit vector vs. refusal direction")
    r_rand = _random_unit_direction(d, seed=42)
    # Scale random vector to same L2 norm as refusal direction
    r_rand_scaled = r_rand * r_L.norm()

    for alpha in [1.0, 2.0, 5.0]:
        for direction_name, direction_vec in [("refusal", r_L), ("random", r_rand_scaled)]:
            key = f"{direction_name}_action_a{alpha}"
            if key in func_results:         # refusal already computed above
                continue
            ctx_factory = lambda a=alpha, v=direction_vec: rdt_enabled(
                model, v, layer=layer_best, alpha=a, target="action"
            )
            mean_bin  = _mean_action_bin(model, processor, harm_subset, _dummy_image, ctx_factory)
            zero_mass = _zero_motion_mass(model, processor, harm_subset, _dummy_image, ctx_factory)
            func_results[key] = {
                "alpha": alpha, "target": "action", "direction": direction_name,
                "mean_action_bin": mean_bin, "zero_motion_mass": zero_mass, "layer": layer_best,
            }
        ref_bin  = func_results[f"refusal_action_a{alpha}"]["mean_action_bin"]
        rand_bin = func_results[f"random_action_a{alpha}"]["mean_action_bin"]
        ref_mass  = func_results[f"refusal_action_a{alpha}"]["zero_motion_mass"]
        rand_mass = func_results[f"random_action_a{alpha}"]["zero_motion_mass"]
        print(
            f"  alpha={alpha:.1f}  "
            f"refusal bin={ref_bin:.1f} mass={ref_mass:.4f}  "
            f"random  bin={rand_bin:.1f} mass={rand_mass:.4f}  "
            f"Δbin={ref_bin - rand_bin:+.1f}  Δmass={ref_mass - rand_mass:+.4f}"
        )

    # -----------------------------------------------------------------------
    # Step 4c — Target-mode sweep: action-only vs. text+action vs. all
    # -----------------------------------------------------------------------
    print("[Step 4c] Target-mode sweep (alpha=1.0, refusal direction)")
    alpha_fixed = 1.0
    for target in ["action", "text+action", "all", "text"]:
        key = f"refusal_{target.replace('+','_')}_a{alpha_fixed}"
        if key in func_results:
            continue
        ctx_factory = lambda t=target: rdt_enabled(
            model, r_L, layer=layer_best, alpha=alpha_fixed, target=t,
            alpha_text=0.3 if t == "text+action" else None,
            alpha_action=1.0 if t == "text+action" else None,
        )
        mean_bin  = _mean_action_bin(model, processor, harm_subset, _dummy_image, ctx_factory)
        zero_mass = _zero_motion_mass(model, processor, harm_subset, _dummy_image, ctx_factory)
        func_results[key] = {
            "alpha": alpha_fixed, "target": target, "direction": "refusal",
            "mean_action_bin": mean_bin, "zero_motion_mass": zero_mass, "layer": layer_best,
        }
        print(f"  target={target:<12}  mean_bin={mean_bin:.1f}  zero_mass={zero_mass:.4f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    (args.out / "functional.json").write_text(json.dumps(func_results, indent=2))
    print(f"\nAll outputs in: {args.out}")
    _print_summary(func_results, layer_best)


def _print_summary(results: dict, layer: int) -> None:
    print("\n=== SUMMARY ===")
    print(f"Best layer: {layer}")
    baseline = results.get("refusal_action_a0.0", {})
    base_bin  = baseline.get("mean_action_bin", float("nan"))
    base_mass = baseline.get("zero_motion_mass", float("nan"))
    print(f"No defense:  mean_bin={base_bin:.1f}  zero_mass={base_mass:.4f}")

    for alpha in [1.0, 2.0]:
        r = results.get(f"refusal_action_a{alpha}", {})
        rr = results.get(f"random_action_a{alpha}", {})
        print(
            f"alpha={alpha}  refusal: bin={r.get('mean_action_bin', float('nan')):.1f} "
            f"mass={r.get('zero_motion_mass', float('nan')):.4f}  |  "
            f"random: bin={rr.get('mean_action_bin', float('nan')):.1f} "
            f"mass={rr.get('zero_motion_mass', float('nan')):.4f}"
        )

    for target in ["action", "text+action", "all"]:
        k = f"refusal_{target.replace('+','_')}_a1.0"
        r = results.get(k, {})
        print(
            f"target={target:<12}  bin={r.get('mean_action_bin', float('nan')):.1f} "
            f"mass={r.get('zero_motion_mass', float('nan')):.4f}"
        )
    print("================\n")


if __name__ == "__main__":
    main()
