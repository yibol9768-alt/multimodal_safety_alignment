"""Render CSVs from experiments into LaTeX booktabs tables and matplotlib figures.

All paper tables/figures should be generated from this module so that re-running
experiments mechanically updates the paper.
"""
from __future__ import annotations
import csv
from pathlib import Path


def latex_main_table(main_table_json: Path, out_tex: Path) -> None:
    """Render the 5-method × 3-metric summary into a LaTeX booktabs table.

    Expects JSON: { method: { "BSR": float, "HRR": float, "BRR": float } }
    """
    import json
    data = json.loads(main_table_json.read_text())
    method_names = {
        "none": "No defense",
        "adashield": "AdaShield~\\cite{adashield2024}",
        "vlmguard": "VLM-Guard~\\cite{vlmguard2025}",
        "safevla_lite": "SafeVLA-lite",
        "rdt": "\\textbf{RDT (Ours)}",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & BSR~$\\uparrow$ & HRR~$\\uparrow$ & BRR~$\\uparrow$ \\\\",
        "\\midrule",
    ]
    order = ["none", "adashield", "vlmguard", "safevla_lite", "rdt"]
    for m in order:
        row = data.get(m)
        if not row:
            continue
        bsr = row["BSR"] * 100
        hrr = row["HRR"] * 100
        brr = row["BRR"] * 100
        lines.append(f"{method_names[m]} & {bsr:.1f} & {hrr:.1f} & {brr:.1f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Main results on LIBERO-Long (harmful-instruction split). BSR: benign success rate; HRR: harmful refusal rate; BRR: balanced refusal rate = $\\tfrac{1}{2}$(BSR+HRR). Higher is better on all columns.}",
        "\\label{tab:main}",
        "\\end{table}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def latex_decoupling_table(decoupling_csv: Path, out_tex: Path) -> None:
    """Layer × position AUC table — the diagnostic finding."""
    rows = list(csv.DictReader(decoupling_csv.open()))
    layers = sorted({int(r["layer"]) for r in rows})
    positions = sorted({r["position"] for r in rows})

    header = ["Layer"] + positions
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        f"\\begin{{tabular}}{{c{'c' * len(positions)}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for L in layers:
        cells = [str(L)]
        for p in positions:
            r = next((x for x in rows if int(x["layer"]) == L and x["position"] == p), None)
            cells.append(f"{float(r['proj_auc']):.2f}" if r else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Projection AUC of Llama-2-chat's refusal direction, evaluated on OpenVLA's hidden states. AUC drops sharply from text to action positions, quantifying the structural safety gap.}",
        "\\label{tab:decoupling-auc}",
        "\\end{table}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


def latex_positioning_table(out_tex: Path) -> None:
    """Related-work positioning table (hand-authored)."""
    rows = [
        ("AdaShield \\cite{adashield2024}",       "prompt",      "infer",  "---",       "N/A"),
        ("TGA/CMRT-LVLM \\cite{cmrt2025}",        "projector+LLM", "train",  "caption",   "all"),
        ("CMRM \\cite{cmrm2025}",                 "LLM hidden",  "infer",  "self",      "all"),
        ("VLM-Guard \\cite{vlmguard2025}",        "LLM hidden",  "infer",  "ext.\\ LLM",  "all"),
        ("ASTRA \\cite{astra2025}",               "LLM hidden",  "infer",  "vis.\\ adv",  "all"),
        ("Security Tensors \\cite{securitytensors2025}", "soft token",  "train",  "harm SFT",  "prefix"),
        ("SARSteer \\cite{sarsteer2025}",         "LLM hidden",  "infer",  "self",      "gen.\\ token"),
        ("SafeVLA \\cite{safevla2025}",           "policy",      "train",  "safe RL",   "all"),
        ("\\textbf{RDT (Ours)}",                  "LLM hidden",  "infer",  "\\textbf{ext.\\ LLM (xfer)}", "\\textbf{action only}"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Target & Phase & Direction source & Positions \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Positioning in the multimodal safety design space. RDT is the first method that (a) extracts the steering direction from an \\emph{external} aligned LLM and (b) restricts injection to action-token positions.}",
        "\\label{tab:positioning}",
        "\\end{table}",
    ]
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    m = sub.add_parser("main")
    m.add_argument("--json", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)

    d = sub.add_parser("decoupling")
    d.add_argument("--csv", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("positioning")
    p.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "main":
        latex_main_table(args.json, args.out)
    elif args.cmd == "decoupling":
        latex_decoupling_table(args.csv, args.out)
    elif args.cmd == "positioning":
        latex_positioning_table(args.out)
