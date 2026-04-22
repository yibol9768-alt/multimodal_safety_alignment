# RDT: Refusal Direction Transplant for VLA Safety

Companion code for *"Refusal Direction Transplant: Transferring Text-Backbone Safety Alignment into Action-Token Pathways of Vision-Language-Action Models"* (EMNLP 2026 submission).

## Claim

OpenVLA is built on Llama-2-7b **base** (never RLHF-aligned) and overwrites the last 256 tokens of the Llama vocabulary as discrete action tokens. This creates a structural safety gap: refusal circuits that exist in Llama-2-7b-**chat** (which shares the same pretraining) have no trigger pathway in OpenVLA's action-token output.

RDT fixes this at inference time: extract the refusal direction from Llama-2-chat, inject it at action-token positions in OpenVLA. Training-free, ~5% latency, composes with other defenses.

## Layout

```
code/
├── README.md
├── requirements.txt
├── config.py                 # paths, hyperparams
├── data_utils.py             # AdvBench + Alpaca loaders
├── refusal_direction.py      # Arditi-style rank-1 + SVD rank-k extraction
├── safety_layer_probe.py     # Bet 1 — Security Tensors-style probe on OpenVLA
├── openvla_utils.py          # load OpenVLA, robust layer-access helpers
├── hidden_collect.py         # forward-hook hidden state collection
├── decoupling_analysis.py    # AUC, Cohen's d, linear probe, viz
├── rdt_intervention.py       # RDT forward hook (core method)
├── action_logit_probe.py     # Bet 3 — what action does refusal want?
├── adaptive_attack.py        # Bet 2 — PGD against RDT (reviewer shield)
└── scripts/
    ├── 01_extract_direction.py
    ├── 02_probe_safety_layers.py
    ├── 03_collect_hidden.py
    ├── 04_analyze_decoupling.py
    ├── 05_sanity_check.py        # Step 1-4 end-to-end (Week 1 deliverable)
    └── run_remote.sh             # kick off on westd WSL
```

## Setup (on westd WSL Ubuntu)

```bash
python3 -m venv /root/venv-rdt
source /root/venv-rdt/bin/activate
pip install --proxy http://172.30.48.1:7890 -r requirements.txt
# PyTorch for sm_120 (RTX 5090 Blackwell) needs torch >= 2.6 cu128:
pip install --proxy http://172.30.48.1:7890 --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

Set `HF_HOME=/root/models` before running any script. Models expected:
- `NousResearch/Llama-2-7b-chat-hf` (non-gated mirror, weights ≡ Meta's Llama-2-7b-chat)
- `openvla/openvla-7b`

## Week 1 deliverable

```bash
python scripts/05_sanity_check.py --out logs/sanity_v0/
```

Produces: `decoupling_auc.png` (motivation Fig. 1), `direction_quality.json`, `layer_sweep.csv`.

Goal: show that Llama-2-chat's refusal direction (layer ~14) achieves AUC > 0.85 on text-token positions but AUC ≈ 0.5–0.6 on action-token positions in OpenVLA. That decoupling *is* the paper's motivation.
