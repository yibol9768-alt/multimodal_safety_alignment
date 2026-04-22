# Refusal Direction Transplant (RDT / RDT+)

**Transferring text-backbone safety alignment into the action-token pathway of Vision-Language-Action models.**

*EMNLP 2026 submission — under review.*

---

## Overview

OpenVLA is built on Llama-2-7b-**base**, an LLM that was never RLHF-aligned, and encodes its 7-DoF robot actions by overwriting the last 256 tokens of the Llama vocabulary with learned discrete bins. After 27 epochs of behavior cloning, these action tokens live in a residual-stream subspace that is nearly orthogonal to the natural-language "harmful / harmless" axis that RLHF alignment would otherwise install. The result is a structural safety gap: the model happily produces actions for any instruction, harmful or not, because its action pathway was never shown a refusal.

**Refusal Direction Transplant (RDT)** closes this gap without retraining. We extract a refusal direction from Llama-2-7b-**chat** — a sibling of OpenVLA's backbone that shares its pretraining initialization but *did* receive RLHF — and inject it at inference time into OpenVLA's action-token positions via a single forward hook.

- **RDT** (base): injects at action-token positions during decode (covers actions `a_2 … a_7`).
- **RDT+**: adds a prefill-phase injection at text-token positions, which covers the first action `a_1` as well, since `a_1` is generated from the text-prefill state before any action tokens exist in the input.

Both variants are training-free, require < 120 lines of code, and add under 5% inference latency.

---

## Key findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Structural safety gap.** Linear probes for harmful/benign achieve AUC > 0.85 at text-token positions in OpenVLA but collapse to AUC ≈ 0.5 at action-token positions. | `code/decoupling_analysis.py` → `decoupling.csv` |
| 2 | **Cross-model geometric transfer.** Llama-2-chat's refusal direction, injected into Llama-2-base-backed OpenVLA, reduces harmful-action compliance by more than 80 percentage points. Shared pretraining initialization preserves residual-stream geometry across RLHF lineage. | Main comparison table (Sec. 4 of paper) |
| 3 | **Refusal in action space.** Under RDT+, OpenVLA's action-logit mass concentrates around bin 128 (zero motion) on harmful instructions — a semantically coherent refusal, not a random perturbation. | `code/action_logit_probe.py` → `functional.json` |
| 4 | **Direction specificity.** Refusal direction outperforms a random unit vector of the same norm by a large margin (Δ_rand > 0), confirming that the cross-model geometry carries the safety signal rather than generic hidden-state noise. | Step 4b of `code/scripts/05_sanity_check.py` |

---

## Method

```
   Llama-2-7b-chat  ──DIM extraction──▶   r_L  (refusal direction at layer L)
                                           │
                                           ▼
   OpenVLA (Llama-2-7b-base backbone)
     Prefill   (generating a_1):    H_L ← H_L + α_text · M_text ⊙ r_L
     Decode    (generating a_2..7): H_L ← H_L + α_act  · M_act  ⊙ r_L
```

- `r_L` is extracted following the difference-in-means protocol of Arditi et al. (rank-1) or an SVD subspace variant (rank-k).
- Two PyTorch forward hooks: one pre-hook on the top-level LLM module to capture `input_ids` (needed to build the position mask), one output hook on decoder layer `L`.
- `M_text` and `M_act` are computed from the action-token id threshold (Llama vocab_size − 256).

---

## Repository layout

```
.
├── code/                          Core implementation
│   ├── rdt_intervention.py        RDT / RDT+ hooks and context manager
│   ├── refusal_direction.py       Difference-in-means + rank-k extraction
│   ├── decoupling_analysis.py     AUC, Cohen's d, linear-probe analysis
│   ├── safety_layer_probe.py      Security-Tensors-style safety-layer probe
│   ├── hidden_collect.py          Collect OpenVLA hidden states per layer
│   ├── action_logit_probe.py      Action-token logit distribution analysis
│   ├── adaptive_attack.py         White-box PGD adaptive attacker
│   ├── openvla_utils.py           OpenVLA loading and layer location helpers
│   ├── config.py                  Paths, model IDs, hyperparameters
│   ├── data_utils.py              AdvBench + Alpaca loaders, VLA prompt builder
│   ├── baseline_adashield.py      AdaShield prompt-level baseline
│   ├── baseline_vlm_guard.py      VLM-Guard hidden-state steering baseline
│   ├── baseline_safevla_lite.py   SafeVLA-lite fine-tuning baseline
│   └── scripts/
│       ├── 01_extract_direction.py    Extract + save refusal directions
│       ├── 02_probe_safety_layers.py  Safety-layer probe (Figure 1)
│       ├── 05_sanity_check.py         End-to-end sanity check (recommended entry)
│       ├── 06_run_main_table.py       Main comparison table
│       ├── 07_run_ablations.py        Ablation sweeps
│       └── run_all.sh                 Full pipeline driver
│
├── paper/                         EMNLP 2026 ACL-format paper (EN + ZH)
├── research/                      Design notes, idea lock, experiment plan
└── requirements.txt               Python dependencies (see below for hardware notes)
```

---

## Getting started

### Hardware

Development was done on a single NVIDIA RTX 5090 (32 GB). OpenVLA fits in bfloat16 on any 24 GB+ GPU. The RTX 5090's `sm_120` architecture requires a PyTorch build with CUDA 12.8+; instructions below use the nightly channel.

### Environment

```bash
# PyTorch (CUDA 12.8+ for Blackwell/sm_120; older GPUs can use the stable cu121 wheel)
pip install --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# OpenVLA-compatible library pins (transformers 5.x removes AutoModelForVision2Seq)
pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.16"

# Remaining dependencies
pip install -r code/requirements.txt
```

### Models

- `NousResearch/Llama-2-7b-chat-hf` — source of the refusal direction.
- `openvla/openvla-7b` — target Vision-Language-Action model.

Both are loaded via `transformers.from_pretrained` with `trust_remote_code=True` (OpenVLA ships a custom `modeling_prismatic.py`). Set `HF_HOME` to a directory with enough space (~30 GB for both checkpoints in fp16).

### Reproducing the sanity check

```bash
cd code
HF_HOME=/path/to/hf/cache \
python scripts/05_sanity_check.py \
    --out /path/to/logs/sanity_v0 \
    --n 64 --n_func 20 \
    --layers 8 10 12 14 16 18 20
```

This runs, in order:

1. Refusal direction extraction from Llama-2-chat (Step 1).
2. Hidden-state collection from OpenVLA at text and action positions (Step 2).
3. Decoupling analysis — projection AUC, Cohen's d, linear probe AUC per layer per position (Step 3).
4. Functional check — alpha sweep over RDT at the best layer, measuring mean action bin and zero-motion mass on harmful prompts (Step 4).
5. Direction-source control — the Δ_rand experiment comparing refusal direction to a random unit vector of the same norm (Step 4b).
6. Target-mode sweep — action-only vs. text+action vs. all positions (Step 4c).

Outputs:

- `decoupling.csv` — AUC / d / probe AUC per (layer, position).
- `functional.json` — mean action bin and zero-motion mass under each intervention.
- `directions.pt` — rank-1 refusal directions per layer.

The critical numbers for the paper's core claim are in `functional.json`: the Δ_rand gap between `refusal_action_a*` and `random_action_a*` entries.

### Full pipeline

After the sanity check passes, the full evaluation reproducible via:

```bash
bash code/scripts/run_all.sh     # direction → probe → sanity → ablations → main table
```

`run_all.sh` writes all stage outputs to a timestamped `logs/run_<YYYYMMDD_HHMMSS>/` directory.

---

## Paper

The draft (English and Chinese parallel versions) lives under `paper/`:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Chinese version (requires xelatex + a CJK font)
xelatex main_zh.tex
```

`paper/main.tex` line 8 switches between `[final]` (local reading, no line numbers) and `[review]` (submission, anonymized with line numbers).

---

## Relationship to prior work

| Work | Direction source | Injection target | Applies to base-backbone VLA? |
|------|------------------|------------------|-------------------------------|
| SARSteer (Liu et al., 2025) | target model itself | all generated tokens | No — base backbone has no refusal to extract |
| VLM-Guard (Chen et al., 2025) | external LLM | last input token | No — does not target action positions |
| Mech-Interp VLA (Wang et al., 2025) | in-distribution VLA data | all positions | Behavior control, not safety |
| **RDT / RDT+ (this work)** | **external RLHF-aligned LLM** | **text + action, phase-aware** | **Designed for this setting** |

---

## Citation

```bibtex
@inproceedings{rdt2026,
  title     = {Refusal Direction Transplant: Transferring Text-Backbone Safety
               Alignment into the Action-Token Pathway of
               Vision-Language-Action Models},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP 2026 (under review)},
  year      = {2026}
}
```

---

## License

Code is released under the MIT license. Paper text is © the authors.
