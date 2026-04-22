#!/bin/bash
# Full pipeline once both models are downloaded.
# Runs: direction extraction -> safety-layer probe -> sanity check
#     -> RDT ablations -> main table
# Assumes models at /root/models/, venv at /root/venv-rdt.

set -euo pipefail

ROOT="${RDT_ROOT:-/root/rdt}"
LOGS="${ROOT}/logs/run_$(date +%Y%m%d_%H%M%S)"
CODE="${ROOT}/code"
mkdir -p "$LOGS"

source /root/venv-rdt/bin/activate
export HF_HOME=/root/models
export HTTP_PROXY=http://172.30.48.1:7890
export HTTPS_PROXY=http://172.30.48.1:7890
export RDT_LOGS="$LOGS"

cd "$CODE"
echo "===================================================================="
echo "Stage 1/5: refusal direction extraction  (Llama-2-chat)"
echo "===================================================================="
python scripts/01_extract_direction.py \
    --out "$LOGS/directions.pt" \
    --rank_k 10 \
    --n 512

echo "===================================================================="
echo "Stage 2/5: safety-layer probe  (Fig 1 motivation)"
echo "===================================================================="
python scripts/02_probe_safety_layers.py --out "$LOGS/probe" --n 64

echo "===================================================================="
echo "Stage 3/5: sanity check end-to-end"
echo "===================================================================="
python scripts/05_sanity_check.py --out "$LOGS/sanity" --n 128

echo "===================================================================="
echo "Stage 4/5: ablation sweeps"
echo "===================================================================="
python scripts/07_run_ablations.py \
    --directions "$LOGS/directions.pt" \
    --out_dir "$LOGS/ablations" \
    --n 128

echo "===================================================================="
echo "Stage 5/5: main comparison table"
echo "===================================================================="
python scripts/06_run_main_table.py \
    --directions "$LOGS/directions.pt" \
    --out "$LOGS/main_table.json" \
    --n 64

echo
echo "All stages complete. Outputs in: $LOGS"
ls -la "$LOGS"
