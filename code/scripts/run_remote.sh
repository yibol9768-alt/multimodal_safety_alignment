#!/bin/bash
# Push the RDT code repo to westd WSL and kick off a sanity check run.
# Designed to tolerate vicp.fun ssh tunnel drops: uses base64-pipe for transfer,
# then nohup + disown to background the run; poll via `tail /root/rdt/logs/...`.
#
# Usage (from Mac in this repo root, i.e. multimodal_safety_alignment/):
#   bash code/scripts/run_remote.sh           # push code + kick off sanity
#   bash code/scripts/run_remote.sh push      # push only
#   bash code/scripts/run_remote.sh run       # run only (code already on westd)
#   bash code/scripts/run_remote.sh tail      # print current log tail
#
set -euo pipefail

ACTION="${1:-full}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # multimodal_safety_alignment/code
REMOTE_PARENT="/root/rdt"
REMOTE_CODE="${REMOTE_PARENT}/code"
LOG_DIR="${REMOTE_PARENT}/logs/sanity_v0"
PROXY="http://172.30.48.1:7890"

wsl_run() {
    # Pipe a WSL bash command in via base64 to dodge quoting hell + scp drops.
    local cmd_b64
    cmd_b64=$(printf '%s' "$1" | base64 -w0 2>/dev/null || printf '%s' "$1" | base64 | tr -d '\n')
    ssh westd "wsl -d Ubuntu -- bash -c \"echo $cmd_b64 | base64 -d | bash\""
}

push_code() {
    echo "[push] tarring code dir"
    local tar_b64
    tar_b64=$(tar -C "$LOCAL_DIR/.." -czf - code | base64 | tr -d '\n')
    echo "[push] tar size = $(printf '%s' "$tar_b64" | wc -c) bytes (b64)"
    wsl_run "mkdir -p $REMOTE_PARENT && cd $REMOTE_PARENT && rm -rf code && echo '$tar_b64' | base64 -d | tar -xzf -"
    wsl_run "ls -la $REMOTE_CODE | head -20"
}

setup_venv_if_missing() {
    wsl_run "
set -e
if [ ! -d /root/venv-rdt ]; then
    python3 -m venv /root/venv-rdt
fi
source /root/venv-rdt/bin/activate
pip install -q --proxy $PROXY --upgrade pip
# PyTorch first (sm_120 / cu128 nightly for 5090)
pip install -q --proxy $PROXY --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu128 || \
    pip install -q --proxy $PROXY torch torchvision
pip install -q --proxy $PROXY -r $REMOTE_CODE/requirements.txt
echo '[venv] ready'
"
}

kick_off_sanity() {
    wsl_run "
mkdir -p $LOG_DIR
cd $REMOTE_CODE
nohup bash -c '
source /root/venv-rdt/bin/activate
export HF_HOME=/root/models
export HTTP_PROXY=$PROXY
export HTTPS_PROXY=$PROXY
python scripts/05_sanity_check.py --out $LOG_DIR
' > $LOG_DIR/run.log 2>&1 < /dev/null &
disown
echo '[kickoff] pid ' \$!
sleep 1
tail -5 $LOG_DIR/run.log 2>/dev/null || echo '(log not yet flushed)'
"
}

tail_log() {
    wsl_run "tail -40 $LOG_DIR/run.log 2>/dev/null || echo '(no log yet at $LOG_DIR/run.log)'"
}

case "$ACTION" in
    push)  push_code ;;
    venv)  setup_venv_if_missing ;;
    run)   kick_off_sanity ;;
    tail)  tail_log ;;
    full)  push_code; setup_venv_if_missing; kick_off_sanity ;;
    *)     echo "unknown action: $ACTION"; exit 1 ;;
esac
