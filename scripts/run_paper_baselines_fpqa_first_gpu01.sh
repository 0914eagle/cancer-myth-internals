#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER_MODEL="${1:?Model required}"
case "$PAPER_MODEL" in qwen25|gemma3) ;; *) exit 2 ;; esac
PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01}"
export PAPER_PYTHON="/data3/heejae/uv/paper_baselines_b81ace2/bin/python"
export PAPER_CONFIG="$PAPER_RUN_ROOT/protocol/runtime.json"
export PAPER_BUNDLE="$PAPER_RUN_ROOT/protocol/full_bundle.json"
export PAPER_OUT="$PAPER_RUN_ROOT/main"
export PAPER_MODELS="$PAPER_MODEL"
export PAPER_WORKER_SCRIPT="scripts/run_paper_baselines_fpqa_first.py"
export PYTHONUNBUFFERED=1
PAPER_STATE="$PAPER_OUT/logs/$PAPER_MODEL-supervisor.state"
printf '%s\n' "pid=$$" "model=$PAPER_MODEL" "phase=all" "state=running" > "$PAPER_STATE"
trap 'PAPER_RC=$?; printf "exit_code=%s\n" "$PAPER_RC" >> "$PAPER_STATE"' EXIT
bash scripts/run_paper_baselines_4090.sh run all
printf 'state=complete\n' >> "$PAPER_STATE"
