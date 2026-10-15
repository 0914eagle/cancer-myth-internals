#!/usr/bin/env bash
# Local A6000 adaptation: one independent fixed -> trained queue per model.
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER_MODEL="${1:?Usage: bash scripts/run_paper_baselines_a6000_gpu01.sh qwen25|gemma3}"
case "$PAPER_MODEL" in qwen25|gemma3) ;; *) exit 2 ;; esac
PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01}"
export PAPER_PYTHON="${PAPER_PYTHON:-/data3/heejae/uv/paper_baselines_b81ace2/bin/python}"
export PAPER_CONFIG="${PAPER_CONFIG:-$PAPER_RUN_ROOT/protocol/runtime.json}"
export PAPER_BUNDLE="${PAPER_BUNDLE:-$PAPER_RUN_ROOT/protocol/full_bundle.json}"
export PAPER_OUT="${PAPER_OUT:-$PAPER_RUN_ROOT/main}"
export PAPER_MODELS="$PAPER_MODEL"
export PAPER_WORKER_SCRIPT="${PAPER_WORKER_SCRIPT:-scripts/run_paper_baselines_mmlu_resilient.py}"
export PYTHONUNBUFFERED=1
mkdir -p "$PAPER_OUT/logs"
PAPER_STATE="$PAPER_OUT/logs/$PAPER_MODEL-supervisor.state"
printf '%s\n' "pid=$$" "model=$PAPER_MODEL" "state=starting" > "$PAPER_STATE"
trap 'PAPER_RC=$?; printf "exit_code=%s\n" "$PAPER_RC" >> "$PAPER_STATE"' EXIT
for PAPER_STAGE in fixed trained; do
    printf 'phase=%s\nstate=running\n' "$PAPER_STAGE" >> "$PAPER_STATE"
    bash scripts/run_paper_baselines_4090.sh run "$PAPER_STAGE"
    printf 'phase=%s\nstate=complete\n' "$PAPER_STAGE" >> "$PAPER_STATE"
done
printf 'state=complete\n' >> "$PAPER_STATE"
