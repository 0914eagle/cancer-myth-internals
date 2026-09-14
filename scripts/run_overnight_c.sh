#!/usr/bin/env bash
# Detaches the coordinator so SSH loss does not stop an overnight run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=0,1
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
stage="${1:-start}"
pilot="${PILOT_DIR:-$ART/results/pilot/gemma2_9b_v1_fitfix}"
quick="${QUICK_DIR:-$pilot/quick45_original_terra_v1}"
out="${OVERNIGHT_DIR:-$pilot/overnight_c_v1}"
hours="${OVERNIGHT_HOURS:-9}"
case "$stage" in
  start)
    python scripts/overnight_c_diagnostics.py prepare \
      --pilot-dir "$pilot" --quick-dir "$quick" --out-dir "$out" \
      --config "${CONFIG:-configs/gemma2_9b.yaml}" \
      --pairs "${OVERNIGHT_PAIRS:-24}" --fit-questions "${OVERNIGHT_FIT_QUESTIONS:-64}"
    nohup python -u scripts/overnight_c_diagnostics.py run \
      --out-dir "$out" --hours "$hours" --gpus 0 1 \
      >> "$out/coordinator.log" 2>&1 < /dev/null &
    pid=$!
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      tail -n 30 "$out/coordinator.log"
      echo "Coordinator exited. Check log/report before starting again."
      exit 1
    fi
    echo "Started coordinator PID $pid. Physical GPUs: 0,1. Limit: $hours hours. GPT calls: 0."
    echo "Status: bash scripts/run_overnight_c.sh status"
    echo "Morning report: $out/report.md"
    echo "Log: $out/coordinator.log"
    ;;
  status|report|stop)
    python scripts/overnight_c_diagnostics.py "$stage" --out-dir "$out"
    ;;
  *) echo 'usage: run_overnight_c.sh start|status|report|stop' >&2; exit 1 ;;
esac
