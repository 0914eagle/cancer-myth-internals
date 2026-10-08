#!/usr/bin/env bash
# Run independent model workers; failures are propagated, not hidden by wait.
set -euo pipefail
cd "$(dirname "$0")/.."
PAPER_PYTHON="${PAPER_PYTHON:-/data1/heejae/uv/cancer_myth_internals/bin/python}"
PAPER_CONFIG="${PAPER_CONFIG:-results/paper_baselines_v3/protocol/runtime.json}"
PAPER_BUNDLE="${PAPER_BUNDLE:-results/paper_baselines_v3/protocol/full_bundle.json}"
PAPER_OUT="${PAPER_OUT:-results/paper_baselines_v3/main}"
export PYTHONPATH="$PWD/.venv/paper_baseline_extras${PYTHONPATH:+:$PYTHONPATH}"
PAPER_COMMAND="${1:-preflight}"
PAPER_PHASE="${2:-fixed}"
mkdir -p "$PAPER_OUT/logs"
# Cross-run locks for each physical GPU. Kernel releases them on exit.
PAPER_GPU_LOCK_DIR="${PAPER_GPU_LOCK_DIR:-/tmp/paper-baselines-gpu-locks}"
mkdir -p "$PAPER_GPU_LOCK_DIR"
for PAPER_GPU in $("$PAPER_PYTHON" - "$PAPER_CONFIG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));g=[i for m in c['models'].values() for i in m['gpus']]
assert len(g)==len(set(g)), 'GPU allocations overlap'
print(*sorted(g))
PY
); do
    exec {PAPER_FD}>"$PAPER_GPU_LOCK_DIR/gpu-$PAPER_GPU.lock"
    flock -n "$PAPER_FD" || { echo "GPU $PAPER_GPU already leased by another baseline suite" >&2; exit 1; }
done
PAPER_PIDS=()
trap 'for p in "${PAPER_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done' INT TERM
for PAPER_MODEL in qwen25 gemma3; do
    "$PAPER_PYTHON" scripts/run_paper_baselines_resilient.py "$PAPER_COMMAND" \
        --phase "$PAPER_PHASE" --config "$PAPER_CONFIG" --bundle "$PAPER_BUNDLE" \
        --model "$PAPER_MODEL" --out "$PAPER_OUT" "${@:3}" \
        > "$PAPER_OUT/logs/$PAPER_MODEL-$PAPER_COMMAND-$PAPER_PHASE.log" 2>&1 &
    PAPER_PIDS+=("$!")
done
PAPER_EXIT=0
for PAPER_PID in "${PAPER_PIDS[@]}"; do wait "$PAPER_PID" || PAPER_EXIT=1; done
"$PAPER_PYTHON" scripts/summarize_paper_baselines.py \
    --config "$PAPER_CONFIG" --bundle "$PAPER_BUNDLE" --out "$PAPER_OUT"
exit "$PAPER_EXIT"
