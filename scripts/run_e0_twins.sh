#!/usr/bin/env bash
set -euo pipefail

# E0b: true-premise twins of every LLM-aligned fpq question (codex, CPU).
# Run after run_e0_rows.sh. Detaches under nohup; FOREGROUND=1 to run inline.
#
#   DATA_ROOT=/data1/heejae bash scripts/run_e0_twins.sh

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
JUDGE_BACKEND="${JUDGE_BACKEND:-codex}"
LIMIT="${LIMIT:-}"

source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "e0_twins_${ROWS_NAME}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${DATA_ROOT}/uv/cancer_myth_internals/bin/activate"
unset CANCER_MYTH_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"

limit_args=()
[[ -n "${LIMIT}" ]] && limit_args=(--limit "${LIMIT}")
python scripts/make_true_twins.py --config configs/default.yaml --questions "${DATA}/${ROWS_NAME}/questions.jsonl" \
  --out-dir "${DATA}/${ROWS_NAME}" --backend "${JUDGE_BACKEND}" "${limit_args[@]}"
cat "${DATA}/${ROWS_NAME}/twins_audit.md"
echo "[next] eyeball ${DATA}/${ROWS_NAME}/twins_sample.md, then STAGES=9 per model"
