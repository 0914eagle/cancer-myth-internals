#!/usr/bin/env bash
set -euo pipefail

# E0: question table + activation rows. CPU + codex/OpenAI (span alignment). Once.
#   JUDGE_BACKEND=codex (default) | openai
# Detaches under nohup (scripts/lib/detach.sh); FOREGROUND=1 to run inline.
#
#   DATA_ROOT=/data1/heejae bash scripts/run_e0_rows.sh

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
ALIGN="${ALIGN:-llm}"
JUDGE_BACKEND="${JUDGE_BACKEND:-codex}"

source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "e0_rows_${ROWS_NAME}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${DATA_ROOT}/uv/cancer_myth_internals/bin/activate"
unset CANCER_MYTH_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"

python scripts/make_rows.py --config configs/default.yaml --out-name "${ROWS_NAME}" --align "${ALIGN}" --align-backend "${JUDGE_BACKEND}"
cat "${DATA}/${ROWS_NAME}/alignment_audit.md"
echo "[next] eyeball ${DATA}/${ROWS_NAME}/sample_alignments.md, then bash scripts/run_e1_4gpu_125.sh"
