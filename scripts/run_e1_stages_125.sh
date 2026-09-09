#!/usr/bin/env bash
set -euo pipefail

# Run a stage subset of E1 for several models in sequence on one GPU set.
# Typical use: phase 1 (generate + extract A/B/D) is done for the small
# models while the 27B is still generating, so their judge / rows / extract E
# / sweep / direction stages (3-7) can start now on the idle card instead of
# waiting for run_e1_4gpu_125.sh to reach its phase 3.
#
#   bash scripts/run_e1_stages_125.sh                                  # llama, qwen, gemma9b; stages 3-8; GPU 0
#   MODELS="gemma2_27b" GPUS=1,2,3 bash scripts/run_e1_stages_125.sh   # one model, three cards
#   STAGES="8" bash scripts/run_e1_stages_125.sh                       # paired C only (needs stages 3-5 done)
#
# Every stage is resumable, so if run_e1_4gpu_125.sh later reaches the same
# model its judge finds nothing left to score and the rest recomputes from
# the files on disk. Do not run this on a model whose judge is being written
# by another process: run_judge.py holds a lock per output file and the
# second writer exits.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
MODELS="${MODELS:-llama31_8b qwen25_7b gemma2_9b}"
GPUS="${GPUS:-0}"
STAGES="${STAGES:-3 4 5 6 7 8}"
RUN_NAME="${RUN_NAME:-e1}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
JUDGE_BACKEND="${JUDGE_BACKEND:-codex}"

source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "e1_stages_$(echo "${STAGES}" | tr -d ' ')_gpu$(echo "${GPUS}" | tr -d ',')"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
LOG_ROOT="${DATA_ROOT}/cancer_myth_internals/logs"
mkdir -p "${LOG_ROOT}"

for model in ${MODELS}; do
  log="${LOG_ROOT}/${RUN_NAME}_${model}_s$(echo "${STAGES}" | tr -d ' ').log"
  echo "[$(date +%H:%M:%S)] ${model} stages ${STAGES} on GPU ${GPUS} -> ${log}"
  DETACHED=1 DATA_ROOT="${DATA_ROOT}" CONFIG="configs/${model}.yaml" GPUS="${GPUS}" STAGES="${STAGES}" \
    RUN_NAME="${RUN_NAME}" ROWS_NAME="${ROWS_NAME}" JUDGE_BACKEND="${JUDGE_BACKEND}" \
    bash scripts/run_e1_model.sh > "${log}" 2>&1 || { echo "[error] ${model} failed; see ${log}" >&2; exit 1; }
  echo "[$(date +%H:%M:%S)] ${model} done"
done
echo "[done] stages ${STAGES} for: ${MODELS}"
