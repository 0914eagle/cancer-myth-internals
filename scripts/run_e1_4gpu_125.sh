#!/usr/bin/env bash
set -euo pipefail

# E1 on server 125 (four RTX 4090 under /data1/heejae).
#
# Phase 1, three single-card workers in parallel:
#   GPU 0  Llama-3.1-8B-Instruct
#   GPU 1  Qwen2.5-7B-Instruct
#   GPU 2  Gemma-2-9B-it
# Phase 2, after phase 1 (needs three cards, bf16, no quantization):
#   GPU 1,2,3  Gemma-2-27B-it
#
# GPU stages (1 generate, 2 extract) run per worker; the API stage (3 judge)
# and the CPU stages (4, 6, 7, 8) run after, in this script, so the OpenAI
# calls are not four-way parallel. Stage 5 (extract E) needs the card again
# and is run per model at the end.
#
#   bash scripts/run_e1_4gpu_125.sh        # detaches itself; prints the log path

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
RUN_NAME="${RUN_NAME:-e1}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
RUN_27B="${RUN_27B:-1}"
LIMIT="${LIMIT:-}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this wrapper is frozen for server 125 (/data1/heejae)" >&2
  exit 2
fi

source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "e1_4gpu_${RUN_NAME}"

cd /home/eagle0914/cancer-myth-internals
LOG_ROOT="${DATA_ROOT}/cancer_myth_internals/logs"
mkdir -p "${LOG_ROOT}"

worker() {
  local config="$1" gpus="$2" stages="$3" tag="$4"
  DETACHED=1 DATA_ROOT="${DATA_ROOT}" CONFIG="${config}" GPUS="${gpus}" STAGES="${stages}" \
    RUN_NAME="${RUN_NAME}" ROWS_NAME="${ROWS_NAME}" LIMIT="${LIMIT}" \
    bash scripts/run_e1_model.sh > "${LOG_ROOT}/${RUN_NAME}_${tag}.log" 2>&1
}

echo "[phase 1] generate + extract A/B/D on GPUs 0,1,2"
worker configs/llama31_8b.yaml 0 "1 2" llama31_8b_p1 & p0=$!
worker configs/qwen25_7b.yaml  1 "1 2" qwen25_7b_p1  & p1=$!
worker configs/gemma2_9b.yaml  2 "1 2" gemma2_9b_p1  & p2=$!
s0=0; s1=0; s2=0
wait "${p0}" || s0=$?
wait "${p1}" || s1=$?
wait "${p2}" || s2=$?
echo "[phase 1] llama=${s0} qwen=${s1} gemma9b=${s2}"
if [[ "${s0}" -ne 0 || "${s1}" -ne 0 || "${s2}" -ne 0 ]]; then
  echo "[error] a phase-1 worker failed; see ${LOG_ROOT}" >&2
  exit 1
fi

if [[ "${RUN_27B}" == "1" ]]; then
  echo "[phase 2] Gemma-2-27B-it on GPUs 1,2,3"
  worker configs/gemma2_27b.yaml 1,2,3 "1 2" gemma2_27b_p1 || { echo "[error] 27B failed" >&2; exit 1; }
fi

echo "[phase 3] judge + rows + extract E + sweep + direction, per model"
MODELS=(llama31_8b:0 qwen25_7b:1 gemma2_9b:2)
[[ "${RUN_27B}" == "1" ]] && MODELS+=(gemma2_27b:1,2,3)
for entry in "${MODELS[@]}"; do
  model="${entry%%:*}"; gpus="${entry##*:}"
  worker "configs/${model}.yaml" "${gpus}" "3 4 5 6 7 8" "${model}_p3" || { echo "[error] ${model} phase 3 failed" >&2; exit 1; }
  echo "[phase 3] ${model} done"
done
echo "[done] E1 on server 125: ${DATA_ROOT}/cancer_myth_internals/results/${RUN_NAME}"
