#!/usr/bin/env bash
# One independent premise SFT on one visible GPU. No torchrun, no DDP.
# Usage: CUDA_VISIBLE_DEVICES=0 bash scripts/run_premise_sft_single.sh cancer RUN_NAME [train options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/premise_sft_env.sh
dataset="${1:?cancer or crepe}"
run_name="${2:?new run name}"
shift 2
[[ "${dataset}" == cancer || "${dataset}" == crepe ]] || { echo 'Dataset must be cancer or crepe'; exit 1; }
[[ "${run_name}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ -n "${CUDA_VISIBLE_DEVICES}" && "${CUDA_VISIBLE_DEVICES}" != *,* ]] || { echo 'Select exactly one GPU'; exit 1; }
# Do not inherit distributed-launch metadata from an unrelated shell job.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
exec "${PREMISE_ENV}/bin/python" -u scripts/premise_sft.py train \
  --data "${PREMISE_ART}/data/${dataset}" --model-lock "${PREMISE_ART}/downloads.json" \
  --objective premise --cohort all --out "${PREMISE_ART}/runs/${run_name}" \
  --batch-size "${MICRO_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-8}" --max-length 2048 "$@"
