#!/usr/bin/env bash
# Usage: bash scripts/run_premise_sft_ddp.sh cancer premise RUN_NAME [extra train options]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/premise_sft_env.sh
dataset="${1:?cancer or crepe}"
objective="${2:?binary, premise, answer, or joint}"
run_name="${3:?new run name}"
shift 3
[[ "${run_name}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 1; }
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
exec "${PREMISE_ENV}/bin/torchrun" --standalone --nproc_per_node=2 scripts/premise_sft.py train \
  --data "${PREMISE_ART}/data/${dataset}" --model-lock "${PREMISE_ART}/downloads.json" \
  --objective "${objective}" --out "${PREMISE_ART}/runs/${run_name}" \
  --batch-size "${MICRO_BATCH:-2}" --grad-accum "${GRAD_ACCUM:-4}" --max-length 2048 "$@"
