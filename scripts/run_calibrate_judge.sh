#!/usr/bin/env bash
set -euo pipefail

# Judge calibration against the GPT-4o scores in all_data.json. Detaches
# under nohup (scripts/lib/detach.sh); FOREGROUND=1 to run inline.
#
#   DATA_ROOT=/data1/heejae bash scripts/run_calibrate_judge.sh                       # codex default model
#   DATA_ROOT=/data1/heejae MODEL=gpt-4o N=20 bash scripts/run_calibrate_judge.sh    # can codex serve gpt-4o?
#   DATA_ROOT=/data1/heejae BACKEND=openai bash scripts/run_calibrate_judge.sh        # OpenAI API, gpt-4o
#
# Output: $ART/reports/judge_calibration/<backend>_<model>/calibration.md

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
BACKEND="${BACKEND:-codex}"
MODEL="${MODEL:-}"
N="${N:-150}"
MODELS="${MODELS:-GPT-4o Claude-3.5-Sonnet DeepSeek-R1}"

source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "calibrate_${BACKEND}_${MODEL:-default}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${DATA_ROOT}/uv/cancer_myth_internals/bin/activate"
unset CANCER_MYTH_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"

model_args=()
[[ -n "${MODEL}" ]] && model_args=(--model "${MODEL}")
# shellcheck disable=SC2086
python scripts/calibrate_judge.py --backend "${BACKEND}" "${model_args[@]}" --models ${MODELS} --n "${N}"
echo "[done] see ${ART}/reports/judge_calibration/"
