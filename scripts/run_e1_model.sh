#!/usr/bin/env bash
set -euo pipefail

# E1 for one backbone on the GPUs named in GPUS. Stages:
#   1 generate   Plain responses (fpq + nfp; tpq shares nfp text)
#   2 extract    positions A/B/D, every hidden-state index
#   3 judge      GPT-4o PCR / NFP / TPQ scores           (API; needs OPENAI_API_KEY)
#   4 rows_e     position-E rows with PCR labels merged
#   5 extract_e  position E, every hidden-state index
#   6 sweep      A readout heatmap
#   7 direction  C direction, cos(A,C), PCR predictability, gate probe
#   8 paired     paired C (reference +1 minus -1 answers, teacher-forced), extract + direction
#
#   CONFIG=configs/llama31_8b.yaml GPUS=0 bash scripts/run_e1_model.sh
#   STAGES="1 2" ...   to run a subset (each stage resumes where it left off)
#
# Detaches under nohup by default (scripts/lib/detach.sh); FOREGROUND=1 to run
# inline. run_e1_4gpu_125.sh calls this with DETACHED=1 so its own workers
# stay attached to the parent that waits on them.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
CONFIG="${CONFIG:?Set CONFIG=configs/<model>.yaml}"
GPUS="${GPUS:?Set GPUS, e.g. 0 or 1,2,3}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
RUN_NAME="${RUN_NAME:-e1}"
STAGES="${STAGES:-1 2 3 4 5 6 7 8}"
LIMIT="${LIMIT:-}"
JUDGE_BACKEND="${JUDGE_BACKEND:-codex}"

_tag="$(basename "${CONFIG}" .yaml)"
[[ -n "${LIMIT}" ]] && _tag="${_tag}_smoke${LIMIT}"
source "$(dirname "${BASH_SOURCE[0]}")/lib/detach.sh" "e1_${_tag}_s$(echo "${STAGES}" | tr -d " ")"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${DATA_ROOT}/uv/cancer_myth_internals/bin/activate"
unset CANCER_MYTH_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
export CUDA_VISIBLE_DEVICES="${GPUS}"
source scripts/env.sh "${DATA_ROOT}"

MODEL="$(python -c "import sys; sys.path.insert(0,'.'); from src.config import load_config, model_short_name; print(model_short_name(load_config('${CONFIG}')))")"
ROWS="${DATA}/${ROWS_NAME}"
RES="${ART}/results/${RUN_NAME}/${MODEL}"
ACT_AD="${ART}/activations/${RUN_NAME}_${MODEL}_ad"
ACT_E="${ART}/activations/${RUN_NAME}_${MODEL}_e"
ACT_PAIR="${ART}/activations/${RUN_NAME}_${MODEL}_pair"
mkdir -p "${RES}" "${ART}/logs"
limit_args=()
[[ -n "${LIMIT}" ]] && limit_args=(--limit "${LIMIT}")

has() { [[ " ${STAGES} " == *" $1 "* ]]; }

test -s "${ROWS}/questions.jsonl" || { echo "[error] run scripts/run_e0_rows.sh first" >&2; exit 2; }
python scripts/check_gpu_setup.py --config "${CONFIG}" --require-free-gb 20

if has 1; then
  echo "[stage 1/8] plain responses (${MODEL})"
  python scripts/run_generate.py --config "${CONFIG}" --questions "${ROWS}/questions.jsonl" \
    --run-name "${RUN_NAME}" --output "${RES}/plain_responses.jsonl" "${limit_args[@]}"
fi

if has 2; then
  echo "[stage 2/8] extract A/B/D (${MODEL})"
  extra=()
  [[ -n "${LIMIT}" ]] && extra=(--limit-prompts "${LIMIT}")
  python -m src.extract_activations --config "${CONFIG}" --input "${ROWS}/activation_rows.jsonl" \
    --output-dir "${ACT_AD}" --layers all --strategies last_subtoken span_mean "${extra[@]}"
fi

if has 3; then
  echo "[stage 3/8] judge via ${JUDGE_BACKEND}"
  if [[ "${JUDGE_BACKEND}" == "openai" ]]; then
    [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[error] OPENAI_API_KEY not set" >&2; exit 2; }
  else
    command -v codex >/dev/null 2>&1 || { echo "[error] codex not on PATH" >&2; exit 2; }
  fi
  python scripts/run_judge.py --config "${CONFIG}" --responses "${RES}/plain_responses.jsonl" \
    --questions "${ROWS}/questions.jsonl" --output "${RES}/plain_judge.jsonl" --backend "${JUDGE_BACKEND}"
  python scripts/summarize_judge.py --scores "${RES}/plain_judge.jsonl" \
    --questions "${ROWS}/questions.jsonl" --output "${RES}/plain_summary.json"
fi

if has 4; then
  echo "[stage 4/8] response rows + labels"
  python scripts/make_response_rows.py --questions "${ROWS}/questions.jsonl" \
    --activation-rows "${ROWS}/activation_rows.jsonl" --responses "${RES}/plain_responses.jsonl" \
    --scores "${RES}/plain_judge.jsonl" --out-dir "${RES}/rows"
fi

if has 5; then
  echo "[stage 5/8] extract E (${MODEL})"
  python -m src.extract_activations --config "${CONFIG}" --input "${RES}/rows/response_rows.jsonl" \
    --output-dir "${ACT_E}" --layers all --strategies span_mean last_subtoken
  # Join the judge labels onto the A/B/D and E manifests (tensors untouched).
  # Idempotent, so an E run extracted before the rows carried labels is fixed
  # here without re-running the backbone.
  python scripts/merge_labels_into_manifests.py --run-dir "${ACT_AD}" --labels "${RES}/rows/labels.jsonl"
  python scripts/merge_labels_into_manifests.py --run-dir "${ACT_E}" --labels "${RES}/rows/labels.jsonl"
fi

if has 6; then
  echo "[stage 6/8] A readout sweep"
  python scripts/run_probe_sweep.py --run-dir "${ACT_AD}" --out-dir "${RES}/probe_sweep"
fi

if has 7; then
  echo "[stage 7/8] C direction"
  python scripts/run_direction_c.py --run-ad "${ACT_AD}" --run-e "${ACT_E}" --out-dir "${RES}/direction_c"
fi

if has 8; then
  echo "[stage 8/8] paired C (${MODEL})"
  python scripts/make_paired_rows.py --config "${CONFIG}" --questions "${ROWS}/questions.jsonl" \
    --responses "${RES}/plain_responses.jsonl" --labels "${RES}/rows/labels.jsonl" \
    --output "${RES}/rows/paired_rows.jsonl"
  python -m src.extract_activations --config "${CONFIG}" --input "${RES}/rows/paired_rows.jsonl" \
    --output-dir "${ACT_PAIR}" --layers all --strategies span_mean
  python scripts/run_direction_pair.py --run-pair "${ACT_PAIR}" --run-ad "${ACT_AD}" \
    --directions "${RES}/direction_c/directions.npz" --out-dir "${RES}/direction_c"
fi
echo "[done] ${MODEL}: ${RES}"
