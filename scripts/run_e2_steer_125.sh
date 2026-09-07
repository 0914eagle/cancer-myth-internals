#!/usr/bin/env bash
set -euo pipefail

# E2 for one model: unconditional vs A-gated steering with the C direction,
# at one layer and a few alphas, on fpq 100 + nfp 100 + all tpq, then judge.
#
#   DATA_ROOT=/data1/heejae CONFIG=configs/llama31_8b.yaml GPUS=0 LAYER=16 \
#     ALPHAS="2 4 8" bash scripts/run_e2_steer_125.sh
#
# Pick LAYER from results/e1/<model>/direction_c/table.md (best pcr_auroc_D)
# and probe_sweep/summary.md (best A readout at D).

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
CONFIG="${CONFIG:?Set CONFIG}"
GPUS="${GPUS:?Set GPUS}"
LAYER="${LAYER:?Set LAYER (hidden-state index)}"
GATE_LAYER="${GATE_LAYER:-${LAYER}}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.0}"
ALPHAS="${ALPHAS:-2 4 8}"
RUN_NAME="${RUN_NAME:-e1}"
E2_NAME="${E2_NAME:-e2}"
ROWS_NAME="${ROWS_NAME:-e1_rows_v1}"
N_FPQ="${N_FPQ:-100}"
N_NFP="${N_NFP:-100}"
JUDGE_BACKEND="${JUDGE_BACKEND:-codex}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "${DATA_ROOT}/uv/cancer_myth_internals/bin/activate"
unset CANCER_MYTH_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
export CUDA_VISIBLE_DEVICES="${GPUS}"
source scripts/env.sh "${DATA_ROOT}"

MODEL="$(python -c "import sys; sys.path.insert(0,'.'); from src.config import load_config, model_short_name; print(model_short_name(load_config('${CONFIG}')))")"
ROWS="${DATA}/${ROWS_NAME}"
DIRS="${ART}/results/${RUN_NAME}/${MODEL}/direction_c/directions.npz"
OUT="${ART}/results/${E2_NAME}/${MODEL}"
mkdir -p "${OUT}"
test -s "${DIRS}" || { echo "[error] missing ${DIRS}; run E1 stage 7 first" >&2; exit 2; }
if [[ "${JUDGE_BACKEND}" == "openai" ]]; then [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "[error] OPENAI_API_KEY not set" >&2; exit 2; }; else command -v codex >/dev/null || { echo "[error] codex not on PATH" >&2; exit 2; }; fi

run_one() {
  local policy="$1" alpha="$2" tag="$3"
  python scripts/run_steer.py --config "${CONFIG}" --questions "${ROWS}/questions.jsonl" \
    --directions "${DIRS}" --layer "${LAYER}" --alpha "${alpha}" --policy "${policy}" \
    --gate-layer "${GATE_LAYER}" --gate-threshold "${GATE_THRESHOLD}" \
    --n-fpq "${N_FPQ}" --n-nfp "${N_NFP}" --seed 17 --output "${OUT}/${tag}.jsonl"
  python scripts/run_judge.py --config "${CONFIG}" --responses "${OUT}/${tag}.jsonl" \
    --questions "${ROWS}/questions.jsonl" --output "${OUT}/${tag}_judge.jsonl" --backend "${JUDGE_BACKEND}"
  python scripts/summarize_judge.py --scores "${OUT}/${tag}_judge.jsonl" \
    --questions "${ROWS}/questions.jsonl" --output "${OUT}/${tag}_summary.json"
}

run_one none 0 "plain_subset"
for a in ${ALPHAS}; do
  run_one unconditional "${a}" "uncond_L${LAYER}_a${a}"
  run_one gated "${a}" "gated_L${LAYER}_a${a}_g${GATE_LAYER}_t${GATE_THRESHOLD}"
done
python - <<EOF
import json, glob, os
out = "${OUT}"
print(f"{'condition':<44} {'PCR':>6} {'PCS':>6} {'NFP':>6} {'TPQ':>6}")
for p in sorted(glob.glob(os.path.join(out, "*_summary.json"))):
    s = json.load(open(p)); f = s.get("fpq", {}); n = s.get("nfp", {}); t = s.get("tpq", {})
    print(f"{os.path.basename(p)[:-13]:<44} {f.get('PCR', float('nan')):6.1f} {f.get('PCS', float('nan')):6.2f} {n.get('NFP', float('nan')):6.1f} {t.get('TPQ_preserved', float('nan')):6.1f}")
EOF
echo "[done] ${OUT}"
