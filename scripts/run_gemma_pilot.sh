#!/usr/bin/env bash
# Explicit stages: no GPU generation or judge calls until that stage is requested.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-}"

stage="${1:?usage: run_gemma_pilot.sh prepare|baselines|report-baselines|fit|sweep|select|test|report}"
config="${CONFIG:-configs/gemma2_9b.yaml}"
pilot="${PILOT_DIR:-$ART/results/pilot/gemma2_9b_v1}"
manifest="$pilot/split/manifest.json"
backend="${JUDGE_BACKEND:-codex}"
judge_model="${JUDGE_MODEL:-gpt-5}"
batch="${BATCH_SIZE:-4}"
max_new="${MAX_NEW_TOKENS:-512}"
review="${REVIEW_TOKENS:-128}"
read -r -a layers <<< "${LAYERS:-14 21 28}"
read -r -a alphas <<< "${ALPHAS:-0 0.02 0.05 0.1}"
if [[ ! "$judge_model" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo 'JUDGE_MODEL must be a filename-safe explicit model ID' >&2
    exit 1
fi
scores_suffix="judge_${backend}_${judge_model}.jsonl"

generate() {
    local partition="$1" method="$2" tag="$3"
    shift 3
    python scripts/run_pilot.py generate --config "$config" --manifest "$manifest" \
        --partition "$partition" --method "$method" --batch-size "$batch" \
        --max-new-tokens "$max_new" --review-tokens "$review" \
        --output "$pilot/$partition/$tag.jsonl" "$@"
}

judge() {
    local partition="$1" tag="$2"
    python scripts/run_judge.py --config "$config" --questions "$pilot/split/$partition.jsonl" \
        --responses "$pilot/$partition/$tag.jsonl" --backend "$backend" --model "$judge_model" \
        --output "$pilot/$partition/${tag}_${scores_suffix}"
}

case "$stage" in
    prepare)
        python scripts/run_pilot.py prepare --config "$config" --out-dir "$pilot/split"
        ;;
    baselines)
        for method in plain fp_identification premise_cot; do
            generate dev "$method" "$method"
            judge dev "$method"
        done
        ;;
    fit)
        python scripts/run_pilot.py fit --config "$config" --manifest "$manifest" \
            --layers "${layers[@]}" --prefix-tokens "${PREFIX_TOKENS:-32}" --out-dir "$pilot/fit"
        ;;
    sweep)
        for layer in "${layers[@]}"; do
            for alpha in "${alphas[@]}"; do
                tag="steering_L${layer}_a${alpha}"
                generate dev steering "$tag" --fit-dir "$pilot/fit" --layer "$layer" --alpha "$alpha"
                judge dev "$tag"
            done
        done
        ;;
    select)
        : "${NFP_MARGIN_PP:?Set a pilot dev tolerance in percentage points before selection (not a statistical guarantee)}"
        candidates=()
        for layer in "${layers[@]}"; do
            for alpha in "${alphas[@]}"; do
                candidates+=("$pilot/dev/steering_L${layer}_a${alpha}_${scores_suffix}")
            done
        done
        python scripts/run_pilot.py select --manifest "$manifest" \
            --plain-scores "$pilot/dev/plain_${scores_suffix}" --candidate-scores "${candidates[@]}" \
            --nfp-margin-pp "$NFP_MARGIN_PP" --output "$pilot/selection.json"
        ;;
    test)
        for method in plain fp_identification premise_cot steering; do
            generate test "$method" "$method" --fit-dir "$pilot/fit" --selection "$pilot/selection.json"
            judge test "$method"
        done
        ;;
    report|report-baselines)
        partition="${PARTITION:-dev}"
        if [[ "$stage" == report-baselines && "$partition" != dev ]]; then
            echo 'report-baselines reports dev only; use report for the locked test comparison' >&2
            exit 1
        fi
        files=()
        for method in plain fp_identification premise_cot; do
            files+=("$pilot/$partition/${method}_${scores_suffix}")
        done
        if [[ "$stage" == report-baselines ]]; then
            report_tag="baselines_dev"
        elif [[ "$partition" == test ]]; then
            report_tag="test"
            files+=("$pilot/test/steering_${scores_suffix}")
        else
            report_tag="$partition"
            for layer in "${layers[@]}"; do
                for alpha in "${alphas[@]}"; do
                    files+=("$pilot/dev/steering_L${layer}_a${alpha}_${scores_suffix}")
                done
            done
        fi
        python scripts/run_pilot.py report --manifest "$manifest" --partition "$partition" \
            --scores "${files[@]}" --output "$pilot/report_${report_tag}_${backend}_${judge_model}.json"
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac
