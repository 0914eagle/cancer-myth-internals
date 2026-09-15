#!/usr/bin/env bash
# Explicit, restartable stages. "gpu" does NOT call an external judge.
set -euo pipefail
SUITE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SUITE_REPO"
export CANCER_MYTH_CODE_ROOT="$SUITE_REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
case "$CUDA_VISIBLE_DEVICES" in 0|1|0,1|1,0) ;; *) echo 'Only physical GPUs 0 and 1 are allowed.' >&2; exit 2;; esac
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
SUITE_CONFIG="${SUITE_CONFIG:-configs/qwen25_7b.yaml}"
SUITE_DIR="${SUITE_DIR:-$ART/results/baselines/qwen25_7b_v1}"
FPQ_REFERENCE_FILE="${FPQ_REFERENCE_FILE:-$ART/data/cancer_myth_fpq_well.jsonl}"
PYTHON="${PYTHON:-python}"
stage="${1:-help}"
case "$stage" in
  test)
    "$PYTHON" -m pytest -q tests/test_baseline_download.py tests/test_baseline_suite.py tests/test_baseline_gates.py tests/test_baseline_generation.py tests/test_baseline_cli.py tests/test_well_eval.py
    ;;
  test-runtime)
    # Tiny randomly initialized CPU models only; no pretrained checkpoint download.
    "$PYTHON" -m pytest -q tests/test_baseline_runtime.py
    ;;
  references)
    "$PYTHON" scripts/download_baseline_references.py --output "$FPQ_REFERENCE_FILE"
    ;;
  prepare)
    : "${PILOT_DIR:?Set PILOT_DIR to the existing pilot with split/manifest.json}"
    [[ -f "$FPQ_REFERENCE_FILE" ]] || { echo 'Run: bash scripts/run_baselines.sh references (dataset only, no model calls)' >&2; exit 2; }
    "$PYTHON" scripts/run_baseline_suite.py prepare --manifest "$PILOT_DIR/split/manifest.json" --out-dir "$SUITE_DIR" --fpq-source "$FPQ_REFERENCE_FILE"
    "$PYTHON" scripts/run_baseline_suite.py splits --out-dir "$SUITE_DIR"
    ;;
  generate|detect|extract)
    "$PYTHON" scripts/run_baseline_suite.py "$stage" --config "$SUITE_CONFIG" --out-dir "$SUITE_DIR"
    ;;
  gpu)
    mkdir -p "$SUITE_DIR/logs"
    # Independent extraction uses GPU1; generation and detection share cached reviews on GPU0.
    if [[ "$CUDA_VISIBLE_DEVICES" == *","* ]]; then
      CUDA_VISIBLE_DEVICES=1 "$PYTHON" scripts/run_baseline_suite.py extract --config "$SUITE_CONFIG" --out-dir "$SUITE_DIR" > "$SUITE_DIR/logs/extract.log" 2>&1 &
      extraction_pid=$!
      generation_status=0
      CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_baseline_suite.py generate --config "$SUITE_CONFIG" --out-dir "$SUITE_DIR" > "$SUITE_DIR/logs/generate.log" 2>&1 || generation_status=$?
      if [[ "$generation_status" -eq 0 ]]; then
        CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/run_baseline_suite.py detect --config "$SUITE_CONFIG" --out-dir "$SUITE_DIR" > "$SUITE_DIR/logs/detect.log" 2>&1 || generation_status=$?
      fi
      extraction_status=0
      wait "$extraction_pid" || extraction_status=$?
      [[ "$generation_status" -eq 0 && "$extraction_status" -eq 0 ]] || { echo "Stage failed; inspect $SUITE_DIR/logs. No judge calls." >&2; exit 1; }
    else
      for gpu_stage in generate detect extract; do
        "$PYTHON" scripts/run_baseline_suite.py "$gpu_stage" --config "$SUITE_CONFIG" --out-dir "$SUITE_DIR" > "$SUITE_DIR/logs/$gpu_stage.log" 2>&1
      done
    fi
    echo "GPU stages finished. Judge calls: 0. Next: gate, judge-plan."
    ;;
  gate)
    "$PYTHON" scripts/run_baseline_suite.py gate --out-dir "$SUITE_DIR"
    ;;
  crossfit)
    "$PYTHON" scripts/run_baseline_suite.py gate --out-dir "$SUITE_DIR" --scheme crossfit --name crossfit_v1
    ;;
  judge-plan)
    answer_files=()
    for method in plain zero_shot_cot fp_identification extract_verify premise_review; do
      answer_files+=("$SUITE_DIR/answers/$method.jsonl")
    done
    "$PYTHON" scripts/evaluate_well.py prepare --questions "$SUITE_DIR/questions.jsonl" --answers "${answer_files[@]}" --out-dir "$SUITE_DIR/well_judge"
    ;;
  judge)
    : "${MAX_JUDGE_CALLS:?Set MAX_JUDGE_CALLS explicitly, e.g. 20. No automatic unbounded judge run.}"
    "$PYTHON" scripts/evaluate_well.py score --out-dir "$SUITE_DIR/well_judge" --max-calls "$MAX_JUDGE_CALLS"
    ;;
  report)
    "$PYTHON" scripts/evaluate_well.py report --out-dir "$SUITE_DIR/well_judge"
    ;;
  status)
    "$PYTHON" scripts/run_baseline_suite.py status --out-dir "$SUITE_DIR"
    ;;
  *)
    echo 'Stages: test | test-runtime | references | prepare | generate | detect | extract | gpu | gate | crossfit | judge-plan | judge | report | status'
    echo 'Defaults: Qwen2.5-7B; existing canonical questions; our holdout; GPUs 0/1 only; judge is separate/capped.'
    ;;
esac
