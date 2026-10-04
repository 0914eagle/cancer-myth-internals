#!/usr/bin/env bash
# Existing correction-recovery methods vs our prompt baselines on the Qwen2.5 Cancer-Myth suite (docs/66 §4-1).
# Generates the two new rows (prewome, extract_verify_plain) on GPU $GPU, judges them with Sonnet into a
# separate judge dir, and prints the report. Plain / premise_review / extract_verify / fp_identification
# rows already exist in $SUITE_DIR/well_judge_claude and are compared from there.
#   GPU=1 nohup bash scripts/run_method_compare_1004.sh > method_compare_1004.log 2>&1 &
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
export SUITE_DIR="${SUITE_DIR:-$ART/results/baselines/qwen25_7b_final1024_v1}"
export JUDGE_MAX="${JUDGE_MAX:-1500}"
export CFG="${CFG:-configs/qwen25_7b.yaml}"
GPU="${GPU:-1}"
METHODS="${METHODS:-prewome extract_verify_plain}"
W="$SUITE_DIR/well_judge_claude_methods1004"
test -s "$SUITE_DIR/questions.jsonl" || { echo "[error] $SUITE_DIR has no questions.jsonl" >&2; exit 2; }

echo "[gen] $METHODS $(date)"
CUDA_VISIBLE_DEVICES=$GPU python scripts/run_baseline_suite.py generate --out-dir "$SUITE_DIR" --config "$CFG" \
  --methods $METHODS
python scripts/run_baseline_suite.py status --out-dir "$SUITE_DIR" | grep -A3 -E 'prewome|extract_verify_plain' || true

answers=()
for m in $METHODS; do answers+=("$SUITE_DIR/answers/$m.jsonl"); done
if [ ! -s "$W/plan.json" ]; then
  python scripts/evaluate_well.py prepare --backend claude --questions "$SUITE_DIR/questions.jsonl" \
    --answers "${answers[@]}" --out-dir "$W"
fi
set +e
while :; do
  python scripts/evaluate_well.py score --out-dir "$W" --max-calls "$JUDGE_MAX" 2>&1 | tail -3 | tee -a "$W/resume.log"
  grep -q "\"remaining_unstarted\": 0" "$W/resume.log" && break
  sleep 900
done
python scripts/evaluate_well.py report --out-dir "$W" | tail -12
echo "[done] $(date) -> $W/report.md ; baseline rows: $SUITE_DIR/well_judge_claude/report.md"
