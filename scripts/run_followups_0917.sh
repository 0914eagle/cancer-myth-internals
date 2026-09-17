#!/usr/bin/env bash
# 9/17 follow-ups, in parallel under nohup from ONE tmux window (docs/33 §4, 31, 32 §7).
#   A (GPU 0): Well-style unconditional FP Identification row -> export -> judge plan -> Sonnet scoring
#              loop (15-min retries; the preflight makes a retry during a usage limit free).
#   B (GPU 1): knowledge probe (31) -> CREPE clone/inspect/extract -> CREPE<->Cancer-Myth transfer eval.
# Usage:  bash scripts/run_followups_0917.sh        (SUITE_DIR must point at the final1024 suite)
# Logs:   $SUITE_DIR/logs/followups_0917/{A_fp_unconditional,B_knowledge_crepe}.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
export SUITE_DIR="${SUITE_DIR:?export SUITE_DIR=.../results/baselines/qwen25_7b_final1024_v1 first}"
export CREPE_DIR="${CREPE_DIR:-$ART/external/CREPE}"
export ROWS="${ROWS:-$DATA/e1_rows_v1}"
export JUDGE_MAX="${JUDGE_MAX:-732}"
LOG="$SUITE_DIR/logs/followups_0917"; mkdir -p "$LOG" "$(dirname "$CREPE_DIR")"
test -s "$SUITE_DIR/questions.jsonl" || { echo "[error] $SUITE_DIR has no questions.jsonl" >&2; exit 2; }
test -s "$ROWS/questions_twins.jsonl" || echo "[warn] $ROWS/questions_twins.jsonl missing; CREPE eval will skip twin rows"

nohup bash -c '
set -uo pipefail
echo "[A] start $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_suite.py generate --config configs/qwen25_7b.yaml \
  --out-dir "$SUITE_DIR" --methods fp_unconditional || { echo "[A] generation failed"; exit 1; }
W="$SUITE_DIR/well_judge_claude_fpu"
python scripts/evaluate_well.py prepare --questions "$SUITE_DIR/questions.jsonl" \
  --answers "$SUITE_DIR/answers/fp_unconditional.jsonl" --out-dir "$W" --backend claude || exit 1
while :; do
  python scripts/evaluate_well.py score --out-dir "$W" --max-calls "$JUDGE_MAX" 2>&1 | tail -3 | tee -a "$W/resume.log"
  grep -q "\"remaining_unstarted\": 0" "$W/resume.log" && break
  sleep 900
done
python scripts/evaluate_well.py report --out-dir "$W" | tail -8
echo "[A] done $(date)"
' > "$LOG/A_fp_unconditional.log" 2>&1 &
echo "[A] pid $!  log $LOG/A_fp_unconditional.log"

nohup bash -c '
set -uo pipefail
echo "[B] start $(date)"
CUDA_VISIBLE_DEVICES=1 python scripts/probe_knowledge.py --suite-dir "$SUITE_DIR" --config configs/qwen25_7b.yaml --name qwen25_7b \
  || echo "[B] knowledge probe failed; continuing to CREPE"
if [ ! -d "$CREPE_DIR" ]; then git clone --depth 1 https://github.com/velocityCavalry/CREPE "$CREPE_DIR" || echo "[B] CREPE clone failed"; fi
python scripts/crepe_transfer.py inspect --crepe-dir "$CREPE_DIR" || { echo "[B] CREPE inspect failed"; exit 1; }
CUDA_VISIBLE_DEVICES=1 python scripts/crepe_transfer.py extract --crepe-dir "$CREPE_DIR" --suite-dir "$SUITE_DIR" \
  --config configs/qwen25_7b.yaml || { echo "[B] CREPE extract failed (label detection? see inspect output above)"; exit 1; }
python scripts/crepe_transfer.py eval --suite-dir "$SUITE_DIR" \
  --twins "$ROWS/questions_twins.jsonl" --twins-questions "$ROWS/questions.jsonl" \
  --paraphrases "$SUITE_DIR/variants/para/questions_para.jsonl" --name v1
echo "[B] done $(date)"
' > "$LOG/B_knowledge_crepe.log" 2>&1 &
echo "[B] pid $!  log $LOG/B_knowledge_crepe.log"
echo "Watch: tail -f $LOG/A_fp_unconditional.log $LOG/B_knowledge_crepe.log"
