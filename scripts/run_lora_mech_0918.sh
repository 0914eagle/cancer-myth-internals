#!/usr/bin/env bash
# 9/18: mechanism scan + twin LoRA (B) in one launch from ONE tmux window (nohup chains).
#   Step 0 (both GPUs): mechanism extract, rows split in two shards; the script waits for both.
#   Chain M (GPU 0): patch L17 -> patch L20 -> eval  => $SUITE_DIR/mechanism/mech_v1/report.md
#   Chain L (GPU 1): twin LoRA data -> train -> generate (adapter, base twins) -> Sonnet judge loop
#                    -> compare; then the FPQ-only control LoRA the same way.
#                    => $SUITE_DIR/lora/{twin_v1,fpqonly_v1}/compare.md
# Usage:  bash scripts/run_lora_mech_0918.sh          (SUITE_DIR must point at the final1024 suite; peft installed)
#         ONLY=L bash scripts/run_lora_mech_0918.sh   relaunch one chain (M or L); SKIP_EXTRACT=1 skips step 0
# Logs:   $SUITE_DIR/logs/lora_mech_0918/{extract0,extract1,M_mechanism,L_lora}.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
export SUITE_DIR="${SUITE_DIR:?export SUITE_DIR=.../results/baselines/qwen25_7b_final1024_v1 first}"
export ROWS="${ROWS:-$DATA/e1_rows_v1}"
export JUDGE_MAX="${JUDGE_MAX:-1500}"
export CFG=configs/qwen25_7b.yaml
LOG="$SUITE_DIR/logs/lora_mech_0918"; mkdir -p "$LOG"
test -s "$SUITE_DIR/questions.jsonl" || { echo "[error] $SUITE_DIR has no questions.jsonl" >&2; exit 2; }
test -s "$ROWS/questions_twins.jsonl" || { echo "[error] $ROWS/questions_twins.jsonl missing" >&2; exit 2; }
test -s "$SUITE_DIR/answers/fp_unconditional.jsonl" || { echo "[error] fp_unconditional answers missing" >&2; exit 2; }
test -s "$SUITE_DIR/well_judge_claude_fpu/judge_plan.json" || { echo "[error] well_judge_claude_fpu (fp_unconditional scores) missing" >&2; exit 2; }
python -c "import peft" 2>/dev/null || { echo "[error] pip install peft first" >&2; exit 2; }

if [[ "${SKIP_EXTRACT:-0}" != "1" ]]; then
  echo "[0] mechanism extract on both GPUs $(date)"
  CUDA_VISIBLE_DEVICES=0 python scripts/mechanism_scan.py extract --suite-dir "$SUITE_DIR" --config "$CFG" \
    --e1-questions "$ROWS/questions.jsonl" --twins "$ROWS/questions_twins.jsonl" --name mech_v1 --shard 0/2 > "$LOG/extract0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 python scripts/mechanism_scan.py extract --suite-dir "$SUITE_DIR" --config "$CFG" \
    --e1-questions "$ROWS/questions.jsonl" --twins "$ROWS/questions_twins.jsonl" --name mech_v1 --shard 1/2 > "$LOG/extract1.log" 2>&1 &
  P1=$!
  wait $P0 || { echo "[0] shard 0 failed; see $LOG/extract0.log" >&2; exit 1; }
  wait $P1 || { echo "[0] shard 1 failed; see $LOG/extract1.log" >&2; exit 1; }
  echo "[0] extract done $(date)"
fi

if [[ "${ONLY:-ML}" == *M* ]]; then
nohup bash -c '
set -uo pipefail
echo "[M] start $(date)"
for L in 17 20; do
  CUDA_VISIBLE_DEVICES=0 python scripts/mechanism_scan.py patch --suite-dir "$SUITE_DIR" --config "$CFG" --name mech_v1 --layers $L || { echo "[M] patch L$L failed"; exit 1; }
done
python scripts/mechanism_scan.py eval --suite-dir "$SUITE_DIR" --name mech_v1 | tail -40
echo "[M] done $(date)  -> $SUITE_DIR/mechanism/mech_v1/report.md"
' > "$LOG/M_mechanism.log" 2>&1 &
echo "[M] pid $!  log $LOG/M_mechanism.log"
fi

if [[ "${ONLY:-ML}" == *L* ]]; then
nohup bash -c '
set -uo pipefail
judge_loop() {  # $1 = judge dir
  while :; do
    python scripts/evaluate_well.py score --out-dir "$1" --max-calls "$JUDGE_MAX" 2>&1 | tail -3 | tee -a "$1/resume.log"
    grep -q "\"remaining_unstarted\": 0" "$1/resume.log" && break
    sleep 900
  done
}
run_one() {  # $1 = run name, $2 = negatives
  OUT="$SUITE_DIR/lora/$1"; mkdir -p "$OUT"
  echo "[L:$1] data $(date)"
  CUDA_VISIBLE_DEVICES=1 python scripts/lora_twin.py data --suite-dir "$SUITE_DIR" --config "$CFG" \
    --twins "$ROWS/questions_twins.jsonl" --e1-questions "$ROWS/questions.jsonl" \
    --well-dir "$SUITE_DIR/well_judge_claude_fpu" --out "$OUT" --negatives "$2" || return 1
  echo "[L:$1] train $(date)"
  CUDA_VISIBLE_DEVICES=1 python scripts/lora_twin.py train --config "$CFG" --out "$OUT" || return 1
  echo "[L:$1] generate adapter $(date)"
  CUDA_VISIBLE_DEVICES=1 python scripts/lora_twin.py generate --config "$CFG" --out "$OUT" --which adapter --suite-dir "$SUITE_DIR" || return 1
  echo "[L:$1] generate base twins $(date)"
  CUDA_VISIBLE_DEVICES=1 python scripts/lora_twin.py generate --config "$CFG" --out "$OUT" --which base --suite-dir "$SUITE_DIR" || return 1
  W="$OUT/well_judge_claude"
  echo "[L:$1] judge $(date)"
  python scripts/evaluate_well.py prepare --backend claude --questions "$OUT/eval_questions.jsonl" \
    --answers "$OUT/eval_adapter/answers/lora_$1.jsonl" "$OUT/eval_base/answers/plain.jsonl" --out-dir "$W" || return 1
  judge_loop "$W"
  python scripts/evaluate_well.py report --out-dir "$W" | tail -8
  python scripts/lora_twin.py compare --out "$OUT" --lora-well "$W" \
    --baseline-well "$SUITE_DIR/well_judge_claude" "$SUITE_DIR/well_judge_claude_fpu" || return 1
  echo "[L:$1] done $(date)  -> $OUT/compare.md"
}
run_one twin_v1 twins || { echo "[L] twin_v1 failed"; exit 1; }
run_one fpqonly_v1 none || { echo "[L] fpqonly_v1 failed"; exit 1; }
echo "[L] all done $(date)"
' > "$LOG/L_lora.log" 2>&1 &
echo "[L] pid $!  log $LOG/L_lora.log"
fi
