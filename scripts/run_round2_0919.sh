#!/usr/bin/env bash
# 9/19 round 2 from ONE tmux window (nohup chains). Judge calls only in the L chain (Sonnet loop).
#   Chain F (GPU 0): signal flow  fit -> scan -> patch -> eval  => $SUITE_DIR/signal_flow/flow_v1/report.md
#   Chain L (GPU 1): twin LoRA round 2: balanced SFT (no false paraphrases) then DPO pairs,
#                    each: data -> train -> generate -> judge -> compare  => $SUITE_DIR/lora/{twin_bal_s4,twin_dpo_s4}/compare.md
# Usage:  nohup bash scripts/run_round2_0919.sh > $ART/launcher_round2.log 2>&1 &
#         ONLY=F|L to launch one chain. Twin Plain answers are copied from lora/twin_s4/twin_plain (no regeneration).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
export SUITE_DIR="${SUITE_DIR:-$ART/results/baselines/qwen25_7b_final1024_v1}"
export ROWS="${ROWS:-$DATA/e1_rows_v1}"
export JUDGE_MAX="${JUDGE_MAX:-1500}"
export MIN_SCORE="${MIN_SCORE:-4}"
export CFG=configs/qwen25_7b.yaml
LOG="$SUITE_DIR/logs/round2_0919"; mkdir -p "$LOG"
test -s "$SUITE_DIR/mechanism/mech_v1/rows.jsonl" || { echo "[error] mechanism rows missing (run mechanism_scan extract first)" >&2; exit 2; }
test -d "$SUITE_DIR/lora/twin_s4/twin_plain" || { echo "[error] lora/twin_s4/twin_plain missing (round 1 data stage)" >&2; exit 2; }
python -c "import peft" 2>/dev/null || { echo "[error] peft missing in $(command -v python)" >&2; exit 2; }
echo "[env] python $(command -v python); suite $SUITE_DIR"

if [[ "${ONLY:-FL}" == *F* ]]; then
nohup bash -c '
set -uo pipefail
echo "[F] start $(date)"
CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py fit --suite-dir "$SUITE_DIR" --config "$CFG" --rows "$SUITE_DIR/mechanism/mech_v1/rows.jsonl" --name flow_v1 || { echo "[F] fit failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py scan --suite-dir "$SUITE_DIR" --config "$CFG" --name flow_v1 || { echo "[F] scan failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 python scripts/signal_flow.py patch --suite-dir "$SUITE_DIR" --config "$CFG" --name flow_v1 || { echo "[F] patch failed"; exit 1; }
python scripts/signal_flow.py eval --suite-dir "$SUITE_DIR" --name flow_v1 | tail -60
echo "[F] done $(date)  -> $SUITE_DIR/signal_flow/flow_v1/report.md"
' > "$LOG/F_signal_flow.log" 2>&1 &
echo "[F] pid $!  log $LOG/F_signal_flow.log"
fi

if [[ "${ONLY:-FL}" == *L* ]]; then
nohup bash -c '
set -uo pipefail
judge_loop() {
  while :; do
    python scripts/evaluate_well.py score --out-dir "$1" --max-calls "$JUDGE_MAX" 2>&1 | tail -3 | tee -a "$1/resume.log"
    grep -q "\"remaining_unstarted\": 0" "$1/resume.log" && break
    sleep 900
  done
}
run_one() {  # $1 = run name, $2 = objective (sft|dpo), $3... = extra data args
  OUT="$SUITE_DIR/lora/$1"; mkdir -p "$OUT"
  [[ -d "$OUT/twin_plain" ]] || cp -r "$SUITE_DIR/lora/twin_s4/twin_plain" "$OUT/twin_plain"
  echo "[L:$1] data $(date)"
  CUDA_VISIBLE_DEVICES=1 python scripts/lora_twin.py data --suite-dir "$SUITE_DIR" --config "$CFG" \
    --twins "$ROWS/questions_twins.jsonl" --e1-questions "$ROWS/questions.jsonl" \
    --well-dir "$SUITE_DIR/well_judge_claude_fpu" --out "$OUT" --negatives twins --min-score "$MIN_SCORE" --objective "$2" "${@:3}" || return 1
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
run_one twin_bal_s4 sft --no-fpara || { echo "[L] twin_bal_s4 failed"; exit 1; }
run_one twin_dpo_s4 dpo || { echo "[L] twin_dpo_s4 failed"; exit 1; }
echo "[L] all done $(date)"
' > "$LOG/L_lora_round2.log" 2>&1 &
echo "[L] pid $!  log $LOG/L_lora_round2.log"
fi
