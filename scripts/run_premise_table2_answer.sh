#!/usr/bin/env bash
# SFT-only explicit-answer follow-up. No Base generation or Base judging.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/premise_sft_env.sh
: "${TABLE2_MAX_CALLS:?Set the explicit SFT Well call cap (1..882)}"
[[ "$TABLE2_MAX_CALLS" =~ ^[0-9]+$ && "$TABLE2_MAX_CALLS" -gt 0 && "$TABLE2_MAX_CALLS" -le 882 ]] || exit 2
TABLE2_OUT="${TABLE2_OUT:-$PREMISE_ART/eval/table2_sft_answer_instruction_20260930_v1}"
PYTHON="$PREMISE_ENV/bin/python"
mkdir -p "$TABLE2_OUT/logs"
echo $$ > "$TABLE2_OUT/logs/pipeline.pid"
status() {
  "$PYTHON" -c 'import datetime,json,sys; from pathlib import Path; p=Path(sys.argv[1]); tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps({"stage":sys.argv[2],"at":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+"\n"); tmp.replace(p)' "$TABLE2_OUT/pipeline_status.json" "$1"
}
trap 'status failed' ERR
status preparing
"$PYTHON" scripts/premise_table2_answer.py prepare --art "$PREMISE_ART" --out "$TABLE2_OUT" \
  --cancer-run cm_premise_all_s17_gpu2_20260930 --crepe-run crepe_premise_s17_gpu3_20260930
status generating
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u scripts/premise_table2_answer.py generate \
  --out "$TABLE2_OUT" --condition cancer > "$TABLE2_OUT/logs/gpu2.log" 2>&1 &
gpu2_pid=$!
echo "$gpu2_pid" > "$TABLE2_OUT/logs/gpu2.pid"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u scripts/premise_table2_answer.py generate \
  --out "$TABLE2_OUT" --condition crepe > "$TABLE2_OUT/logs/gpu3.log" 2>&1 &
gpu3_pid=$!
echo "$gpu3_pid" > "$TABLE2_OUT/logs/gpu3.pid"
gpu2_status=0; wait "$gpu2_pid" || gpu2_status=$?
gpu3_status=0; wait "$gpu3_pid" || gpu3_status=$?
[[ "$gpu2_status" -eq 0 && "$gpu3_status" -eq 0 ]] || { status generation_failed; exit 1; }
status judge_preparing
"$PYTHON" scripts/premise_table2_answer.py judge-plan --out "$TABLE2_OUT"
status scoring
"$PYTHON" -u scripts/premise_table2_answer.py score --out "$TABLE2_OUT" --max-calls "$TABLE2_MAX_CALLS" \
  > "$TABLE2_OUT/logs/well.log" 2>&1
"$PYTHON" scripts/premise_table2_answer.py report --out "$TABLE2_OUT"
"$PYTHON" -c 'import json,sys; from pathlib import Path; d=json.loads((Path(sys.argv[1])/"table2.json").read_text()); sys.exit(0 if d["complete"] else 1)' "$TABLE2_OUT"
status complete
