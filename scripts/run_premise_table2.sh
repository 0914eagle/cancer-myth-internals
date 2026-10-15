#!/usr/bin/env bash
# Matched nine-row generation -> bounded Sonnet Well scoring -> Table 2.
# Launch this entire script in an independent session (setsid); see doc 40 §9.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/premise_sft_env.sh
: "${TABLE2_MAX_CALLS:?Set the explicit Well call cap (at most 1323 for this test)}"
[[ "$TABLE2_MAX_CALLS" =~ ^[0-9]+$ && "$TABLE2_MAX_CALLS" -gt 0 && "$TABLE2_MAX_CALLS" -le 1323 ]] || {
  echo 'TABLE2_MAX_CALLS must be 1..1323'; exit 2;
}
TABLE2_OUT="${TABLE2_OUT:-$PREMISE_ART/eval/table2_prompt_sft_20260930_v1}"
CM_RUN="${CM_RUN:-cm_premise_all_s17_gpu2_20260930}"
CREPE_RUN="${CREPE_RUN:-crepe_premise_s17_gpu3_20260930}"
JUDGE_MODEL="${JUDGE_MODEL:-claude-sonnet-5}"
PYTHON="$PREMISE_ENV/bin/python"
mkdir -p "$TABLE2_OUT/logs"
echo $$ > "$TABLE2_OUT/logs/pipeline.pid"
status_file="$TABLE2_OUT/pipeline_status.json"
status() {
  "$PYTHON" -c 'import datetime,json,sys; from pathlib import Path; p=Path(sys.argv[1]); value={"stage":sys.argv[2],"at":datetime.datetime.now(datetime.timezone.utc).isoformat()}; temp=p.with_suffix(".tmp"); temp.write_text(json.dumps(value,indent=2)+"\n"); temp.replace(p)' "$status_file" "$1"
}
trap 'status failed' ERR
status preparing
"$PYTHON" scripts/premise_table2.py prepare --art "$PREMISE_ART" --out "$TABLE2_OUT" \
  --cancer-run "$CM_RUN" --crepe-run "$CREPE_RUN"
status generating
gpu2() {
  # Inspect Plain for both models before spending time on the prompted conditions.
  for condition in cancer base; do
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u scripts/premise_table2.py generate \
      --out "$TABLE2_OUT" --condition "$condition" --methods plain || return $?
  done
  for condition in cancer base; do
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" -u scripts/premise_table2.py generate \
      --out "$TABLE2_OUT" --condition "$condition" --methods zero_shot_cot fp_unconditional || return $?
  done
}
gpu3() {
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u scripts/premise_table2.py generate \
    --out "$TABLE2_OUT" --condition crepe --methods plain zero_shot_cot fp_unconditional
}
gpu2 > "$TABLE2_OUT/logs/gpu2.log" 2>&1 &
gpu2_pid=$!
echo "$gpu2_pid" > "$TABLE2_OUT/logs/gpu2.pid"
gpu3 > "$TABLE2_OUT/logs/gpu3.log" 2>&1 &
gpu3_pid=$!
echo "$gpu3_pid" > "$TABLE2_OUT/logs/gpu3.pid"
gpu2_status=0; wait "$gpu2_pid" || gpu2_status=$?
gpu3_status=0; wait "$gpu3_pid" || gpu3_status=$?
[[ "$gpu2_status" -eq 0 && "$gpu3_status" -eq 0 ]] || {
  status generation_failed; echo "Generation failed: GPU2=$gpu2_status GPU3=$gpu3_status"; exit 1;
}
status judge_preparing
"$PYTHON" scripts/premise_table2.py judge-plan --out "$TABLE2_OUT" --backend claude --model "$JUDGE_MODEL"
status scoring
"$PYTHON" -u scripts/premise_table2.py score --out "$TABLE2_OUT" --max-calls "$TABLE2_MAX_CALLS" \
  > "$TABLE2_OUT/logs/well.log" 2>&1
"$PYTHON" scripts/premise_table2.py report --out "$TABLE2_OUT"
"$PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); d=json.loads((p/"table2.json").read_text()); print("All Well judgments valid:", d["complete"]); sys.exit(0 if d["complete"] else 1)' "$TABLE2_OUT"
status complete
