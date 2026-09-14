#!/usr/bin/env bash
# Gate pilot only. No generation, judge checks, or judge calls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh "${DATA_ROOT:-}"
pilot="${PILOT_DIR:?Set PILOT_DIR to the completed baseline pilot}"
gate="${GATE_DIR:-$pilot/gate_L21_v1}"
nfp="${NFP_EVAL_DIR:-$pilot/nfp_baselines_terra_v2_once}"
stage="${1:-all}"
case "$stage" in
  all|extract|fit|report) ;;
  *) echo 'usage: run_gemma_gate.sh [all|extract|fit|report]'; exit 1 ;;
esac
if [[ "$stage" == all || "$stage" == report ]]; then
  for file in "$nfp/plan.json" "$nfp/run.json" "$nfp/attempts.jsonl"; do
    [[ -f "$file" ]] || { echo "Missing prerequisite: $file"; exit 1; }
  done
  for method in plain fp_identification premise_cot; do
    for suffix in .jsonl .jsonl.run.json _judge_codex_gpt-5.6-sol.jsonl _judge_codex_gpt-5.6-sol.jsonl.run.json; do
      [[ -f "$pilot/dev/$method$suffix" ]] || { echo "Missing prerequisite: $pilot/dev/$method$suffix"; exit 1; }
    done
  done
fi
if [[ "$stage" == all || "$stage" == extract ]]; then
  python scripts/pilot_gate.py extract --pilot-dir "$pilot" --out-dir "$gate" --config "${CONFIG:-configs/gemma2_9b.yaml}"
fi
if [[ "$stage" == all || "$stage" == fit ]]; then
  python scripts/pilot_gate.py fit --manifest "$pilot/split/manifest.json" --out-dir "$gate"
fi
if [[ "$stage" == all || "$stage" == report ]]; then
  python scripts/pilot_gate.py report --pilot-dir "$pilot" --out-dir "$gate" --nfp-dir "$nfp"
fi
