#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:?cm, crepe, or cm-answers}" in
  cm)
    for objective in binary premise; do
      bash scripts/run_premise_sft_ddp.sh cancer "${objective}" "cm_${objective}_all_s17"
    done
    ;;
  cm-answers)
    for objective in answer premise joint binary; do
      name="cm_${objective}_s17"
      bash scripts/run_premise_sft_ddp.sh cancer "${objective}" "${name}" --cohort answer_matched
    done
    ;;
  crepe)
    for objective in binary premise; do
      bash scripts/run_premise_sft_ddp.sh crepe "${objective}" "crepe_${objective}_s17"
    done
    ;;
  *) echo 'usage: bash scripts/run_premise_sft_matrix.sh cm|crepe|cm-answers'; exit 1 ;;
esac
