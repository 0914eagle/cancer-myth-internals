#!/usr/bin/env bash
# List jobs launched through scripts/lib/detach.sh: running or not, last log line.
#
#   bash scripts/jobs.sh            # last 15
#   bash scripts/jobs.sh 40         # last 40
#   bash scripts/jobs.sh gpu        # plus nvidia-smi summary

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
JOBS="${DATA_ROOT}/cancer_myth_internals/logs/jobs.tsv"
N=15
[[ "${1:-}" =~ ^[0-9]+$ ]] && N="$1"

if [[ ! -s "${JOBS}" ]]; then
  echo "no jobs recorded in ${JOBS}"
else
  printf '%-20s %-8s %-8s %-24s %s\n' "started" "pid" "state" "tag" "last line"
  tail -n "${N}" "${JOBS}" | while IFS=$'\t' read -r ts pid tag log; do
    if kill -0 "${pid}" 2>/dev/null; then state="RUNNING"; else state="done"; fi
    last="$(tail -n 1 "${log}" 2>/dev/null | cut -c1-90)"
    printf '%-20s %-8s %-8s %-24s %s\n' "${ts:0:19}" "${pid}" "${state}" "${tag:0:24}" "${last}"
  done
fi

if [[ "${1:-}" == "gpu" ]] || [[ "${2:-}" == "gpu" ]]; then
  echo
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "  GPU %s  %s / %s  util %s\n", $1, $2, $3, $4}'
fi
