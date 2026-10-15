#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXPERIMENT_OUT="${1:-results/fpqa_prompting/well_scope_factorial_luna6_20261004_v1}"
.venv/fpqa_prompting/bin/python -u scripts/run_well_scope_factorial.py --out "$EXPERIMENT_OUT" --smoke --workers 2
.venv/fpqa_prompting/bin/python -u scripts/run_well_scope_factorial.py --out "$EXPERIMENT_OUT" --workers 4
