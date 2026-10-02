#!/usr/bin/env bash
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Independent CPU environment: leave CUDA/PyTorch and running GPU jobs untouched.
FPQA_ROOT="${FPQA_ROOT:-$CODE_ROOT/external/fpqa_20261002}"
FPQA_ENV="${FPQA_ENV:-$CODE_ROOT/.venv/fpqa_prompting}"
if [[ ! -x "$FPQA_ENV/bin/python" ]]; then
  uv venv --python 3.12 "$FPQA_ENV"
fi
uv pip install --python "$FPQA_ENV/bin/python" -r "$CODE_ROOT/configs/fpqa_prompting/requirements.txt"
FPQA_DATA="${FPQA_DATA:-$CODE_ROOT/results/fpqa_prompting/data}"
if [[ ! -f "$FPQA_DATA/manifest.json" ]]; then
  "$FPQA_ENV/bin/python" "$CODE_ROOT/scripts/prepare_fpqa_prompt_data.py" \
    --sources "$FPQA_ROOT" --out "$FPQA_DATA" --download
fi
