#!/usr/bin/env bash
# Browser device approval -> pinned Gemma download -> train-only smoke -> suite.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data3/heejae/hf_cache
export PYTHONUNBUFFERED=1
export PAPER_PYTHON=/data3/heejae/uv/paper_baselines_b81ace2/bin/python
export PAPER_CONFIG=/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/protocol/runtime.json
export PAPER_BUNDLE=/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/protocol/full_bundle.json
export PAPER_OUT=/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/main
export PAPER_MODELS=gemma3
PAPER_HF_PYTHON=/data3/heejae/uv/hf_device_login/bin/python
exec {PAPER_AUTH_FD}>/tmp/paper-gemma-device-bootstrap.lock
flock -n "$PAPER_AUTH_FD" || { echo 'A Gemma device login bootstrap is already active'; exit 1; }
printf 'stage=waiting_for_browser_authorization\n'
/data3/heejae/uv/hf_device_login/bin/hf auth login --force --format agent
printf 'stage=downloading_pinned_checkpoint\n'
"$PAPER_HF_PYTHON" - "$PAPER_CONFIG" <<'PY'
import json,sys
from huggingface_hub import snapshot_download
config=json.load(open(sys.argv[1])); spec=config['models']['gemma3']
path=snapshot_download(spec['model_id'],revision=spec['revision'],cache_dir=config['cache_root'],
    allow_patterns=['*.json','*.safetensors','*.model','*.jinja','*.txt','tokenizer*'])
print('pinned_checkpoint_ready:',path,flush=True)
PY
printf 'stage=train_only_generation_and_Well_smoke\n'
bash scripts/run_paper_baselines_4090.sh smoke fixed --methods plain,direct_gate
printf 'stage=starting_full_baselines\n'
exec bash scripts/run_paper_baselines_a6000_gpu01.sh gemma3
