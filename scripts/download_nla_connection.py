"""Download the released, revision-pinned NLA weights; never train or alter them."""
import argparse
import datetime
import json
from pathlib import Path
from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
p.add_argument('--model', required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
out = Path(c['artifact_root']) / a.model
out.mkdir(parents=True, exist_ok=True)
def state(**kw):
    payload = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **kw)
    tmp = out / 'download_status.tmp'
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(out / 'download_status.json')
try:
    checkpoints = {}
    for role in ('av', 'ar'):
        spec = c['models'][a.model][role]
        state(state='downloading', role=role, **spec)
        checkpoints[role] = snapshot_download(spec['repo'], revision=spec['revision'],
            cache_dir=c['cache_root'], max_workers=4,
            allow_patterns=['*.json','*.safetensors','*.yaml','*.jinja','*.txt','*.model'])
    (out / 'checkpoints.json').write_text(json.dumps(checkpoints, indent=2) + '\n')
    state(state='complete', checkpoints=checkpoints)
except Exception as error:
    state(state='failed', error_type=type(error).__name__, error=str(error))
    raise
