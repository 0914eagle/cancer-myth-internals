"""Bounded-disk shard download/conversion; preserve public checkpoint provenance."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from huggingface_hub import HfApi, hf_hub_download

p = argparse.ArgumentParser()
p.add_argument('--specs', type=Path, required=True)
a = p.parse_args()
spec = json.loads(a.specs.read_text())['nla_av']
repo, revision = spec['repo'], spec['revision']
root = Path('/data3/heejae/hf_cache/converted') / ('nla-gemma3-27b-L41-av-bf16-' + revision)
stage = root / '_download_staging'
stage.mkdir(parents=True, exist_ok=True)
status_path = root / 'conversion_status.json'
state = json.loads(status_path.read_text()) if status_path.exists() else dict(repo=repo, revision=revision, shards={})
assert state['repo'] == repo and state['revision'] == revision
api = HfApi()
files = api.model_info(repo, revision=revision, files_metadata=True).siblings
metadata = [f.rfilename for f in files if '/' not in f.rfilename and not f.rfilename.endswith('.safetensors')
            and f.rfilename.endswith(('.json', '.yaml', '.model', '.jinja'))]
for filename in metadata:
    local = hf_hub_download(repo, filename, revision=revision, local_dir=stage)
    shutil.copyfile(local, root / filename)
index = json.loads((root / 'model.safetensors.index.json').read_text())
weights = sorted(set(index['weight_map'].values()))
sizes = {f.rfilename: f.size for f in files}
largest = max(sizes[name] for name in weights)
remaining = sum(sizes[name] // 2 for name in weights if name not in state['shards'])
assert remaining + largest + 8 * 1024**3 < shutil.disk_usage('/data3').free
for filename in weights:
    if filename in state['shards']:
        assert (root / filename).exists()
        continue
    state.update(state='downloading', current_shard=filename, utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status_path.write_text(json.dumps(state, indent=2) + '\n')
    local = Path(hf_hub_download(repo, filename, revision=revision, local_dir=stage))
    assert local.stat().st_size == sizes[filename]
    subprocess.run(['/data3/heejae/uv/medical_nla/bin/python',
        str(Path(__file__).with_name('convert_nla_shard_bf16.py')),
        '--input', str(local), '--output', str(root / filename)], check=True)
    converted = json.loads((root / filename).with_suffix('.conversion.json').read_text())
    state['shards'][filename] = converted
    status_path.write_text(json.dumps(state, indent=2) + '\n')
    # Only this task's newly downloaded staging shard is removed after conversion is sealed.
    assert local.parent.resolve() == stage.resolve()
    local.unlink()
    print('Converted', len(state['shards']), '/', len(weights), flush=True)
index['metadata']['total_size'] = sum(s['tensor_bytes'] for s in state['shards'].values())
(root / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
config = json.loads((root / 'config.json').read_text())
config['torch_dtype'] = 'bfloat16'
(root / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
state.update(state='complete', local_path=str(root), conversion='Same BF16 rounding used for BF16 inference; no training',
             original_revision=revision)
status_path.write_text(json.dumps(state, indent=2) + '\n')
print('NLA27 BF16 ready', root, flush=True)
