"""Verify the publisher and cached kernel files before using a pinned local kernel."""
import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi
from kernels.utils import _check_trust_remote_code

ROOT = Path(__file__).resolve().parents[1]
REPO = 'kernels-community/finegrained-fp8'
REVISION = '8c178950e8710e26a2210e4e909cd02dc16c8715'
CACHE = Path('/data1/heejae/hf_cache/hub/kernels--kernels-community--finegrained-fp8/snapshots') / REVISION
OUT = ROOT / 'results/qwen38/full_transformers_20260926_v1/kernel_pin.json'

api = HfApi(token=False)
org = api.get_organization_overview('kernels-community')
assert getattr(org, 'trustedKernelPublisher', False) is True
_check_trust_remote_code(REPO, False)
verified = []
for entry in api.list_repo_tree(REPO, repo_type='kernel', revision=REVISION, recursive=True):
    name = entry.path
    if not name.startswith('build/torch-cuda/') or not hasattr(entry, 'blob_id'):
        continue
    path = CACHE / name
    data = path.read_bytes()
    assert len(data) == entry.size, name
    # Kernel sources/metadata are ordinary Git blobs, not LFS weights.
    assert not getattr(entry, 'lfs', None), 'Unexpected LFS kernel file'
    git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert git_sha == entry.blob_id, f'Cached kernel differs from pinned upstream: {name}'
    verified.append({'path': name, 'git_blob': git_sha, 'sha256': hashlib.sha256(data).hexdigest()})
assert len(verified) >= 10
manifest = {'repo': REPO, 'revision': REVISION, 'cache': str(CACHE),
            'publisher_trusted': True, 'verified_at': time.time(), 'files': verified}
OUT.write_text(json.dumps(manifest, indent=2) + '\n')
print(f'Publisher trust and {len(verified)} cached files verified against pinned upstream revision.')
