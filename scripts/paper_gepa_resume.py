"""Explicit, hash-pinned scheduling-only migration of frozen baseline caches.

No model, prompt, annotation, split, decoding or dependency differences are
accepted. Keep the old cache identity, and record the actual new execution code
in the separate migration and optimization policy artifacts.
"""
import copy
import json
from pathlib import Path

from src.paper_baselines import digest, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
POLICY = 'gepa_parallel_resume.json'
BASE_FILE = 'scripts/run_paper_baselines.py'
ADAPTER_FILE = 'run_paper_baselines_v4.py'
HELPER_FILE = 'paper_gepa_parallel.py'
PINNED_FILES = (BASE_FILE, 'scripts/' + ADAPTER_FILE,
                'scripts/' + HELPER_FILE, 'scripts/paper_gepa_resume.py')
# Audited predecessor: the running serial evaluator before this change.
OLD_BASE = 'd16aaf4fffc3684871ef4664ee73353a9887f5dd775c4b9f8cc3bde71e982c0c'
OLD_ADAPTER = 'bc9a91e57dcc93766ec1234fd112520d406899543ca1e2ae5c4045e4c0e7909a'


def load_policy(out):
    path = Path(out) / POLICY
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    if policy['version'] != 1:
        raise ValueError('Unknown GEPA scheduling migration')
    actual = {f: sha(ROOT / f) for f in PINNED_FILES}
    if actual != policy['execution_sha256']:
        raise ValueError('GEPA migration execution code changed; review before resume')
    old = json.loads((Path(out) / 'identity.json').read_text())
    if old != policy['base_identity']:
        raise ValueError('GEPA migration frozen identity changed')
    if old['code_sha256'][BASE_FILE] != OLD_BASE or policy['adapter_identity'][ADAPTER_FILE] != OLD_ADAPTER:
        raise ValueError('GEPA migration predecessor is not the audited serial worker')
    return policy


def preserve_cache_identity(out, current, stage):
    policy = load_policy(out)
    if policy is None:
        return current
    normalized = copy.deepcopy(current)
    old = policy['base_identity']
    normalized['code_sha256'][BASE_FILE] = old['code_sha256'][BASE_FILE]
    expected = copy.deepcopy(old)
    if stage == 'adapter':
        adapter = normalized['v4_adapter_sha256']
        adapter[ADAPTER_FILE] = policy['adapter_identity'][ADAPTER_FILE]
        adapter.pop(HELPER_FILE)
        expected['v4_adapter_sha256'] = policy['adapter_identity']
    elif stage != 'base':
        raise ValueError('Unknown identity stage')
    if normalized != expected:
        raise ValueError('Non-scheduling settings changed; refusing frozen-cache reuse')
    write_json(Path(out) / 'execution_policies' / ('parallel_resume_' + stage + '.json'), {
        'migration_sha256': sha(Path(out) / POLICY),
        'actual_identity': current, 'cache_identity_sha256': digest(expected),
        'scope': 'Only evaluator scheduling changed; batch-dependent outputs may vary'}, immutable=True)
    return expected


def prepare_policy(out):
    out = Path(out)
    if (out / POLICY).exists():
        return load_policy(out)
    old = json.loads((out / 'identity.json').read_text())
    matches = []
    for p in (out / 'execution_policies').glob('*.json'):
        data = json.loads(p.read_text())
        if data.get(ADAPTER_FILE) == OLD_ADAPTER:
            matches.append(data)
    if not matches or any(x != matches[0] for x in matches):
        raise ValueError('Cannot identify the audited original V4 adapter')
    if old['code_sha256'][BASE_FILE] != OLD_BASE:
        raise ValueError('Unexpected original base worker')
    # Reject any unrelated code edits even before constructing a worker.
    for filename, checksum in old['code_sha256'].items():
        if filename != BASE_FILE and sha(ROOT / filename) != checksum:
            raise ValueError('Unrelated base code changed: ' + filename)
    for filename, checksum in matches[0].items():
        if filename != ADAPTER_FILE and sha(ROOT / 'scripts' / filename) != checksum:
            raise ValueError('Unrelated adapter code changed: ' + filename)
    value = {'version': 1, 'base_identity': old, 'adapter_identity': matches[0],
             'execution_sha256': {f: sha(ROOT / f) for f in PINNED_FILES},
             'evaluation_workers': 4, 'pair_workers': 2,
             'preserve': 'Existing generation, judge, classifier and selected-prompt artifacts'}
    write_json(out / POLICY, value, immutable=True)
    return load_policy(out)
