import copy
import json
from pathlib import Path

import pytest

from scripts import paper_gepa_resume as r
from scripts.queue_paper_gepa_parallel import safe_generation_boundary


def identities():
    old = {'config': {'seed': 42}, 'code_sha256': {r.BASE_FILE: r.OLD_BASE, 'other.py': 'same'}}
    adapter = {r.ADAPTER_FILE: r.OLD_ADAPTER, 'other_adapter.py': 'same'}
    policy = {'base_identity': old, 'adapter_identity': adapter}
    current = copy.deepcopy(old)
    current['code_sha256'][r.BASE_FILE] = 'new'
    return old, adapter, policy, current


def test_resume_preserves_cache_only_for_scheduling_changes(tmp_path, monkeypatch):
    old, adapter, policy, current = identities()
    monkeypatch.setattr(r, 'load_policy', lambda out: policy)
    (tmp_path / r.POLICY).write_text('{}')
    assert r.preserve_cache_identity(tmp_path, current, 'base') == old
    current = {**old, 'v4_adapter_sha256': {**adapter, r.ADAPTER_FILE: 'new', r.HELPER_FILE: 'new'}}
    result = r.preserve_cache_identity(tmp_path, current, 'adapter')
    assert result == {**old, 'v4_adapter_sha256': adapter}
    assert json.loads((tmp_path / 'execution_policies/parallel_resume_adapter.json').read_text())['actual_identity'] == current


@pytest.mark.parametrize('change', ['config', 'code', 'adapter'])
def test_resume_rejects_unrelated_changes(tmp_path, monkeypatch, change):
    old, adapter, policy, current = identities()
    monkeypatch.setattr(r, 'load_policy', lambda out: policy)
    current['v4_adapter_sha256'] = {**adapter, r.ADAPTER_FILE: 'new', r.HELPER_FILE: 'new'}
    if change == 'config': current['config']['seed'] = 43
    elif change == 'code': current['code_sha256']['other.py'] = 'changed'
    else: current['v4_adapter_sha256']['other_adapter.py'] = 'changed'
    with pytest.raises(ValueError, match='Non-scheduling'):
        r.preserve_cache_identity(tmp_path, current, 'adapter')


def test_no_manifest_leaves_normal_identity_checks_intact(tmp_path):
    identity = {'anything': True}
    assert r.preserve_cache_identity(tmp_path, identity, 'base') is identity


def test_policy_rejects_code_modified_after_registration(tmp_path):
    old, adapter, _, _ = identities()
    (tmp_path / 'identity.json').write_text(json.dumps(old))
    (tmp_path / r.POLICY).write_text(json.dumps({'version': 1, 'base_identity': old,
        'adapter_identity': adapter, 'execution_sha256': {f: 'wrong' for f in r.PINNED_FILES}}))
    with pytest.raises(ValueError, match='execution code changed'):
        r.load_policy(tmp_path)


def test_waits_for_selected_and_never_stops_an_optimizer(tmp_path):
    progress = {'state': 'running', 'stage': 'table12', 'method': 'gepa_gate',
                'dataset': 'cancer_myth', 'split': 'fold0', 'processed': 0, 'total': 242}
    assert not safe_generation_boundary(tmp_path, progress, 128)
    selected = tmp_path / 'gepa/cancer_myth/fold0/gepa_gate/selected.json'
    selected.parent.mkdir(parents=True)
    selected.write_text('{}')
    assert safe_generation_boundary(tmp_path, progress, 128)
    assert not safe_generation_boundary(tmp_path, {**progress, 'processed': 200}, 128)
    assert not safe_generation_boundary(tmp_path, {**progress, 'method': 'gepa'}, 128)
    assert not safe_generation_boundary(tmp_path, {**progress, 'stage': 'table3'}, 128)
    assert not safe_generation_boundary(tmp_path, {**progress, 'state': 'failed'}, 128)
