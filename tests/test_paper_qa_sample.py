import copy
import json
from types import SimpleNamespace

import pytest

from scripts import paper_qa_sample as sample


def bundle():
    b = {'qa': [{'dataset': d, 'id': f'{d}:{i}', 'gold': i % 4}
                for d, n in [('large', 600), ('small', 308)] for i in range(n)],
         'splits': [{'train': ['train1'], 'test': ['test1']}], 'rows': []}
    b['content_sha256'] = sample.digest(b)
    return b


def test_common_sample_counts_and_repeatability():
    b = bundle()
    before = copy.deepcopy(b)
    m = sample.make_manifest(b)
    rows = sample.select_rows(b, m)
    assert len(rows) == 808
    assert len({(r['dataset'], r['id']) for r in rows}) == 808
    assert sum(r['dataset'] == 'small' for r in rows) == 308
    assert sample.make_manifest(b) == m
    assert b == before
    assert sample.make_manifest(b, seed=99)['datasets']['large']['ids'] != m['datasets']['large']['ids']


def test_rejects_population_and_manifest_changes():
    b = bundle()
    m = sample.make_manifest(b)
    altered = copy.deepcopy(b)
    altered['qa'][0]['gold'] = 99
    with pytest.raises(ValueError, match='population mismatch'):
        sample.select_rows(altered, m)
    m['datasets']['large']['ids'][0] = 'not_a_question'
    with pytest.raises(ValueError, match='hash mismatch'):
        sample.select_rows(b, m)


@pytest.mark.parametrize('table2_only', [False, True])
def test_execution_filters_qa_only_without_restoring_table2_qa(tmp_path, monkeypatch, table2_only):
    b = bundle()
    source = tmp_path/'bundle.json'
    source.write_text(json.dumps(b))
    manifest = tmp_path/'sample.json'
    manifest.write_text(json.dumps(sample.make_manifest(b)))
    monkeypatch.setattr(sample, 'MANIFEST', manifest)
    class Worker:
        def execute(self):
            return self.bundle
    runner = SimpleNamespace(V4Worker=Worker)
    sample.install_qa_subset(runner)
    installed = Worker.execute
    sample.install_qa_subset(runner)
    assert Worker.execute is installed
    worker = Worker()
    worker.bundle = copy.deepcopy(b)
    if table2_only:
        worker.bundle['qa'] = []
    worker.args = SimpleNamespace(bundle=source)
    worker.out = tmp_path/'worker'
    result = worker.execute()
    assert len(result['qa']) == (0 if table2_only else 808)
    assert result['splits'] == b['splits']
    assert result['rows'] == b['rows']
    assert json.loads(source.read_text()) == b
    assert (worker.out/'execution_policies/table3_sample500.json').exists()
