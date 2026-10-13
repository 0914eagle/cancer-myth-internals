from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from scripts.paper_gepa_parallel import evaluate_pair, write_quota_status


def pair():
    return [{'id': 'f', 'label': 1, 'question': 'false'},
            {'id': 'n', 'label': 0, 'question': 'normal'}]


def test_pair_overlaps_and_retains_input_order_and_score(tmp_path):
    both_generating = threading.Barrier(2)
    normal_graded = threading.Event()
    finished = []

    def generate(method, question, dataset, candidate):
        both_generating.wait(timeout=5)
        return {'answer': question}

    def judge(row, answer):
        if row['label']:
            assert normal_graded.wait(timeout=5)
        finished.append(row['id'])
        if not row['label']:
            normal_graded.set()
        return (1 if row['label'] else 5), 'reason'

    w = SimpleNamespace(out=tmp_path, pipeline=SimpleNamespace(run=generate), judge=judge)
    with ThreadPoolExecutor(2) as pool:
        score, feedback = evaluate_pair(w, 'gepa', 'cancer_myth', {}, pair(), pool)
    assert finished == ['n', 'f']
    assert [r['id'] for r in feedback['examples']] == ['f', 'n']
    assert score == pytest.approx(.6)


def test_stop_after_generation_prevents_new_judge_call(tmp_path):
    def generate(*a, **k):
        (tmp_path / 'STOP').touch()
        return {'answer': 'a'}
    w = SimpleNamespace(out=tmp_path, pipeline=SimpleNamespace(run=generate),
                        judge=lambda *a: pytest.fail('Judge must not start after STOP'))
    with pytest.raises(RuntimeError, match='STOP requested'):
        evaluate_pair(w, 'gepa', 'cancer_myth', {}, pair())


def test_failure_is_not_converted_to_a_low_score(tmp_path):
    def judge(*a):
        raise ValueError('bad rating')
    w = SimpleNamespace(out=tmp_path, pipeline=SimpleNamespace(run=lambda *a, **k: {'answer': 'a'}),
                        judge=judge)
    with ThreadPoolExecutor(2) as pool:
        with pytest.raises(ValueError, match='bad rating'):
            evaluate_pair(w, 'gepa', 'cancer_myth', {}, pair(), pool)


def test_concurrent_quota_writes_keep_all_requests(tmp_path):
    w = SimpleNamespace(out=tmp_path)
    def write(i):
        write_quota_status(w, str(i), {'state': 'waiting_quota', 'next_retry_epoch': i})
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(write, range(40)))
    files = list((tmp_path / 'optimization_quota_requests').glob('*.json'))
    assert len(files) == 40
    for p in files:
        record = json.loads(p.read_text())
        assert record['request_id'] == p.stem
        assert record['next_retry_epoch'] == int(p.stem)
    assert json.loads((tmp_path / 'optimization_quota.json').read_text())['state'] == 'waiting_quota'
    assert not list(tmp_path.rglob('*.tmp'))


def test_worker_wiring_parallel_budget_and_feedback(tmp_path, monkeypatch):
    """Exercise Worker.optimize without GPU/API or an installed GEPA dependency."""
    from scripts.run_paper_baselines import Worker
    captured = {}

    def optimize_anything(**kwargs):
        captured.update(kwargs)
        # Model the adapter's evaluator pool, independently of the row pool.
        with ThreadPoolExecutor(kwargs['config'].engine.max_workers) as pool:
            values = list(pool.map(lambda e: kwargs['evaluator'](kwargs['seed_candidate'], e),
                                   kwargs['valset']))
        assert all(score == .8 for score, _ in values)
        assert all([x['label'] for x in fb['examples']] == [1, 0] for _, fb in values)
        return SimpleNamespace(best_candidate=kwargs['seed_candidate'], best_idx=0)

    module = SimpleNamespace(GEPAConfig=SimpleNamespace, EngineConfig=SimpleNamespace,
                             ReflectionConfig=SimpleNamespace, optimize_anything=optimize_anything)
    monkeypatch.setitem(sys.modules, 'gepa', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'gepa.optimize_anything', module)
    tokenizer = SimpleNamespace(encode=lambda v, **k: v.split())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    w = object.__new__(Worker)
    w.config = {'cache_root': str(tmp_path), 'gepa': {'metric_budget': 24, 'seed': 42,
        'dev_per_class_max': 2, 'tokenizer_revision': 'test',
        'prompt_tokenizer': 'fake', 'prompt_token_limit': 512}}
    w.out = tmp_path
    w.identity = {'test': True}
    w.reg = {'prompts': {'plain': 'seed'}}
    w.gepa_evaluation_workers = 4
    w.gepa_pair_workers = 2
    w.lookup = {str(i): {'id': str(i), 'label': i % 2, 'question': str(i)} for i in range(12)}
    w.pipeline = SimpleNamespace(run=lambda *a, **k: {'answer': 'a'})
    w.judge = lambda *a: (4, 'reason')
    split = {'dataset': 'cancer_myth', 'name': 'fold0',
             'train': [str(i) for i in range(8)], 'dev': [str(i) for i in range(8, 12)]}
    assert w.optimize(split, 'gepa') == {'answer': 'seed'}
    config = captured['config']
    assert config.engine.parallel is True
    assert config.engine.max_workers == 4
    assert config.engine.max_metric_calls == 12
    assert config.reflection.reflection_minibatch_size == 2
    captured.clear()
    assert w.optimize(split, 'gepa') == {'answer': 'seed'}
    assert not captured  # Completed selections are reused, not optimized twice.


def test_same_external_request_is_cached_once_under_parallel_calls(tmp_path, monkeypatch):
    from scripts import run_paper_baselines_v4 as runner
    calls = []
    class FakeCLI:
        def __init__(self, *a, **k): pass
        def __call__(self, messages):
            calls.append(messages)
            return 'Rating: 4', {}
    monkeypatch.setattr(runner, 'PooledCLI', FakeCLI)
    w = object.__new__(runner.V4Worker)
    w.out = tmp_path / 'model'
    w.args = SimpleNamespace(out=tmp_path)
    w.external = {'judge_model': {'model': 'fake'}}
    with ThreadPoolExecutor(8) as pool:
        answers = list(pool.map(lambda _: w.external_call('judge', [{'role': 'user', 'content': 'q'}]), range(16)))
    assert answers == ['Rating: 4'] * 16
    assert len(calls) == 1


def test_quota_retry_recovers_and_stop_interrupts_wait(tmp_path, monkeypatch):
    from scripts import run_paper_baselines_v4 as runner
    count = []
    class FakeCLI:
        def __init__(self, *a, **k): pass
        def __call__(self, messages):
            count.append(1)
            if len(count) == 1:
                raise RuntimeError('rate_limit_error')
            return 'Rating: 4', {}
    monkeypatch.setattr(runner, 'PooledCLI', FakeCLI)
    monkeypatch.setattr(runner, 'INTERVAL', 0)
    w = object.__new__(runner.V4Worker)
    w.out = tmp_path / 'model'
    w.args = SimpleNamespace(out=tmp_path)
    w.external = {'judge_model': {'model': 'fake'}}
    assert w.external_call('judge', [{'role': 'user', 'content': 'q'}]) == 'Rating: 4'
    assert len(count) == 2
    assert json.loads((w.out / 'optimization_quota.json').read_text())['state'] == 'recovered'
    count.clear()
    (w.out / 'STOP').touch()
    monkeypatch.setattr(runner, 'INTERVAL', 1)
    with pytest.raises(RuntimeError, match='STOP requested'):
        w.external_call('judge', [{'role': 'user', 'content': 'different request'}])
