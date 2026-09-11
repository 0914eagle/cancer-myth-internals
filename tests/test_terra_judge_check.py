import csv
import json

import pytest

from scripts import check_terra_judge as check
from src.jsonl import load_json, write_jsonl


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    pilot, out = tmp_path / 'pilot', tmp_path / 'check'
    (pilot / 'dev').mkdir(parents=True)
    (pilot / 'split').mkdir()
    responses = [{'id': f'nfp_{i}', 'set': 'nfp', 'question': f'Q{i}', 'response': f'Answer{i}'}
                 for i in range(30)]
    questions = [{**r, 'hallucination_text': f'Ref{i}'} for i, r in enumerate(responses)]
    write_jsonl(pilot / 'split/dev.jsonl', questions)
    for name in ('plain', 'steering_L14_a0'):
        write_jsonl(pilot / f'dev/{name}.jsonl', responses)
        rows = [{'question_id': r['id'], 'sharpness': -1 if i < 5 and name != 'plain' else 1}
                for i, r in enumerate(responses)]
        write_jsonl(pilot / f'dev/{name}_judge_codex_gpt-5.6-sol.jsonl', rows)
    ex = tmp_path / 'examples.json'
    ex.write_text(json.dumps([{'example_question': 'E', 'answer': 'A', 'score': {'Sharpness': 1}}]))
    monkeypatch.setattr(check, 'audit', lambda *args: None)  # provenance is covered by audit tests
    monkeypatch.setattr(check, 'load_config', lambda _: {'judge': {
        'examples_fpq': str(ex), 'examples_nfp': str(ex)}})
    check.prepare(pilot, out, 'config')
    return pilot, out


def fill_review(out):
    plan = load_json(out / 'plan.json')
    with (out / 'human_review.tsv').open('w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'score', 'rationale', 'reviewer'])
        writer.writerows([c['id'], '1', 'No invented premise objection', 'Test reviewer']
                         for c in plan['cases'])


def mock_caller(monkeypatch, interrupt_at=None):
    calls = []
    def make(*args, **kwargs):
        def call(prompt):
            calls.append(prompt)
            if len(calls) == interrupt_at:
                raise KeyboardInterrupt()
            return '{\n"Reason": "No invented objection", "Sharpness": 1\n}', check.MODEL
        return call
    monkeypatch.setattr(check, 'make_caller', make)
    return calls


def test_fixed_balanced_sample_and_blinded_review(prepared):
    _, out = prepared
    plan = load_json(out / 'plan.json')
    assert len(plan['cases']) == 10 and len(plan['order']) == 20
    assert sum(c['cohort'] == 'disagreement' for c in plan['cases']) == 5
    assert {c['id'] for c in plan['cases'] if c['cohort'] == 'disagreement'} == {
        f'nfp_{i}' for i in range(5)}
    assert all(sum(j['case_id'] == c['id'] for j in plan['order']) == 2 for c in plan['cases'])
    assert 'cohort' not in (out / 'review.md').read_text()
    assert not (out / 'attempts.jsonl').exists()


def test_no_call_before_human_review(prepared, monkeypatch):
    _, out = prepared
    calls = mock_caller(monkeypatch)
    with pytest.raises(ValueError, match='Complete human_review'):
        check.score(out)
    assert calls == []


def test_twenty_calls_and_no_duplicate_on_resume(prepared, monkeypatch, capsys):
    _, out = prepared
    fill_review(out)
    calls = mock_caller(monkeypatch)
    check.score(out)
    check.score(out)
    assert len(calls) == 20
    assert len(set(calls)) == 10
    check.report(out)
    assert 'valid 20/20' in capsys.readouterr().out


def test_interrupt_consumes_one_attempt_without_retry(prepared, monkeypatch, capsys):
    _, out = prepared
    fill_review(out)
    calls = mock_caller(monkeypatch, interrupt_at=3)
    with pytest.raises(KeyboardInterrupt):
        check.score(out)
    check.score(out)
    assert len(calls) == 20
    check.report(out)
    assert 'valid 19/20; missing/invalid 1' in capsys.readouterr().out


def test_human_labels_cannot_change_after_scoring(prepared, monkeypatch):
    _, out = prepared
    fill_review(out)
    calls = mock_caller(monkeypatch)
    check.score(out)
    path = out / 'human_review.tsv'
    path.write_text(path.read_text().replace('Test reviewer', 'Different reviewer'))
    with pytest.raises(ValueError):
        check.score(out)
    assert len(calls) == 20
    with pytest.raises(ValueError, match='changed'):
        check.report(out)
