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


@pytest.mark.parametrize('raw', [
    '{"Reason":"ok","Sharpness":1}',
    '{\n"Sharpness":-1,"Reason":"reason with {braces}"\n}',
    '```json\n{"Reason":"ok","Sharpness":1}\n```',
    'Result:\n{"Sharpness":1,"Reason":"ok"}\nDone.',
    '{\r\n"Reason":"ok","Sharpness":1\r\n}',
])
def test_json_parser_accepts_formatting_variations(raw):
    parsed, ok = check.parse_nfp_json(raw)
    assert ok and type(parsed['Sharpness']) is int


@pytest.mark.parametrize('raw', [
    '', 'Sharpness: 1', '{"Reason":"ok","Sharpness":0}',
    '{"Reason":"ok","Sharpness":true}', '{"Reason":"ok","Sharpness":"1"}',
    '{"Reason":"ok","Sharpness":1.0}', '{"Sharpness":1}',
    '{"Reason":"ok","Sharpness":1,"Sharpness":-1}',
    '{"Reason":"ok","Sharpness":1} {"Reason":"no","Sharpness":-1}',
    '{"Reason":"ok","Sharpness":1} {"Reason":"ok","Sharpness":1}',
    '[{"Reason":"ok","Sharpness":1}]',
    '{broken: {"Reason":"ok","Sharpness":1}}',
    '{"nested":{"Reason":"ok","Sharpness":1}}',
])
def test_json_parser_rejects_invalid_or_ambiguous_scores(raw):
    assert check.parse_nfp_json(raw) == ({}, False)


def test_offline_reparse_recovers_17_without_calls_or_ledger_edits(prepared, monkeypatch, capsys):
    from src.judge_prompts import parse_score

    _, out = prepared
    fill_review(out)
    parser_v2 = check.parse_nfp_json
    monkeypatch.setattr(check, 'parse_nfp_json', parse_score)
    calls = []
    def make(*args, **kwargs):
        def call(prompt):
            calls.append(prompt)
            raw = json.dumps({'Reason': 'ok', 'Sharpness': 1}, indent=2 if len(calls) <= 3 else None)
            return raw, check.MODEL
        return call
    monkeypatch.setattr(check, 'make_caller', make)
    check.score(out)
    ledger = out / 'attempts.jsonl'
    before = ledger.read_bytes()
    check.report(out)
    assert 'valid 3/20' in capsys.readouterr().out
    monkeypatch.setattr(check, 'parse_nfp_json', parser_v2)
    def no_call(*args, **kwargs):
        raise AssertionError('Offline report must not initialize any model')
    monkeypatch.setattr(check, 'make_caller', no_call)
    check.report(out, reparse=True)
    text = capsys.readouterr().out
    assert 'valid 20/20' in text and 'recovered=17' in text
    assert 'changed scores=0' in text
    assert ledger.read_bytes() == before and len(calls) == 20


def test_reparse_keeps_invalid_score_unresolved(prepared, monkeypatch, capsys):
    _, out = prepared
    fill_review(out)
    monkeypatch.setattr(check, 'make_caller', lambda *a, **kw: lambda p: (
        '{"Reason":"not an NFP score","Sharpness":0}', check.MODEL))
    check.score(out)
    check.report(out, reparse=True)
    text = capsys.readouterr().out
    assert 'valid 0/20' in text and 'Unresolved:' in text
    assert 'not an NFP score' in text


def test_revise_preserves_source_and_approved_sample(prepared):
    _, source = prepared
    fill_review(source)
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    target = source.parent / 'check_v2'
    check.revise(source, target)
    original, revised = load_json(source / 'plan.json'), load_json(target / 'plan.json')
    assert revised['protocol'] == check.PROTOCOL_V2
    assert revised['order'] == original['order']
    assert [(c['id'], c['answer']) for c in revised['cases']] == [(c['id'], c['answer']) for c in original['cases']]
    assert revised['cases'][0]['prompt'] != original['cases'][0]['prompt']
    assert check.human_labels(target, revised) == check.human_labels(source, original)
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    check.revise(source, target)  # deterministic offline resume
    assert not (target / 'attempts.jsonl').exists()
    with pytest.raises(ValueError, match='separate'):
        check.revise(source, source)


def test_template_substitution_does_not_reinterpret_input():
    template = '{{question}}|{{possible_hallucination}}|{{answer}}'
    case = {'question': '{{answer}}', 'reference': '{}', 'answer': '{{question}}'}
    assert check.render_v2(template, case) == '{{answer}}|{}|{{question}}'


@pytest.mark.parametrize('parsed,answer,expected', [
    ({'Sharpness': 1, 'AnswerEvidence': '', 'InventedPremise': ''}, 'help', None),
    ({'Sharpness': -1, 'AnswerEvidence': 'You assume all', 'InventedPremise': 'all need X'},
     'You assume all need X.', None),
    ({'Sharpness': -1, 'AnswerEvidence': 'missing', 'InventedPremise': 'all need X'},
     'Helpful answer', 'exact substring'),
    ({'Sharpness': -1}, 'Helpful answer', 'Missing'),
])
def test_evidence_contract(parsed, answer, expected):
    issue = check.evidence_issue(parsed, answer)
    assert issue is None if expected is None else expected in issue


def test_v2_twenty_calls_and_flagged_evidence_does_not_change_score(prepared, monkeypatch, capsys):
    _, source = prepared
    fill_review(source)
    target = source.parent / 'check_v2'
    check.revise(source, target)
    calls = []
    def make(*a, **kw):
        def call(prompt):
            calls.append(prompt)
            assert 'Protocol: nfp-role-clarified-v2' in prompt
            return json.dumps({'Sharpness': -1, 'Reason': 'test',
                               'AnswerEvidence': 'fabricated quote', 'InventedPremise': 'test'}), check.MODEL
        return call
    monkeypatch.setattr(check, 'make_caller', make)
    check.score(target)
    check.score(target)
    assert len(calls) == 20
    check.report(target, reparse=True)
    text = capsys.readouterr().out
    assert 'Evidence-format issues: 20' in text
    assert 'valid 20/20' in text and 'human agreement 0/10' in text
    assert 'fabricated quote' in text


def test_v2_modified_prompt_rejected_before_calls(prepared, monkeypatch):
    _, source = prepared
    fill_review(source)
    target = source.parent / 'check_v2'
    check.revise(source, target)
    plan = load_json(target / 'plan.json')
    plan['cases'][0]['prompt'] = 'changed'
    (target / 'plan.json').write_text(json.dumps(plan))
    calls = mock_caller(monkeypatch)
    with pytest.raises(ValueError, match='Rendered prompt'):
        check.score(target)
    assert calls == []
