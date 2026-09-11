import json
from pathlib import Path

import pytest

from scripts import reevaluate_pilot_nfp as mod
from scripts.check_terra_judge import MODEL, PROTOCOL_V2
from src.jsonl import load_json, read_jsonl, write_jsonl
from src.pilot import digest, file_digest, make_manifest


@pytest.fixture
def sample(tmp_path):
    pilot, source, out = tmp_path / 'pilot', tmp_path / 'source', tmp_path / 'out'
    (pilot / 'dev').mkdir(parents=True)
    (pilot / 'split').mkdir()
    source.mkdir()
    questions = [{'id': f'n{i:02}', 'set': 'nfp', 'partition': 'dev', 'question': f'Question {i}',
                  'hallucination_text': f'Reference {i}', 'group_id': f'g{i}'} for i in range(30)]
    questions.append({'id': 'f0', 'set': 'fpq', 'partition': 'dev', 'question': 'FPQ', 'group_id': 'gf'})
    manifest = make_manifest(questions, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash='ref')
    (pilot / 'split/manifest.json').write_text(json.dumps(manifest))
    write_jsonl(pilot / 'split/dev.jsonl', questions)
    for method in mod.METHODS:
        run = {'method': method, 'partition': 'dev', 'manifest_hash': manifest['manifest_hash']}
        rows = [{'id': q['id'], 'set': q['set'], 'question': q['question'],
                 'response': 'Invented belief' if method == 'fp_identification' and q['id'] in {f'n{i:02}' for i in range(10)} else 'Helpful answer',
                 'run_hash': digest(run)} for q in questions]
        path = pilot / 'dev' / f'{method}.jsonl'
        write_jsonl(path, rows)
        Path(str(path) + '.run.json').write_text(json.dumps(run))
    template = (mod.ROOT / 'prompts/nfp_role_clarified_v2.txt').read_text()
    base = {'protocol': PROTOCOL_V2, 'model': MODEL, 'backend': 'codex', 'cases': [],
            'prompt_template': template, 'prompt_template_hash': digest(template),
            'transport_hash': file_digest(mod.ROOT / 'src/llm_backend.py')}
    (source / 'plan.json').write_text(json.dumps(base))
    (source / 'human_review.tsv').write_text('Frozen source labels')
    (source / 'run.json').write_text(json.dumps({'plan_hash': digest(base),
        'human_hash': file_digest(source / 'human_review.tsv')}))
    return pilot, source, out


def test_dedup_single_pass_mapping_and_report(sample, monkeypatch):
    pilot, source, out = sample
    before = (pilot / 'dev/plain.jsonl').read_bytes()
    mod.prepare(pilot, source, out)
    mod.prepare(pilot, source, out)
    plan = load_json(out / 'plan.json')
    assert plan['budget'] == 40  # 30 unique questions + 10 different answers
    assert plan['prompt_template'] == load_json(source / 'plan.json')['prompt_template']
    assert not (out / 'attempts.jsonl').exists()
    calls = []
    def call(prompt):
        calls.append(prompt)
        negative = prompt.endswith('Invented belief') or 'Invented belief' in prompt
        return json.dumps({'Sharpness': -1 if negative else 1, 'Reason': 'reviewed',
                           'AnswerEvidence': 'Invented belief' if negative else '',
                           'InventedPremise': 'Unsupported belief' if negative else ''}), MODEL
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: call)
    mod.score(out)
    mod.score(out)
    assert len(calls) == 40
    scores = mod.report(out)
    assert scores['plain'] == scores['premise_cot']
    assert sum(v == -1 for v in scores['fp_identification'].values()) == 10
    report = (out / 'report.md').read_text()
    assert '| fp_identification | 30/30 | 20 | 66.7 | 30 | 0/10 | -33.3 |' in report
    assert 'FPQ reevaluation' in report
    assert before == (pilot / 'dev/plain.jsonl').read_bytes()


def test_missing_not_imputed_and_no_retry(sample, monkeypatch):
    pilot, source, out = sample
    mod.prepare(pilot, source, out)
    calls = []
    def interrupted(prompt):
        calls.append(prompt)
        raise KeyboardInterrupt()
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: interrupted)
    with pytest.raises(KeyboardInterrupt):
        mod.score(out)
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: lambda prompt: (calls.append(prompt) or 'invalid', MODEL))
    mod.score(out)
    mod.score(out)
    assert len(calls) == 40
    assert len(list(read_jsonl(out / 'attempts.jsonl'))) == 79
    scores = mod.report(out)
    assert all(v is None for values in scores.values() for v in values.values())
    assert '| plain | 0/30 | 0 | incomplete | 0 | 0/0 | NA |' in (out / 'report.md').read_text()


def test_evidence_flags_keep_scores(sample, monkeypatch):
    pilot, source, out = sample
    mod.prepare(pilot, source, out)
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: lambda p: (json.dumps({
        'Sharpness': -1, 'Reason': 'reason', 'AnswerEvidence': 'absent quote', 'InventedPremise': 'belief'}), MODEL))
    mod.score(out)
    scores = mod.report(out)
    assert all(v == -1 for v in scores['plain'].values())
    assert 'evidence flags 40' in (out / 'report.md').read_text()


def test_changed_plan_or_input_rejected(sample, monkeypatch):
    pilot, source, out = sample
    mod.prepare(pilot, source, out)
    path = pilot / 'dev/plain.jsonl'
    rows = list(read_jsonl(path))
    rows[0]['response'] = 'Changed answer'
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        mod.prepare(pilot, source, out)
    plan = load_json(out / 'plan.json')
    plan['mapping']['plain']['n00'] = plan['mapping']['plain']['n01']
    (out / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: pytest.fail('must not call'))
    with pytest.raises(ValueError, match='Mapping'):
        mod.score(out)


def test_wrong_model_stops_after_one_call(sample, monkeypatch):
    pilot, source, out = sample
    mod.prepare(pilot, source, out)
    monkeypatch.setattr(mod, 'make_caller', lambda *a, **k: lambda p: ('{"Sharpness":1,"Reason":"ok"}', 'other'))
    with pytest.raises(SystemExit, match='Unexpected model'):
        mod.score(out)
    assert len(list(read_jsonl(out / 'attempts.jsonl'))) == 2
