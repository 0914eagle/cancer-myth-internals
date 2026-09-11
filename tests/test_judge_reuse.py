import json
from pathlib import Path

import pytest

from scripts import run_judge
from scripts.audit_pilot_judge import audit
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import digest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    folder = tmp_path / 'dev'
    folder.mkdir()
    (tmp_path / 'split').mkdir()
    qpath = tmp_path / 'split' / 'dev.jsonl'
    write_jsonl(qpath, [{'id': 'n1', 'set': 'nfp', 'question': 'Q?', 'hallucination_text': 'H'}])
    examples = tmp_path / 'examples.json'
    examples.write_text(json.dumps([{'example_question': 'Example?', 'example_assumption': 'Fact',
                                     'answer': 'Example answer', 'score': {'Sharpness': 1, 'Reason': 'ok'}}]))
    monkeypatch.setattr(run_judge, 'load_config', lambda _: {
        'judge': {'examples_fpq': str(examples), 'examples_nfp': str(examples)}})
    calls = []
    def caller(*args, **kwargs):
        def call(prompt):
            calls.append(prompt)
            return '{\n"Reason": "original", "Sharpness": 1\n}', 'gpt-5.6-terra'
        return call
    monkeypatch.setattr(run_judge, 'make_caller', caller)
    monkeypatch.setattr(run_judge, 'acquire_lock', lambda _: None)
    def run(name, answer='Answer', reuse=None, dry=False):
        response = folder / f'{name}.jsonl'
        run_spec = {'method': name, 'identity': 'fixed'}
        if not response.exists():
            write_jsonl(response, [{'id': 'n1', 'set': 'nfp', 'question': 'Q?',
                                   'response': answer, 'run_hash': digest(run_spec)}])
            Path(str(response) + '.run.json').write_text(json.dumps(run_spec))
        out = folder / f'{name}_judge_codex_gpt-5.6-terra.jsonl'
        args = ['judge', '--responses', str(response), '--questions', str(qpath),
                '--output', str(out), '--model', 'gpt-5.6-terra', '--backend', 'codex']
        if reuse:
            args += ['--reuse-scores', str(reuse)]
        if dry:
            args += ['--dry-run']
        monkeypatch.setattr('sys.argv', args)
        run_judge.main()
        return out
    return tmp_path, calls, run


def test_identical_answer_reuses_without_call_and_rebinds_provenance(setup):
    _, calls, run = setup
    plain = run('plain')
    target = run('steering_L14_a0', reuse=plain)
    assert len(calls) == 1
    row = next(read_jsonl(target))
    signature = json.loads(Path(str(target) + '.run.json').read_text())
    assert row['judge_run_hash'] == digest(signature)
    assert row['response_run_hash'] == digest(signature['response_run'])
    assert row['reason'] == 'original'
    assert row['score_reuse']['source_score_id'] == 'n1::n1'
    before = target.read_bytes()
    run('steering_L14_a0', reuse=plain)
    assert target.read_bytes() == before and len(calls) == 1


def test_changed_answer_requires_call(setup):
    _, calls, run = setup
    plain = run('plain')
    target = run('changed', answer='Different', reuse=plain)
    assert len(calls) == 2
    assert 'score_reuse' not in next(read_jsonl(target))


@pytest.mark.parametrize('field', ['requested_model', 'questions_hash', 'examples_hash', 'rubric_hash', 'temperature'])
def test_rejects_different_judge_inputs(setup, field):
    _, _, run = setup
    plain = run('plain')
    signature = json.loads(Path(str(plain) + '.run.json').read_text())
    signature[field] = 'different'
    with pytest.raises(ValueError, match='differs'):
        run_judge.load_reuse_scores(plain, signature)


def test_rejects_broken_provenance(setup):
    _, _, run = setup
    plain = run('plain')
    row = next(read_jsonl(plain))
    row['judge_run_hash'] = 'wrong'
    write_jsonl(plain, [row])
    signature = json.loads(Path(str(plain) + '.run.json').read_text())
    with pytest.raises(ValueError, match='provenance'):
        run_judge.load_reuse_scores(plain, signature)


def test_dry_run_no_calls_or_score_file(setup, capsys):
    _, calls, run = setup
    plain = run('plain')
    target = run('steering_L14_a0', reuse=plain, dry=True)
    assert len(calls) == 1 and not target.exists()
    assert 'over 0 calls' in capsys.readouterr().out


def test_audit_lists_disagreement_without_changing_files(setup):
    root, calls, run = setup
    plain = run('plain')
    target = run('steering_L14_a0')
    row = next(read_jsonl(target))
    row.update(sharpness=-1, reason='second judgment')
    write_jsonl(target, [row])
    before = [p.read_bytes() for p in (plain, target)]
    text = audit(root, 'gpt-5.6-terra', 'steering_L14_a0')
    assert 'score disagreements on identical answers=1' in text
    assert 'original' in text and 'second judgment' in text and 'Answer' in text
    assert before == [p.read_bytes() for p in (plain, target)]
    assert len(calls) == 2
