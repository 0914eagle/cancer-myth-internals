import json
import threading
import time

from scripts import run_well_parallel as runner


def prepare(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    questions = [{'id': f'q{i}', 'set': 'fpq' if i % 2 == 0 else 'nfp',
                  'question': f'Original question {i}?', 'premise_text': 'Reference premise.',
                  'correction': 'Reference correction.'} for i in range(12)]
    answers = [{'id': r['id'], 'method': 'plain', 'response': 'Saved answer.'} for r in questions]
    qpath, apath = tmp_path / 'questions.jsonl', tmp_path / 'answers.jsonl'
    qpath.write_text(''.join(json.dumps(r) + '\n' for r in questions))
    apath.write_text(''.join(json.dumps(r) + '\n' for r in answers))
    out = tmp_path / 'judge'
    runner.we.prepare(qpath, [apath], out, backend='claude', model='claude-sonnet-5')
    return out


def test_parallel_ledger_and_resume_no_duplicate_judgments(tmp_path, monkeypatch):
    out = prepare(tmp_path, monkeypatch)
    state = {'active': 0, 'peak': 0, 'calls': []}
    lock = threading.Lock()

    def caller(prompt):
        with lock:
            state['calls'].append(prompt)
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        time.sleep(.03)
        with lock:
            state['active'] -= 1
        return 'Explanation.\nRating: 4', 'claude-sonnet-5', {'result': 'Explanation.\nRating: 4'}

    first = runner.score_parallel(out, 5, caller=caller, preflight=False)
    assert first['valid'] == 5 and state['peak'] == 5
    final = runner.score_parallel(out, 732, caller=caller, preflight=False)
    assert final['valid'] == 12 and final['state'] == 'complete'
    assert len(state['calls']) == len(set(state['calls'])) == 12
    events = [json.loads(line) for line in (out / 'attempts.jsonl').read_text().splitlines()]
    assert len(events) == 24
    plan, plan_hash = runner.we.load_plan(out)
    _, scores = runner.we.read_ledger(out, plan, plan_hash)
    assert len(scores) == 12 and set(scores.values()) == {4}


def test_wrong_model_halts_submission_and_preserves_invalid_ledger(tmp_path, monkeypatch):
    out = prepare(tmp_path, monkeypatch)
    def caller(prompt):
        time.sleep(.03)
        return 'Rating: 5', 'wrong-model', {'result': 'Rating: 5'}
    status = runner.score_parallel(out, 732, caller=caller, preflight=False)
    assert status['state'] == 'stopped'
    assert status['valid'] == 0
    assert status['attempted'] <= 5
    assert status['finished'] == status['attempted']
