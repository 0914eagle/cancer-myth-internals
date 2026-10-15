import json
import threading
import time

import pytest

from scripts import run_frontier_luna as runner


def events(extra=None):
    items = [{'type': 'thread.started', 'thread_id': 'test'}, {'type': 'turn.started'}]
    if extra:
        items.append(extra)
    items += [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'Answer.'}},
              {'type': 'turn.completed', 'usage': {}}]
    return '\n'.join(json.dumps(x) for x in items)


@pytest.mark.parametrize('event', [
    {'type': 'turn.failed'},
    {'type': 'item.completed', 'item': {'type': 'command_execution'}},
    {'type': 'item.completed', 'item': {'type': 'web_search'}},
    {'type': 'item.completed', 'item': {'type': 'mcp_tool_call'}},
    {'type': 'item.completed', 'item': {'type': 'error'}},
    {'type': 'turn.completed', 'usage': {}},
])
def test_reject_tools_errors_and_extra_turn(event):
    with pytest.raises(ValueError):
        runner.parse_events(events(event), 'Answer.')


def test_final_answer_matches_events():
    assert len(runner.parse_events(events(), 'Answer.')) == 4
    with pytest.raises(ValueError):
        runner.parse_events(events(), 'Different answer.')


def test_twenty_workers_and_stop(tmp_path):
    runner.tr.STOP.clear()
    lock = threading.Lock()
    state = {'active': 0, 'peak': 0}
    returned = []
    def worker(i):
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
        time.sleep(.04)
        with lock:
            state['active'] -= 1
        return i
    runner.schedule(tmp_path, range(40), worker, lambda i, f: returned.append(f.result()), 20)
    assert sorted(returned) == list(range(40)) and state['peak'] == 20
    (tmp_path / 'STOP').touch()
    with pytest.raises(RuntimeError, match='STOP'):
        runner.schedule(tmp_path, range(2), lambda _: pytest.fail('Unexpected call'), lambda *a: None, 20)


def test_mixed_condition_resume_and_prompt_identity(tmp_path, monkeypatch):
    runner.tr.STOP.clear()
    inputs = [{'id': f'{kind}_{i}', 'question': f'Question {kind} {i}?'}
              for kind in ('fpq', 'nfp') for i in range(3)]
    specs = {m: {'method': m, 'model': runner.MODEL, 'workers': 20, 'user_template': p}
             for m, p in runner.PROMPTS.items()}
    for m in specs:
        runner.tr.write(tmp_path / m / 'evaluation_questions.json', inputs)
    calls = []
    def invoke(out, method, row, spec, opts):
        calls.append((method, row['id']))
        record = {**row, 'method': method, 'signature': runner.tr.digest([spec, row]),
                  'status': 'complete', 'response': 'Answer.', 'model_used': runner.MODEL}
        runner.tr.write(out / method / 'records' / (row['id'] + '.json'), record)
        return record
    monkeypatch.setattr(runner, 'invoke', invoke)
    runner.generation(tmp_path, inputs, specs, [], 10, 4)
    runner.generation(tmp_path, inputs, specs, [], 20, 0)
    runner.generation(tmp_path, inputs, specs, [], 20, 0)
    assert len(calls) == len(set(calls)) == 12
    for m in specs:
        answers = runner.tr.read_rows(tmp_path / m / 'answers_eval732.jsonl')
        assert len(answers) == 6 and all(r['method'] == m for r in answers)


def test_two_condition_judge_ledger_resume(tmp_path, monkeypatch):
    runner.tr.STOP.clear()
    runner.tr.write(tmp_path / 'manifest.json', {'test': True})
    questions = [{'id': f'q{i}', 'question': f'Unique question {i}?',
                  'set': 'fpq' if i < 583 else 'nfp',
                  'premise_text': 'Reference premise.', 'correction': 'Reference correction.'} for i in range(732)]
    (tmp_path / 'evaluation_questions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in questions))
    for method in runner.PROMPTS:
        folder = tmp_path / method
        folder.mkdir()
        answers = [{'id': q['id'], 'method': method, 'response': f'{method} answer.'} for q in questions]
        (folder / 'answers_eval732.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
    seen = []
    def caller(prompt, model, out):
        seen.append(prompt)
        return 'Rating: 4', runner.JUDGE, {'result': 'Rating: 4'}
    monkeypatch.setattr(runner, 'judge_call', caller)
    runner.judging(tmp_path, 10, 10)
    runner.judging(tmp_path, 10, 10)
    assert len(seen) == len(set(seen)) == 20
    judge = tmp_path / 'well_judge_sonnet_v1'
    plan, h = runner.we.load_plan(judge)
    events, scores = runner.we.read_ledger(judge, plan, h)
    assert len(events) == len(scores) == 20
    assert len(plan['jobs']) == 1464
    assert set(plan['mapping']) == set(runner.PROMPTS)
