import json
import threading
import time
from types import SimpleNamespace

import pytest

from scripts import run_frontier_claude as runner


def reply(**changes):
    data = {'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'A full answer.',
            'stop_reason': 'end_turn', 'num_turns': 1, 'modelUsage': {'claude-sonnet-5': {}},
            'usage': {'server_tool_use': {'web_search_requests': 0}}}
    data.update(changes)
    return json.dumps(data)


@pytest.mark.parametrize('changes', [
    {'stop_reason': 'max_tokens'}, {'is_error': True}, {'result': ''},
    {'modelUsage': {'another-model': {}}}, {'num_turns': 2},
    {'usage': {'server_tool_use': {'web_search_requests': 1}}},
])
def test_rejects_incomplete_wrong_model_and_tool_results(changes):
    with pytest.raises(ValueError):
        runner.parse_result(reply(**changes), 'claude-sonnet-5')


def test_accepts_dated_model_without_fallback():
    _, served = runner.parse_result(reply(modelUsage={'claude-sonnet-5-20260922': {}}), 'claude-sonnet-5')
    assert served == 'claude-sonnet-5-20260922'


def setup_run(tmp_path, monkeypatch):
    runner.STOP.clear()
    monkeypatch.setattr(runner.signal, 'signal', lambda *a: None)
    (tmp_path / 'isolated_cwd').mkdir()
    inputs = [{'id': f'q{i}', 'question': f'Question {i}?'} for i in range(12)]
    runner.write(tmp_path / 'evaluation_questions.json', inputs[:10])
    return SimpleNamespace(out_dir=tmp_path, limit=0), {'command': ['fake'], 'model': 'claude-sonnet-5', 'workers': 5}, inputs


def test_five_independent_requests_and_resume(tmp_path, monkeypatch):
    args, spec, inputs = setup_run(tmp_path, monkeypatch)
    lock = threading.Lock()
    state = {'active': 0, 'peak': 0, 'prompts': []}

    def fake(cmd, prompt, cwd, timeout):
        assert timeout == 240
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            state['prompts'].append(prompt)
        time.sleep(.03)
        with lock:
            state['active'] -= 1
        return reply()

    monkeypatch.setattr(runner, 'invoke', fake)
    assert runner.generate(args, spec, inputs) == 0
    assert state['peak'] == 5
    assert sorted(state['prompts']) == sorted(r['question'] for r in inputs)
    assert len((tmp_path / 'answers_eval732.jsonl').read_text().splitlines()) == 10
    assert runner.generate(args, spec, inputs) == 0
    assert len(state['prompts']) == 12


def test_stop_marker_prevents_even_first_request(tmp_path, monkeypatch):
    args, spec, inputs = setup_run(tmp_path, monkeypatch)
    (tmp_path / 'STOP').touch()
    monkeypatch.setattr(runner, 'invoke', lambda *a: pytest.fail('Must not launch'))
    with pytest.raises(RuntimeError, match='STOP marker'):
        runner.generate(args, spec, inputs)
