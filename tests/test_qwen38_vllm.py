import io
import json
import pytest

from scripts import run_qwen38_vllm as run


def test_modes_share_sampling_but_separate_reasoning_and_final(monkeypatch):
    requests = []
    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        requests.append(body)
        thinking = body['chat_template_kwargs']['enable_thinking']
        return io.StringIO(json.dumps({'model': run.MODEL, 'choices': [{
            'finish_reason': 'stop', 'message': {'content': 'FINAL ANSWER',
            'reasoning': 'PRIVATE REASONING' if thinking else None}}]}))
    monkeypatch.setattr(run.urllib.request, 'urlopen', fake_urlopen)
    task = {'id': 'q', 'stage': 'plain', 'system': 'system', 'prompt': 'question'}
    off = run.completion('http://localhost/v1', task, False, 8192)
    on = run.completion('http://localhost/v1', task, True, 8192)
    assert off['response'] == on['response'] == 'FINAL ANSWER'
    assert not off['reasoning'] and on['reasoning'] == 'PRIVATE REASONING'
    for key in ('messages', 'seed', 'max_tokens', *run.SAMPLING):
        assert requests[0][key] == requests[1][key]


def test_reasoning_leak_in_final_fails(monkeypatch):
    monkeypatch.setattr(run.urllib.request, 'urlopen', lambda *a, **kw: io.StringIO(json.dumps({
        'model': run.MODEL, 'choices': [{'finish_reason': 'stop',
        'message': {'content': '<think>analysis</think>answer'}}]})))
    with pytest.raises(ValueError, match='parser'):
        run.completion('http://localhost/v1', {'id': 'q', 'stage': 'plain',
                       'system': 's', 'prompt': 'p'}, True, 8192)


def test_truncation_not_counted_as_completed_answer(tmp_path, monkeypatch):
    def truncated(*args):
        return {'truncated': True, 'finish_reason': 'length', 'response': 'unfinished'}
    monkeypatch.setattr(run, 'completion', truncated)
    result = run.invoke(tmp_path, {'id': 'q', 'stage': 'plain'}, True, 'unused', 'hash')
    assert result['status'] == 'invalid'
    assert len(result['attempts']) == 2
    assert 'result' not in result


def test_judge_records_qwen_identity_and_only_final(tmp_path, monkeypatch):
    from scripts import judge_qwen38_answers as judge
    from scripts.run_gemma4_diagnostics import jsonl
    (tmp_path / 'plan.json').write_text(json.dumps({'model': run.MODEL}))
    rows = [{'id': 'nf', 'set': 'nfp', 'question': 'A normal question?'}]
    jsonl(tmp_path / 'evaluation_questions.jsonl', rows)
    jsonl(tmp_path / 'answers/plain.jsonl', [{'id': 'nf', 'method': 'plain', 'response': 'FINAL'}])
    prompts = []
    def fake_call(prompt, model, out):
        prompts.append(prompt)
        return ('OK' if prompt == 'Reply with the single word OK.' else 'Rating: 4',
                'claude-sonnet-5', {'mock': True})
    monkeypatch.setattr(judge, 'call', fake_call)
    judge.tr.STOP.clear()
    judge.judge_method(tmp_path, 'plain', 1)
    state = json.loads((tmp_path / 'well/plain/status.json').read_text())
    assert state['generator'] == run.MODEL and state['valid'] == 1
    assert 'FINAL' in prompts[-1]
