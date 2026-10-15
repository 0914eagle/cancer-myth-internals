import json
import threading
import time

import pytest

from scripts import run_frontier_corrections as runner


def fixture(tmp_path):
    folder = tmp_path / 'balanced'
    (folder / 'isolated_cwd').mkdir(parents=True)
    inputs = [{'id': f'q{i}', 'question': f'Original question {i}?'} for i in range(12)]
    runner.tr.write(folder / 'evaluation_questions.json', inputs)
    spec = {'method': 'balanced', 'model': runner.MODEL, 'command': ['fake'],
            'workers': 5, 'user_template': runner.PROMPTS['balanced']}
    runner.tr.STOP.clear()
    return folder, inputs, spec


def test_prompt_isolation_original_question_and_resume(tmp_path, monkeypatch):
    folder, inputs, spec = fixture(tmp_path)
    state = {'active': 0, 'peak': 0, 'prompts': []}
    lock = threading.Lock()

    def invoke(cmd, prompt, cwd, timeout):
        with lock:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            state['prompts'].append(prompt)
        time.sleep(.03)
        with lock:
            state['active'] -= 1
        return json.dumps({'type': 'result', 'subtype': 'success', 'result': 'An answer.',
                           'stop_reason': 'end_turn', 'num_turns': 1,
                           'modelUsage': {runner.MODEL: {}}})

    monkeypatch.setattr(runner.tr, 'invoke', invoke)
    first = runner.generate(tmp_path, folder, spec, inputs, {r['id'] for r in inputs[:5]})
    assert first['counts']['complete'] == 5
    final = runner.generate(tmp_path, folder, spec, inputs)
    assert final['state'] == 'complete' and state['peak'] == 5
    runner.generate(tmp_path, folder, spec, inputs)
    assert len(state['prompts']) == 12
    assert set(state['prompts']) == {spec['user_template'].format(question=r['question']) for r in inputs}
    records = runner.tr.read_rows(folder / 'answers_eval732.jsonl')
    assert {r['id']: r['question'] for r in records} == {r['id']: r['question'] for r in inputs}
    assert {r['method'] for r in records} == {'balanced'}
    # Generation output remains directly consumable by the pinned Well evaluator.
    questions = [{'id': r['id'], 'question': r['question'], 'set': 'fpq',
                  'premise_text': 'Hidden reference.', 'correction': 'Hidden correction.'} for r in inputs]
    qpath = tmp_path / 'questions.jsonl'
    qpath.write_text(''.join(json.dumps(r) + '\n' for r in questions))
    plan = runner.we.prepare(qpath, [folder / 'answers_eval732.jsonl'], tmp_path / 'judge',
                             backend='claude', model=runner.MODEL)
    assert plan['answer_rows'] == 12


def test_pipeline_stop_blocks_calls(tmp_path, monkeypatch):
    folder, inputs, spec = fixture(tmp_path)
    (tmp_path / 'STOP').touch()
    monkeypatch.setattr(runner.tr, 'invoke', lambda *a: pytest.fail('Unexpected model call'))
    with pytest.raises(ValueError, match='STOP'):
        runner.generate(tmp_path, folder, spec, inputs)


def test_unconditional_exact_existing_prompt():
    from src.baseline_generation import FP_ANSWER
    assert runner.PROMPTS['fp_unconditional'] == FP_ANSWER
