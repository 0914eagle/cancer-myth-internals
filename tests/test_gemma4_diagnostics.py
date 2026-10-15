from scripts.run_gemma4_diagnostics import prompt_for, parse
from scripts.summarize_gemma4_diagnostics import score_summary
import pytest


def test_gold_fields_are_not_in_ordinary_inference():
    row = {'id': 'x', 'question': 'QUESTION', 'set': 'fpq',
           'premise_text': 'GOLD_PREMISE', 'correction': 'GOLD_CORRECTION'}
    for stage in ('direct', 'plain', 'review', 'zero_shot_cot_one_step', 'fp_unconditional', 'extract'):
        task = prompt_for(stage, row, {}, {})
        assert 'GOLD_' not in task['prompt']
        assert 'set' not in task and 'correction' not in task


def test_invalid_decision_does_not_become_negative():
    with pytest.raises(ValueError):
        parse('direct', '{"has_false_premise": false, "false_probability": 0.9}', {})
    with pytest.raises(ValueError):
        parse('direct', 'I cannot decide.', {})


def test_format_recovery_keeps_explicit_verdict_but_rejects_multiple_objects():
    r = parse('direct', '- has_false_premise: true\n- false_probability: 0.9', {})
    assert r['has_false_premise'] and r['format_recovered']
    r = parse('direct', 'Explanation\n```json\n{"has_false_premise": false, "false_probability": 0}\n```', {})
    assert r['has_false_premise'] is False and r['format_recovered']
    with pytest.raises(ValueError):
        parse('direct', '{"has_false_premise": false, "false_probability": 0} '
              '{"has_false_premise": true, "false_probability": 1}', {})


def test_verification_missing_items_rejected_and_empty_extraction_negative():
    with pytest.raises(ValueError):
        parse('verify_extracted', '{"judgments": []}', {'premises': ['claim']})
    result = parse('verify_extracted', '{"judgments": []}', {'premises': []})
    assert result['has_false_premise'] is False


def test_dependency_failure_is_not_silently_replaced():
    row = {'id': 'x', 'question': 'QUESTION'}
    assert prompt_for('cot', row, {}, {}) is None
    assert prompt_for('verify_annotated', row, {}, {}) is None


def test_zero_and_missing_have_distinct_denominators():
    result = score_summary([0, 4, 5, None], 4)
    assert result['valid'] == 3 and result['missing'] == 1
    assert result['mean_0_to_5'] == 3
    assert result['mean_1_to_5'] == 4.5
    assert result['at_least_4_fraction_valid'] == 2 / 3


def test_judge_ledger_uses_gemma_identity_and_existing_rubric(tmp_path, monkeypatch):
    import json
    from scripts import judge_gemma4_answers as judge
    from scripts.run_gemma4_diagnostics import jsonl
    from src import well_eval
    rows = [{'id': 'fp', 'set': 'fpq', 'question': 'A unique evaluation question?',
             'premise_text': 'A reference claim.', 'correction': 'Its reference correction.'},
            {'id': 'nf', 'set': 'nfp', 'question': 'A different normal evaluation question?'}]
    jsonl(tmp_path / 'evaluation_questions.jsonl', rows)
    jsonl(tmp_path / 'answers/plain.jsonl',
          [{'id': r['id'], 'method': 'plain', 'response': 'An answer.'} for r in rows])
    def fake_call(prompt, model, out):
        answer = 'OK' if prompt == 'Reply with the single word OK.' else 'Rating: 4'
        return answer, 'claude-sonnet-5', {'mock': True}
    monkeypatch.setattr(judge, 'call', fake_call)
    judge.tr.STOP.clear()
    judge.judge_method(tmp_path, 'plain', 2)
    status = json.loads((tmp_path / 'well/plain/status.json').read_text())
    assert status['valid'] == 2 and status['state'] == 'complete'
    assert status['generator'] == 'google/gemma-4-12B-it'
    assert status['same_model_generator_and_judge'] is False
    plan, h = well_eval.load_plan(tmp_path / 'well/plain')
    _, scores = well_eval.read_ledger(tmp_path / 'well/plain', plan, h)
    assert list(scores.values()) == [4, 4]
