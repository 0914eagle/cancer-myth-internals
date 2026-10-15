import json
import pytest
from scripts.qwen38_transformers_runtime import split_response
from scripts.run_qwen38_transformers import prepare


def test_thinking_is_separate_from_final_and_required_only_in_on():
    assert split_response('work here</think>\n{"has_false_premise": false}', True) == (
        'work here', '{"has_false_premise": false}')
    assert split_response('An answer.', False) == ('', 'An answer.')
    for text, thinking in [('unfinished reasoning', True), ('</think>answer', True),
                           ('<think>hidden</think>answer', False), ('thinking</think>', True)]:
        with pytest.raises(ValueError):
            split_response(text, thinking)


def test_paired_plans_share_inventory_prompts_and_budget(tmp_path):
    off, rows, _ = prepare(tmp_path / 'off', 'thinking_off', 8)
    on, rows_on, _ = prepare(tmp_path / 'on', 'thinking_on', 8)
    assert rows == rows_on and len(rows) == 732
    for key in ('prompts', 'systems', 'sampling', 'max_new_tokens', 'source_sha256', 'stages'):
        assert off[key] == on[key]
    assert off['thinking_enabled'] is False and on['thinking_enabled'] is True
    assert off['engine'] == on['engine'] == 'transformers'
    assert on['reasoning_effort'] == 'medium'
    inputs = json.loads((tmp_path / 'off/generation_inputs.json').read_text())
    assert all(set(row) == {'id', 'question'} for row in inputs)
