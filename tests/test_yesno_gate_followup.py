import json
import math
import pytest
from scripts.run_yesno_gate_followup import decision, verbalizers, render
from scripts.wait_yesno_gate_followup import ready, METHODS


def test_logits_and_tie_rule():
    assert decision(3, 2)['has_false_premise'] is True
    assert decision(-3, -2)['has_false_premise'] is False
    assert decision(0, 0)['has_false_premise'] is False
    assert decision(1000, -1000)['score'] == 1
    with pytest.raises(ValueError):
        decision(math.nan, 0)


def test_multitoken_verbalizer_rejected():
    class Tok:
        def __call__(self, text, **kw):
            return {'input_ids': [1, 2] if text == 'Yes' else [3]}
    with pytest.raises(ValueError):
        verbalizers(Tok())


def test_user_only_and_no_thinking():
    class Tok:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{'role': 'user', 'content': 'test'}]
            assert kwargs['enable_thinking'] is False
            assert kwargs['add_generation_prompt'] is True
            return 'rendered'
    assert render(Tok(), 'test') == 'rendered'


def test_queue_requires_entire_current_suite(tmp_path):
    def write(p, state):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'state': state}))
    write(tmp_path / 'status.json', 'generation_finished')
    for mode in ('thinking_off', 'thinking_on'):
        for method in METHODS:
            assert not ready(tmp_path)
            write(tmp_path / mode / 'well' / method / 'status.json', 'complete')
    assert ready(tmp_path)
    write(tmp_path / 'status.json', 'failed')
    assert not ready(tmp_path)
