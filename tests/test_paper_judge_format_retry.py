from types import SimpleNamespace
import pytest
from scripts import judge_paper_queue as q


def test_worker_control_changes_without_restart(tmp_path):
    import json
    path=tmp_path/'workers.json'
    assert q.worker_count(path,16)==16
    path.write_text(json.dumps({'workers':32}))
    assert q.worker_count(path,16)==32
    path.write_text(json.dumps({'workers':0}))
    with pytest.raises(ValueError):q.worker_count(path,16)


def test_provider_refusal_is_not_quota_or_generic_transport_error():
    assert q.provider_refusal(RuntimeError("API Error: Haiku 5.5 can't help with this. [general_harms]"))
    assert not q.provider_refusal(RuntimeError('HTTP 429 rate limit'))
    assert not q.provider_refusal(RuntimeError('request timeout'))


def test_malformed_output_retried_without_changing_prompt(tmp_path,monkeypatch):
    replies=iter(['Rating: 2\nRating: 4','Rating: 3']);calls=[]
    def call(self,role,messages):
        calls.append(messages);return next(replies)
    monkeypatch.setattr(q.Worker,'external_call',call)
    client=SimpleNamespace(out=tmp_path,external={'judge_model':{'model':'haiku'}})
    messages=[{'role':'user','content':'same rubric'}]
    assert q.validated_judge_call(client,'judge',messages)=='Rating: 3'
    assert calls==[messages,messages]
    assert len(list((tmp_path/'invalid_judge_calls').glob('*.json')))==1


def test_repeated_malformed_output_has_no_fabricated_score(tmp_path,monkeypatch):
    monkeypatch.setattr(q.Worker,'external_call',lambda *args:'No rating')
    client=SimpleNamespace(out=tmp_path,external={'judge_model':{'model':'haiku'}})
    with pytest.raises(q.JudgeFormatError):q.validated_judge_call(client,'judge',[])
    assert len(list((tmp_path/'invalid_judge_calls').glob('*.json')))==3
