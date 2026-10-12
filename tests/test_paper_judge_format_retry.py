from types import SimpleNamespace
import pytest
from scripts import judge_paper_queue as q


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
