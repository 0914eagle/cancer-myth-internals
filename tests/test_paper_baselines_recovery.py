import json
from types import SimpleNamespace
import pytest
from scripts.run_paper_baselines_recovery import validated_optimization_call,recoverable_item_error
from src.paper_baselines import digest,sha,write_json
from scripts.run_paper_baselines_recovery import ROOT


def test_bad_cached_optimization_grade_is_archived_then_retried(tmp_path):
    cfg={'model':'judge'};messages=[{'role':'user','content':'unchanged rubric'}]
    w=SimpleNamespace(args=SimpleNamespace(out=tmp_path),external={'judge_model':cfg})
    key=digest({'config':cfg,'messages':messages,'role':'judge','backend_sha':sha(ROOT/'src/fpqa_cli_backend.py')})
    p=tmp_path/'shared/optimization_calls'/(key+'.json')
    write_json(p,{'text':'Rating: 4\nRating: 4'})
    calls=[]
    def original(worker,role,msg):
        calls.append(msg)
        if p.exists():return json.loads(p.read_text())['text']
        write_json(p,{'text':'Rating: 4'});return 'Rating: 4'
    assert validated_optimization_call(w,'judge',messages,original)=='Rating: 4'
    assert calls==[messages,messages]
    assert len(list((tmp_path/'shared/invalid_optimization_calls').glob('*.json')))==1
    assert json.loads(p.read_text())['text']=='Rating: 4'


def test_failed_judge_retries_do_not_invent_score(tmp_path):
    w=SimpleNamespace(args=SimpleNamespace(out=tmp_path),external={'judge_model':{}})
    with pytest.raises(ValueError,match='Three malformed'):
        validated_optimization_call(w,'judge',[],lambda *a:'no numeric rating')
    assert len(list((tmp_path/'shared/invalid_optimization_calls').glob('*.json')))==3


def test_only_empty_output_added_to_item_failure_policy():
    old=lambda error:None
    assert recoverable_item_error(ValueError('Empty generation'),old)=='empty_generation'
    assert recoverable_item_error(RuntimeError('CUDA out of memory'),old) is None
    assert recoverable_item_error(ValueError('Invalid API key'),old) is None


def test_empty_item_failure_is_persisted_and_next_item_continues(tmp_path,monkeypatch):
    from scripts import run_paper_baselines_resilient as r
    old=r.item_error;monkeypatch.setattr(r,'item_error',lambda e:recoverable_item_error(e,old))
    worker=SimpleNamespace(out=tmp_path,args=SimpleNamespace(command='run'))
    calls=[]
    def process(w,row,*args):
        calls.append(row['id'])
        if row['id']=='empty':raise ValueError('Empty generation')
    for id in ['empty','empty','good']:
        r.process_with_failures(worker,{'id':id,'dataset':'cancer_myth','label':1},
                                {'dataset':'cancer_myth','name':'fold0'},'prewome',None,None,False,process)
    assert calls==['empty','good']
    saved=json.loads(next((tmp_path/'failures').rglob('*.json')).read_text())
    assert saved['failure_kind']=='empty_generation'
    assert 'rating' not in saved
