import json
from types import SimpleNamespace
from scripts.prepare_paper_v4 import sample
from scripts.paper_v4_pipeline import EvaluatedPipeline
from scripts.judge_paper_queue import pending,score_one
from src.paper_baselines import write_json


def test_sampling_balanced_disjoint_reproducible():
    lookup={str(i):{'label':i%2} for i in range(2200)}
    ids=list(lookup)
    picked=sample(ids,lookup,500,42)
    assert len(picked)==len(set(picked))==1000
    assert sum(lookup[i]['label'] for i in picked)==500
    assert sample(list(reversed(ids)),lookup,500,42)==picked
    assert picked!=sample(ids,lookup,500,43)


def test_detection_saved_before_fresh_answer_and_reused(tmp_path):
    calls=[]
    def call(messages):
        calls.append(messages)
        if messages[0]['content']=='detect':return 'No'
        return 'new response'
    worker=SimpleNamespace(out=tmp_path)
    registry={'prompts':{'direct_detector':'detect','plain':'plain','correct_route':'correct'}}
    p=EvaluatedPipeline(registry,call,None,None,worker)
    p.context.value=({'id':'x','dataset':'crepe','label':0},{'name':'official'},False)
    first=p.run('direct_gate','Q','crepe')
    assert first['verdict']==0 and first['answer']=='new response'
    assert 'Verdict: No' in calls[1][1]['content']
    assert list((tmp_path/'table1').rglob('*.json'))
    p.run('direct_gate','Q','crepe')
    assert len(calls)==3 # stored detection, another real response call
    assert calls[2][0]['content']=='plain'


def test_haiku_regrades_sonnet_without_overwriting_source(tmp_path):
    out=tmp_path/'answers';grades=tmp_path/'haiku'
    path=out/'qwen25/table12/crepe/official/plain/a.json'
    original={'id':'x','question':'Q','answer':'A','rating':5,'judge_config':{'model':'sonnet'}}
    write_json(path,original)
    lookup={'x':{'id':'x','question':'Q','label':0}}
    config={'model':'haiku'}
    client=SimpleNamespace(external={'judge_model':config},judge=lambda row,answer:(2,'Rating: 2'))
    assert len(list(pending(out,['qwen25'],grades,config,lookup)))==1
    dest=grades/path.relative_to(out)
    assert score_one(client,path,lookup,dest)
    assert json.loads(path.read_text())==original
    assert json.loads(dest.read_text())['rating']==2
    assert not list(pending(out,['qwen25'],grades,config,lookup))
    original['answer']='Changed';write_json(path,original)
    assert len(list(pending(out,['qwen25'],grades,config,lookup)))==1


def test_detection_first_does_not_generate_response_and_later_reuses(tmp_path):
    calls=[]
    def call(messages):
        calls.append(messages)
        return 'Yes' if messages[0]['content']=='detect' else 'fresh answer'
    worker=SimpleNamespace(out=tmp_path)
    p=EvaluatedPipeline({'prompts':{'direct_detector':'detect','plain':'plain','correct_route':'correct'}},call,None,None,worker)
    p.context.value=({'id':'x','dataset':'crepe','label':1},{'name':'official'},False)
    p.context.detection_only=True
    r=p.run('direct_gate','Q','crepe')
    assert r['verdict']==1 and 'answer' not in r and len(calls)==1
    p.context.detection_only=False
    r=p.run('direct_gate','Q','crepe')
    assert r['answer']=='fresh answer' and len(calls)==2
    assert 'Verdict: Yes' in calls[1][1]['content']


def test_parallel_judge_keeps_success_when_peer_is_rate_limited(tmp_path):
    import threading
    from scripts.judge_paper_queue import score_group
    barrier=threading.Barrier(2)
    out=tmp_path/'answers';grades=tmp_path/'grades';lookup={}
    jobs=[]
    for i in range(2):
        key=str(i);path=out/'m/table12'/f'{i}.json'
        row={'id':key,'question':'Q','label':i};lookup[key]=row
        write_json(path,{**row,'answer':'A'});jobs.append(('m',path))
    def judge(row,answer):
        barrier.wait(timeout=5)
        if row['id']=='0':raise RuntimeError('rate_limit_error')
        return 5,'Rating: 5'
    client=SimpleNamespace(external={'judge_model':{'model':'haiku'}},judge=judge)
    results=list(score_group(jobs,{'m':client},lookup,out,grades,2))
    assert sum(r[2] for r in results)==1
    assert sum(r[3] is not None for r in results)==1
    assert (grades/'m/table12/1.json').exists()
    assert not (grades/'m/table12/0.json').exists()
