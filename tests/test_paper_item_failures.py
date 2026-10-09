import json
from types import SimpleNamespace
import pytest
from scripts.run_paper_baselines_resilient import process_with_failures, item_failure_path
from scripts.summarize_paper_baselines import aggregate
from src.paper_baselines import digest, write_json


def test_skip_persists_and_next_question_runs(tmp_path):
    w=SimpleNamespace(out=tmp_path,args=SimpleNamespace(command='run'))
    split={'dataset':'cancer_myth','name':'fold0'}
    calls=[]
    def process(w,row,*args):
        calls.append(row['id'])
        if row['id']=='bad':
            raise RuntimeError('Generation truncated; retained task has no completed answer')
    for i in ['bad','bad','good']:
        row={'id':i,'dataset':'mmlu'}
        process_with_failures(w,row,split,'plain',None,None,True,process)
    assert calls==['bad','good']
    row={'id':'bad','dataset':'mmlu'}
    p=item_failure_path(w,row,split,'plain',True)
    assert json.loads(p.read_text())['failure_kind']=='output_length_limit'
    assert p==item_failure_path(w,row,{'dataset':'crepe','name':'official'},'plain',True)


@pytest.mark.parametrize('message',['CUDA out of memory','STOP requested','Invalid API key'])
def test_fatal_errors_not_swallowed(tmp_path,message):
    w=SimpleNamespace(out=tmp_path,args=SimpleNamespace(command='run'))
    def process(*args): raise RuntimeError(message)
    with pytest.raises(RuntimeError,match=message):
        process_with_failures(w,{'id':'x','dataset':'mmlu'},
                             {'dataset':'cancer_myth','name':'fold0'},'plain',None,None,True,process)
    assert not list(tmp_path.rglob('*.json'))


def test_aggregate_keeps_failure_denominators(tmp_path):
    b={'content_sha256':'hash','rows':[{'id':'p','label':1},{'id':'n','label':0}],
       'splits':[{'dataset':'cancer_myth','name':'fold0','test':['p','n']}],
       'qa':[{'id':'q1','dataset':'mmlu'},{'id':'q2','dataset':'mmlu'}],'audit':{}}
    c={'models':{'m':{}},'methods':['plain'],'qa_datasets':['mmlu'],'protocol_id':'test'}
    write_json(tmp_path/'m/table12/cancer_myth/fold0/plain/p.json',{'id':'p','label':1,'rating':5})
    write_json(tmp_path/'m/failures/table12/cancer_myth/fold0/plain/n.json',{'id':'n','label':0})
    f=digest(['mmlu','q1'])+'.json'
    write_json(tmp_path/'m/table3/shared/fixed/plain'/f,{'id':'q1','correct':True})
    f=digest(['mmlu','q2'])+'.json'
    write_json(tmp_path/'m/failures/table3/shared/fixed/plain'/f,{'id':'q2','state':'generation_failed'})
    tables=aggregate(tmp_path,b,c)
    t2=next(r for r in tables[2] if r['dataset']=='cancer_myth')
    assert t2['state']=='complete_with_errors'
    assert t2['NFP_Well_ge4']==0 and t2['NFP_scored']==0
    t3=next(r for r in tables[3] if r['source']=='cancer_myth')
    assert t3['accuracy']==50 and t3['failed_system_questions']==1
    assert t3['state']=='complete_with_errors'
