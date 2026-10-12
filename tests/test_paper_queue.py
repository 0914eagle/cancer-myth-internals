import json
from types import SimpleNamespace
from scripts.paper_generation_schedule import jobs,execute
from scripts.judge_paper_queue import pending,score_one
from src.paper_baselines import write_json


def test_all_fixed_questions_before_qa_or_gepa():
    splits=[{'name':'f0','test':['a']},{'name':'f1','test':['b']},{'name':'crepe','test':['c']}]
    plan=list(jobs(['plain','balanced','tfidf_gate','gepa'],splits))
    assert [(m,s['name'],qa) for m,s,qa in plan[:6]]==[
        ('plain','f0',False),('plain','f1',False),('plain','crepe',False),
        ('balanced','f0',False),('balanced','f1',False),('balanced','crepe',False)]
    assert all(not qa for _,_,qa in plan[:9])
    assert all(m=='gepa' for m,_,_ in plan[18:])


def test_generation_does_not_call_judge(tmp_path):
    rows=[{'id':'a'},{'id':'b'}];calls=[]
    class Fake:
        args=SimpleNamespace(generate_only=False)
        out=tmp_path
        identity={}
        reg={'transport_common':'common','prompts':{}}
        methods=['plain','balanced']
        lookup={r['id']:r for r in rows}
        bundle={'splits':[{'name':'f0','dataset':'cancer_myth','test':['a']},
                          {'name':'f1','dataset':'cancer_myth','test':['b']}],
                'qa':[{'id':'qa'}]}
        task=None
        bridge=None
        def process(self,row,split,method,candidate,gate,qa):
            assert self.args.generate_only
            calls.append((method,row['id'],qa))
    execute(Fake())
    assert calls[:4]==[('plain','a',False),('plain','b',False),('balanced','a',False),('balanced','b',False)]
    assert json.loads((tmp_path/'generation_progress.json').read_text())['state']=='complete'


def test_queue_fairness_score_reuse_and_no_native_qa(tmp_path):
    for model in ['a','b']:
        for i in range(2):
            write_json(tmp_path/model/'table12'/f'{i}.json',{'id':str(i),'answer':'answer'})
        write_json(tmp_path/model/'table3'/'q.json',{'id':'qa','answer':'Answer: A'})
    items=list(pending(tmp_path,['a','b']))
    assert [m for m,_ in items]==['a','b','a','b']
    calls=[]
    class Client:
        external={'judge_model':{'model':'fixed'}}
        def judge(self,row,answer):calls.append(row['id']);return 4,'Rating: 4'
    path=items[0][1]
    assert score_one(Client(),path,{'0':{'id':'0'}})
    assert not score_one(Client(),path,{'0':{'id':'0'}})
    assert calls==['0']
    assert len(list(pending(tmp_path,['a','b'])))==3
