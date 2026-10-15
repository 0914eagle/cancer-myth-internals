"""End-to-end held-out runner test with concurrent fake transports, no model calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from scripts.evaluate_well_upstream_test import test_jobs as make_jobs, upstream_row


def test_round_robin_and_question_only_schema():
    rows={'a':[{'id':'1'}],'b':[{'id':'2'},{'id':'3'}]}
    prompts={ds:{'plain':'p','gepa':'g'} for ds in rows}
    jobs=make_jobs(rows,prompts)
    assert [(d,m,r['id']) for d,m,r in jobs[:4]]==[('a','plain','1'),('a','gepa','1'),('b','plain','2'),('b','gepa','2')]
    assert len(jobs)==6
    r=upstream_row({'id':'x','question':'q','label':0,'presuppositions':[],'reference_answer':'gold'})
    assert r['_gepa_question_type']=='tpq' and r['few_shot_data']==[]


def test_frozen_test_runner_parallel_end_to_end(tmp_path):
    root=Path(__file__).resolve().parents[1]
    data=tmp_path/'data';manifest={'files':{}}
    for ds in ('crepe','cancer_myth'):
        for split in ('train','dev','test'):
            rows=[{'id':f'{ds}_{split}_{y}','dataset':ds,'label':y,'split':split,
                   'question':f'{ds} {split} question {y}?','presuppositions':['gold'],
                   'reference_answer':'reference','corrections':[]} for y in (1,0)]
            p=data/ds/f'{split}.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('\n'.join(map(json.dumps,rows))+'\n')
            manifest['files'][f'{ds}/{split}.jsonl']=hashlib.sha256(p.read_bytes()).hexdigest()
    (data/'manifest.json').write_text(json.dumps(manifest))
    code=r'''
import sys, runpy
from src import fpqa_well_upstream as adapter
class Fake:
    def __init__(self,config): assert config['exact_well_messages']
    def __call__(self,messages):
        if messages[0]['content']=='You are a helpful assistant.':
            return 'Rating: 4', {}
        assert 'reference' not in messages[1]['content']
        return 'Mixed Case Answer', {}
adapter.CLIBackend=Fake
sys.argv=['scripts/evaluate_well_upstream_test.py','--optimization-root',sys.argv[1],
          '--out',sys.argv[2],'--crepe-data',sys.argv[3],'--cancer-data',sys.argv[3],'--workers','5']
runpy.run_path('scripts/evaluate_well_upstream_test.py',run_name='__main__')
'''
    out=tmp_path/'out'
    result=subprocess.run([sys.executable,'-c',code,
        str(root/'results/fpqa_prompting/well_upstream_cli_v3_20261003'),str(out),str(data)],
        cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'metrics.json').read_text())
    assert report['completed']==report['total']==8
    assert report['state']=='complete'
    for p in out.glob('*/*/*/record.json'):
        r=json.loads(p.read_text());feedback=r['feedback']
        assert feedback['answer']=='mixed case answer'
        expected={'crepe':{1:'CREPEFPQ',0:'CREPETPQ'},'cancer_myth':{1:'CancerMyth',0:'CancerMythNFP'}}
        assert feedback['score_template']==expected[r['dataset']][r['label']]+'ResponseLevelScoreTemplate'
        assert r['score']==4
    # Completed records must be reused on explicit recovery, without model calls.
    second=code.replace("'--workers','5']", "'--workers','5','--resume']").replace("assert config['exact_well_messages']", "raise AssertionError('Completed records must not construct model clients')")
    result=subprocess.run([sys.executable,'-c',second,
        str(root/'results/fpqa_prompting/well_upstream_cli_v3_20261003'),str(out),str(data)],
        cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
