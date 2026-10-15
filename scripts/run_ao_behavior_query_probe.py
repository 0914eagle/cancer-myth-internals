"""CREPE50: compare semantic, stance, and prospective response descriptions."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import P,OUT
from scripts.reader_gepa_io import save
from src.paper_baselines import digest

DIRECTORY=OUT/'ao_behavior_probe_crepe_20261014'
BEHAVIOR_QUERY=(
    "Describe how the model appears likely to handle the assumptions in the user's question when answering. "
    "Identify any relevant assumption and the response it seems to be preparing. "
    "Describe its likely behavior, not what it should do. "
    "Do not judge the premise using your own knowledge. "
    "If unclear from the activations, say so."
)


def export(records):
    target=ROOT.parents[1]/'results/ao_behavior_comparison_crepe_20261014'
    target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in records:
        judgment=P/'judgments'/(r['source_native_id']+'.json')
        judge=json.loads(judgment.read_text()) if judgment.exists() else {}
        rows.append(dict(question_id=r['row_id'],question_type='FPQ' if r['label'] else 'NFP',question=r['question'],
            neutral_text=r['neutral']['explanation'],stance_text=r['stance']['explanation'],
            behavior_text=r['behavior']['explanation'],actual_answer=r['actual_answer'],
            well_score=judge.get('score'),source_native_id=r['source_native_id'],
            source_layer_zero_based=31,token_window_size=r['token_window_size'],
            activation_sha256=r['activation_sha256'],source_text_given_to_ao=False))
    (target/'crepe50.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    with (target/'crepe50.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(target/'protocol.json',json.loads((DIRECTORY/'protocol.json').read_text()))
    code=target/'code';code.mkdir(exist_ok=True)
    (code/Path(__file__).name).write_text(Path(__file__).read_text())
    return target


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpus',default='4,5');args=parser.parse_args()
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    lock=(DIRECTORY/'probe.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=DIRECTORY/'status.json';worker=None;log=None
    queue=DIRECTORY/'worker';reader=queue/'readers/ao'
    try:
        previous=json.loads((OUT/'ao_query_probe/query_outputs.json').read_text())
        previous=sorted([r for r in previous if r['dataset']=='crepe'],key=lambda r:r['row_id'])
        assert len(previous)==50 and len({r['row_id'] for r in previous})==50
        assert sum(r['label']==1 for r in previous)==25
        config=json.loads((P/'runtime.json').read_text())
        protocol=dict(questions=50,FPQ=25,NFP=25,dataset='crepe',source_prompt='Existing fixed Plain seed',
            neutral_query=config['ao_prompt'],stance_query=json.loads((OUT/'ao_query_probe/protocol.json').read_text())['stance_query'],
            behavior_query=BEHAVIOR_QUERY,layer_zero_based=31,hf_hidden_state_index=32,
            position='Question end, last up to20 question tokens, first prefill before answer tokens',
            activation_control='Exact same saved activation tensor for all three query conditions',
            source_text_given_to_ao=False,new_source_generation=False,generation=dict(do_sample=False,max_new_tokens=256,batch_size=8),
            model='google/gemma-3-27b-it',no_gepa_selection_use=True,
            scope='Fixed previously examined train50; exploratory query change, not heldout confirmation',
            outcome='Specific assumption, described next-answer behavior, agreement with actual answer, and useful information absent from answer')
        path=DIRECTORY/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:save(path,protocol)
        jobs={}
        for r in previous:
            assert r['stance']['source_activation_path']==r['source_activation_path']
            assert r['neutral']['bytewise_verified'] is True
            ident=digest(['ao_behavior_probe_v1',r['source_native_id'],BEHAVIOR_QUERY]);jobs[r['row_id']]=ident
            save(reader/'requests'/(ident+'.json'),dict(id=ident,native_id=r['source_native_id'],
                activation_path=r['source_activation_path'],query=BEHAVIOR_QUERY))
        if any(not (reader/'results'/(i+'.json')).exists() for i in jobs.values()):
            save(DIRECTORY/'worker_runtime.json',{**config,'artifact_root':str(queue)})
            (reader/'STOP').unlink(missing_ok=True)
            log=(DIRECTORY/'worker.log').open('a')
            worker=subprocess.Popen(['/data3/heejae/uv/fpqa_ao_20261012/bin/python','-u',
                'scripts/reader_gepa_verbalizer_worker.py','--config',str(DIRECTORY/'worker_runtime.json'),
                '--kind','ao','--gpus',args.gpus],cwd=ROOT,
                env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'},stdout=log,stderr=log)
        records=[];trajectories=[];started=time.monotonic()
        save(status,dict(state='running',complete=0,expected=50,gpus=args.gpus,updated_utc=time.time()))
        for r in previous:
            result=reader/'results'/(jobs[r['row_id']]+'.json')
            while not result.exists():
                if worker is not None and worker.poll() is not None:raise RuntimeError('Owned AO behavior worker exited')
                if time.monotonic()-started>1800:raise TimeoutError('AO behavior probe exceeded1800seconds')
                time.sleep(1)
            behavior=json.loads(result.read_text())
            assert behavior['query']==BEHAVIOR_QUERY and behavior['source_text_provided'] is False
            assert behavior['source_native_id']==r['source_native_id']
            assert behavior['source_activation_path']==r['source_activation_path']
            n=json.loads((P/'native/results'/(r['source_native_id']+'.json')).read_text())
            record={**r,'behavior':behavior,'token_window_size':len(n['ao_source_token_ids']),
                'activation_sha256':hashlib.sha256(Path(r['source_activation_path']).read_bytes()).hexdigest()}
            records.append(record)
            trajectories.append(dict(id='ao_behavior_query/'+r['source_native_id'],dataset='crepe',reader='ao',
                row_id=r['row_id'],label=r['label'],question=r['question'],candidate_key='behavior_query_control',
                starting_prompt=False,source_native_id=r['source_native_id'],
                seed_reader_explanation=r['stance']['explanation'],current_reader_explanation=behavior['explanation'],
                seed_answer=r['actual_answer'],current_answer=r['actual_answer']))
            save(DIRECTORY/'query_outputs_partial.json',records)
            save(status,dict(state='running',complete=len(records),expected=50,gpus=args.gpus,updated_utc=time.time()))
        save(DIRECTORY/'query_outputs.json',records);save(DIRECTORY/'reader_trajectories.json',trajectories)
        target=export(records)
        save(status,dict(state='complete',complete=50,expected=50,export=str(target),updated_utc=time.time()))
        print(json.dumps(dict(state='complete',export=str(target))),flush=True)
    except Exception as error:
        save(status,dict(state='failed',error=repr(error),updated_utc=time.time()));raise
    finally:
        if worker is not None and worker.poll() is None:
            save(reader/'STOP',dict(reason='Owned behavior-query probe finished; normal shutdown'))
            worker.wait(timeout=120)
        if log is not None:log.close()


if __name__=='__main__':main()
