"""Two AO queries on the exact same saved100 seed activation windows."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_gepa_io import PilotIO,save
from scripts.reader_feedback_controls import P,OUT,STANCE_QUERY
from src.paper_baselines import digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',default='2,3')
    args=parser.parse_args()
    directory=OUT/'ao_query_probe'
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'probe.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text()).get('state')=='complete':
        return
    io=PilotIO(P/'runtime.json')
    status=directory/'status.json'
    worker=None;worker_log=None
    # A private queue permits this fixed-activation probe to run alongside GEPA.
    # Never consume, stop, or rewrite the GEPA reader's shared queue.
    queue_root=directory/'worker'
    reader_root=queue_root/'readers/ao'
    try:
        ids=json.loads((P/'native/cohort_request_ids.json').read_text())
        assert len(ids)==100
        native=[json.loads((P/'native/results'/(i+'.json')).read_text()) for i in ids]
        native.sort(key=lambda n:(n['dataset']!='crepe',n['row_id']))
        assert all(n['reader_reuse']['ao'].get('bitwise_equal') for n in native)
        assert all(Path(n['ao_activation_path']).is_file() for n in native)
        lookup={r['id']:r for r in json.loads(Path(io.c['bundle']).read_text())['rows']}
        protocol=dict(questions=100,original_query=io.c['ao_prompt'],stance_query=STANCE_QUERY,
            layer_zero_based=31,hf_hidden_state_index=32,source_position='question end, last up to20 question tokens',
            source_activation_reuse='Exact same saved tensor path for both queries; no new source generation.',
            stance_instruction='Read evidence of source acceptance/doubt/rejection; no objective factual judgment or presumed error.',
            no_gepa_selection_use=True)
        path=directory/'protocol.json'
        if path.exists():assert json.loads(path.read_text())==protocol
        else:save(path,protocol)
        jobs={}
        for n in native:
            ident=digest(['ao_stance_probe_v1',n['id'],STANCE_QUERY])
            job=dict(id=ident,native_id=n['id'],activation_path=n['ao_activation_path'],query=STANCE_QUERY)
            request=reader_root/'requests'/(ident+'.json')
            if not request.exists():save(request,job)
            jobs[n['id']]=ident
        if any(not (reader_root/'results'/(i+'.json')).exists() for i in jobs.values()):
            config={**io.c,'artifact_root':str(queue_root)}
            save(directory/'worker_runtime.json',config)
            (reader_root/'STOP').unlink(missing_ok=True)
            save(directory/'execution.json',dict(gpus=args.gpus,queue_root=str(queue_root),
                priority='CREPE50 first, then Cancer-Myth50',source_generation=False,updated_utc=time.time()))
            worker_log=(directory/'worker.log').open('a')
            worker=subprocess.Popen(['/data3/heejae/uv/fpqa_ao_20261012/bin/python','-u',
                'scripts/reader_gepa_verbalizer_worker.py','--config',str(directory/'worker_runtime.json'),
                '--kind','ao','--gpus',args.gpus],cwd=ROOT,
                env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'},
                stdout=worker_log,stderr=worker_log)
        records=[];joined=[];started=time.monotonic()
        for n in native:
            ident=jobs[n['id']];result=reader_root/'results'/(ident+'.json')
            while not result.exists():
                if worker is not None and worker.poll() is not None:raise RuntimeError('AO query reader exited')
                if time.monotonic()-started>3600:raise TimeoutError('AO query probe exceeded3600seconds')
                time.sleep(1)
            stance=json.loads(result.read_text());assert stance['query']==STANCE_QUERY
            assert stance['source_native_id']==n['id']
            assert stance['source_activation_path']==n['ao_activation_path']
            neutral=io.reader('ao',n)
            row=lookup[n['row_id']]
            records.append(dict(row_id=n['row_id'],dataset=n['dataset'],label=row['label'],question=n['question'],
                actual_answer=n['answer'],source_native_id=n['id'],source_activation_path=n['ao_activation_path'],
                neutral=neutral,stance=stance))
            # Reuse the exact-quote-validated anonymous content auditor. The two
            # answers and activations are identical; only reader queries differ.
            joined.append(dict(id='ao_query/'+n['id'],dataset=n['dataset'],reader='ao',row_id=n['row_id'],
                label=row['label'],question=n['question'],candidate_key='query_control',starting_prompt=False,
                source_native_id=n['id'],seed_reader_explanation=neutral['explanation'],
                current_reader_explanation=stance['explanation'],seed_answer=n['answer'],current_answer=n['answer']))
            save(directory/'query_outputs_partial.json',records)
            save(status,dict(state='running',complete=len(records),expected=100,
                crepe_complete=sum(r['dataset']=='crepe' for r in records),updated_utc=time.time()))
        save(directory/'query_outputs.json',records)
        save(directory/'reader_trajectories.json',joined)
        save(status,dict(state='complete',complete=100,expected=100,updated_utc=time.time()))
    except Exception as e:
        save(status,dict(state='failed',error=repr(e),updated_utc=time.time()));raise
    finally:
        if worker is not None and worker.poll() is None:
            save(reader_root/'STOP',dict(reason='Owned AO query probe finished; normal shutdown'))
            worker.wait(timeout=120)
        if worker_log is not None:worker_log.close()


if __name__=='__main__':main()
