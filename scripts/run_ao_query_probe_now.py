"""Run the registered query-only comparison independently of GEPA controls."""
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
from scripts.reader_feedback_controls import OUT
from scripts.reader_gepa_io import save
from scripts.supervise_reader_gepa_pilot import retry_decision


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpus',default='4,5')
    args=parser.parse_args()
    directory=OUT/'ao_query_probe';directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'immediate_controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=directory/'immediate_controller_status.json'
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HOME':'/data3/heejae/hf_cache'}
    steps=[('run_ao_stance_query_probe.py',['--gpus',args.gpus],'status.json'),
           ('audit_reader_gepa_trajectories.py',['--input-root',str(directory)],'trajectory_audit_status.json'),
           ('summarize_ao_query_probe.py',[],None)]
    try:
        for script,arguments,result_name in steps:
            failures=0;progress=0
            while True:
                save(status,dict(state='running',stage=script,pid=os.getpid(),updated_utc=time.time()))
                result=subprocess.run([sys.executable,'-u','scripts/'+script,*arguments],cwd=ROOT,env=env)
                if result.returncode==0:break
                detail=directory/result_name if result_name else None
                error=json.loads(detail.read_text()).get('error','') if detail and detail.exists() else 'Unstructured failure in '+script
                current=max((p.stat().st_mtime_ns for p in (directory/'audit_calls').glob('*.json')),default=0)
                retry,failures,kind=retry_decision(error,failures,current>progress);progress=current
                if not retry:raise RuntimeError(error)
                deadline=time.time()+600
                save(status,dict(state='waiting_provider',stage=script,error=error,failure_kind=kind,
                    next_retry_utc=deadline,updated_utc=time.time()))
                while time.time()<deadline:time.sleep(5)
        save(status,dict(state='complete',updated_utc=time.time()))
    except Exception as error:
        save(status,dict(state='attention_required',error=repr(error),updated_utc=time.time()))
        raise


if __name__=='__main__':main()
