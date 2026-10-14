"""Run registered controls automatically after the original pilot completes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import P,OUT,prepare
from scripts.reader_gepa_io import save
from scripts.supervise_reader_gepa_pilot import retry_decision,provider_progress


def main():
    prepare();OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=OUT/'controller_status.json';native=None;driver=None;handles=[]
    def state(stage,**kwargs):save(status,dict(state='running',stage=stage,pid=os.getpid(),updated_utc=time.time(),**kwargs))
    def start(script,log,args=None):
        handle=(OUT/log).open('a');handles.append(handle)
        return subprocess.Popen(['/data3/heejae/uv/paper_baselines_b81ace2/bin/python','-u','scripts/'+script,*(args or [])],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=handle,start_new_session=True,
            env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'})
    def successful_provider_timestamp():
        extra=OUT/'ao_query_probe/audit_calls'
        return max(provider_progress(P),max((f.stat().st_mtime_ns for f in extra.glob('*.json')),default=0))
    def phase(script,log,result_path,args=None):
        nonlocal driver
        if result_path.exists() and json.loads(result_path.read_text()).get('state')=='complete':return
        failures=0;last_progress=successful_provider_timestamp()
        while True:
            state(script,source_pid=native.pid if native is not None and native.poll() is None else None)
            driver=start(script,log,args)
            while driver.poll() is None:
                if native is not None and native.poll() is not None:raise RuntimeError('Control source worker exited unexpectedly')
                time.sleep(5)
            if driver.returncode==0:return
            failure=json.loads(result_path.read_text()) if result_path.exists() else {}
            error=str(failure.get('error','Driver exited without a structured error; inspect '+log))
            progress=successful_provider_timestamp()
            retry,failures,kind=retry_decision(error,failures,progress>last_progress);last_progress=progress
            if not retry:raise RuntimeError(error)
            deadline=time.time()+600
            save(status,dict(state='waiting_provider',stage=script,error=error,failure_kind=kind,
                next_retry_utc=deadline,retry_interval_seconds=600,updated_utc=time.time()))
            while time.time()<deadline:time.sleep(5)
    try:
        while True:
            pipeline=json.loads((P/'pipeline_status.json').read_text())
            supervisor=json.loads((P/'supervisor_status.json').read_text())
            if pipeline.get('state')==supervisor.get('state')=='complete':break
            state('waiting_original_pilot',original_state=pipeline.get('state'),original_stage=pipeline.get('stage'),
                  detail='Original AO and separate heldout Haiku must finish first; no new source/reader process started.')
            time.sleep(10)
        subprocess.run(['python3','scripts/package_reader_experiments.py'],cwd=ROOT,check=True)
        result=OUT/'status.json'
        if not result.exists() or json.loads(result.read_text()).get('state')!='complete':
            (P/'native/STOP').unlink(missing_ok=True)
            native=start('reader_gepa_native_worker.py','native_worker.log',['--config',str(P/'runtime.json'),'--gpus','0,1'])
            phase('reader_feedback_controls.py','controls.log',result)
            save(P/'native/STOP',{'reason':'All registered control answer generation complete; normal shutdown'})
            while native.poll() is None:time.sleep(2)
            native=None
        phase('run_ao_stance_query_probe.py','ao_query_probe.log',OUT/'ao_query_probe/status.json')
        phase('audit_reader_gepa_trajectories.py','ao_query_audit.log',OUT/'ao_query_probe/trajectory_audit_status.json',
              ['--input-root',str(OUT/'ao_query_probe')])
        subprocess.run([sys.executable,'scripts/summarize_ao_query_probe.py'],cwd=ROOT,check=True)
        save(status,dict(state='complete',stage='all_controls_and_query_audit',updated_utc=time.time()))
        subprocess.run(['python3','scripts/package_reader_experiments.py'],cwd=ROOT,check=True)
    except Exception as e:
        save(status,dict(state='attention_required',error=repr(e),updated_utc=time.time(),cached_results_preserved=True))
        raise
    finally:
        if native is not None and native.poll() is None:
            save(P/'native/STOP',{'reason':'Control supervisor exiting; normal source shutdown'})
        for h in handles:h.close()


if __name__=='__main__':main()
