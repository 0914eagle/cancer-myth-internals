"""Own the isolated GPU workers and resume provider failures from saved state."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.gepa_ao_feedback_ablation import OUT,prepare,NATIVE_GPUS,READER_GPUS
from scripts.reader_gepa_io import save
from scripts.supervise_reader_gepa_pilot import retry_decision,provider_progress


def main():
    prepare()
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HOME':'/data3/heejae/hf_cache'}
    status=OUT/'supervisor_status.json';handles=[];native=None
    def start(script,args,log):
        handle=(OUT/log).open('a');handles.append(handle)
        return subprocess.Popen([sys.executable,'-u','scripts/'+script,*args],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=handle,stderr=handle,start_new_session=True)
    try:
        state_path=OUT/'status.json'
        if state_path.exists() and json.loads(state_path.read_text()).get('state')=='complete':
            save(status,dict(state='complete',updated_utc=time.time()));return
        (OUT/'native/STOP').unlink(missing_ok=True)
        native=start('reader_gepa_native_worker.py',['--config',str(OUT/'runtime.json'),'--gpus',NATIVE_GPUS],'native_worker.log')
        failures=0;progress=provider_progress(OUT)
        while True:
            driver=start('gepa_ao_feedback_ablation.py',[],'pipeline.log')
            save(status,dict(state='running',native_pid=native.pid,driver_pid=driver.pid,native_gpus=NATIVE_GPUS,reader_gpus=READER_GPUS,updated_utc=time.time()))
            while driver.poll() is None:
                if native.poll() is not None:raise RuntimeError('Isolated source worker exited; inspect native_worker.log')
                time.sleep(3)
            if driver.returncode==0:
                save(status,dict(state='complete',updated_utc=time.time()));return
            result=json.loads(state_path.read_text()) if state_path.exists() else {}
            error=str(result.get('error','Driver failed without structured status'))
            current=provider_progress(OUT)
            retry,failures,kind=retry_decision(error,failures,current>progress);progress=current
            if not retry:raise RuntimeError(error)
            deadline=time.time()+600
            save(status,dict(state='waiting_provider',error=error,failure_kind=kind,next_retry_utc=deadline,
                native_pid=native.pid,updated_utc=time.time()))
            while time.time()<deadline:
                if native.poll() is not None:raise RuntimeError('Isolated source worker exited during retry wait')
                time.sleep(5)
    except Exception as error:
        save(status,dict(state='attention_required',error=repr(error),updated_utc=time.time()));raise
    finally:
        if native is not None and native.poll() is None:
            save(OUT/'native/STOP',dict(reason='Isolated feedback-ablation supervisor finished; normal shutdown'))
            native.wait(timeout=120)
        for handle in handles:handle.close()


if __name__=='__main__':main()
