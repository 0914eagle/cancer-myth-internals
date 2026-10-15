"""Resume the preserved AO-only suite on GPU0-3 after semantic controls finish."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_gepa_io import save

BASE=ROOT/'results/nla_ao_parallel_20261012'
FOLLOWUP=BASE/'gepa_ao_behavior_feedback_crepe_luna_20261014'


def ready():
    for dataset in ['cancer_myth','crepe']:
        directory=BASE/f'gepa_ao_semantic_feedback_{dataset}_luna_20261014'
        for filename in ['status.json','supervisor_status.json']:
            path=directory/filename
            if not path.exists() or json.loads(path.read_text()).get('state')!='complete':return False
        path=directory/'native/status.json'
        if not path.exists() or json.loads(path.read_text()).get('state')!='stopped':return False
        path=directory/'readers/ao/status.json'
        if path.exists() and json.loads(path.read_text()).get('state')!='stopped':return False
    # The paused execution on GPU4-7 must also have released its model workers.
    for filename in ['native/status.json','readers/ao/status.json']:
        path=FOLLOWUP/filename
        if path.exists() and json.loads(path.read_text()).get('state')!='stopped':return False
    return True


def main():
    policy=BASE/'qwen8_migration_policy.json'
    if policy.exists() and json.loads(policy.read_text())['state']=='8b_followup_selected':
        print('27B behavior suite deferred by user; followup moved to Qwen3-8B',flush=True)
        return
    lock=(BASE/'gpu03_followup_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=BASE/'gpu03_followup_queue_status.json'
    save(status,dict(state='waiting',predecessors=['semantic Cancer-Myth Luna','semantic CREPE Luna'],
        followup=str(FOLLOWUP),followup_native_gpus='0,1',followup_reader_gpus='2,3',
        reason='User requested GPU4-7 job resume only after GPU0-3 experiments finish',updated_utc=time.time()))
    while not ready():time.sleep(5)
    env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HOME':'/data3/heejae/hf_cache',
        'FPQA_GEPA_DATASET':'crepe','FPQA_GEPA_QUERY':'behavior',
        'FPQA_GEPA_NATIVE_GPUS':'0,1','FPQA_GEPA_READER_GPUS':'2,3'}
    save(status,dict(state='running',followup=str(FOLLOWUP),native_gpus='0,1',reader_gpus='2,3',updated_utc=time.time()))
    try:
        subprocess.run([sys.executable,'-u','scripts/supervise_gepa_ao_feedback_ablation.py'],cwd=ROOT,env=env,check=True)
        save(status,dict(state='complete',followup=str(FOLLOWUP),updated_utc=time.time()))
    except Exception as error:
        save(status,dict(state='attention_required',error=repr(error),followup=str(FOLLOWUP),updated_utc=time.time()));raise


if __name__=='__main__':main()
