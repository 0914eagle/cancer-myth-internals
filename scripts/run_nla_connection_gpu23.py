"""Independent per-GPU extraction -> existing AV -> official AR smoke queue."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--config', type=Path,required=True)
p.add_argument('--model',required=True)
a = p.parse_args()
c = json.loads(a.config.read_text());spec = c['models'][a.model]
out = Path(c['artifact_root'])/a.model;out.mkdir(parents=True,exist_ok=True)
lock_dir=Path('/tmp/paper-baselines-gpu-locks');lock_dir.mkdir(exist_ok=True)
lock=(lock_dir/f"gpu-{spec['gpu']}.lock").open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env = {**os.environ,'CUDA_VISIBLE_DEVICES':str(spec['gpu']),
    'HF_HOME':'/data3/heejae/hf_cache','PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'}


def state(stage, **kw):
    item = dict(stage=stage,pid=os.getpid(),model=a.model,gpu=spec['gpu'],
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    tmp=out/'pipeline_status.tmp';tmp.write_text(json.dumps(item,indent=2)+'\n')
    tmp.replace(out/'pipeline_status.json')
    print(json.dumps(item),flush=True)


def run(stage, cmd):
    state(stage,state='running')
    with (out/(stage+'.log')).open('a') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,check=True)


baseline_python='/data3/heejae/uv/paper_baselines_b81ace2/bin/python'
nla_python='/data3/heejae/uv/medical_nla/bin/python'
common=['--config',str(a.config.resolve()),'--model',a.model]
try:
    if not (out/'extraction_status.json').exists():
        run('extract',[baseline_python,'-u','scripts/nla_connection.py','extract',*common])
    state('waiting_for_checkpoints',state='waiting')
    while not (out/'checkpoints.json').exists():
        status_path=out/'download_status.json'
        if status_path.exists() and json.loads(status_path.read_text())['state']=='failed':
            raise RuntimeError('Official checkpoint download failed; see download_status.json')
        time.sleep(30)
    manifest=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
    av_file=out/'av_results.jsonl'
    if not av_file.exists() or len(av_file.read_text().splitlines()) != len(manifest):
        run('av',[nla_python,'-u','scripts/run_existing_nla_av.py',
            '--config',str(out/'av_config.yaml'),'--manifest',str(out/'manifest.jsonl'),
            '--output',str(av_file)])
    if not (out/'summary.json').exists():
        run('ar',[nla_python,'-u','scripts/nla_connection.py','score',*common])
    state('complete',state='complete',summary=json.loads((out/'summary.json').read_text()))
except Exception as error:
    state('failed',state='failed',error_type=type(error).__name__,error=str(error))
    raise
