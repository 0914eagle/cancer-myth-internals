"""Run capture then canonical NLA verbalization for a training-only span audit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True)
a=p.parse_args();c=json.loads(a.config.read_text());spec=c['models'][a.model]
out=Path(c['artifact_root'])/a.model;out.mkdir(parents=True,exist_ok=True)
locks=[]
for gpu in spec['gpus']:
 f=Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
common=['--config',str(a.config.resolve()),'--model',a.model]
env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','PYTHONUNBUFFERED':'1'}
commands=[('capture',['/data3/heejae/uv/paper_baselines_b81ace2/bin/python','-u','scripts/nla_gold_span_capture.py',*common])]
av_script='scripts/nla_av_multigpu.py' if len(spec['gpus'])>1 else 'scripts/nla_av_batched.py'
commands.append(('av',['/data3/heejae/uv/medical_nla/bin/python','-u',av_script,*common,
 '--gpu',','.join(map(str,spec['gpus'])),'--batch-size','16' if len(spec['gpus'])>1 else '32',
 '--manifest',str(out/'manifest.jsonl'),'--output',str(out/'av_results.jsonl')]))
status=out/'pipeline_status.json'
try:
 for stage,cmd in commands:
  status.write_text(json.dumps(dict(state='running',stage=stage,pid=os.getpid()))+'\n')
  with (out/(stage+'.log')).open('a') as log:
   subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=log,check=True)
 status.write_text(json.dumps(dict(state='complete',stage='av',pid=os.getpid()))+'\n')
except Exception as error:
 status.write_text(json.dumps(dict(state='failed',error=str(error)))+'\n');raise
