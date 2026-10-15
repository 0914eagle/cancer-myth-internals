"""GPU queue: real answer generation/capture, released AV, released AR."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True)
a=p.parse_args();c=json.loads(a.config.read_text());spec=c['models'][a.model]
out=Path(c['artifact_root'])/a.model;out.mkdir(parents=True,exist_ok=True)
lock_dir=Path('/tmp/paper-baselines-gpu-locks');lock_dir.mkdir(exist_ok=True)
lock=(lock_dir/f"gpu-{spec['gpu']}.lock").open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
env={**os.environ,'CUDA_VISIBLE_DEVICES':str(spec['gpu']),'HF_HOME':'/data3/heejae/hf_cache',
    'PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'}


def status(stage,**kw):
    data=dict(pid=os.getpid(),model=a.model,gpu=spec['gpu'],stage=stage,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    tmp=out/'pipeline_status.tmp';tmp.write_text(json.dumps(data,indent=2)+'\n')
    tmp.replace(out/'pipeline_status.json');print(json.dumps(data),flush=True)


def run(stage,cmd):
    status(stage,state='running')
    with (out/(stage+'.log')).open('a') as log:
        subprocess.run(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,check=True)


try:
    baseline_python='/data3/heejae/uv/paper_baselines_b81ace2/bin/python'
    nla_python='/data3/heejae/uv/medical_nla/bin/python'
    common=['--config',str(a.config.resolve()),'--model',a.model]
    gen=out/'generation_status.json'
    if not gen.exists() or json.loads(gen.read_text())['state']!='complete':
        run('generate',[baseline_python,'-u','scripts/nla_generation_capture.py',*common])
    manifest=[json.loads(x) for x in (out/'manifest.jsonl').read_text().splitlines()]
    av=out/'av_results.jsonl'
    if not av.exists() or len(av.read_text().splitlines())!=len(manifest):
        run('av',[nla_python,'-u','scripts/run_existing_nla_av.py','--config',str(out/'av_config.yaml'),
            '--manifest',str(out/'manifest.jsonl'),'--output',str(av)])
    if not (out/'summary.json').exists():
        run('ar',[nla_python,'-u','scripts/nla_connection.py','score',*common])
    answers={r['case_id']:r for r in [json.loads(x) for x in (out/'answers.jsonl').read_text().splitlines()]}
    targets={r['id']:r for r in manifest}
    rows=[json.loads(x) for x in (out/'scored.jsonl').read_text().splitlines()]
    fields=['model','base_id','source','category','condition','position_family','selected_token_text',
        'question','answer','answer_completed','nla_output','recon_mse','recon_cos','question_text_control_mse']
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            target=targets[row['id']];answer=answers[target['case_id']]
            writer.writerow({k:({**row,'model':a.model,'selected_token_text':target['selected_token_text'],
                'question':answer['question'],'answer':answer['answer'],'answer_completed':answer['completed']}).get(k) for k in fields})
    status('complete',state='complete',generation=json.loads(gen.read_text()),
        summary=json.loads((out/'summary.json').read_text()),well_scoring='pending; GPU-only queue')
except Exception as error:
    status('failed',state='failed',error_type=type(error).__name__,error=str(error));raise
