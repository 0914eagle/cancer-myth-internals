"""Full train coverage, with existing sparse cases retained and NLA priority order."""
import argparse
import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True)
p.add_argument('--replace-pilot',type=int)
a=p.parse_args();c=json.loads(a.config.read_text());spec=c['models'][a.model]
out=Path(c['artifact_root'])/a.model;out.mkdir(parents=True,exist_ok=True)
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


def handoff(pid):
    proc=Path(f'/proc/{pid}')
    if not proc.exists():return
    command=(proc/'cmdline').read_bytes().split(b'\0')
    if b'scripts/run_nla_generation_comparison_gpu23.py' not in command or a.model.encode() not in command:
        raise RuntimeError('Refusing to replace an unrelated process')
    if os.getpgid(pid)!=pid:raise RuntimeError('Unexpected pilot process group')
    previous=Path(c['reuse_root'])/a.model/'generation_status.json'
    def count():
        if not previous.exists():return 0
        return json.loads(previous.read_text()).get('completed_cases',0)
    initial=count();status('handoff',state='waiting_for_current_case',previous_pid=pid,saved_cases=initial)
    deadline=time.monotonic()+60
    while proc.exists() and count()==initial and time.monotonic()<deadline:time.sleep(.2)
    # Sealed cases survive; any unsealed current case is regenerated on resume.
    members=[]
    for candidate in Path('/proc').iterdir():
        if candidate.name.isdigit():
            try:
                if os.getpgid(int(candidate.name))==pid:members.append(int(candidate.name))
            except ProcessLookupError:pass
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:pass
    for _ in range(100):
        alive=[]
        for member in members:
            try:
                stat=Path(f'/proc/{member}/stat').read_text();state=stat.rsplit(')',1)[1].strip().split()[0]
                if state!='Z':alive.append(member)
            except FileNotFoundError:pass
        if not alive:break
        time.sleep(.1)
    else:raise RuntimeError('Pilot processes did not exit; GPU not reused')
    (out/'handoff.json').write_text(json.dumps(dict(previous_pid=pid,saved_cases=count(),
        reason='Replace 20-question scope with all train rows; sealed cases retained'),indent=2)+'\n')


def prioritize():
    answers={}
    for line in (out/'answers.jsonl').open():
        row=json.loads(line);answers[row['case_id']]=row
    paired={}
    for row in answers.values():paired.setdefault((row['dataset'],row['id']),{})[row['condition']]=row
    markers=re.compile(r'false premise|misconception|not necessarily|incorrect|not accurate|not true|not automatically|does not|doesn.t',re.I)
    ranks={}
    for pair in paired.values():
        plain=pair['plain'];balanced=pair['balanced']
        # Search order only; these string markers never become correctness labels.
        priority=0 if markers.search(balanced['answer']) and not markers.search(plain['answer']) else 1 if balanced['answer']!=plain['answer'] else 2
        for row in pair.values():ranks[row['case_id']]=priority
    rows=[json.loads(x) for x in (out/'manifest.jsonl').open()]
    positions={'prompt_keyword_1':0,'prompt_keyword_2':1,'answer_content_token_32':2,
        'answer_content_token_8':3,'answer_content_token_1':4,'question_last_token':5,'assistant_boundary_last_token':6}
    rows.sort(key=lambda r:(ranks[r['case_id']],r['source'],r['base_id'],positions[r['position_family']],r['condition']))
    target=out/'manifest_priority.jsonl'
    with target.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'nla_priority_policy.json').write_text(json.dumps(dict(
        coverage='All saved sparse vectors; priority changes order only',
        priority='Balanced-only correction markers, then differing answers, then remaining pairs',
        correctness='Unknown until Well scoring; markers are not correctness labels'),indent=2)+'\n')
    return target,len(rows)


try:
    if a.replace_pilot:handoff(a.replace_pilot)
    lock_dir=Path('/tmp/paper-baselines-gpu-locks');lock_dir.mkdir(exist_ok=True)
    lock=(lock_dir/f"gpu-{spec['gpu']}.lock").open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    baseline_python='/data3/heejae/uv/paper_baselines_b81ace2/bin/python'
    nla_python='/data3/heejae/uv/medical_nla/bin/python'
    common=['--config',str(a.config.resolve()),'--model',a.model]
    gen=out/'generation_status.json'
    if not gen.exists() or json.loads(gen.read_text())['state']!='complete':
        run('generate',[baseline_python,'-u','scripts/nla_generation_capture_full.py',*common])
    manifest,n=prioritize();av=out/'av_results.jsonl'
    if not av.exists() or sum(1 for _ in av.open())!=n:
        run('av',[nla_python,'-u','scripts/run_existing_nla_av.py','--config',str(out/'av_config.yaml'),
            '--manifest',str(manifest),'--output',str(av)])
    if not (out/'summary.json').exists():
        run('ar',[nla_python,'-u','scripts/nla_full_reconstruction.py',*common])
    status('complete',state='complete',generation=json.loads(gen.read_text()),
        summary=json.loads((out/'summary.json').read_text()),well_scoring='pending')
except Exception as error:
    status('failed',state='failed',error_type=type(error).__name__,error=str(error));raise
