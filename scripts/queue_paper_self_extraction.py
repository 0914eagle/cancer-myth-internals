"""Handoff after each model's GEPA Table 2 finishes; retain every Sol artifact."""
import argparse,json,subprocess,time,sys,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import write_json
PYTHON='/data1/heejae/uv/cancer_myth_internals/bin/python'
BASE=ROOT/'results/paper_baselines_v4'
MODELS={'qwen25':'qwen','gemma3':'gemma','luna6':'luna'}

def ready_for_handoff(progress):
    return progress.get('stage')=='table3' and progress.get('state') in ('running','complete')

def unit(model):return 'paper-v4-'+MODELS[model]+'-sol'+('-gpu123' if model=='gemma3' else '')+'-20261012.service'
def pid(model):return subprocess.check_output(['systemctl','--user','show',unit(model),'-p','MainPID','--value'],text=True).strip()
def command(model,table='table2',action='run'):
    return [PYTHON,'scripts/run_paper_self_extraction.py',action,'--model',model,'--config',str(BASE/'protocol/runtime.json'),'--bundle',str(BASE/'protocol/bundle.json'),'--out',str(BASE/'self_extraction'),'--table',table]

def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    out=BASE/'self_extraction';out.mkdir(exist_ok=True)
    lock=(out/'.handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Share the four GPT slots with the existing suite.
    (out/'shared').mkdir(exist_ok=True)
    slots=out/'shared/gpt_slots'
    if not slots.exists():slots.symlink_to(BASE/'main/shared/gpt_slots',target_is_directory=True)
    status=out/'handoff_status.json'
    state=json.loads(status.read_text()) if status.exists() else {m:{'state':'waiting_gepa'} for m in MODELS}
    children={};judge=None
    while True:
        for m in MODELS:
            s=state[m];stage=s['state'];old=BASE/'main'/m
            if stage=='waiting_gepa':
                progress=json.loads((old/'generation_progress.json').read_text())
                if ready_for_handoff(progress):
                    if (old/'STOP').exists():continue # don't take ownership of another stop request
                    (old/'STOP').write_text('Self-extraction handoff after GEPA Table 2\n')
                    s.update(state='draining',requested_epoch=time.time())
            elif stage=='draining':
                if pid(m) not in ('','0'):continue
                subprocess.run(['systemctl','--user','stop',unit(m)],check=True)
                s['state']='table2'
            elif stage in ('table2','table3'):
                if m not in children:
                    children[m]=subprocess.Popen(command(m,stage),cwd=ROOT)
                rc=children[m].poll()
                if rc is not None:
                    del children[m]
                    if rc:raise RuntimeError(f'{m} {stage} failed ({rc}); resumable output retained')
                    if stage=='table2':
                        # Resume native QA without the obsolete Sol pipeline conditions.
                        subprocess.run([PYTHON,'scripts/launch_paper_v4_sol.py',MODELS[m],'--methods','plain,balanced,direct_gate,cot_gate,cot,tfidf_gate,probe_gate,gepa_gate,gepa,gepa_both'],cwd=ROOT,check=True)
                        s['state']='waiting_native_qa'
                    else:s['state']='complete'
            elif stage=='waiting_native_qa':
                progress=json.loads((old/'generation_progress.json').read_text())
                if progress.get('state')=='complete' and pid(m) in ('','0'):s['state']='table3'
        write_json(status,dict(state))
        if a.once:return
        # Start one independent streaming judge when all identities are initialized.
        if judge is None:
            for m in MODELS:
                if not (out/m/'identity.json').exists():subprocess.run(command(m,action='init'),cwd=ROOT,check=True)
            args=[PYTHON,'scripts/judge_paper_queue.py','--config',str(BASE/'protocol/runtime.json'),'--bundle',str(BASE/'protocol/bundle.json'),'--out',str(out),'--judge-config',str(BASE/'protocol/haiku.json'),'--judgments-out',str(BASE/'self_extraction_haiku'),'--workers','128','--workers-control',str(BASE/'haiku/workers_control.json')]
            judge=subprocess.Popen(args,cwd=ROOT)
        if judge.poll() is not None:raise RuntimeError('Self-extraction judge exited; supervisor will retry')
        time.sleep(30)

if __name__=='__main__':main()
