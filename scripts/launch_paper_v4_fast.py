"""Resume the authorized v4 suite with larger batches and detection-first scheduling.

Existing workers must drain through STOP before replacement; never kills live calls.
Frozen runtime, split manifest, prompts and completed outputs are unchanged.
"""
import argparse,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data1/heejae/uv/cancer_myth_internals/bin/python'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['qwen','gemma','luna','haiku']);a=p.parse_args()
    old=f'paper-v4-{a.role}-cli-20261012.service';new=f'paper-v4-{a.role}-fast-20261012.service'
    for unit in (old,new):
        result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],capture_output=True,text=True)
        if result.stdout.strip() not in ('','0'):raise SystemExit(f'{unit} still running; wait for saved-call drain')
    load=subprocess.run(['systemctl','--user','show',old,'-p','LoadState','--value'],capture_output=True,text=True)
    if load.stdout.strip() not in ('','not-found'):subprocess.run(['systemctl','--user','stop',old],check=True)
    protocol=Path('results/paper_baselines_v4/protocol');out=Path('results/paper_baselines_v4/main')
    common=['--config',str(protocol/'runtime.json'),'--bundle',str(protocol/'bundle.json'),'--out',str(out)]
    if a.role=='haiku':
        stop=ROOT/'results/paper_baselines_v4/haiku/STOP_JUDGE'
        command=[PYTHON,'scripts/judge_paper_queue.py',*common,'--judge-config',str(protocol/'haiku.json'),'--judgments-out','results/paper_baselines_v4/haiku','--workers','4']
    else:
        model={'qwen':'qwen25','gemma':'gemma3','luna':'luna6'}[a.role];stop=ROOT/out/model/'STOP'
        command=[PYTHON,'scripts/run_paper_baselines_v4.py','run','--model',model,*common,'--detect-first']
        if a.role!='luna':command+=['--max-batch','16' if a.role=='qwen' else '8','--token-budget','147456' if a.role=='qwen' else '98304']
    stop.unlink(missing_ok=True)
    subprocess.run(['systemd-run','--user',f'--unit={new}','--property=Restart=on-failure','--property=RestartSec=600',
        f'--property=WorkingDirectory={ROOT}',
        '--setenv=PATH=/home/eagle0914/.local/bin:/home/eagle0914/.npm-global/bin:/usr/local/bin:/usr/bin:/bin',
        f'--setenv=PYTHONPATH={ROOT}/.venv/paper_baseline_extras',*command],check=True)
    print(json.dumps({'unit':new,'command':command,'split_manifest_unchanged':True}))

if __name__=='__main__':main()
