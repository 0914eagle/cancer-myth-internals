"""Resume the suite after draining old workers and archiving Luna-extracted pipeline answers."""
import argparse,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/data1/heejae/uv/cancer_myth_internals/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('role',choices=['qwen','gemma','luna','extraction','haiku']);a=p.parse_args()
    new=f'paper-v4-{a.role}-sol'+('-gpu123' if a.role=='gemma' else '')+'-20261012.service'
    units=[f'paper-v4-{a.role}-{suffix}-20261012.service' for suffix in ('cli','fast','prefetch','sol','sol-gpu23')]
    for unit in dict.fromkeys([*units,new]):
        state=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','--value'],capture_output=True,text=True)
        if state.stdout.strip() not in ('','0'):raise SystemExit(f'{unit} is still running; drain first')
        load=subprocess.run(['systemctl','--user','show',unit,'-p','LoadState','--value'],capture_output=True,text=True)
        if load.stdout.strip() not in ('','not-found'):subprocess.run(['systemctl','--user','stop',unit],check=True)
    protocol=Path('results/paper_baselines_v4/protocol');out=Path('results/paper_baselines_v4/main')
    smoke=json.loads((ROOT/out/'shared/sol_extraction/smoke.json').read_text())
    if smoke['state']!='complete' or smoke['model']!='gpt-6-sol':raise SystemExit('Sol train-item smoke required')
    common=['--config',str(protocol/'runtime.json'),'--bundle',str(protocol/'bundle.json'),'--out',str(out)]
    if a.role=='haiku':
        stop=ROOT/'results/paper_baselines_v4/haiku/STOP_JUDGE'
        command=[PYTHON,'scripts/judge_paper_queue.py',*common,'--judge-config',str(protocol/'haiku.json'),'--judgments-out','results/paper_baselines_v4/haiku','--workers','8']
    elif a.role=='extraction':
        stop=ROOT/out/'shared/sol_extraction/STOP'
        command=[PYTHON,'scripts/prepare_shared_sol_extractions.py',*common]
    else:
        model={'qwen':'qwen25','gemma':'gemma3','luna':'luna6'}[a.role];stop=ROOT/out/model/'STOP'
        script='scripts/run_paper_gemma_placement.py' if a.role=='gemma' else 'scripts/run_paper_baselines_sol.py'
        command=[PYTHON,script,'run','--model',model,*common,'--detect-first']
        if a.role!='luna':command+=['--max-batch','16' if a.role=='qwen' else '32','--token-budget','147456' if a.role=='qwen' else '393216']
    stop.unlink(missing_ok=True)
    subprocess.run(['systemd-run','--user',f'--unit={new}','--property=Restart=on-failure','--property=RestartSec=600',
        f'--property=WorkingDirectory={ROOT}',
        '--setenv=PATH=/home/eagle0914/.local/bin:/home/eagle0914/.npm-global/bin:/usr/local/bin:/usr/bin:/bin',
        '--setenv=HF_HUB_OFFLINE=1',
        f'--setenv=PYTHONPATH={ROOT}/.venv/paper_baseline_extras',*command],check=True)
    print(json.dumps({'unit':new,'command':command,'split_manifest_unchanged':True}))


if __name__=='__main__':main()
