"""Drain saved FPQA answers independently of GPU generation. No generation calls."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_paper_baselines import Worker, check_bundle
from scripts.run_paper_baselines_resilient import quota_error, locked_external_call, INTERVAL
from src.paper_baselines import WellBridge, write_json, sha


class JudgeClient:
    judge=Worker.judge
    def __init__(self,config,out):
        self.out=out
        self.external=json.loads((ROOT/config['external_config']).read_text())
        self.bridge=WellBridge(config)
        identity=json.loads((out/'identity.json').read_text())
        if self.external!=identity['external_config']:
            raise ValueError('Frozen judge configuration mismatch')
    def external_call(self,role,messages):
        return locked_external_call(self,role,messages,Worker.external_call)


def pending(out,models):
    """Round robin across models so a fast generator cannot starve its peer."""
    per_model=[]
    for model in models:
        paths=[]
        for p in sorted((out/model/'table12').rglob('*.json')):
            r=json.loads(p.read_text())
            if 'rating' not in r:paths.append(p)
        per_model.append((model,paths))
    for i in range(max((len(p) for _,p in per_model),default=0)):
        for model,paths in per_model:
            if i<len(paths):yield model,paths[i]


def score_one(client,path,lookup):
    r=json.loads(path.read_text())
    if 'rating' in r:return False
    rating,reason=client.judge(lookup[r['id']],r['answer'])
    r.update(rating=rating,judge_reason=reason,judge_config=client.external['judge_model'])
    write_json(path,r)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text());bundle=check_bundle(a.bundle)
    lookup={r['id']:r for r in bundle['rows']}
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'.judge_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clients={m:JudgeClient(config,a.out/m) for m in config['models']}
    status=a.out/'judge_queue_status.json'
    state={'state':'running','code_sha256':sha(Path(__file__)),'scored_this_run':0}
    deadline=0
    if status.exists():
        old=json.loads(status.read_text())
        if old.get('state')=='waiting_quota':deadline=old['next_retry_epoch']
    try:
        while True:
            if (a.out/'STOP_JUDGE').exists():
                state.update(state='stopped');write_json(status,state);return
            if time.time()<deadline:
                time.sleep(min(1,deadline-time.time()));continue
            queue=list(pending(a.out,clients))
            state.update(state='running' if queue else 'idle',pending=len(queue),updated_epoch=time.time())
            write_json(status,state)
            if not queue:
                time.sleep(10);continue
            for model,path in queue:
                if (a.out/'STOP_JUDGE').exists():break
                try:
                    changed=score_one(clients[model],path,lookup)
                except (RuntimeError,ValueError) as error:
                    if not quota_error(error):raise
                    deadline=time.time()+INTERVAL
                    state.update(state='waiting_quota',model=model,next_retry_epoch=deadline)
                    write_json(status,state)
                    print(f'[judge queue] Quota limit; retry in {INTERVAL}s. GPU generation continues.',flush=True)
                    break
                state['scored_this_run']+=int(changed)
                state['pending']-=1
                state.update(updated_epoch=time.time(),state='running')
                write_json(status,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));write_json(status,state);raise


if __name__=='__main__':main()
