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
from src.paper_baselines import WellBridge, write_json, sha, digest


class JudgeClient:
    judge=Worker.judge
    def __init__(self,config,out,judge_config=None):
        self.out=out
        self.external=json.loads((ROOT/config['external_config']).read_text())
        self.bridge=WellBridge(config)
        identity=json.loads((out/'identity.json').read_text())
        if self.external!=identity['external_config']:
            raise ValueError('Frozen judge configuration mismatch')
        if judge_config is not None:self.external['judge_model']=judge_config
    def external_call(self,role,messages):
        return locked_external_call(self,role,messages,Worker.external_call)


def evaluation_key(record,judge_config,row):
    return digest({'question':record['question'],'answer':record['answer'],'judge':judge_config,'annotation':row})


def pending(out,models,judgments_out=None,judge_config=None,lookup=None):
    """Round robin across models so a fast generator cannot starve its peer."""
    per_model=[]
    for model in models:
        paths=[]
        for p in sorted((out/model/'table12').rglob('*.json')):
            r=json.loads(p.read_text())
            if lookup is not None and r['id'] not in lookup:continue
            if judgments_out is None:
                if 'rating' not in r:paths.append(p)
            else:
                dest=judgments_out/p.relative_to(out)
                key=evaluation_key(r,judge_config,lookup[r['id']])
                if not dest.exists() or json.loads(dest.read_text()).get('evaluation_key')!=key:paths.append(p)
        per_model.append((model,paths))
    for i in range(max((len(p) for _,p in per_model),default=0)):
        for model,paths in per_model:
            if i<len(paths):yield model,paths[i]


def score_one(client,path,lookup,destination=None):
    r=json.loads(path.read_text())
    key=None
    if destination is None:
        if 'rating' in r:return False
    else:
        key=evaluation_key(r,client.external['judge_model'],lookup[r['id']])
        if destination.exists() and json.loads(destination.read_text()).get('evaluation_key')==key:return False
    rating,reason=client.judge(lookup[r['id']],r['answer'])
    r.update(rating=rating,judge_reason=reason,judge_config=client.external['judge_model'])
    if destination is not None:
        r.update(evaluation_key=key,source_answer_path=str(path))
    write_json(destination or path,r)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--judgments-out',type=Path);p.add_argument('--judge-config',type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text());bundle=check_bundle(a.bundle)
    lookup={r['id']:r for r in bundle['rows']}
    if bool(a.judgments_out)!=bool(a.judge_config):p.error('Provide both --judgments-out and --judge-config')
    judge_config=json.loads(a.judge_config.read_text()) if a.judge_config else None
    queue_root=a.judgments_out or a.out
    queue_root.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(queue_root/'.judge_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clients={m:JudgeClient(config,a.out/m,judge_config) for m in config['models']}
    status=queue_root/'judge_queue_status.json'
    state={'state':'running','code_sha256':sha(Path(__file__)),'scored_this_run':0,'judge_config':judge_config,'regrade_all':bool(a.judgments_out)}
    deadline=0
    if status.exists():
        old=json.loads(status.read_text())
        if old.get('state')=='waiting_quota':deadline=old['next_retry_epoch']
    try:
        while True:
            if (queue_root/'STOP_JUDGE').exists():
                state.update(state='stopped');write_json(status,state);return
            if time.time()<deadline:
                time.sleep(min(1,deadline-time.time()));continue
            queue=list(pending(a.out,clients,a.judgments_out,judge_config,lookup if a.judgments_out else None))
            state.update(state='running' if queue else 'idle',pending=len(queue),updated_epoch=time.time())
            write_json(status,state)
            if not queue:
                time.sleep(10);continue
            for model,path in queue:
                if (queue_root/'STOP_JUDGE').exists():break
                try:
                    changed=score_one(clients[model],path,lookup,(a.judgments_out/path.relative_to(a.out)) if a.judgments_out else None)
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
