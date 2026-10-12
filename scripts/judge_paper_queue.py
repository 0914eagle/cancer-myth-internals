"""Drain saved FPQA answers independently of GPU generation. No generation calls."""
from concurrent.futures import ThreadPoolExecutor,as_completed
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


class JudgeFormatError(ValueError):
    pass


def worker_count(path,default):
    value=json.loads(path.read_text())['workers'] if path and path.exists() else default
    if type(value) is not int or value<1:raise ValueError('workers must be a positive integer')
    return value


def provider_refusal(error):
    text=str(error)
    return "can't help with this" in text and ('[general_harms]' in text or 'anthropic.com/legal/aup' in text)


def validated_judge_call(client,role,messages):
    """Retry malformed ratings, preserving the raw output without changing the rubric."""
    from src.fpqa_prompting import parse_rating
    for attempt in range(3):
        text=Worker.external_call(client,role,messages)
        try:parse_rating(text)
        except ValueError:
            identity={'role':role,'config':client.external[role+'_model'],'messages':messages,
                      'code':sha(ROOT/'src/fpqa_cli_backend.py')}
            path=client.out/'external_calls'/(digest(identity)+'.json')
            archive=client.out/'invalid_judge_calls'/f'{digest(identity)}-{time.time_ns()}.json'
            archive.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():path.replace(archive)
            else:write_json(archive,{'request':identity,'text':text})
        else:return text
    raise JudgeFormatError('Three malformed ratings; raw responses archived, no score assigned')


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
        return locked_external_call(self,role,messages,validated_judge_call if role=='judge' else Worker.external_call)


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


def score_group(jobs,clients,lookup,out,judgments_out,workers):
    """At most workers requests; preserve successful grades if a peer fails."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(score_one,clients[m],p,lookup,judgments_out/p.relative_to(out) if judgments_out else None):(m,p) for m,p in jobs}
        for future in as_completed(futures):
            model,path=futures[future]
            try:changed=future.result()
            except Exception as error:yield model,path,False,error
            else:yield model,path,changed,None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--judgments-out',type=Path);p.add_argument('--judge-config',type=Path)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--workers-control',type=Path)
    a=p.parse_args()
    if a.workers<1:p.error('workers must be positive')
    config=json.loads(a.config.read_text());bundle=check_bundle(a.bundle)
    lookup={r['id']:r for r in bundle['rows']}
    if bool(a.judgments_out)!=bool(a.judge_config):p.error('Provide both --judgments-out and --judge-config')
    judge_config=json.loads(a.judge_config.read_text()) if a.judge_config else None
    queue_root=a.judgments_out or a.out
    queue_root.mkdir(parents=True,exist_ok=True)
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(queue_root/'.judge_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    clients={m:JudgeClient(config,a.out/m,judge_config) for m in config['models']}
    status=queue_root/'judge_queue_status.json'
    state={'state':'running','code_sha256':sha(Path(__file__)),'scored_this_run':0,'judge_config':judge_config,'regrade_all':bool(a.judgments_out),'workers':worker_count(a.workers_control,a.workers),'started_epoch':time.time()}
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
            queue=[]
            for model,path in pending(a.out,clients,a.judgments_out,judge_config,lookup if a.judgments_out else None):
                refusal=queue_root/'provider_refusals'/path.relative_to(a.out)
                if refusal.exists():
                    record=json.loads(path.read_text())
                    key=evaluation_key(record,clients[model].external['judge_model'],lookup[record['id']])
                    if json.loads(refusal.read_text()).get('evaluation_key')==key:continue
                failure=queue_root/'format_failures'/path.relative_to(a.out)
                if failure.exists() and json.loads(failure.read_text()).get('retry_after_epoch',0)>time.time():continue
                queue.append((model,path))
            state.update(state='running' if queue else 'idle',pending=len(queue),updated_epoch=time.time())
            write_json(status,state)
            if not queue:
                time.sleep(10);continue
            offset=0
            while offset<len(queue):
                if (queue_root/'STOP_JUDGE').exists():break
                workers=worker_count(a.workers_control,a.workers)
                jobs=queue[offset:offset+workers];offset+=len(jobs)
                group_started=time.time();group_scored=0
                state.update(workers=workers,group_started_epoch=group_started)
                write_json(status,state)
                limited=False;fatal=None
                for model,path,changed,error in score_group(jobs,clients,lookup,a.out,a.judgments_out,workers):
                    if error is not None:
                        if provider_refusal(error):
                            record=json.loads(path.read_text())
                            write_json(queue_root/'provider_refusals'/path.relative_to(a.out),{
                                'source_answer_path':str(path),'error':str(error),'score_assigned':False,
                                'evaluation_key':evaluation_key(record,clients[model].external['judge_model'],lookup[record['id']]),
                                'action':'manual review required; do not automatically retry provider refusal'})
                            state['provider_refusals_this_run']=state.get('provider_refusals_this_run',0)+1
                            state['pending']-=1
                            write_json(status,state)
                            continue
                        if isinstance(error,JudgeFormatError):
                            write_json(queue_root/'format_failures'/path.relative_to(a.out),{
                                'source_answer_path':str(path),'error':str(error),
                                'retry_after_epoch':time.time()+INTERVAL,'score_assigned':False})
                            state['format_failures_this_run']=state.get('format_failures_this_run',0)+1
                            state['pending']-=1
                            write_json(status,state)
                            continue
                        if not quota_error(error):fatal=error;continue
                        limited=True;deadline=time.time()+INTERVAL
                        state.update(state='waiting_quota',model=model,next_retry_epoch=deadline,
                                     last_quota_error=str(error),last_quota_epoch=time.time())
                        write_json(status,state)
                        continue
                    state['scored_this_run']+=int(changed)
                    group_scored+=int(changed)
                    (queue_root/'format_failures'/path.relative_to(a.out)).unlink(missing_ok=True)
                    state['pending']-=1
                    state.update(updated_epoch=time.time())
                    write_json(status,state)
                if fatal is not None:raise fatal
                state.update(last_group_seconds=time.time()-group_started,last_group_scored=group_scored)
                write_json(status,state)
                if limited:
                    if a.workers_control:
                        target=max(1,min(worker_count(a.workers_control,workers),workers//2))
                        write_json(a.workers_control,{'workers':target,'reason':'reduce after provider quota error','updated_epoch':time.time()})
                    print(f'[judge queue] Quota limit; retry in {INTERVAL}s. GPU generation continues.',flush=True)
                    break
    except BaseException as error:
        state.update(state='failed',error=str(error));write_json(status,state);raise


if __name__=='__main__':main()
