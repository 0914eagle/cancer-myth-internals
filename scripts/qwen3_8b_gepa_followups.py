"""Run the deferred CREPE semantic/behavior GEPA controls on Qwen3-8B, Luna only."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import qwen3_8b_ao_crepe as worker
from scripts.reader_gepa_io import save,load_prompt_tokenizer
from scripts.run_ao_yesno_crepe_probe import external,LUNA
from src.fpqa_prompting import parse_rating,well_response_judge
from src.fpqa_cli_backend import CLIBackend

BASE=ROOT/'results'


def prepare(mode):
    _,native_protocol=worker.rows_and_protocol()
    old=json.loads((worker.PILOT/'plan.json').read_text())
    config=json.loads((worker.PILOT/'runtime.json').read_text())
    lookup={r['id']:r for r in json.loads(Path(config['bundle']).read_text())['rows']}
    split=old['splits']['crepe']
    arms=['text','text_ao','text_ao_shuffled']+(['ao_only'] if mode=='behavior' else [])
    donor={}
    for label in [1,0]:
        ids=sorted(i for i in split['train_ids'] if lookup[i]['label']==label)
        random.Random(f'42:crepe:{label}').shuffle(ids)
        donor.update({i:ids[(j+1)%len(ids)] for j,i in enumerate(ids)})
    assert all(a!=b and lookup[a]['group_id']!=lookup[b]['group_id'] and lookup[a]['label']==lookup[b]['label'] for a,b in donor.items())
    out=BASE/f'qwen3_8b_gepa_{mode}_crepe_luna_20261014'
    query=worker.QUERIES['neutral' if mode=='semantic' else 'behavior']
    protocol=dict(model='Qwen/Qwen3-8B',source_checkpoint=native_protocol['source_checkpoint'],
        ao_checkpoint=native_protocol['ao_checkpoint'],mode=mode,arms=arms,
        starting_prompt=old['starting_prompt'],splits=split,donor_map=donor,ao_query=query,
        ao_layer=18,source_positions='Actual native generation prefill, last up to20 question tokens',
        source_gpu=4,reader_gpu=5,source_batch_max=16,reader_batch_max=64,
        source_thinking=False,dtype='bfloat16',source_max_new_tokens=8192,
        question_metric_cap=100,pair_metric_cap=50,max_candidate_proposals=4,
        reflection_minibatch_pairs=2,prompt_token_limit=512,diagnostic_token_limit=512,seed=42,
        gpt=LUNA,final_judge=json.loads((worker.PILOT/'haiku_runtime.json').read_text())['judge_model'],
        ao_only='Reflection gets current prompt and AO descriptions only; no per-question text, answer, label, reference, score or grading reason.',
        objective='Actual-answer Well scores in every arm, including AO-only; reader scores are not objectives.',
        selection='Luna candidate-dev only; freeze before pinned Haiku existing8 heldout-dev evaluation.',
        donor_control='Different same-label train group under identical candidate. Prefill-only if donor native answer is not cached. Donor answers not supplied or scored.',
        prior_exposure='Same previously examined train50/dev8/heldout8. Exploratory, not independent confirmation or official test.',
        migration='No Gemma/Sol optimizer states, scores, prompts or activations reused; original model names kept in preserved runs.')
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:save(path,protocol)
    return out,protocol,native_protocol,lookup,config


def run(mode):
    out,p,np,lookup,config=prepare(mode)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from gepa.optimize_anything import GEPAConfig,EngineConfig,ReflectionConfig,optimize_anything
    baseline=json.loads(Path(config['baseline_config']).read_text())
    tok=load_prompt_tokenizer(baseline)
    owned=[];logs=[]
    for name in ['native/STOP','readers/ao/STOP']:(worker.OUT/name).unlink(missing_ok=True)
    def status(stage,**kwargs):save(out/'status.json',dict(state='running',stage=stage,updated_utc=time.time(),**kwargs))
    def wait(path,worker_index=0):
        started=time.monotonic()
        while not path.exists():
            if owned[worker_index].poll() is not None:raise RuntimeError('Owned8B GEPA worker exited')
            if time.monotonic()-started>1800:raise TimeoutError('8B GEPA queue request exceeded1800seconds')
            time.sleep(.2)
        return json.loads(path.read_text())
    def native_batch(rows,prompt,capture_only=False):
        ids=[worker.request(r,prompt,np,capture_only=capture_only) for r in rows]
        return [wait(worker.OUT/'native/results'/(i+'.json')) for i in ids]
    def native(row,prompt,capture_only=False):return native_batch([row],prompt,capture_only)[0]
    def judge(row,n):
        path=out.parent/'qwen3_8b_luna_judge_cache_20261014'/(n['id']+'.json')
        if path.exists():return json.loads(path.read_text())
        if not n['completed']:r=dict(score=0,reason='Incomplete answer; no external judge call.',native_id=n['id'])
        else:
            text=external(well_response_judge(ROOT/baseline['well_root'],row,n['answer']),output_dir=out)
            rating=parse_rating(text)
            if rating is None:raise ValueError('Invalid Luna Well score')
            r=dict(score=rating,reason=text,native_id=n['id'],model=LUNA)
        save(path,r);return r
    def description(n):
        kind='neutral' if mode=='semantic' else 'behavior'
        ident=worker.sha([n['id'],kind,p['ao_query'],'qwen3_8b_ao_v1'])
        save(worker.OUT/'readers/ao/requests'/(ident+'.json'),dict(id=ident,native_id=n['id'],
            activation_path=n['ao_activation_path'],query=p['ao_query'],query_kind=kind))
        r=wait(worker.OUT/'readers/ao/results'/(ident+'.json'),worker_index=1)
        assert r['source_native_id']==n['id'] and r['source_text_provided'] is False
        return r
    def pairs(ids,partition):
        groups=[[lookup[i] for i in ids if lookup[i]['label']==label] for label in [1,0]]
        for g in groups:random.Random(42).shuffle(g)
        return [dict(pair=[a,b],partition=partition) for a,b in zip(*groups)]
    try:
        for role in ['native','reader']:
            log=(out/(role+'_worker.log')).open('a');logs.append(log)
            owned.append(subprocess.Popen([worker.PYTHON,'-u',str(ROOT/'scripts/qwen3_8b_ao_crepe.py'),'--role',role],
                cwd=ROOT,env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'},stdout=log,stderr=log))
        selected={}
        for arm in p['arms']:
            directory=out/'optimization'/arm;directory.mkdir(parents=True,exist_ok=True)
            if (directory/'selected.json').exists():selected[arm]=json.loads((directory/'selected.json').read_text());continue
            status('optimization',arm=arm)
            def evaluate(candidate,example):
                if set(candidate)!={'answer'}:raise ValueError('Candidate components changed')
                if len(tok.encode(candidate['answer'],add_special_tokens=False))>512:return 0.,{'error':'Candidate exceeds512token limit'}
                rows=example['pair'];generated=native_batch(rows,candidate['answer']);feedback=[];objective=[]
                for row,n in zip(rows,generated):
                    j=judge(row,n);info={}
                    if example['partition']=='train':
                        if arm!='ao_only':info=dict(question=row['question'],answer=n['answer'],label=row['label'],
                            presuppositions=row['presuppositions'],reference_answer=row['reference_answer'],rating=j['score'],grading_reason=j['reason'])
                        if arm!='text':
                            donor=lookup[p['donor_map'][row['id']]] if arm=='text_ao_shuffled' else row
                            dn=n if donor['id']==row['id'] else native(donor,candidate['answer'],capture_only=True)
                            extra=description(dn);tokens=tok.encode(extra['explanation'],add_special_tokens=False)
                            info['ao_description']=tok.decode(tokens[:512])
                            save(directory/'extras'/(n['id']+'.json'),dict(extra,recipient_id=row['id'],donor_id=donor['id'],
                                supplied_diagnostic=info['ao_description'],clipped=len(tokens)>512))
                        if arm=='ao_only':assert set(info)=={'ao_description'}
                    feedback.append(info);objective.append(dict(row_id=row['id'],native_id=n['id'],score=j['score']))
                score=sum(x['score']/5 for x in objective)/2
                with (directory/'evaluations.jsonl').open('a') as f:f.write(json.dumps(dict(candidate=candidate,
                    partition=example['partition'],score=score,objective_records=objective,reflection_feedback=feedback,timestamp=time.time()),ensure_ascii=False)+'\n')
                return score,{'examples':feedback}
            def reflect(messages):
                if isinstance(messages,str):messages=[dict(role='user',content=messages)]
                if not any(m['role']=='system' for m in messages):messages=[dict(role='system',content='Improve the supplied prompt using training feedback. Return the requested candidate.')]+messages
                assert [m['role'] for m in messages]==['system','user']
                with (directory/'reflection_payloads.jsonl').open('a') as f:f.write(json.dumps(dict(messages=messages,arm=arm,timestamp=time.time()),ensure_ascii=False)+'\n')
                return external(messages,output_dir=out)
            result=optimize_anything(seed_candidate={'answer':p['starting_prompt']},evaluator=evaluate,
                dataset=pairs(p['splits']['train_ids'],'train'),valset=pairs(p['splits']['candidate_dev_ids'],'dev'),
                objective='Maximize balanced FPQ correction and normal-question preservation. Only questions are available at inference.',
                background='Optimize only the answer prompt. No tools or retrieval. The prompt must fit512tokenizer tokens. '
                    'Use only the supplied feedback. Activation descriptions may be unreliable and are not evidence of source beliefs. '
                    'Do not assume that a described future response actually occurred. Do not copy example-specific facts into a general instruction.',
                config=GEPAConfig(engine=EngineConfig(run_dir=str(directory/'engine'),seed=42,max_metric_calls=50,
                    max_candidate_proposals=4,parallel=False,max_workers=1,cache_evaluation=False),
                    reflection=ReflectionConfig(reflection_lm=reflect,reflection_minibatch_size=2)))
            assert set(result.best_candidate)=={'answer'} and len(tok.encode(result.best_candidate['answer'],add_special_tokens=False))<=512
            selected[arm]=dict(arm=arm,candidate=result.best_candidate,best_idx=result.best_idx,selection='Frozen Luna candidate-dev selection')
            save(directory/'selected.json',selected[arm]);print('Selected',mode,arm,result.best_idx,flush=True)
        save(out/'frozen_candidates.json',selected)
        save(worker.OUT/'readers/ao/STOP',dict(reason='All8B candidates frozen; no reader during final evaluation'))
        haiku=p['final_judge'];scored=[]
        for arm in p['arms']:
            for ident in p['splits']['heldout_validation_ids']:
                status('heldout',arm=arm,row_id=ident)
                n=native(lookup[ident],selected[arm]['candidate']['answer'])
                path=out.parent/'qwen3_8b_haiku_cache_20261014'/(worker.sha([haiku,n['id']])+'.json')
                if path.exists():j=json.loads(path.read_text())
                elif not n['completed']:j=dict(score=0,reason='Incomplete answer',incomplete=True)
                else:
                    raw,meta=CLIBackend(haiku)(well_response_judge(ROOT/baseline['well_root'],lookup[ident],n['answer']))
                    rating=parse_rating(raw)
                    if rating is None:raise ValueError('Invalid Haiku score')
                    j=dict(score=rating,reason=raw,metadata=meta,model=haiku)
                save(path,j);scored.append(dict(arm=arm,row_id=ident,label=lookup[ident]['label'],native=n,judgment=j))
                save(out/'heldout_scored_partial.json',scored)
        summary=[]
        for arm in p['arms']:
            f=[r['judgment']['score'] for r in scored if r['arm']==arm and r['label']==1]
            n=[r['judgment']['score'] for r in scored if r['arm']==arm and r['label']==0]
            assert len(f)==len(n)==4
            summary.append(dict(arm=arm,fpq_mean=sum(f)/4,nfp_mean=sum(n)/4,fpq_n=4,nfp_n=4))
        save(out/'heldout_scored.json',scored);save(out/'comparison_summary.json',dict(results=summary,
            limits='Exploratory existing8 heldout-dev, not official test; no Haiku reselection'))
        save(out/'status.json',dict(state='complete',records=len(scored),updated_utc=time.time()))
    finally:
        for name in ['native/STOP','readers/ao/STOP']:save(worker.OUT/name,dict(reason='Owned8B GEPA suite finished; normal shutdown'))
        for proc in owned:
            if proc.poll() is None:proc.wait(timeout=120)
        for log in logs:log.close()


def queue():
    path=BASE/'qwen3_8b_followup_queue_status.json'
    lock=(BASE/'qwen3_8b_followup_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        p=worker.OUT/'status.json'
        status=json.loads(p.read_text()) if p.exists() else {}
        if status.get('state')=='complete':break
        if status.get('state')=='attention_required':
            save(path,dict(state='waiting_for_initial_probe_recovery',reason=status['error'],updated_utc=time.time()))
        else:save(path,dict(state='waiting_for_initial_probe',updated_utc=time.time()))
        time.sleep(5)
    for mode in ['semantic','behavior']:
        out,p,_,_,_=prepare(mode)
        errors=0
        while True:
            save(path,dict(state='running',mode=mode,native_gpu=4,reader_gpu=5,updated_utc=time.time()))
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode],cwd=ROOT)
            if result.returncode==0:break
            errors+=1
            if errors>=3:
                save(path,dict(state='attention_required',mode=mode,error='Three consecutive failures; inspect suite logs',updated_utc=time.time()))
                return
            deadline=time.time()+600
            save(path,dict(state='waiting_provider',mode=mode,next_retry_utc=deadline,attempt=errors,updated_utc=time.time()))
            while time.time()<deadline:time.sleep(5)
    save(path,dict(state='complete',updated_utc=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['semantic','behavior'],default='semantic')
    parser.add_argument('--queue',action='store_true');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:
        for mode in ['semantic','behavior']:
            out,p,_,_,_=prepare(mode);print('Prepared',out,p['arms'],flush=True)
    elif args.queue:queue()
    else:
        try:run(args.mode)
        except Exception as error:
            out,_,_,_,_=prepare(args.mode)
            save(out/'status.json',dict(state='attention_required',error=repr(error),updated_utc=time.time()))
            raise
