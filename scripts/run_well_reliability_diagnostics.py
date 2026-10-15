#!/usr/bin/env python3
"""Frozen-input response repetitions and blinded rescoring; never overwrite old scores.

prepare writes an immutable job manifest. run resumes accepted calls by exact request.
No verifier is rerun in this experiment: pipeline reproducibility is a different estimand.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import freeze, load_rows, response_metrics
from src.fpqa_well_upstream import UncachedCaller, FatalTransportError, load_upstream, make_args, verify_source

BASE = ROOT/'results/fpqa_prompting'
SCOPE = BASE/'well_scope_factorial_luna6_20261004_v1'
UNCERTAINTY = BASE/'well_uncertainty_luna6_20261004_v1'
TEST = BASE/'cancer_data_test100_v2/cancer_myth/test.jsonl'
WELL = ROOT/'external/fpqa_20261002/Well'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(s): return hashlib.sha256(s.encode()).hexdigest()

def prepare(out):
    source = read(SCOPE/'run.json')
    ids = set(source['ids'])
    rows = {r['id']: r for r in load_rows(TEST) if r['id'] in ids}
    assert len(rows) == 199
    reference = Path(read(BASE/'well_pipelines_luna6_20261004/run.json')['reference_root'])
    plains = {read(p)['id']: p for p in (reference/'cancer_myth/plain').glob('*/record.json')}
    records, requests, provenance = {}, {}, {}
    for i in sorted(ids):
        records[i], requests[i] = {}, {}
        for name, p in [('B1', UNCERTAINTY/'items'/i/'B/record.json'),
                        ('B2', SCOPE/'items'/i/'B/record.json'),
                        ('DE1', SCOPE/'items'/i/'DE/record.json'), ('Plain1', plains[i])]:
            r = read(p)
            if name == 'Plain1':
                r = {**r, 'answer': r['feedback']['answer'], 'question': r['feedback']['question']}
                q = reference/r['attempt']/'calls/000001_task.request.json'
            else:
                q = p.parent.parent/'stages'/('DE' if name == 'DE1' else 'B')/'answer/request.json'
            assert r['question'] == rows[i]['question'] and r['label'] == rows[i]['label']
            assert read(q)['config'] == source['config']['task_model']
            records[i][name] = r; requests[i][name] = read(q)['messages']
            provenance[str(p)] = sha(p); provenance[str(q)] = sha(q)
        assert requests[i]['B1'] == requests[i]['B2']
        for name in records[i]: freeze(out/'historical'/i/f'{name}.json', records[i][name])
    flipped = sorted(i for i in ids if (records[i]['B1']['score'] >= 4) != (records[i]['B2']['score'] >= 4))
    assert Counter(rows[i]['label'] for i in flipped) == {1: 12, 0: 24}
    controls = []
    for label in [1, 0]:
        eligible = [i for i in ids if i not in flipped and rows[i]['label'] == label]
        controls += sorted(eligible, key=lambda i: rank('reliability-control-v1:'+i))[:12]
    audit = flipped + controls
    jobs = []
    # Each of 120 saved answers gets TWO independent rescores. Original score is retained.
    for i in audit:
        for origin in ['B1', 'B2']:
            for repeat in [1, 2]:
                jobs.append({'key': f'rejudge/{i}/{origin}/{repeat}', 'phase': 'rejudge',
                    'id': i, 'condition': origin, 'repeat': repeat,
                    'answer': records[i][origin]['answer']})
    jobs.sort(key=lambda j: rank('rejudge-order-v1:'+j['key']))
    fresh = []
    for i in sorted(ids):
        for condition, source_name, repeats in [('Plain', 'Plain1', [2, 3]), ('B', 'B2', [3]), ('DE', 'DE1', [2, 3])]:
            for repeat in repeats:
                fresh.append({'key': f'repeat/{i}/{condition}/{repeat}', 'phase': 'repeat',
                    'id': i, 'condition': condition, 'repeat': repeat,
                    'messages': requests[i][source_name]})
    jobs += sorted(fresh, key=lambda j: rank('repeat-order-v1:'+j['key']))
    plan = {'protocol': 'frozen_verification_reliability_v1', 'config': source['config'],
        'source_hashes': provenance, 'test_sha256': sha(TEST), 'well_source_sha256': verify_source(WELL),
        'code_sha256': {p: sha(ROOT/p) for p in ['scripts/run_well_reliability_diagnostics.py',
            'src/fpqa_cli_backend.py', 'src/fpqa_well_upstream.py']},
        'rows': list(rows.values()), 'flipped_ids': flipped, 'control_ids': controls,
        'audit_ids': audit, 'jobs': jobs,
        'scope': 'Exposed development cohort. Frozen verifier/claim outputs; response-generation reliability only.',
        'selection': 'All 36 flipped cases plus 24 deterministic stable controls. Audit is outcome-enriched, not a prevalence sample.',
        'repeats': 'Plain/DE: two fresh + one historical; B: one fresh + two historical. Historical runs not contemporaneously randomized.',
        'judge': 'Exact Well rubric, method and previous scores hidden, same model. Two rescores per saved answer; keep all originals.',
        'sampling': 'No supported temperature or generation seed in this CLI. Hash seeds control order/subset selection only.',
        'diagnostic': 'Separate reviewed claim manifest required before diagnostic generation; never infer all claims from question labels.'}
    freeze(out/'run.json', plan)
    freeze(out/'audit_inputs.json', [{'id': i, 'row': rows[i], 'selection': 'flipped' if i in flipped else 'control',
        'claims': records[i]['B1']['claims'], 'verdicts': records[i]['B1']['verdicts'],
        'answers': {k: {f: records[i][k].get(f) for f in ['answer', 'score', 'score_explanation']} for k in ['B1','B2']}}
        for i in audit])
    print(json.dumps({'audit_cases': len(audit), 'new_judge_only_calls': 240,
        'new_answers': len(fresh), 'new_answer_judges': len(fresh), 'verifier_calls': 0}))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','summarize'])
    p.add_argument('--out',type=Path,required=True); p.add_argument('--workers',type=int,default=2)
    p.add_argument('--phase',choices=['all','rejudge','repeat','diagnostic'],default='all')
    a=p.parse_args(); out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if a.command=='prepare': prepare(out);return
    plan=read(out/'run.json');rows={r['id']:r for r in plan['rows']}
    jobs=plan['jobs']
    if (out/'diagnostic_jobs.json').exists(): jobs += read(out/'diagnostic_jobs.json')['jobs']
    if a.phase!='all': jobs=[j for j in jobs if j['phase']==a.phase]
    def status(state):
        rs=[read(out/'jobs'/j['key']/'record.json') for j in jobs if (out/'jobs'/j['key']/'record.json').exists()]
        groups=defaultdict(list)
        for r in rs: groups[f"{r['phase']}/{r['condition']}/{r['repeat']}"].append(r)
        s={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),'target':len(jobs),'completed':len(rs),
           'completed_by_phase':dict(Counter(r['phase'] for r in rs)),
           'metrics':{k:response_metrics(v) for k,v in groups.items()}}
        tmp=out/f'status_{a.phase}.tmp';tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(out/f'status_{a.phase}.json');return s
    if a.command=='summarize':print(json.dumps(status('snapshot')));return
    assert 1<=a.workers<=4
    lock=(out/f'.lock_{a.phase}').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'STOP').exists(): raise RuntimeError('STOP exists')
    for path, checksum in {**plan['source_hashes'],**plan['code_sha256']}.items():
        if sha(ROOT/path)!=checksum: raise ValueError(f'Changed source: {path}')
    if verify_source(WELL)!=plan['well_source_sha256'] or sha(TEST)!=plan['test_sha256']:raise ValueError('Sources changed')
    config=plan['config'];local=threading.local();stop=threading.Event()
    balanced,_,_=load_upstream(WELL,lambda role,messages:local.call('judge',role,messages))
    args=make_args(balanced,'CancerMyth',out/'unused',out,config)
    from constant.constant import MODEL_FINAL_ANSWER_KEY
    from response.response import FinalAnswerResponse
    def process(job):
        loc=out/'jobs'/job['key'];target=loc/'record.json'
        if target.exists():return
        row=rows[job['id']]
        def call(stage,role,messages):
            if stop.is_set() or (out/'STOP').exists():raise FatalTransportError('STOP')
            f=loc/'stages'/stage;freeze(f/'request.json',{'role':role,'config':config[f'{role}_model'],'messages':messages})
            if (f/'result.json').exists():return read(f/'result.json')['text']
            raw=UncachedCaller(config,f)(role,messages)
            if not raw.strip():raise ValueError('Empty response')
            freeze(f/'result.json',{'text':raw});return raw
        try:
            if 'answer' in job: answer=job['answer']
            else: answer=FinalAnswerResponse.model_validate_plain_text(call('answer','task',job['messages']),model_role=args.model_role).get()
            local.call=call
            evaluator=balanced.BalancedDirectQAGEPAEvaluator(args,'gemini')
            evaluator.ScoreTemplateClass=evaluator.template_classes['fpq' if row['label'] else 'tpq']
            dp={'id':row['id'],'question':row['question'],'answer':row['reference_answer'],
                'presuppositions':row['presuppositions'],'few_shot_data':[],'passages':[],MODEL_FINAL_ANSWER_KEY:answer}
            score,reason=evaluator.score_response(dp)
            freeze(target,{'id':row['id'],'label':row['label'],'phase':job['phase'],'condition':job['condition'],
                'repeat':job['repeat'],'answer':answer,'score':score,'score_explanation':reason})
            print(f"COMPLETE {job['key']} score={score}",flush=True)
        except BaseException as e:
            stop.set();freeze(loc/'FAILED.json',{'type':type(e).__name__,'message':str(e)});raise
    status('running')
    try:
        # Finish rejudging first; do not let queued generation obscure diagnostic progress.
        for phase in ['rejudge','repeat','diagnostic']:
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                futures=[pool.submit(process,j) for j in jobs if j['phase']==phase]
                for f in as_completed(futures):f.result();status('running')
        s=status('complete');freeze(out/f'metrics_{a.phase}.json',s)
    except BaseException:status('failed');raise

if __name__=='__main__':main()
