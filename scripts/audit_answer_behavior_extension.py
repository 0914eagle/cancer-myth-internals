"""Audit all saved answer conditions missing from the original full audit.

Do not resubmit the original 7,318 answers. Start after that runner completes.
Uses exactly the existing reference-assisted answer coding prompt/validator.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import time

from scripts import audit_answer_behavior_full as transport

ROOT=Path(__file__).resolve().parents[1]
PRIMARY=transport.OUT
OUT=ROOT/'results/audits/answer_behavior_extension_20260930_v1'
SOURCE=ROOT/'results/analysis/full_behavior_census_20260930_v2/individual_answers.jsonl'


def prepare():
    original=json.loads((PRIMARY/'manifest.json').read_text())
    known={(r['model'],r['id'],'plain' if r['side']=='plain' else r['method']) for r in original}
    assert len(known)==7318
    inputs=[];manifest=[];seen=set()
    for line in SOURCE.read_text().splitlines():
        r=json.loads(line);identity=(r['model'],r['id'],r['method'])
        assert identity not in seen;seen.add(identity)
        if identity in known:continue
        key='x_'+transport.digest('/'.join(identity))[:20]
        ref=r['reference']
        reference=(dict(kind='target_error',premise=ref['premise'] or '',correction=ref['correction'] or '')
                   if r['dataset']=='fpq' else dict(kind='normal_question_context',premises=ref['tpq_premises']))
        inputs.append(dict(key=key,question=r['question'],answer=r['answer'],reference=reference))
        manifest.append(dict(key=key,model=r['model'],id=r['id'],method=r['method'],dataset=r['dataset'],
                             score=r['score'],source_job=r['job'],answer_sha256=r['answer_sha256']))
    assert known<=seen and len(seen)==13906 and len(inputs)==6588
    inputs.sort(key=lambda r:transport.digest('extension-order:'+r['key']))
    # Never place two answers to the same question in the same request.
    meta={r['key']:r for r in manifest};pending=list(inputs);ordered=[]
    while pending:
        batch=[];qids=set()
        for item in list(pending):
            q=meta[item['key']]['id']
            if q in qids:continue
            batch.append(item);qids.add(q);pending.remove(item)
            if len(batch)==3:break
        assert len(batch)==min(3,len(pending)+len(batch))
        ordered.extend(batch)
    inputs=ordered;manifest=[meta[r['key']] for r in inputs]
    plan=dict(n=len(inputs),source=str(SOURCE),source_sha256=transport.sha(SOURCE),
        existing_excluded=len(known),total_saved_answers=len(seen),model=transport.MODEL,
        prompt=transport.PROMPT,system=transport.SYSTEM,batch_size=3,workers=8,
        counts=[dict(model=k[0],method=k[1],dataset=k[2],n=v) for k,v in
                sorted(Counter((r['model'],r['method'],r['dataset']) for r in manifest).items())],
        run_after=str(PRIMARY/'status.json'),
        limits=['AI coding, not clinical truth or internal causal explanation.',
                'All available saved conditions, not a new matched-prompt model experiment.',
                'Question/reference/one answer supplied; model/method/scores/decisions hidden.'])
    for name,value in [('plan.json',plan),('inputs.json',inputs),('manifest.json',manifest)]:
        transport.write(OUT/name,value,True)
    return inputs,manifest


def report(inputs,manifest,results,state,**extra):
    status=dict(state='complete' if len(results)==len(inputs) else state,complete=len(results)==len(inputs),
                valid=len(results),expected=len(inputs),remaining=len(inputs)-len(results),pid=os.getpid(),at=time.time(),**extra)
    transport.write(OUT/'status.json',status)
    transport.write(OUT/'codings.json',list(results.values()))
    counts=Counter((m['model'],m['method'],m['dataset']) for m in manifest if m['key'] in results)
    transport.write(OUT/'coverage.json',dict(**status,counts=[dict(model=k[0],method=k[1],dataset=k[2],n=v) for k,v in counts.items()]))
    return status


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run-after-primary']);args=ap.parse_args()
    inputs,manifest=prepare()
    if args.action=='prepare':print(json.dumps(dict(expected=len(inputs),out=str(OUT))));return
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/'STOP').exists():
            try:state=json.loads((PRIMARY/'status.json').read_text())
            except json.JSONDecodeError:
                time.sleep(5);continue
            if state.get('complete') and state.get('valid')==7318:break
            report(inputs,manifest,{},'waiting_existing_primary',primary_valid=state.get('valid'),primary_state=state.get('state'))
            time.sleep(60)
        if (OUT/'STOP').exists():
            report(inputs,manifest,{},'stopped_by_user');return
        transport.OUT=OUT;transport.report=report
        transport.run(inputs,manifest,8)


if __name__=='__main__':main()
