#!/usr/bin/env python3
"""Make reference-aligned diagnostic requests only from direct reviewed claim labels.

Keeps fixed B claim strings, exact response system/examples/action and original judge.
Model-verdict comparator uses B's three response repetitions, not new duplicate calls.
NFP corrected+content equals corrected by design and shares responses, not fake repeats.
"""
import json
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fpqa_prompting import freeze
from src.fpqa_uncertainty import parse_verdict

def make_messages(original, claims, verdicts, correction=None, target_indices=()):
    messages=json.loads(json.dumps(original))
    prefix, tail=messages[1]['content'].split('\nFeedback: ',1)
    old, action=tail.rsplit('\nAction: ',1)
    feedback=json.loads(old)
    if [r['claim'] for r in feedback]!=claims:raise ValueError('Original feedback differs from reviewed claims')
    replacement=[]
    for i,(claim,verdict) in enumerate(zip(claims,verdicts)):
        parse_verdict(json.dumps(verdict))
        item={'claim':claim,**verdict}
        if correction is not None and i in target_indices:item['reference_correction']=correction
        replacement.append(item)
    messages[1]['content']=prefix+'\nFeedback: '+json.dumps(replacement,ensure_ascii=False)+'\nAction: '+action
    return messages

def main():
    out=ROOT/'results/fpqa_prompting/well_reliability_20261005_v1'
    path=ROOT/'docs/reviews/well_reliability_2026-10-05/diagnostic_claim_review.json'
    review=json.loads(path.read_text());plan=json.loads((out/'run.json').read_text())
    rows={r['id']:r for r in plan['rows']}
    audits={r['id']:r for r in json.loads((out/'audit_inputs.json').read_text())}
    cases={r['id']:r for r in review['cases']}
    if len(cases)!=len(review['cases']) or set(cases)&set(review['excluded']):raise ValueError('Duplicate inclusion/exclusion')
    if set(cases)|set(review['excluded'])!=set(plan['audit_ids']):raise ValueError('Audit cohort not fully accounted for')
    requests={j['id']:j['messages'] for j in plan['jobs'] if j['phase']=='repeat' and j['condition']=='B'}
    jobs=[];sources={}
    for i,c in cases.items():
        a=audits[i];claims=a['claims']
        if len(c['claims'])!=len(claims):raise ValueError(f'{i}: claim count mismatch')
        vs=[{'verdict':v,'claim_relation':rel,'personal_report':personal,'reason':review['reason_templates'][reason],'source_ids':[]}
            for v,rel,personal,reason in c['claims']]
        target=c.get('target_indices',[])
        if rows[i]['label'] and not target:raise ValueError('Included FPQ must have reviewed target')
        if any(not 0<=k<len(claims) or vs[k]['verdict']!='false' for k in target):raise ValueError('Invalid target mapping')
        conditions=['reviewed','reviewed_plus_correction'] if rows[i]['label'] else ['reviewed']
        for condition in conditions:
            msgs=make_messages(requests[i],claims,vs,
                rows[i]['reference_answer'] if condition=='reviewed_plus_correction' else None,target)
            for repeat in [1,2,3]:jobs.append({'key':f'diagnostic/{i}/{condition}/{repeat}',
                'phase':'diagnostic','id':i,'condition':condition,'repeat':repeat,'messages':msgs})
        sources[i]={'claims':claims,'reviewed_verdicts':vs,'review':c,
            'model_verdict_comparator':['historical/B1','historical/B2','repeat/B/3'],
            'content_arm_alias':None if rows[i]['label'] else 'reviewed (identical information; no extra generation)'}
    jobs.sort(key=lambda j:hashlib.sha256(('diagnostic-order-v1:'+j['key']).encode()).hexdigest())
    result={'protocol':review['protocol'],'review_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'case_count':len(cases),'fpq':sum(rows[i]['label'] for i in cases),'nfp':sum(not rows[i]['label'] for i in cases),
        'excluded_count':len(review['excluded']),'scope':review['qualification'],
        'fixed':'B claim strings, response system/few-shot/action and judge; only claim feedback changes.',
        'changing':'Reference-aligned verdict/relation/context feedback, then correction content. Not a pure one-token-verdict manipulation.',
        'selection':review['selection'],'sources':sources,'excluded':review['excluded'],'jobs':jobs}
    freeze(out/'diagnostic_jobs.json',result)
    print(json.dumps({k:result[k] for k in ['case_count','fpq','nfp','excluded_count']}));print('fresh answer/judge pairs',len(jobs))

if __name__=='__main__':main()
