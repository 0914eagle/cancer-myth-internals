#!/usr/bin/env python3
"""Read-only-inference summary with paired question bootstrap, never pooled pseudo-items."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import random
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fpqa_prompting/well_reliability_20261005_v1'
DOC=ROOT/'docs/reviews/well_reliability_2026-10-05'
def read(p):return json.loads(p.read_text())
def estimate(deltas):
    if not deltas:return None
    rng=random.Random(20261005);n=len(deltas)
    boot=sorted(sum(rng.choices(deltas,k=n))/n for _ in range(10000))
    return {'n_questions':n,'mean_paired_difference':statistics.mean(deltas),
        'question_bootstrap_95ci':[boot[249],boot[9749]],
        'note':'Paired resampling of question-level mean differences, conditional on observed runs. Three runs do not characterize all run-level variability.'}

def main():
    plan=read(OUT/'run.json');rows={r['id']:r for r in plan['rows']}
    records=defaultdict(dict)
    for i in rows:
        for name,condition,rep in [('Plain1','Plain',1),('B1','B',1),('B2','B',2),('DE1','DE',1)]:
            records[condition].setdefault(i,{})[rep]=read(OUT/'historical'/i/f'{name}.json')
    for p in (OUT/'jobs/repeat').glob('*/*/*/record.json'):
        r=read(p);records[r['condition']].setdefault(r['id'],{})[r['repeat']]=r
    comparisons={}
    for before,after in [('Plain','B'),('Plain','DE'),('B','DE')]:
        comparisons[f'{before}->{after}']={}
        for label,name in [(1,'fpq'),(0,'nfp')]:
            ids=[i for i in rows if rows[i]['label']==label and len(records[before].get(i,{}))==3 and len(records[after].get(i,{}))==3]
            d=[statistics.mean(r['score']>=4 for r in records[after][i].values())-statistics.mean(r['score']>=4 for r in records[before][i].values()) for i in ids]
            comparisons[f'{before}->{after}'][name]={'complete_question_count':len(ids),'expected':99 if label else 100,
                'estimate':estimate(d) if len(ids)==(99 if label else 100) else None}
    noise=[]
    for i in plan['audit_ids']:
        for origin in ['B1','B2']:
            scores=[read(OUT/'historical'/i/f'{origin}.json')['score']]
            for rep in [1,2]:
                p=OUT/'jobs/rejudge'/i/origin/str(rep)/'record.json'
                if p.exists():scores.append(read(p)['score'])
            noise.append({'id':i,'origin':origin,'selection':'flipped' if i in plan['flipped_ids'] else 'stable_control',
                'scores':scores,'complete':len(scores)==3,'threshold_changed':len(set(s>=4 for s in scores))>1})
    review=read(DOC/'manual_flip_review.json')
    summary={'response_repetitions':{c:{str(rep):{name:{'n':len(xs),'ge4':statistics.mean(x['score']>=4 for x in xs),
        'mean_well':statistics.mean(x['score'] for x in xs)}
        for label,name in [(1,'fpq'),(0,'nfp')] if (xs:=[rs[rep] for i,rs in byid.items() if rep in rs and rows[i]['label']==label])}
        for rep in [1,2,3]} for c,byid in records.items()},
        'paired_comparisons':comparisons,'rescore_records':noise,
        'rescore_completed_answers':sum(x['complete'] for x in noise),
        'manual_review_counts':dict(Counter(r['category'] for r in review['cases'])),
        'manual_by_label':{label:dict(Counter(r['category'] for r in review['cases'] if r['id'].startswith(label))) for label in ['fpq','nfp']}}
    if (OUT/'diagnostic_jobs.json').exists():
        diag=read(OUT/'diagnostic_jobs.json');dr=defaultdict(lambda:defaultdict(dict))
        for path in (OUT/'jobs/diagnostic').glob('*/*/*/record.json'):
            r=read(path);dr[r['condition']][r['id']][r['repeat']]=r
        dr['model']= {i:records['B'][i] for i in diag['sources']}
        summary['diagnostic']={'reviewed_cases':diag['case_count'],'fpq':diag['fpq'],'nfp':diag['nfp'],
            'excluded':diag['excluded_count'],'comparisons':{},
            'note':'Reference-aligned developmental pilot; not validated medical oracle. All repeats required before paired estimates. Content arm shares NFP reviewed outputs.'}
        for before,after in [('model','reviewed'),('reviewed','reviewed_plus_correction')]:
            for label,name in [(1,'fpq'),(0,'nfp')]:
                expected=[i for i in diag['sources'] if rows[i]['label']==label]
                alias='reviewed' if label==0 and after=='reviewed_plus_correction' else after
                ids=[i for i in expected if len(dr[before].get(i,{}))==3 and len(dr[alias].get(i,{}))==3]
                d=[statistics.mean(r['score']>=4 for r in dr[alias][i].values())-statistics.mean(r['score']>=4 for r in dr[before][i].values()) for i in ids]
                summary['diagnostic']['comparisons'][f'{before}->{after}/{name}']={'complete':len(ids),'expected':len(expected),
                    'estimate':estimate(d) if len(ids)==len(expected) else None,'same_arm_alias':alias!=after}
    (OUT/'reliability_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    audit={r['id']:r for r in read(OUT/'audit_inputs.json')}
    lines=['# 동일 검증 결과에서 답변·점수가 바뀐 36쌍 직접 대조','',
        '모델: GPT-6 Luna medium. 조건: B 삼분 판정의 저장된 동일 주장·판정·이유. 두 실행의 입력 199개를 전부 대조해 동일함을 확인. 점수는 원래 Well 채점 결과이며 아래 독해로 바꾸지 않음.',
        '', '선정: 4점 경계를 넘은 FPQ 12쌍, NFP 24쌍. 이는 전환 사례의 구성이지 전체 오류율이 아님. 아래 분류는 주 에이전트의 직접 독해이며 독립 의료 검증이나 확정적인 오채점 판정이 아님.','']
    for r in review['cases']:
        a=audit[r['id']];lines += [f"## {r['id']}",'',a['row']['question'],'',f"- 주된 차이: {r['category']}",f"- 독해: {r['note']}",'']
        for s in ['B1','B2']:lines += [f"### {s}: 기존 점수 {a['answers'][s]['score']}",'',a['answers'][s]['answer'],'']
    (DOC/'casebook.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:summary[k] for k in ['rescore_completed_answers','manual_review_counts','manual_by_label']},ensure_ascii=False))

if __name__=='__main__':main()
