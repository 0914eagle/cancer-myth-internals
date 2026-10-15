#!/usr/bin/env python3
"""Saved-output audit; never generates answers, changes scores, or calls a judge."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=['A_saved','B','C0','C1']
def read(p): return json.loads(p.read_text())
def route(vs):
    if any(v['verdict']=='false' and v['claim_relation'] in ['explicit','implied'] for v in vs): return 'false_in_scope'
    if any(v['verdict']=='false' for v in vs): return 'false_only_changed_or_unclear'
    if any(v['verdict']=='unknown' for v in vs): return 'no_false_has_unknown'
    return 'all_supported'
def state(r): return [(v['verdict'],v['claim_relation']) for v in r['verdicts']]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=ROOT/'results/fpqa_prompting/well_uncertainty_luna6_20261004_v1')
    p.add_argument('--out',type=Path,default=ROOT/'docs/reviews/well_uncertainty_failure_analysis_2026-10-04')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    assert read(a.run/'status.json')['state']=='complete'
    ms={};hashes={}
    for m in METHODS:
        files=sorted(a.run.glob(f'items/*/{m}/record.json'))
        ms[m]={read(f)['id']:read(f) for f in files}
        assert len(ms[m])==199
        hashes.update({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    assert all(set(ms[m])==set(ms['B']) for m in METHODS)
    gold={x['id']:x for x in map(json.loads,(ROOT/'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl').read_text().splitlines())}
    groups={}
    comparisons={}
    for x,y in [('A_saved','B'),('B','C1'),('C0','C1')]:
        for label,name in [(1,'fpq'),(0,'nfp')]:
            ids=[i for i,r in ms[x].items() if r['label']==label]
            gain=[i for i in ids if ms[x][i]['score']<4<=ms[y][i]['score']]
            loss=[i for i in ids if ms[y][i]['score']<4<=ms[x][i]['score']]
            k=f'{name}_{x}_to_{y}'
            groups[k+'_gain']=sorted(gain);groups[k+'_loss']=sorted(loss)
            n=len(gain)+len(loss)
            pv=min(1.,2*sum(math.comb(n,j) for j in range(min(len(gain),len(loss))+1))/2**n) if n else 1.
            comparisons[k]={'n':len(ids),'gain':len(gain),'loss':len(loss),'net':len(gain)-len(loss),
                'exact_mcnemar_p_unadjusted':pv,
                'gains_same_verdict_relation':sum(state(ms[x][i])==state(ms[y][i]) for i in gain) if x!='A_saved' else None,
                'losses_same_verdict_relation':sum(state(ms[x][i])==state(ms[y][i]) for i in loss) if x!='A_saved' else None}
    summary={'groups':groups,'comparisons':comparisons,'methods':{}}
    for m in METHODS:
        d={}
        for label,name in [(1,'fpq'),(0,'nfp')]:
            rr=[r for r in ms[m].values() if r['label']==label]
            entry={'n':len(rr),'ge4':sum(r['score']>=4 for r in rr)}
            if m!='A_saved':
                entry['claim_verdicts']=dict(Counter(v['verdict'] for r in rr for v in r['verdicts']))
                entry['claim_relations']=dict(Counter(v['claim_relation'] for r in rr for v in r['verdicts']))
                entry['routes']={t:{'n':sum(route(r['verdicts'])==t for r in rr),
                    'ge4':sum(route(r['verdicts'])==t and r['score']>=4 for r in rr)} for t in
                    ['false_in_scope','false_only_changed_or_unclear','no_false_has_unknown','all_supported']}
            d[name]=entry
        summary['methods'][m]=d
    lost=groups['fpq_A_saved_to_B_loss']
    summary['fpq_21_B_routes']=dict(Counter(route(ms['B'][i]['verdicts']) for i in lost))
    summary['fpq_21_B_has_changed']=sum(any(v['claim_relation']=='changed' for v in ms['B'][i]['verdicts']) for i in lost)
    summary['fpq_B_C1_gains_with_false']=sum(any(v['verdict']=='false' for v in ms['C1'][i]['verdicts']) for i in groups['fpq_B_to_C1_gain'])
    selected=set(lost+groups['nfp_B_to_C1_loss']+groups['fpq_B_to_C1_loss'])
    assert len(selected)==37
    notes_path=a.out/'manual_notes.jsonl'
    if notes_path.exists():
        notes=[json.loads(s) for s in notes_path.read_text().splitlines()]
        assert {n['id'] for n in notes}==selected
        for n in notes:
            for q in n.get('quotes',[]):
                assert q['text'] in ms[q['method']][n['id']][q['field']],(n['id'],q)
        summary['manual_notes']=len(notes)
    evidence=[];book=[]
    for i in sorted(ms['B']):
        for m in METHODS:
            assert ms[m][i]['question']==gold[i]['question']
            assert ms[m][i]['claims']==ms['B'][i]['claims']
        retrieved=[read(f) for f in sorted((a.run/'items'/i/'retrieval').glob('*.json'))]
        evidence.append({'id':i,'annotation':gold[i],**{m:ms[m][i] for m in METHODS},'retrieved':retrieved})
        if i in selected:
            book.extend([f'## {i}',gold[i]['question'],'주석: '+json.dumps(gold[i]['presuppositions'],ensure_ascii=False),
                         '교정 설명: '+str(gold[i]['reference_answer'])])
            for m in METHODS:
                r=ms[m][i];book.extend([f'### {m} 점수 {r["score"]}',
                    json.dumps(r.get('verdicts',r.get('factcheck_results')),ensure_ascii=False,indent=2),r['answer'],r['score_explanation']])
    (a.out/'evidence.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in evidence))
    (a.out/'casebook.md').write_text('\n\n'.join(book)+'\n')
    (a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (a.out/'source_manifest.json').write_text(json.dumps({'files':hashes,'run_sha256':hashlib.sha256((a.run/'run.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'All 199 matched records counted; direct answer reading documented for 37 unique selected questions. No new calls or score overrides.'},indent=2)+'\n')
    print(json.dumps({'matched':199,'direct_read_unique':37,'fpq_21_B_routes':summary['fpq_21_B_routes'],
                     'fpq_gains_new_false':summary['fpq_B_C1_gains_with_false']},ensure_ascii=False))
if __name__=='__main__':main()
