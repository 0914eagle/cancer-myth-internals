"""Freeze completed Claude answer audits and analyze within-question changes offline."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from scripts import audit_answer_behavior_full as a


def summary(rows):
    return dict(n=len(rows), well_ge4=sum(r['score'] >= 4 for r in rows),
                stance=dict(Counter(r['coding']['stance'] for r in rows)),
                response=dict(Counter(r['coding']['response'] for r in rows)),
                grounding=dict(Counter(r['coding']['target_grounding'] for r in rows)),
                alignment=dict(Counter(r['coding']['reference_alignment']['status'] for r in rows)),
                flags=dict(Counter(f['tag'] for r in rows for f in r['coding']['flags'])))


def pair_summary(pairs):
    return dict(n=len(pairs),
        well=dict(Counter(f"{int(p['plain']['score']>=4)}->{int(p['answer']['score']>=4)}" for p in pairs)),
        stance=dict(Counter(p['plain']['coding']['stance']+'->'+p['answer']['coding']['stance'] for p in pairs)),
        alignment=dict(Counter(p['plain']['coding']['reference_alignment']['status']+'->'+p['answer']['coding']['reference_alignment']['status'] for p in pairs)),
        response=dict(Counter(p['plain']['coding']['response']+'->'+p['answer']['coding']['response'] for p in pairs)),
        plain=summary([p['plain'] for p in pairs]), answer=summary([p['answer'] for p in pairs]))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); out=args.out
    out.mkdir(parents=True,exist_ok=True)
    inputs=json.loads((a.OUT/'inputs.json').read_text()); ix={r['key']:r for r in inputs}
    meta={r['key']:r for r in json.loads((a.OUT/'manifest.json').read_text())}
    old=json.loads((a.ROOT/'results/analysis/answer_behavior_partial_1404_20260930/snapshot.json').read_text())
    if (out/'snapshot.json').exists():
        snap=json.loads((out/'snapshot.json').read_text())
    else:
        files=sorted((a.OUT/'batches').glob('*/result.json'))
        snap={'sources':{str(f):a.sha(f) for f in files},'input_sha256':a.sha(a.OUT/'inputs.json'),
              'manifest_sha256':a.sha(a.OUT/'manifest.json'),'previous_snapshot_n':old['valid']}
        a.write(out/'snapshot.json',snap,True)
    assert snap['input_sha256']==a.sha(a.OUT/'inputs.json')
    assert snap['manifest_sha256']==a.sha(a.OUT/'manifest.json')
    codes={}
    for name,h in snap['sources'].items():
        p=Path(name); assert a.sha(p)==h
        obj=json.loads(p.read_text()); bi=int(p.parent.name); batch=inputs[bi*3:bi*3+3]
        assert obj['prompt_sha256']==a.digest(a.PROMPT+json.dumps(batch,ensure_ascii=False))
        validated,repairs=a.validate(obj['items'],batch); assert not repairs
        native=json.loads((Path(obj['attempt'])/'stdout.json').read_text())
        assert native.get('subtype')=='success' and not native.get('is_error') and native.get('stop_reason')=='end_turn'
        assert any(m==a.MODEL or m.startswith(a.MODEL+'-') for m in native['modelUsage'])
        for r in validated:
            assert r['key'] not in codes
            codes[r['key']]=r
    rows=[dict(meta[k],coding=c,is_new=k not in old['keys']) for k,c in sorted(codes.items())]
    grouped=defaultdict(dict)
    for r in rows: grouped[r['pair_key']][r['side']]=r
    pairs=[dict(pair_key=k,**v) for k,v in sorted(grouped.items()) if set(v)=={'plain','answer'}]
    strata=[]
    for model in a.PRIMARY:
        for ds in ['fpq','nfp']:
            for side in ['plain','answer']:
                rr=[r for r in rows if (r['model'],r['dataset'],r['side'])==(model,ds,side)]
                strata.append(dict(model=model,dataset=ds,side=side,method='plain' if side=='plain' else a.PRIMARY[model],
                                   new_n=sum(r['is_new'] for r in rr),**summary(rr)))
    paired=[]
    for model in a.PRIMARY:
        for ds in ['fpq','nfp']:
            pp=[p for p in pairs if p['plain']['model']==model and p['plain']['dataset']==ds]
            paired.append(dict(model=model,dataset=ds,method=a.PRIMARY[model],**pair_summary(pp)))
    transitions={}
    for ds in ['fpq','nfp']:
        for start in [False,True]:
            for end in [False,True]:
                pp=[p for p in pairs if p['plain']['dataset']==ds and (p['plain']['score']>=4)==start and (p['answer']['score']>=4)==end]
                transitions[f'{ds}_{int(start)}->{int(end)}']=pair_summary(pp)
    mismatches={}
    for ds in ['fpq','nfp']:
        for side in ['plain','answer']:
            rr=[r for r in rows if r['dataset']==ds and r['side']==side and r['score']<4]
            mismatches[ds+'_'+side+'_low']=summary(rr)
    report=dict(answers=len(rows),expected=len(meta),new_answers=sum(r['is_new'] for r in rows),
                newly_completed_pairs=sum(p['plain']['is_new'] or p['answer']['is_new'] for p in pairs),
                primary_conditions=a.PRIMARY,strata=strata,paired=paired,transitions=transitions,low_score_behavior=mismatches,
                limits=['Incomplete frozen subset, not final population prevalence or controlled model ranking.',
                        '4 model families, 5 settings; Qwen3.8 thinking OFF and ON are separate.',
                        'Use paired comparisons; unpaired completion cohorts differ.',
                        'AI codes can disagree with Well or misread scope. Neither is independent clinical ground truth.',
                        'Descriptive answer behavior, not gate verdicts or internal causal evidence.'])
    # Keep pair count separate from paired tables.
    report['completed_pairs']=len(pairs)
    a.write(out/'analysis.json',report)
    a.write(out/'codings_snapshot.json',rows)
    (out/'pairs.jsonl').write_text(''.join(json.dumps(dict(p,question=ix[p['plain']['key']]['question'],
        reference=ix[p['plain']['key']]['reference'],plain_text=ix[p['plain']['key']]['answer'],
        answer_text=ix[p['answer']['key']]['answer']),ensure_ascii=False)+'\n' for p in pairs))
    with (out/'coverage.csv').open('w') as f:
        cols=['model','dataset','side','method','n','new_n','well_ge4']; w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        w.writerows({k:r[k] for k in cols} for r in strata)
    print(json.dumps({k:report[k] for k in ['answers','new_answers','completed_pairs','newly_completed_pairs','primary_conditions']},indent=2))


if __name__=='__main__':main()
