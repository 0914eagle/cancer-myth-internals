"""Summarize the completed independent boundary audit; reject incomplete coverage."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/audits/correction_boundaries_20261001_v1'
OUT=ROOT/'docs/reviews/boundary_followup_2026-10-01'


def main():
    jobs=json.loads((RUN/'jobs.json').read_text())
    mapping={r['key']:r for r in json.loads((RUN/'private_mapping.json').read_text())}
    inputs={r['key']:r for r in json.loads((RUN/'blind_inputs.json').read_text())}
    results=[]; pending=[]
    for j in jobs:
        p=RUN/'records'/j['job']/'record.json'
        if not p.exists():pending.append(j['job']);continue
        r=json.loads(p.read_text())
        if r['status']!='complete':pending.append(j['job']);continue
        for a in r['result']:
            results.append(dict(**mapping[a['key']],assessment=a,
                question=inputs[a['key']]['question'],answer=inputs[a['key']]['answer']))
    if pending:
        raise SystemExit(f'Incomplete: {len(results)}/598. No partial-cohort rate reported; {len(pending)} jobs remaining/invalid.')
    assert len(results)==len({r['key'] for r in results})==598
    fpq=defaultdict(dict);nfps=defaultdict(list)
    for r in results:
        if r['cohort']=='fpq178': fpq[r['model'],r['id']][r['side']]=r
        else:nfps[r['model'],r['cohort']].append(r)
    assert len(fpq)==178 and all(set(x)=={'plain','alternative'} for x in fpq.values())
    by_model={}
    for model in sorted({k[0] for k in fpq}):
        pairs=[v for (m,q),v in fpq.items() if m==model]
        by_model[model]=dict(n_pairs=len(pairs),
            plain_relations=Counter(v['plain']['assessment']['relation'] for v in pairs),
            alternative_relations=Counter(v['alternative']['assessment']['relation'] for v in pairs),
            transitions=Counter(v['plain']['assessment']['relation']+' -> '+v['alternative']['assessment']['relation'] for v in pairs),
            plain_reacceptance=Counter(v['plain']['assessment']['later_reacceptance'] for v in pairs),
            alternative_reacceptance=Counter(v['alternative']['assessment']['later_reacceptance'] for v in pairs))
    nfp=[]
    for (model,cohort),rr in sorted(nfps.items()):
        nfp.append(dict(model=model,cohort=cohort,n=len(rr),
            stance=Counter(r['assessment']['stance'] for r in rr),
            scope=Counter(r['assessment']['scope'] for r in rr)))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'independent_audit_summary.json').write_text(json.dumps(dict(fpq=by_model,nfp=nfp,
        n_answers=598,warning='Independent AI text audit; no clinical gold validation or Well relabeling.'),ensure_ascii=False,indent=2)+'\n')
    (OUT/'independent_audit_full.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(complete=598,fpq_pairs=178,nfp_answers=242,output=str(OUT)),ensure_ascii=False))


if __name__=='__main__':main()
