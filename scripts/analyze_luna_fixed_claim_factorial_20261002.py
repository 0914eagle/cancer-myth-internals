"""Paired, source-cluster bootstrap analysis of the frozen Luna 2x2 run."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts import run_luna_fixed_claim_factorial_20261001 as runner

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/frontier_cli/luna_fixed_claim_factorial_20261001_v1'
OUT = ROOT/'docs/reviews/luna_fixed_claim_factorial_2026-10-02'
CONDITIONS = runner.CONDITIONS
DESIGN = {'fpq_2','fpq_30','nfp_1005','nfp_1133'}
PAIRS = [('control','content'),('control','criterion'),('control','both'),
         ('criterion','both'),('content','both')]
B = 20000


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def csvwrite(name, rows):
    if not rows:
        return
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def bootstrap(ids, values, source, rng):
    groups = sorted({source[i]['group_id'] for i in ids})
    pos = {g:j for j,g in enumerate(groups)}
    sizes = np.zeros(len(groups)); totals = np.zeros(len(groups))
    for i, v in zip(ids, values):
        j = pos[source[i]['group_id']]
        sizes[j] += 1; totals[j] += v
    # Resample whole source groups, retaining paired condition differences.
    draws = rng.integers(len(groups), size=(B, len(groups)))
    means = totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)
    return dict(estimate_pp=float(np.mean(values)*100),
                ci95_pp=(np.quantile(means,[.025,.975])*100).tolist(),
                source_groups=len(groups)), means


def witness(x):
    return x['assessment']=='false' and x['question_relation']=='relied_on'


def witness_change(a, b):
    kinds=set()
    for x,y in zip(a['claim_assessments'],b['claim_assessments']):
        assert x['claim_id']==y['claim_id']
        if not witness(y) or witness(x): continue
        if x['assessment']=='false': kinds.add('relation_only')
        elif x['question_relation']=='relied_on': kinds.add('assessment_only')
        else:kinds.add('assessment_and_relation')
    if not kinds:
        kinds.add('already_qualifying_candidate' if any(witness(x) for x in a['claim_assessments'])
                  else 'no_qualifying_candidate_after')
    return '+'.join(sorted(kinds))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source={r['id']:r for r in json.loads((RUN/'evaluation_questions.json').read_text())}
    plan=json.loads((RUN/'plan.json').read_text())
    records={c:{} for c in CONDITIONS}; failed=[]; hashes={}; discrepancies=[]
    for c in CONDITIONS:
        for i,q in source.items():
            path=RUN/c/i/'record.json'; rec=json.loads(path.read_text())
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            job=dict(condition=c,id=i,question=q['question'],candidates=q['candidates'])
            assert rec['prompt']==runner.prompt_for(plan,job), (c,i,'prompt changed')
            if rec['status']!='complete':
                failed.append(dict(condition=c,id=i,label=q['set'],error=rec.get('error')));continue
            value,warnings=runner.validate(json.dumps(rec['result']),job)
            assert warnings==rec['warnings']
            records[c][i]=value
            if warnings:
                discrepancies.append(dict(condition=c,id=i,label=q['set'],
                    gate=value['has_false_premise'],qualifying_candidates=sum(witness(x) for x in value['claim_assessments']),
                    warnings='|'.join(warnings)))
    common=sorted(set.intersection(*[set(records[c]) for c in CONDITIONS]))
    assert len(common)==721 and len(failed)==2
    rng=np.random.default_rng(20261002)
    rates=[]
    for scope,ii in [('all_valid',None),('four_way_common',common),('exclude_design_examples',[i for i in common if i not in DESIGN])]:
        for c in CONDITIONS:
            for label in ['fpq','nfp']:
                ids=[i for i in (records[c] if ii is None else ii) if source[i]['set']==label]
                yes=sum(records[c][i]['has_false_premise'] for i in ids)
                rates.append(dict(scope=scope,condition=c,label=label,n=len(ids),yes=yes,rate_pct=100*yes/len(ids)))
    contrasts=[]; transitions=[]; deltas=[]; changed=[]; group_results={}; signatures=[]
    for a,b in PAIRS:
        # Factor comparisons use the four-way complete cohort, per protocol.
        pair_ids=sorted(set(records[a])&set(records[b])) if a=='control' else common
        label_draws={}
        for label in ['fpq','nfp']:
            ids=[i for i in pair_ids if source[i]['set']==label]
            values=[int(records[b][i]['has_false_premise'])-int(records[a][i]['has_false_premise']) for i in ids]
            stats,draws=bootstrap(ids,values,source,rng);label_draws[label]=draws
            counts=Counter(f"{int(records[a][i]['has_false_premise'])}->{int(records[b][i]['has_false_premise'])}" for i in ids)
            contrasts.append(dict(before=a,after=b,label=label,n=len(ids),**stats,
                no_to_no=counts['0->0'],no_to_yes=counts['0->1'],yes_to_no=counts['1->0'],yes_to_yes=counts['1->1']))
            sig=Counter()
            for i in ids:
                x,y=records[a][i],records[b][i];gate=f"{int(x['has_false_premise'])}->{int(y['has_false_premise'])}"
                for u,v in zip(x['claim_assessments'],y['claim_assessments']):
                    transitions.append(dict(before=a,after=b,id=i,label=label,gate_transition=gate,claim_id=u['claim_id'],
                        assessment_before=u['assessment'],assessment_after=v['assessment'],
                        relation_before=u['question_relation'],relation_after=v['question_relation']))
                if gate=='0->1':sig[witness_change(x,y)]+=1
                if gate in {'0->1','1->0'}:
                    deltas.append(dict(before=a,after=b,id=i,label=label,transition=gate,
                        gain_signature=witness_change(x,y) if gate=='0->1' else '',
                        question=source[i]['question']))
            for k,n in sig.items():signatures.append(dict(before=a,after=b,label=label,gain_signature=k,n=n))
        jd=label_draws['fpq']-label_draws['nfp']
        two=contrasts[-2:]
        group_results[b+'-'+a]=dict(delta_J_pp=two[0]['estimate_pp']-two[1]['estimate_pp'],
                                    ci95_pp=(np.quantile(jd,[.025,.975])*100).tolist())
    interactions=[]
    for label in ['fpq','nfp']:
        ids=[i for i in common if source[i]['set']==label]
        vals=[int(records['both'][i]['has_false_premise'])-int(records['content'][i]['has_false_premise'])
              -int(records['criterion'][i]['has_false_premise'])+int(records['control'][i]['has_false_premise']) for i in ids]
        s,_=bootstrap(ids,vals,source,rng);interactions.append(dict(label=label,n=len(ids),**s))
    historical={r['id']:r for r in csv.DictReader((ROOT/'results/frontier_cli/luna_explicit_claim_gate_20261001_v1/historical_repeat_items.csv').open())}
    old=[]
    for label in ['fpq','nfp']:
        ids=[i for i in common if source[i]['set']==label and historical[i]['stable_yes_to_no']=='1']
        old.append(dict(label=label,n=len(ids),yes_by_condition={c:sum(records[c][i]['has_false_premise'] for i in ids) for c in CONDITIONS}))
    wide=[]
    for i in sorted(source):
        wide.append(dict(id=i,label=source[i]['set'],group_id=source[i]['group_id'],design_example=i in DESIGN,
            **{c:int(records[c][i]['has_false_premise']) if i in records[c] else '' for c in CONDITIONS}))
        valid={c:records[c][i] for c in CONDITIONS if i in records[c]}
        if len({v['has_false_premise'] for v in valid.values()})>1:
            changed.append(dict(id=i,label=source[i]['set'],group_id=source[i]['group_id'],
                question=source[i]['question'],candidates=source[i]['candidates'],
                reference=dict(premise=source[i].get('premise_text'),correction=source[i].get('correction')),results=valid))
    misses=[]
    for c in CONDITIONS:
        for label in ['fpq','nfp']:
            rr=[records[c][i] for i in common if source[i]['set']==label and not records[c][i]['has_false_premise']]
            misses.append(dict(condition=c,label=label,no_total=len(rr),
                no_false_candidate=sum(not any(x['assessment']=='false' for x in r['claim_assessments']) for r in rr),
                false_but_not_relied_on=sum(any(x['assessment']=='false' for x in r['claim_assessments']) and not any(witness(x) for x in r['claim_assessments']) for r in rr),
                qualifying_false_but_no=sum(any(witness(x) for x in r['claim_assessments']) for r in rr)))
    aggregates=[]
    for a,b in PAIRS:
        for label in ['fpq','nfp']:
            rr=[r for r in transitions if r['before']==a and r['after']==b and r['label']==label]
            counter=Counter((r['gate_transition'],r['assessment_before'],r['assessment_after'],r['relation_before'],r['relation_after']) for r in rr)
            for key,n in counter.items():
                aggregates.append(dict(before=a,after=b,label=label,gate_transition=key[0],assessment_before=key[1],assessment_after=key[2],relation_before=key[3],relation_after=key[4],candidates=n))
    for name,rr in [('rates.csv',rates),('paired_contrasts.csv',contrasts),('interactions.csv',interactions),
                    ('all_723_decisions.csv',wide),('failed_records.csv',failed),('gate_candidate_disagreements.csv',discrepancies),
                    ('changed_gate_contrasts.csv',deltas),('gain_signatures.csv',signatures),('candidate_transition_counts.csv',aggregates),
                    ('no_decision_diagnostics.csv',misses)]:
        csvwrite(name,rr)
    dump('delta_J.json',group_results);dump('historical_stable_switches.json',old)
    (OUT/'all_changed_questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changed))
    dump('design_examples.json',[dict(source=source[i],results={c:records[c][i] for c in CONDITIONS}) for i in sorted(DESIGN)])
    example_ids=['fpq_2','fpq_30','fpq_51','fpq_56','fpq_550','fpq_611','fpq_253','fpq_350','fpq_566','fpq_747','nfp_1001','nfp_1005','nfp_1075','nfp_1133']
    dump('focused_examples.json',[dict(source=source[i],results={c:records[c][i] for c in CONDITIONS}) for i in example_ids])
    dump('manifest.json',dict(input_sha256=hashes,plan_sha256=hashlib.sha256((RUN/'plan.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model=plan['model'],new_calls=0,
        complete=2890,failed=failed,common_questions=len(common),changed_questions=len(changed),
        bootstrap=dict(seed=20261002,replicates=B,unit='source group, paired by condition, within each label',interval='percentile 95%, unadjusted exploratory intervals'),
        warning='One draw per condition; not generation-variance CI or held-out method performance. J is one operating point, not AUROC. Candidate transitions are model self-reports, not recovered mechanisms. Historical subgroup is selected, not main evaluation.'))
    print(json.dumps(dict(common_rates=[r for r in rates if r['scope']=='four_way_common'],
        contrasts=contrasts,interactions=interactions,delta_J=group_results,changed_questions=len(changed),
        signatures=signatures,historical=old),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
