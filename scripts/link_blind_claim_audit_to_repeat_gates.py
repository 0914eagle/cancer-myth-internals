"""Join completed label-blind Terra wording audit to all 732 repeated Luna gates.

Uses preliminary claim types as annotations, not medical truth or causal labels.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
Q=ROOT/'results/audits/measurement_validity_20261001_v1/questions'
OLD=ROOT/'docs/reviews/core_failures_2026-10-01'
OUT=ROOT/'docs/reviews/boundary_followup_2026-10-01'


def savecsv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mapping={r['key']:r for r in json.loads((Q/'private_mapping.json').read_text())}
    questions={r['key']:r['question'] for r in json.loads((Q/'blind_inputs.json').read_text())}
    audited={}
    for job in json.loads((Q/'jobs.json').read_text()):
        r=json.loads((Q/'records'/job['job']/'record.json').read_text())
        assert r['status']=='complete'
        for a in r['result']:
            qid=mapping[a['key']]['id'];assert qid not in audited
            audited[qid]=a
            assert all(c['quote'] in questions[a['key']] for c in a['claims'])
    assert len(audited)==732
    rows=[]
    for r in csv.DictReader((OLD/'luna_repeat_items.csv').open()):
        a=audited[r['id']]
        b1,b2,c1,c2=[int(r[k]) for k in ['baseline_r1','baseline_r2','context_r1','context_r2']]
        group=('unstable' if b1!=b2 or c1!=c2 else
            'stable_yes_to_no' if b1 and not c1 else 'stable_no_to_yes' if c1 and not b1 else
            'stable_yes' if b1 else 'stable_no')
        types={c['claim_type'] for c in a['claims']}
        rows.append(dict(id=r['id'],dataset=r['dataset'],group=group,
            baseline_mean=(b1+b2)/2,context_mean=(c1+c2)/2,
            question=questions[a['key']],has_general='general_claim' in types,
            has_personal='individual_context' in types,has_inquiry='inquiry' in types,
            has_ambiguous='ambiguous' in types,
            apparent_error_flag=any(c['preliminary_status']=='apparent_error' for c in a['claims']),
            claims=json.dumps(a['claims'],ensure_ascii=False),ambiguity=a['ambiguity']))
    assert len(rows)==732
    savecsv('all_732_claim_gate_links.csv',rows)
    grouped=defaultdict(list)
    for r in rows:grouped[r['dataset'],r['group']].append(r)
    summary=[]
    for (ds,group),rr in sorted(grouped.items()):
        summary.append(dict(dataset=ds,group=group,n=len(rr),
            **{k:sum(r[k] for r in rr) for k in ['has_general','has_personal','has_inquiry','has_ambiguous','apparent_error_flag']}))
    savecsv('claim_types_by_gate_transition.csv',summary)
    pairs=[json.loads(l) for l in (ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl').open()]
    refs={r['id']:r['plain']['reference'] for r in pairs}
    stable=[dict(r,reference=json.dumps(refs[r['id']],ensure_ascii=False)) for r in rows if r['group']=='stable_yes_to_no']
    assert Counter(r['dataset'] for r in stable)=={'fpq':103,'nfp':25}
    savecsv('stable_128_questions_claims_references.csv',stable)
    note=dict(n=732,claim_auditor='gpt-5.6-terra',gate_model='gpt-5.6-luna',
        gate_conditions='Direct baseline vs personal-context protection, each repeated twice',
        new_calls=0,claim_inputs='Question only; labels, references, answers and scores hidden.',
        limitations=['Claim types are AI wording annotations, not clinical ground truth.',
            'No general_claim tag does not prove absence of a false premise.',
            'Gate transition strata were selected after observing results; descriptive associations only.',
            'Personal-context and general-claim categories can co-occur.'])
    sources=[Q/'plan.json',Q/'jobs.json',Q/'private_mapping.json',Q/'blind_inputs.json',
        OLD/'luna_repeat_items.csv',ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl',
        Path(__file__)]
    sources += [Q/'records'/j['job']/'record.json' for j in json.loads((Q/'jobs.json').read_text())]
    note['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'claim_link_protocol.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__': main()
