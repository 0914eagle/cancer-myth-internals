"""Validate manual two-axis records against saved originals and aggregate them.

This script does not assign semantic labels or infer clinical truth.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/reviews/k3_behavior_scope_2026-10-04'
MASTER = ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
REL = ('explicit','contextual_premise','strengthened_or_changed','ambiguous','not_applicable')
CONTENT = ('reported_personal_circumstance','user_inference_or_belief','general_claim',
           'request_or_option','personal_or_inference_unclear','none')

def write_csv(name, rows):
    with (OUT/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def main():
    master = {(p['model'],p['id']): p for p in map(json.loads,MASTER.open())}
    paths = [OUT/'two_axis_review_records.json', OUT/'nfp17_recovery_success_reading.json']
    low, high = [json.loads(p.read_text()) for p in paths]
    keys = set()
    for r in low+high:
        key=(r['model'],r['id'],r['side']); assert key not in keys;keys.add(key)
        a = master[key[:2]][key[2]]
        assert r['question']==a['question']
        assert r['answer_sha256']==a['answer_sha256']==hashlib.sha256(a['answer'].encode()).hexdigest()
        assert r['answer_evidence'] in a['answer']
        assert r['relation'] in REL and r['content'] in CONTENT
        assert not r['medical_truth_adjudicated'] and not r['well_rescored']
        r['original_well_score']=a['score']
    expected={(p['model'],p['id'],s) for p in master.values() if p['dataset']=='nfp'
              for s in ('plain','alternative') if p[s]['score']<4}
    assert {(r['model'],r['id'],r['side']) for r in low}==expected and len(low)==230
    recovery={(p['model'],p['id']) for p in master.values() if p['dataset']=='nfp'
              and p['plain']['score']<4 and p['alternative']['score']>=4}
    assert {(r['model'],r['id']) for r in high}==recovery and len(high)==17
    census=[]
    for model,side in sorted({(r['model'],r['side']) for r in low}):
        rr=[r for r in low if (r['model'],r['side'])==(model,side)]
        rc=Counter(r['relation'] for r in rr);cc=Counter(r['content'] for r in rr)
        census.append(dict(model=model,side=side,method=rr[0]['method'],n=len(rr),
                           **{'relation_'+k:rc[k] for k in REL},**{'content_'+k:cc[k] for k in CONTENT}))
    write_csv('two_axis_low230_by_model.csv',census)
    crossed=Counter((r['relation'],r['content']) for r in low)
    write_csv('two_axis_low230_cross.csv',[dict(relation=a,content=b,n=n) for (a,b),n in sorted(crossed.items())])
    before={(r['model'],r['id'],r['side']):r for r in low}
    paired=[]
    for a in high:
        b=before[a['model'],a['id'],'plain']
        paired.append(dict(model=a['model'],id=a['id'],plain_score=b['original_well_score'],
                           alternative_score=a['original_well_score'],plain_relation=b['relation'],
                           alternative_relation=a['relation'],plain_content=b['content'],alternative_content=a['content'],
                           same_two_axes=(b['relation'],b['content'])==(a['relation'],a['content']),
                           note_ko=a['note_ko']))
    write_csv('nfp17_recovery_paired_axes.csv',paired)
    summary=dict(low_records=len(low),unique_low_questions=len({r['id'] for r in low}),
                 relation_counts=dict(Counter(r['relation'] for r in low)),
                 content_counts=dict(Counter(r['content'] for r in low)),
                 old57_relation_counts=dict(Counter(r['relation'] for r in low if r['source_cohort']=='old57_calibration')),
                 high_recovery_records=len(high),recovery_same_two_axes=sum(r['same_two_axes'] for r in paired),
                 limitations=['230 low records reclassify existing full-reading records and quoted evidence, not a new independent full rereading.',
                   '17 high answers freshly read in full; all low-to-high NFP pairs, not representative of all high answers.',
                   'Primary target only; not exhaustive claim-level annotation. No clinical verdict or Well regrading.',
                   'Two-axis agreement does not mean same content, strength, correctness or response quality.',
                   'No failure-vs-success prevalence claim: the 121 deterioration Plain controls and broader FPQ success controls are not uniformly recoded.'],
                 source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MASTER]+paths})
    (OUT/'two_axis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
