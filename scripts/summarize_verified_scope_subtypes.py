"""Aggregate a manual re-reading of all 57 previously verified N1 pairs.

The assignment lists are the current assistant's textual judgments, not a
keyword classifier. N1 is a selected score-loss cohort, not all NFP failures.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/reviews/direct_reading_2026-10-02'
OUT=ROOT/'docs/reviews/failure_type_denominators_2026-10-02'
GROUPS={
    'personal_to_population':[179,180,182,183,190,192,193,194,195,198,205,206,207,208,213,219,223,226,229,234,238,241,247,251,259,261,263,266,267,276,277,281,282],
    'relation_or_scope_change':[181,189,202,212,216,224,230,240,245,253,257,270,280,289],
    'mixed_or_ambiguous':[214,217,220,256,279,287],
    'referent_time_or_scenario_change':[268,292,294,298],
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    taxonomy=json.loads((SOURCE/'taxonomy.json').read_text())
    n1={x['review_index'] for x in taxonomy['assignments'] if x['category']=='N1'}
    notes={x['review_index']:x for x in map(json.loads,(SOURCE/'reading_notes.jsonl').open())}
    evidence={x['review_index']:x for x in json.loads((SOURCE/'reviewed_evidence.json').read_text())}
    idx={i:g for g,ids in GROUPS.items() for i in ids}
    assert len(idx)==sum(map(len,GROUPS.values()))==57 and set(idx)==n1
    rows=[]
    for i in sorted(idx):
        e=evidence[i];n=notes[i]
        assert n['alternative_quote'] in e['alternative']['answer']
        assert e['plain']['score']>=4 and e['alternative']['score']<4
        rows.append(dict(review_index=i,model=e['model'],id=e['id'],method=e['method'],
            subtype=idx[i],question=e['question'],alternative_quote=n['alternative_quote'],
            previous_full_reading_note=n['finding_ko'],
            subtype_review_scope='Re-read question, exact answer quote and prior full-answer comparison note; not a fresh clinical adjudication.'))
    (OUT/'manual_scope_subtypes57.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    full=json.loads((SOURCE/'summary.json').read_text())
    counts=Counter((r['model'],r['subtype']) for r in rows)
    pairs=[json.loads(l) for l in (ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl').open()]
    result=[]
    for m in ['qwen25','gemma','luna','qwen38_off','qwen38_on']:
        selected=next(x['total'] for x in full['coverage'] if x['model']==m and x['cohort']=='nfp_loss')
        all_low=sum(x['model']==m and x['dataset']=='nfp' and x['alternative']['score']<4 for x in pairs)
        result.append(dict(model=m,all_alternative_low=all_low,reviewed_plain_high_to_alt_low=selected,
            outside_selected_cohort=all_low-selected,n1_total=sum(counts[m,g] for g in GROUPS),
            **{g:counts[m,g] for g in GROUPS},
            note='Subtype counts are within selected N1 pairs, not exhaustive nonpersonal error counts. Other original primary categories may have secondary scope changes.'))
    with (OUT/'manual_scope_subtypes_by_model.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(result)
    (OUT/'manual_scope_subtype_protocol.json').write_text(json.dumps(dict(
        scope='All 57 N1 pairs, themselves selected from 121 Plain>=4 to alternative<4 NFP pairs.',
        new_calls=0,clinically_adjudicated=False,
        definitions={
            'personal_to_population':'Individual report/plan re-evaluated via typicality, other cases or universal necessity.',
            'relation_or_scope_change':'Added exclusivity/certainty/independent efficacy, relative-to-absolute comparison, temporal-to-causal link, or changed request/action; not just person-to-population.',
            'mixed_or_ambiguous':'Multiple transformations or original scope is ambiguous; do not force into nonpersonal category.',
            'referent_time_or_scenario_change':'Past belief treated as current, future child changed to current pregnancy, or stated renal scenario changed.'},
        limitations=['Original N1 is one main change per selected pair, not an exhaustive multi-label audit.',
                     'Counts do not certify that objections were medically wrong or caused the Well score.',
                     'These are a post-hoc manual partition. No model/score blinding.',
                     'Non-N1 answers are not automatically free of these transformations.']),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
