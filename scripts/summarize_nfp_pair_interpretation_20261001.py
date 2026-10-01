"""Derive observable pair transitions and cross-audit agreement, retaining all 121 pairs."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/reviews/nfp_plain_pair_review_2026-10-01'
RUN=ROOT/'results/audits/nfp_plain_pairs_20261001_v1'


def main():
    rows=json.loads((OUT/'paired_review_full.json').read_text())
    assert len(rows)==121
    solo={(r['model'],r['id']):r for r in json.loads((ROOT/'docs/reviews/boundary_followup_2026-10-01/independent_audit_full.json').read_text()) if r['cohort']=='loss'}
    original={(r['model'],r['id']):r for r in json.loads((ROOT/'results/analysis/core_failures_20261001_v1/all_121_nfp_losses_with_judges.json').read_text())}
    def flag(r,s):return r[s]['stance']=='challenge' and r[s]['scope'] in {'strengthened','different_target'}
    table=[]; ledger=[]
    for model in ['qwen25','gemma','luna','qwen38_off','qwen38_on','ALL']:
        rr=[r for r in rows if model=='ALL' or r['model']==model]
        table.append(dict(model=model,n=len(rr),
            new_challenge=sum(r['new_challenge'] for r in rr),
            both_challenge=sum(r['plain']['stance']==r['alternative']['stance']=='challenge' for r in rr),
            alternative_not_challenge=sum(r['alternative']['stance']!='challenge' for r in rr),
            same_substantive_caveat=sum(r['same_substantive_caveat']=='yes' for r in rr),
            new_flagged_challenge=sum(flag(r,'alternative') and not flag(r,'plain') for r in rr),
            retained_flagged_challenge=sum(flag(r,'alternative') and flag(r,'plain') for r in rr),
            removed_flagged_challenge=sum(flag(r,'plain') and not flag(r,'alternative') for r in rr)))
    for r in rows:
        key=r['model'],r['id'];old=original[key]
        ledger.append(dict(model=r['model'],method=r['method'],id=r['id'],
            plain_score=r['plain_score'],alternative_score=r['alternative_score'],
            plain_stance=r['plain']['stance'],alternative_stance=r['alternative']['stance'],
            plain_scope=r['plain']['scope'],alternative_scope=r['alternative']['scope'],
            same_substantive_caveat=r['same_substantive_caveat'],
            previous_sonnet_plain=old['plain']['coding']['stance'],
            previous_sonnet_alternative=old['alternative']['coding']['stance'],
            previous_terra_solo_alternative=solo[key]['assessment']['stance'],
            question=r['question'],plain_quote=r['plain']['answer_quote'],
            alternative_quote=r['alternative']['answer_quote'],difference_ko=r['difference_ko'],
            plain_well_reason=old['well_reasons']['plain'],alternative_well_reason=old['well_reasons']['alternative']))
    for name,rr in [('transitions_by_model.csv',table),('all_121_linked_evidence.csv',ledger)]:
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
    comparison=dict(n=121,
        terra_solo_to_paired_alternative=Counter(r['previous_terra_solo_alternative']+' -> '+r['alternative_stance'] for r in ledger),
        sonnet_to_paired_plain=Counter(r['previous_sonnet_plain']+' -> '+r['plain_stance'] for r in ledger),
        sonnet_to_paired_alternative=Counter(r['previous_sonnet_alternative']+' -> '+r['alternative_stance'] for r in ledger),
        warning='Different auditor prompts and single/paired contexts. This is coding disagreement, not a change in generated answers.')
    (OUT/'coding_agreement.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    plan=json.loads((RUN/'plan.json').read_text())
    public={k:v for k,v in plan.items() if k!='options'}
    public['ab_plain_first']=sum(r['a']=='plain' for r in rows)
    public['unique_questions']=len({r['id'] for r in rows})
    public['recovery']=json.loads((RUN/'final_quote_recovery/manifest.json').read_text())
    (OUT/'protocol.json').write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(table,ensure_ascii=False))


if __name__=='__main__':main()
