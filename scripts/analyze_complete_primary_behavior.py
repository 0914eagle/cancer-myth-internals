"""Analyze every completed primary pair; no generation or clinical relabeling.

Uses a frozen census and all 732 blind question-scope codings. Retains controls,
unclear/ambiguous codes and individual IDs instead of extrapolating examples.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PRIMARY={'qwen25':'premise_review','gemma':'premise_review_answer','luna':'balanced',
         'qwen38_off':'zero_shot_cot_one_step','qwen38_on':'zero_shot_cot_one_step'}


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def counts(rows):
    return dict(n=len(rows),stance=Counter(r['coding']['stance'] for r in rows),
        response=Counter(r['coding']['response'] for r in rows),
        grounding=Counter(r['coding']['target_grounding'] for r in rows),
        target=Counter(r['coding']['reference_alignment']['status'] for r in rows),
        reference_quality=Counter(r['coding']['reference_alignment']['reference_quality'] for r in rows),
        reference_question_link=Counter(r['coding']['reference_alignment']['reference_question_link'] for r in rows),
        flags=Counter(f['tag'] for r in rows for f in r['coding']['flags']),ids=[r['id'] for r in rows])


def pair_counts(rows):
    result=dict(n=len(rows),ids=[r['id'] for r in rows])
    for field in ['stance','response','target_grounding']:
        result[field]=Counter(r['plain']['coding'][field]+' -> '+r['alternative']['coding'][field] for r in rows)
    result['target']=Counter(r['plain']['coding']['reference_alignment']['status']+' -> '+r['alternative']['coding']['reference_alignment']['status'] for r in rows)
    for flag in ['scope_strengthening','context_shift','personal_context_reopened','request_to_assertion','unverifiable_as_false']:
        result[flag]=Counter(f'{int(flag in {f["tag"] for f in r["plain"]["coding"]["flags"]})}->{int(flag in {f["tag"] for f in r["alternative"]["coding"]["flags"]})}' for r in rows)
    result['new_challenge']=sum(r['plain']['coding']['stance']!='challenge' and r['alternative']['coding']['stance']=='challenge' for r in rows)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--census',type=Path,required=True);ap.add_argument('--out-dir',type=Path,required=True)
    args=ap.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((ROOT/'results/audits/answer_behavior_full_20260929_v2/manifest.json').read_text())
    identities={(r['model'],r['id'],'plain' if r['side']=='plain' else r['method']) for r in manifest}
    answers={}
    for line in (args.census/'individual_answers.jsonl').read_text().splitlines():
        r=json.loads(line)
        if (r['model'],r['id'],r['method']) in identities:
            assert r['coding'] is not None
            answers[r['model'],r['id'],r['method']]=r
    assert len(answers)==7318
    scopefile=ROOT/'results/analysis/full_behavior_completion_20260930_v1/question_scope_items.json'
    scope={r['id']:r for r in json.loads(scopefile.read_text())};assert len(scope)==732
    strata=[];paired=[];pair_rows=[];features=[]
    for model,method in PRIMARY.items():
        for ds in ['fpq','nfp']:
            for cond in ['plain',method]:
                for high in [False,True]:
                    rr=[r for (m,q,c),r in answers.items() if m==model and c==cond and r['dataset']==ds and (r['score']>=4)==high]
                    strata.append(dict(model=model,method=cond,dataset=ds,well_high=high,**counts(rr)))
            group=[]
            for (m,q,c),p in answers.items():
                if m!=model or c!='plain' or p['dataset']!=ds:continue
                a=answers[m,q,method];assert p['question']==a['question'] and p['reference']==a['reference']
                s=scope[q];assert s['dataset']==ds
                row=dict(model=model,id=q,dataset=ds,method=method,plain=p,alternative=a,
                    transition=f'{int(p["score"]>=4)}->{int(a["score"]>=4)}',question_features=s['features'])
                group.append(row);pair_rows.append(row)
            for trans in ['0->0','0->1','1->0','1->1']:
                rr=[r for r in group if r['transition']==trans]
                paired.append(dict(model=model,method=method,dataset=ds,transition=trans,**pair_counts(rr)))
            for feature in next(iter(scope.values()))['features']:
                for present in [False,True]:
                    rr=[r for r in group if r['question_features'][feature]==present]
                    n=len(rr);plain_high=sum(r['plain']['score']>=4 for r in rr)
                    transitions=Counter(r['transition'] for r in rr)
                    features.append(dict(model=model,method=method,dataset=ds,feature=feature,present=present,n=n,
                        plain_high=plain_high,alternative_high=sum(r['alternative']['score']>=4 for r in rr),
                        transitions=transitions,loss_among_plain_high=transitions['1->0']/plain_high if plain_high else None,
                        ids=[r['id'] for r in rr]))
    assert len(pair_rows)==3659
    dump(out/'individual_strata.json',strata);dump(out/'paired_strata.json',paired);dump(out/'question_feature_associations.json',features)
    with (out/'all_primary_pairs.jsonl').open('w') as f:
        for r in pair_rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    # Same-input signatures: no aggregation that treats Qwen OFF/ON as independent models.
    patterns=[]
    for q in sorted(scope):
        rows=[r for r in pair_rows if r['id']==q]
        patterns.append(dict(id=q,dataset=scope[q]['dataset'],settings={r['model']:dict(
            method=r['method'],transition=r['transition'],plain_stance=r['plain']['coding']['stance'],
            alternative_stance=r['alternative']['coding']['stance'],
            plain_target=r['plain']['coding']['reference_alignment']['status'],
            alternative_target=r['alternative']['coding']['reference_alignment']['status']) for r in rows}))
    dump(out/'cross_model_item_patterns.json',patterns)
    # Do not silently discard awkward cases: expose every target-rating disagreement
    # and every unclear reference. These are review queues, not relabelings.
    uncertain=[]
    for r in answers.values():
        code=r['coding'];ref=code['reference_alignment'];reasons=[]
        if r['dataset']=='fpq':
            if ref['status']=='corrects_target' and r['score']<4:reasons.append('coded_correction_but_low_well')
            if ref['status']!='corrects_target' and r['score']>=4:reasons.append('high_well_without_coded_correction')
            if ref['reference_quality']!='usable' or ref['reference_question_link'] in ['not_locatable','unclear']:reasons.append('reference_boundary')
        if code['stance']=='unclear' or code['target_grounding']=='unclear' or code['response']=='unclear':reasons.append('unclear_behavior')
        if reasons:uncertain.append(dict(model=r['model'],method=r['method'],id=r['id'],dataset=r['dataset'],reasons=reasons))
    dump(out/'uncertain_cases.json',uncertain)
    concise=[]
    for s in paired:
        if s['dataset']=='fpq' and s['transition']=='0->1':
            same=s['target'].get('corrects_target -> corrects_target',0)
            new=sum(v for k,v in s['target'].items() if k.endswith(' -> corrects_target') and not k.startswith('corrects_target ->'))
            concise.append(dict(model=s['model'],method=s['method'],group='FPQ score gains',n=s['n'],new_coded_correction=new,already_coded_correction=same,new_challenge=s['new_challenge']))
        if s['dataset']=='nfp' and s['transition']=='1->0':
            substantive=sum(v for k,v in s['response'].items() if k.endswith(' -> substantive'))
            withheld=sum(v for k,v in s['response'].items() if k.endswith(' -> withheld'))
            concise.append(dict(model=s['model'],method=s['method'],group='NFP score losses',n=s['n'],still_substantive=substantive,withheld=withheld,new_challenge=s['new_challenge']))
    dump(out/'key_comparisons.json',concise)
    summary=dict(answers=7318,pairs=3659,questions=732,primary_conditions=PRIMARY,
        uncertain_rows=len(uncertain),uncertain_reasons=Counter(x for r in uncertain for x in r['reasons']),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.census/'summary.json',scopefile,Path(__file__)]},
        limitations=['All numbers refer to completed primary conditions; additional conditions remain separate.',
            'Sonnet coding is provisional semantic evidence, not medical truth or verified internal reasoning.',
            'Scores are original Well ratings; do not equate a threshold crossing with a new correction.',
            'Question-level features can concern another claim than the challenged target.',
            'Different model prompts/methods; no causal ranking or pooled prevalence across settings.',
            'Feature associations are exploratory and overlapping, not causal effects.'])
    dump(out/'summary.json',summary)
    print(json.dumps(concise,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sources','limitations']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
