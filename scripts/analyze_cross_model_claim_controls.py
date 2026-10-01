"""Full saved-result census, with success controls and question-only AI codes.

No model calls. Descriptive associations are not causal estimates or new audits.
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
SCOPE = ROOT / 'results/audits/question_scope_full_20260930_v1'
OUT = ROOT / 'docs/reviews/cross_model_claim_controls_2026-10-02'
MODELS = ['qwen25', 'gemma', 'luna', 'qwen38_off', 'qwen38_on']
FEATURES = ['has_general_claim', 'has_explicit_diagnosis_to_outcome_inference',
            'has_queried_claim', 'has_ambiguous_claim']


def csv_write(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in SRC.open()]
    assert len(pairs) == 3659
    by = {(x['model'], x['id']): x for x in pairs}
    assert len(by) == len(pairs)
    common = set.intersection(*[{x['id'] for x in pairs if x['model'] == m} for m in MODELS])
    assert len(common) == 731
    questions = {}
    for x in pairs:
        q = x['plain']['question']
        assert q == x['alternative']['question']
        assert x['id'] not in questions or questions[x['id']] == q
        questions[x['id']] = q
    manifest = {x['key']: x for x in json.loads((SCOPE / 'manifest.json').read_text())}
    scope_codes = json.loads((SCOPE / 'codings.json').read_text())
    assert len(scope_codes) == 732
    quote_issues = []
    # No labels/references/answers/scores in this question-only export.
    with (OUT / 'question_only_existing_ai_codes.jsonl').open('w') as f:
        for c in scope_codes:
            ident = manifest[c['key']]['id']; q = questions[ident]
            for claim in c['claims']:
                if claim['quote'] not in q: quote_issues.append([ident, claim['quote']])
            f.write(json.dumps(dict(key=c['key'], question=q, existing_ai_code=c), ensure_ascii=False) + '\n')
    csv_write('question_key_map.csv', [dict(key=k, id=v['id']) for k,v in manifest.items()])
    flat, cells, associations, transitions = [], [], [], []
    feature_lookup = {}
    for x in pairs:
        old = feature_lookup.setdefault(x['id'], x['question_features'])
        assert old == x['question_features']
        for cond in ['plain', 'alternative']:
            a = x[cond]; c = a['coding']; tags = {z['tag'] for z in c['flags']}
            flat.append(dict(model=x['model'], id=x['id'], dataset=x['dataset'], condition=cond,
                             method=a['method'], common_item=x['id'] in common, score=a['score'],
                             well_ge4=a['score'] >= 4, existing_alignment=c['reference_alignment']['status'],
                             stance=c['stance'], target_grounding=c['target_grounding'],
                             scope_strengthening='scope_strengthening' in tags,
                             request_to_assertion='request_to_assertion' in tags,
                             unverifiable_as_false='unverifiable_as_false' in tags,
                             **x['question_features']))
    assert len(flat) == 7318
    csv_write('all_answer_outcomes.csv', flat)
    csv_write('common_items_wide.csv', [dict(id=i, dataset=by[(MODELS[0], i)]['dataset'], **{
        m+'_'+cond+'_score': by[(m, i)][cond]['score'] for m in MODELS for cond in ['plain','alternative']}) for i in sorted(common)])
    for ds in ['fpq','nfp']:
        for m in MODELS:
            for cond in ['plain','alternative']:
                subset = [r for r in flat if r['dataset']==ds and r['model']==m and r['condition']==cond and r['common_item']]
                for pred in ['scope_strengthening','request_to_assertion','unverifiable_as_false']:
                    for val in [False,True]:
                        rr = [r for r in subset if r[pred] == val]
                        cells.append(dict(dataset=ds,model=m,condition=cond,tag=pred,present=val,
                                          n=len(rr),well_ge4=sum(r['well_ge4'] for r in rr),
                                          well_lt4=sum(not r['well_ge4'] for r in rr)))
                for feat in FEATURES:
                    n0 = [r for r in subset if not r[feat]]; n1 = [r for r in subset if r[feat]]
                    for outcome in (['well_ge4','existing_corrects_target'] if ds=='fpq' else ['well_lt4']):
                        def value(r):
                            if outcome=='well_ge4': return r['well_ge4']
                            if outcome=='well_lt4': return not r['well_ge4']
                            return r['existing_alignment']=='corrects_target'
                        k0=sum(value(r) for r in n0); k1=sum(value(r) for r in n1)
                        # Stratified item bootstrap of an unadjusted difference, not a treatment effect.
                        rng=np.random.default_rng(20261002)
                        draws=rng.binomial(len(n1),k1/len(n1),4000)/len(n1)-rng.binomial(len(n0),k0/len(n0),4000)/len(n0)
                        ci=np.quantile(draws,[.025,.975])
                        associations.append(dict(dataset=ds,model=m,condition=cond,feature=feat,outcome=outcome,
                            absent_n=len(n0),absent_k=k0,present_n=len(n1),present_k=k1,
                            difference_pp=100*(k1/len(n1)-k0/len(n0)),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1]))
            rr=[x for x in pairs if x['model']==m and x['dataset']==ds and x['id'] in common]
            for pred in ['scope_strengthening','request_to_assertion','unverifiable_as_false']:
                counts=Counter()
                for x in rr:
                    tag=lambda a:any(t['tag']==pred for t in x[a]['coding']['flags'])
                    counts[(tag('plain'),tag('alternative'),x['plain']['score']>=4,x['alternative']['score']>=4)]+=1
                for (p,a,ps,ass),n in counts.items():
                    transitions.append(dict(dataset=ds,model=m,tag=pred,plain_tag=p,alternative_tag=a,
                                            plain_well_ge4=ps,alternative_well_ge4=ass,n=n))
    csv_write('existing_tag_success_controls.csv',cells)
    csv_write('question_feature_associations.csv',associations)
    csv_write('within_item_tag_transitions.csv',transitions)
    targets=Counter((x['model'],cond,x[cond]['coding']['reference_alignment']['status'])
                    for x in pairs if x['dataset']=='fpq' and x['id'] in common for cond in ['plain','alternative'])
    csv_write('fpq_alignment_counts.csv',[dict(model=m,condition=c,existing_code=k,n=n) for (m,c,k),n in targets.items()])
    focused=['fpq_8','fpq_30','nfp_1075','nfp_1059']
    with (OUT / 'focused_same_item_pairs.jsonl').open('w') as f:
        for i in focused:
            for m in MODELS: f.write(json.dumps(by[(m,i)],ensure_ascii=False)+'\n')
    qpath=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/answers/premise_review.jsonl')
    qreviews={r['id']:r for r in map(json.loads,qpath.open())}
    reviews=[]
    for i in focused:
        q=qreviews[i]
        assert q['response']==by[('qwen25',i)]['alternative']['answer']
        reviews.append(dict(model='qwen25',id=i,source=str(qpath),review=q['details']['review']['text']))
        gp=ROOT/f'results/gemma4/full_20260926_v2/records/premise_review_answer/{i}.json'
        g=json.loads(gp.read_text())
        assert g['response']==by[('gemma',i)]['alternative']['answer']
        reviews.append(dict(model='gemma',id=i,source=str(gp.relative_to(ROOT)),review=g['prompt'].split('\n\nPremise review:\n')[1]))
    dump('focused_intermediate_reviews.json',reviews)
    dump('validation.json',dict(source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
        source_pairs=len(pairs),source_answers=len(flat),unique_questions=len(questions),
        common_questions=len(common),common_fpq=sum(i.startswith('fpq') for i in common),
        common_nfp=sum(i.startswith('nfp') for i in common),scope_quote_issues=quote_issues,
        excluded_from_common=[dict(model=m,id=i) for m,i in by if i not in common],
        new_model_calls=0,new_medical_adjudications=0,
        warning='Existing AI codes are reused, not newly verified for all answers. Feature associations use any claim in a question, not necessarily the annotated target. Exploratory CIs are unadjusted for multiple comparisons.'))
    print(json.dumps(dict(pairs=len(pairs),answers=len(flat),common_items=len(common),quote_issues=len(quote_issues))))


if __name__=='__main__': main()
