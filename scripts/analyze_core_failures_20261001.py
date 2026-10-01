"""Full primary-cohort transitions, original judge joins, and paired Luna repeats.

No new LLM judgments, label replacement, threshold tuning, or inferred chain of
thought. Existing Sonnet annotations remain provisional behavioral codes.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/analysis/core_failures_20261001_v1'
DOC = ROOT / 'docs/reviews/core_failures_2026-10-01'
PAIRS = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
REPEAT = ROOT / 'results/audits/measurement_validity_20261001_v1/luna_repeats'
PRIMARY = {'qwen25': 'premise_review', 'gemma': 'premise_review_answer',
           'luna': 'balanced', 'qwen38_off': 'zero_shot_cot_one_step',
           'qwen38_on': 'zero_shot_cot_one_step'}
FLAGS = ['scope_strengthening', 'personal_context_reopened', 'request_to_assertion',
         'context_shift', 'unverifiable_as_false']
SEED = 20261001
B = 5000


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def flags(answer):
    return {f['tag'] for f in answer['coding']['flags']}


def target(answer):
    return answer['coding']['reference_alignment']['status']


def phase_stats(rows):
    result = {'n': len(rows)}
    for side in ('plain', 'alternative'):
        answers = [r[side] for r in rows]
        result[side] = dict(well_scores=Counter(a['score'] for a in answers),
            target=Counter(target(a) for a in answers),
            stance=Counter(a['coding']['stance'] for a in answers),
            grounding=Counter(a['coding']['target_grounding'] for a in answers),
            response=Counter(a['coding']['response'] for a in answers),
            flags={tag: sum(tag in flags(a) for a in answers) for tag in FLAGS})
    result['target_transitions'] = Counter(target(r['plain'])+' -> '+target(r['alternative']) for r in rows)
    result['stance_transitions'] = Counter(r['plain']['coding']['stance']+' -> '+r['alternative']['coding']['stance'] for r in rows)
    result['flag_transitions'] = {tag: Counter(str(int(tag in flags(r['plain'])))+'->'+str(int(tag in flags(r['alternative']))) for r in rows) for tag in FLAGS}
    return result


def judge_loader():
    cache = {}
    def load(a):
        p = Path(a['well_dir']) / 'attempts.jsonl'
        if p not in cache:
            cache[p] = {}
            for line in p.open():
                r = json.loads(line)
                if r.get('valid'): cache[p][r['id']] = r
        r = cache[p][a['job']]
        assert r['score'] == a['score']
        return r['raw']
    return load, cache


def paired_repeats(pairs):
    mapping = {r['key']: r for r in json.loads((REPEAT / 'private_mapping.json').read_text())}
    jobs = json.loads((REPEAT / 'jobs.json').read_text())
    plan = json.loads((REPEAT / 'plan.json').read_text())
    inventory = json.loads((ROOT / 'results/frontier_cli/sonnet_plain_20260922_v1/evaluation_questions.json').read_text())
    original = {q['id']: q for q in inventory}
    features = {}
    for r in pairs:
        assert r['id'] not in features or features[r['id']] == r['question_features']
        features[r['id']] = r['question_features']
    values = defaultdict(dict)
    record_hashes = {}
    for job in jobs:
        record_path = REPEAT / 'records' / job['job'] / 'record.json'
        record_hashes[job['job']] = hashlib.sha256(record_path.read_bytes()).hexdigest()
        r = json.loads(record_path.read_text())
        assert r['status'] == 'complete'
        item = job['items'][0]; q = mapping[item['key']]
        assert item['question'] == original[q['id']]['question']
        assert r['prompt'] == plan['gate_prompts'][job['condition']].format(question=item['question'])
        key = (job['condition'], job['repeat'])
        assert key not in values[q['id']]
        values[q['id']][key] = int(r['result']['has_false_premise'])
    assert len(jobs) == 2928 and len(values) == 732 and all(len(v)==4 for v in values.values())
    items = []
    for qid, v in sorted(values.items()):
        b1,b2,c1,c2 = [v[k] for k in [('baseline',1),('baseline',2),('context_rule',1),('context_rule',2)]]
        items.append(dict(id=qid, dataset=original[qid]['set'], group_id=original[qid]['group_id'],
            baseline_r1=b1,baseline_r2=b2,context_r1=c1,context_r2=c2,
            baseline_mean=(b1+b2)/2,context_mean=(c1+c2)/2,
            delta=(c1+c2-b1-b2)/2,stable_yes_to_no=int(b1==b2==1 and c1==c2==0),
            stable_no_to_yes=int(b1==b2==0 and c1==c2==1),
            baseline_unstable=int(b1!=b2),context_unstable=int(c1!=c2),
            **features[qid]))
    csvout(DOC / 'luna_repeat_items.csv',items)
    rng = np.random.default_rng(SEED)
    results,boot = {},{}
    for ds in ('fpq','nfp'):
        rr=[r for r in items if r['dataset']==ds]
        groups=defaultdict(list)
        for r in rr:groups[r['group_id']].append(r)
        group_rows=list(groups.values())
        totals=np.array([[len(g),sum(x['baseline_mean'] for x in g),sum(x['context_mean'] for x in g)] for g in group_rows],dtype=float)
        sampled=totals[rng.integers(0,len(totals),size=(B,len(totals)))].sum(axis=1)
        rates=sampled[:,1:]/sampled[:,0,None];boot[ds]=rates
        results[ds]=dict(n=len(rr),source_groups=len(groups),
            baseline_r1_yes=sum(x['baseline_r1'] for x in rr),baseline_r2_yes=sum(x['baseline_r2'] for x in rr),
            context_r1_yes=sum(x['context_r1'] for x in rr),context_r2_yes=sum(x['context_r2'] for x in rr),
            baseline_mean_rate=float(np.mean([x['baseline_mean'] for x in rr])),
            context_mean_rate=float(np.mean([x['context_mean'] for x in rr])),
            mean_delta=float(np.mean([x['delta'] for x in rr])),
            delta_ci95=np.quantile(rates[:,1]-rates[:,0],[.025,.975]).tolist(),
            baseline_disagreements=sum(x['baseline_unstable'] for x in rr),
            context_disagreements=sum(x['context_unstable'] for x in rr),
            stable_yes_to_no=sum(x['stable_yes_to_no'] for x in rr),
            stable_no_to_yes=sum(x['stable_no_to_yes'] for x in rr),
            count_transitions=Counter(str(x['baseline_r1']+x['baseline_r2'])+'->'+str(x['context_r1']+x['context_r2']) for x in rr))
    jd=(boot['fpq'][:,1]-boot['fpq'][:,0])-(boot['nfp'][:,1]-boot['nfp'][:,0])
    results['youden_j'] = dict(baseline=results['fpq']['baseline_mean_rate']-results['nfp']['baseline_mean_rate'],
        context=results['fpq']['context_mean_rate']-results['nfp']['context_mean_rate'],
        delta_ci95=np.quantile(jd,[.025,.975]).tolist(),
        warning='Single operating-point summary; unchanged J does not prove unchanged discrimination or threshold-only mechanism.')
    results['protocol']=dict(model='gpt-5.6-luna',effort='medium',task='Direct JSON boolean gate, no answer generation',
        conditions=['baseline','context_rule'],repeats=2,bootstrap=B,seed=SEED,
        resampling='Source-group clustered and label-stratified bootstrap; both conditions and repetitions kept paired.',
        limitation='Two repeats estimate this run-to-run variability only; not all temporal/backend variation. Features are AI annotations, not causal labels.')
    strata=[]
    for ds in ('fpq','nfp'):
        for f in features[items[0]['id']]:
            for value in (False,True):
                rr=[r for r in items if r['dataset']==ds and r[f]==value]
                strata.append(dict(dataset=ds,feature=f,present=value,n=len(rr),
                    baseline_mean=sum(r['baseline_mean'] for r in rr)/len(rr) if rr else None,
                    context_mean=sum(r['context_mean'] for r in rr)/len(rr) if rr else None,
                    stable_yes_to_no=sum(r['stable_yes_to_no'] for r in rr)))
    dump(DOC / 'luna_repeat_summary.json',results)
    dump(DOC / 'luna_repeat_protocol.json', dict(system=plan['system'],
        gate_prompts=plan['gate_prompts'], protocol=results['protocol'],
        record_sha256=record_hashes))
    stable = []
    for r in items:
        if r['stable_yes_to_no'] or r['stable_no_to_yes']:
            stable.append(dict(r, question=original[r['id']]['question']))
    csvout(DOC / 'luna_stable_switch_questions.csv', stable)
    csvout(DOC / 'luna_repeat_feature_associations.csv',strata)
    return results


def main():
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    pairs=[json.loads(line) for line in PAIRS.open()]
    assert len(pairs)==3659 and all(r['method']==PRIMARY[r['model']] for r in pairs)
    judge,cache=judge_loader()
    phases=[]; disag=[];losses=[];joined=[];flags_control=[]
    for model,method in PRIMARY.items():
        for ds in ('fpq','nfp'):
            allrows=[r for r in pairs if r['model']==model and r['dataset']==ds]
            for transition in ('all','0->0','0->1','1->0','1->1'):
                rr=[r for r in allrows if transition=='all' or r['transition']==transition]
                phases.append(dict(model=model,method=method,dataset=ds,well_transition=transition,**phase_stats(rr)))
            for tag in FLAGS:
                for high in (False,True):
                    for side in ('plain','alternative'):
                        rr=[r for r in allrows if (r[side]['score']>=4)==high]
                        flags_control.append(dict(model=model,method='plain' if side=='plain' else method,
                            dataset=ds,well_high=high,tag=tag,n=len(rr),tagged=sum(tag in flags(r[side]) for r in rr)))
    for r in pairs:
        p,a=r['plain'],r['alternative']
        is178=r['dataset']=='fpq' and r['transition']=='0->1' and target(p)==target(a)=='corrects_target'
        isnfp=r['dataset']=='nfp' and r['transition']=='1->0'
        if is178 or isnfp:
            item=dict(r,well_reasons={side:judge(r[side]) for side in ('plain','alternative')})
            (disag if is178 else losses).append(item)
        joined.append(dict(model=r['model'],method=r['method'],id=r['id'],dataset=r['dataset'],
            well_transition=r['transition'],plain_score=p['score'],alternative_score=a['score'],
            plain_target=target(p),alternative_target=target(a),plain_stance=p['coding']['stance'],
            alternative_stance=a['coding']['stance'],
            plain_flags=';'.join(sorted(flags(p))),alternative_flags=';'.join(sorted(flags(a))),
            disagreement178=is178,nfp_loss121=isnfp))
    assert len(disag)==178 and len(losses)==121
    csvout(DOC/'all_3659_pair_transitions.csv',joined)
    csvout(DOC/'flag_controls.csv',flags_control)
    dump(DOC/'behavior_transitions.json',phases)
    dump(OUT/'all_178_disagreements_with_judges.json',disag)
    dump(OUT/'all_121_nfp_losses_with_judges.json',losses)
    # Reviewable evidence for every selected pair, not only illustrative cases.
    for name, rows in [('fpq_178', disag), ('nfp_121', losses)]:
        evidence = []
        for r in rows:
            e = dict(model=r['model'], alternative_method=r['method'], id=r['id'],
                question=r['plain']['question'], reference=json.dumps(r['plain']['reference'],ensure_ascii=False))
            for side in ('plain', 'alternative'):
                a=r[side]; c=a['coding']; ref=c['reference_alignment']
                e.update({side+'_score': a['score'], side+'_answer': a['answer'],
                    side+'_target_code': ref['status'], side+'_target_quote': ref['answer_quote'],
                    side+'_target_note': ref['note_ko'], side+'_stance_code': c['stance'],
                    side+'_stance_quote': c['stance_quote'], side+'_well_reason': r['well_reasons'][side]})
            evidence.append(e)
        csvout(DOC / (name+'_evidence.csv'), evidence)
    movement=[]
    for r in phases:
        if r['well_transition'] != 'all': continue
        cohort=[p for p in phases if p['model']==r['model'] and p['dataset']==r['dataset']]
        transitions={p['well_transition']:p['n'] for p in cohort}
        movement.append(dict(model=r['model'],method=r['method'],dataset=r['dataset'],n=r['n'],
            both_low=transitions['0->0'],gain=transitions['0->1'],loss=transitions['1->0'],
            both_high=transitions['1->1'],net_high=transitions['0->1']-transitions['1->0']))
    csvout(DOC/'well_movements.csv',movement)
    overview={}
    for model in PRIMARY:
        d=[r for r in disag if r['model']==model]
        overview[model]=dict(n=len(d),plain_scores=Counter(r['plain']['score'] for r in d),
            alternative_scores=Counter(r['alternative']['score'] for r in d),
            references=Counter(r['plain']['coding']['reference_alignment']['reference_quality'] for r in d))
    dump(DOC/'disagreement178_summary.json',dict(by_model=overview,n=178,unique_questions=len({r['id'] for r in disag}),
         plain_scores=Counter(r['plain']['score'] for r in disag),
         note='Different evaluations, not 178 established successful corrections or 178 unique questions.'))
    # Every selected case is exposed with unchanged references, full answers and both judge reasons.
    for title,rows,name in [('FPQ 178 evaluation disagreements',disag,'fpq_178_evidence.md'),
                            ('NFP 121 score losses',losses,'nfp_121_evidence.md')]:
        lines=['# '+title,'','Original evidence only. No new semantic labels or corrected Well scores.','']
        for r in rows:
            lines+=['## '+r['model']+' / '+r['method']+' / '+r['id'],'',
                    '**Question**',r['plain']['question'],'','**Reference**',
                    '```json',json.dumps(r['plain']['reference'],ensure_ascii=False,indent=2),'```','']
            for side in ('plain','alternative'):
                a=r[side];c=a['coding']
                lines+=['### '+side+' — Well '+str(a['score']),'',a['answer'],'',
                        '**Original Well reason**',r['well_reasons'][side],'',
                        '**Original Sonnet behavior coding**','```json',json.dumps(c,ensure_ascii=False,indent=2),'```','']
        (OUT/name).write_text('\n'.join(lines))
    repeat=paired_repeats(pairs)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PAIRS,Path(__file__),REPEAT/'plan.json',REPEAT/'jobs.json']}
    hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in cache})
    dump(DOC/'manifest.json',dict(primary_pairs=3659,answers=7318,conditions=PRIMARY,source_sha256=hashes,
        full_evidence=str(OUT),new_model_calls=0,
        limitations=['Behavior codes are provisional Sonnet annotations, not independently adjudicated errors.',
          'Different alternative prompts across models; no pooled causal ranking.',
          'Flag co-occurrence is not proof of a causal mechanism; individual_context flags need not be mistakes.',
          'Full cohorts and unchanged/high-score controls retained; selected 178/121 are not prevalence samples.']))
    print(json.dumps(dict(disagreement178=overview,luna_repeats=repeat),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
