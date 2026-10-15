"""Join saved K3, answer behavior, and training targets. No model calls.

K3 categories describe performance with supplied alternatives, NOT knowledge.
Existing AI grounding tags are not a fresh semantic audit or causal labels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('plain', 'alternative')
STATES = ('choice_both_correct', 'choice_order_disagreement', 'choice_both_wrong', 'choice_missing')
ALIGN = ('misses_target', 'endorses_target_error', 'related_without_correction', 'corrects_target', 'unclear')

def jsonl(p):
    return [json.loads(line) for line in p.open() if line.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def textsha(t):
    return hashlib.sha256(t.encode()).hexdigest()

def write_csv(out, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/name).open('w') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', type=Path, default=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1'))
    ap.add_argument('--out', type=Path, default=ROOT/'docs/reviews/k3_behavior_scope_2026-10-04')
    args = ap.parse_args(); out = args.out; out.mkdir(parents=True, exist_ok=True)
    rawp = args.suite/'knowledge/qwen25_7b/knowledge.jsonl'
    labelp = args.suite/'knowledge/qwen25_7b/labels.jsonl'
    qp = args.suite/'questions.jsonl'
    pairp = ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
    idxp = ROOT/'docs/reviews/integrated_failure_census_2026-10-02/all_7318_answer_index.csv'
    dp = args.suite/'lora/twin_dpo_s4/train.jsonl'
    sp = args.suite/'lora/twin_bal_s4/train.jsonl'
    sources = [rawp,labelp,qp,pairp,idxp,dp,sp]
    questions = {r['id']:r for r in jsonl(qp)}
    records = jsonl(rawp); seen = set(); fc = defaultdict(dict)
    for r in records:
        key = r['id'], r['statement'], r['order']
        assert key not in seen, ('duplicate knowledge record', key)
        seen.add(key)
        if r['statement']=='pair':
            assert r['order'] in ('mc','cm') and r['verdict'] in ('A','B')
            q=questions[r['id']]
            a,b=(q['premise_text'],q['correction']) if r['order']=='mc' else (q['correction'],q['premise_text'])
            assert r['text']==f'A: {a[:80]} | B: {b[:80]}'
            fc[r['id']][r['order']]=r
    labels = {r['id']:r for r in jsonl(labelp)}
    k3={}
    for qid,q in questions.items():
        if q['set']!='fpq': continue
        rr=fc.get(qid,{})
        if set(rr)!=set(('mc','cm')): state=STATES[3]
        else:
            n=int(rr['mc']['verdict']=='B')+int(rr['cm']['verdict']=='A')
            state={2:STATES[0],1:STATES[1],0:STATES[2]}[n]
            assert labels[qid]['knows_fc']=={2:'yes',1:'unsure',0:'no'}[n]
        k3[qid]=state
    pairs=jsonl(pairp)
    index={(r['model'],r['id'],r['side']):r for r in csv.DictReader(idxp.open())}
    answers=[]; qwen=[]; keyseen=set()
    for p in pairs:
        for side in SIDES:
            a=p[side]; c=a['coding']; key=(p['model'],p['id'],side)
            assert key not in keyseen; keyseen.add(key)
            assert a['question']==questions[p['id']]['question']
            assert textsha(a['answer'])==a['answer_sha256']==index[key]['answer_sha256']
            assert a['score']==int(index[key]['original_score'])
            row=dict(model=p['model'],id=p['id'],dataset=p['dataset'],side=side,method=a['method'],
                     score=a['score'],audit_score=int(index[key]['audit_overlay_score']),
                     alignment=c['reference_alignment']['status'],stance=c['stance'],response=c['response'],
                     grounding=c['target_grounding'],flags=';'.join(f['tag'] for f in c['flags']),
                     existing_manual_pair=index[key]['manual_pair_reviewed'],answer_sha256=a['answer_sha256'])
            answers.append(row)
            if p['model']=='qwen25' and p['dataset']=='fpq':
                rr=fc.get(p['id'],{})
                qwen.append(dict(**row,k3=k3[p['id']],partition=questions[p['id']]['partition'],
                                 mc=rr.get('mc',{}).get('verdict',''),cm=rr.get('cm',{}).get('verdict','')))
    assert len(answers)==7318 and len(qwen)==1164
    write_csv(out,'k3_qwen_answer_join.csv',qwen)
    counts=[]
    for side in SIDES:
        for state in STATES:
            rr=[r for r in qwen if r['side']==side and r['k3']==state]
            for group in ('all','well_below4','well_atleast4'):
                gg=[r for r in rr if group=='all' or (r['score']<4)==(group=='well_below4')]
                co=Counter(r['alignment'] for r in gg)
                counts.append(dict(side=side,method=rr[0]['method'] if rr else '',k3=state,well_group=group,
                                   n=len(gg),well_atleast4=sum(r['score']>=4 for r in gg),
                                   **{a:co[a] for a in ALIGN},other=sum(v for k,v in co.items() if k not in ALIGN)))
    write_csv(out,'k3_behavior_counts.csv',counts)
    transitions=[]
    for state in STATES:
        rr=[p for p in pairs if p['model']=='qwen25' and p['dataset']=='fpq' and k3[p['id']]==state]
        cc=Counter((p['plain']['score']>=4,p['alternative']['score']>=4) for p in rr)
        transitions.append(dict(k3=state,n=len(rr),both_below4=cc[False,False],improved=cc[False,True],
                                worsened=cc[True,False],both_atleast4=cc[True,True]))
    write_csv(out,'k3_paired_transitions.csv',transitions)
    # Descriptive bootstrap on items, not answer rows; comparisons are exploratory.
    rng=np.random.default_rng(20261004); contrasts=[]
    for side in SIDES:
        for metric in ('well_atleast4','endorses_target_error','misses_target','related_without_correction'):
            def vals(group):
                return np.array([int(r['score']>=4) if metric=='well_atleast4' else int(r['alignment']==metric)
                    for r in qwen if r['side']==side and (r['k3']==STATES[0] if group=='pass' else r['k3'] in STATES[1:3])])
            x,y=vals('pass'),vals('other')
            boot=rng.choice(x,(10000,len(x))).mean(1)-rng.choice(y,(10000,len(y))).mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            contrasts.append(dict(side=side,metric=metric,pass_n=len(x),other_n=len(y),pass_rate=float(x.mean()),
                                  other_rate=float(y.mean()),difference=float(x.mean()-y.mean()),ci_low=float(lo),ci_high=float(hi),
                                  boundary_warning='percentile_bootstrap_degenerate_for_zero_event_group' if y.sum()==0 or x.sum()==0 else '',
                                  interpretation='exploratory_association_not_knowledge_or_causation'))
    write_csv(out,'k3_exploratory_contrasts.csv',contrasts)
    # Full four-cell census using existing coding only, with challenge-only denominator.
    grounding=[]
    for model in sorted({r['model'] for r in answers}):
        for side in SIDES:
            for ds in ('fpq','nfp'):
                for ok in (False,True):
                    for denom in ('all_answers','challenge_only'):
                        rr=[r for r in answers if r['model']==model and r['side']==side and r['dataset']==ds
                            and (r['score']>=4)==ok and (denom=='all_answers' or r['stance']=='challenge')]
                        cc=Counter(r['grounding'] for r in rr)
                        grounding.append(dict(model=model,side=side,dataset=ds,well_group='atleast4' if ok else 'below4',
                            denominator=denom,n=len(rr),explicit=cc['explicit'],implicit=cc['implicit']+cc['implied'],
                            shifted=cc['strengthened_or_shifted'],absent=cc['absent'],
                            other=sum(v for k,v in cc.items() if k not in ('explicit','implicit','implied','strengthened_or_shifted','absent')),
                            grounding_counts=json.dumps(cc,ensure_ascii=False),
                            caution='existing_AI_code_not_exhaustive_semantic_judgment'))
    write_csv(out,'all_models_existing_grounding_four_cells.csv',grounding)
    # Check correspondence with the previously read subset; neither is a gold label.
    manualp=ROOT/'docs/reviews/integrated_failure_census_2026-10-02/manual_299_pair_registry.csv'
    sources.append(manualp)
    manual=list(csv.DictReader(manualp.open()))
    overlap=[]
    for m in manual:
        if m['dataset']!='nfp' or m['category']!='N1':continue
        a=index[m['model'],m['id'],'alternative']
        overlap.append(dict(model=m['model'],id=m['id'],old_reading_category=m['category'],
            old_reading_subtype=m['scope_subtype'],existing_grounding=a['grounding'],
            existing_broad_scope_signal=a['scope_signal'],existing_personal_reopened=a['personal_reopened'],
            caution='same_selected_subset_code_correspondence_not_gold_standard_accuracy'))
    assert len(overlap)==57
    write_csv(out,'old57_scope_code_correspondence.csv',overlap)
    # Confirm the exact chosen texts, not just the two training-set sizes.
    sft={r['id']:r for r in jsonl(sp)}; dpo={r['id']:r for r in jsonl(dp)}
    assert set(sft)==set(dpo)
    training=[]
    hashes=defaultdict(list)
    for r in answers: hashes[r['answer_sha256']].append(r)
    for qid,r in dpo.items():
        s=sft[qid]; assert r['question']==s['question'] and r['origin']==s['origin']
        matches=hashes.get(textsha(r['chosen']),[])
        training.append(dict(id=qid,origin=r['origin'],kind=r['kind'],judge_score=r['judge_score'],
            same_chosen_as_sft=r['chosen']==s['target'],chosen_equals_rejected=r['chosen']==r['rejected'],
            chosen_sha256=textsha(r['chosen']),sft_target_sha256=textsha(s['target']),
            rejected_sha256=textsha(r['rejected']),chosen_existing_coding_matches=len(matches),
            coding_matches=json.dumps([(v['model'],v['id'],v['side']) for v in matches]),
            chosen_words=len(r['chosen'].split()),rejected_words=len(r['rejected'].split()),
            chosen_literal_false_premise='false premise' in r['chosen'].lower(),
            rejected_literal_false_premise='false premise' in r['rejected'].lower(),
            caution='semantic_quality_not_inferred_from_Well_selection_or_text_equality'))
    write_csv(out,'all_242_training_pair_identity.csv',training)
    # Reading packets: original Q and full answer, scores/codes/reference withheld.
    outside=[]
    for p in pairs:
        for side in SIDES:
            a=p[side];ix=index[p['model'],p['id'],side]
            if p['dataset']=='nfp' and a['score']<4 and ix['manual_pair_reviewed']=='False':
                outside.append(dict(model=p['model'],id=p['id'],side=side,method=a['method'],
                                    question=a['question'],answer=a['answer'],answer_sha256=a['answer_sha256']))
    assert len(outside)==109
    dump(out/'nfp_109_reading_packet.json',outside)
    queue=[]
    newreads=[]
    for name in ('new_luna28_full_reading.json','new_qwen81_full_reading.json'):
        newreadp=out/name
        if newreadp.exists():
            sources.append(newreadp)
            newreads.extend(json.loads(newreadp.read_text()))
    newkeys={(r['model'],r['id'],r['side']):r for r in newreads}
    assert len(newkeys)==len(newreads)
    newcensus=[]
    relation_names=('explicit_target','licensed_relation','personal_reconsideration',
                    'stronger_or_different_target','ambiguous_relation','advice_or_qualification','negative_answer')
    for model,side in sorted({(r['model'],r['side']) for r in newreads}):
        rr=[r for r in newreads if (r['model'],r['side'])==(model,side)]
        cc=Counter(r['primary_relation'] for r in rr)
        newcensus.append(dict(model=model,side=side,n=len(rr),**{k:cc[k] for k in relation_names}))
    write_csv(out,'new109_relation_census.csv',newcensus)
    newtrans=[]
    for model,qid in sorted({(r['model'],r['id']) for r in newreads}):
        if (model,qid,'plain') in newkeys and (model,qid,'alternative') in newkeys:
            p,a=newkeys[model,qid,'plain'],newkeys[model,qid,'alternative']
            newtrans.append(dict(model=model,id=qid,plain_relation=p['primary_relation'],
                alternative_relation=a['primary_relation'],same_primary_relation=p['primary_relation']==a['primary_relation'],
                caution='both_original_Well_below4_selected_pairs_not_all_method_changes'))
    write_csv(out,'new109_both_low_pairs.csv',newtrans)
    for p in pairs:
        for side in SIDES:
            a=p[side];c=a['coding']
            queue.append(dict(model=p['model'],id=p['id'],dataset=p['dataset'],side=side,
                method=a['method'],well_group='atleast4' if a['score']>=4 else 'below4',
                existing_stance=c['stance'],existing_grounding=c['target_grounding'],
                target_ko=c['target_ko'],question_quote=c['question_quote'],answer_quote=c['stance_quote'],
                question=a['question'],answer_sha256=a['answer_sha256'],
                new_semantic_audit='full_answer_read' if (p['model'],p['id'],side) in newkeys else 'pending'))
    write_csv(out,'all_answer_semantic_review_inventory.csv',queue)
    summary=dict(k3_records=len(records),k3_question_counts=dict(Counter(k3.values())),
        missing_k3=[k for k,v in k3.items() if v==STATES[3]],
        qwen_missing_answer=sorted(set(k3)-{r['id'] for r in qwen}),
        qwen_matched_k3=dict(Counter(r['k3'] for r in qwen if r['side']=='plain')),
        train_rows=len(training),same_chosen=sum(r['same_chosen_as_sft'] for r in training),
        chosen_equal_rejected=sum(r['chosen_equals_rejected'] for r in training),
        chosen_coding_matches=sum(r['chosen_existing_coding_matches']>0 for r in training),
        outside_manual_109=dict(Counter(r['model']+'|'+r['side'] for r in outside)))
    summary['training_score_counts']={str(k):v for k,v in Counter((r['kind'],str(r['judge_score'])) for r in training).items()}
    summary['old57_with_broad_scope_tag']=sum(r['existing_broad_scope_signal'] in ('True','1') for r in overlap)
    summary['training_origins']=len({r['origin'] for r in training})
    summary['new_full_readings']=len(newreads)
    summary['new_unique_questions']=len({r['id'] for r in newreads})
    summary['new_relation_counts']=dict(Counter(r['primary_relation'] for r in newreads))
    summary['new_both_low_pairs']=len(newtrans)
    summary['new_both_low_pairs_same_primary_relation']=sum(r['same_primary_relation'] for r in newtrans)
    for key,r in newkeys.items():
        assert key in index and r['answer_sha256']==index[key]['answer_sha256']
        assert textsha(r['answer'])==r['answer_sha256']
        assert r['question']==questions[r['id']]['question']
        assert r['answer_evidence'] in r['answer']
        assert r['primary_relation'] in relation_names
        assert not r['medical_truth_adjudicated'] and not r['well_rescored']
        assert index[key]['manual_pair_reviewed']=='False'
    if len(newreads)==109:
        assert set(newkeys)=={(r['model'],r['id'],r['side']) for r in outside}
        assert len(newtrans)==46
    for name in ('fpq4_success_counterexamples.json','training4_full_reading.json','reading_protocol.md'):
        p=out/name
        if p.exists():sources.append(p)
    dump(out/'summary.json',summary)
    dump(out/'manifest.json',dict(sources={str(p):sha(p) for p in sources},script_sha256=sha(Path(__file__)),
                                 no_model_calls=True,bootstrap_seed=20261004,bootstrap_replicates=10000))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\nK3 behavior all rows:')
    for r in counts:
        if r['well_group']=='all':print(r)
    print('\nTransitions:',transitions)

if __name__=='__main__':main()
