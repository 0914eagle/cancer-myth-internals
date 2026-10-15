"""Census of all saved answer conditions and all context-rule decisions.

No generation/rescoring. Keep incomplete semantic audits separate from complete
Well transitions. Run: python3 -m scripts.analyze_full_behavior_census --out-dir PATH
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts import analyze_cross_model_routing_losses as src
from scripts.analyze_ungated_answer_transitions import CONFIG, Q38
from scripts import audit_answer_behavior_full as audit
from src import well_eval as we

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def tally(rows):
    cells = Counter(f'{int(r["plain_score"] >= 4)}->{int(r["answer_score"] >= 4)}' for r in rows)
    return dict(n=len(rows), plain_high=sum(r['plain_score'] >= 4 for r in rows),
                answer_high=sum(r['answer_score'] >= 4 for r in rows),
                gains=cells['0->1'], losses=cells['1->0'],
                both_low=cells['0->0'], both_high=cells['1->1'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {m: dict(v) for m, v in CONFIG.items()}
    for model, mode in [('qwen38_off', 'thinking_off'), ('qwen38_on', 'thinking_on')]:
        for method in ['premise_review_answer', 'fp_unconditional']:
            config[model][method] = Q38 / mode / 'well' / method

    ledger = src.read(src.LEDGER, True)
    ix = {(r['model'], r['id']): r for r in ledger}
    questions = {r['id']: r for r in ledger if r['model'] == 'gemma'}
    assert len(questions) == 732
    plan, _ = we.load_plan(src.G / 'well/plain')
    rubric = plan['sources']['templates']['derived_templates']
    inputs = src.read(audit.OUT / 'inputs.json')
    manifest = src.read(audit.OUT / 'manifest.json')
    inp = {r['key']: r for r in inputs}
    codes = audit.cached(inputs)
    # Freeze completed items without modifying the live runner's status/files.
    code_sources = {}
    for p in sorted((audit.OUT / 'batches').glob('*/result.json')):
        packet = json.loads(p.read_text())
        if all(c['key'] in codes for c in packet['items']):
            code_sources[str(p)] = audit.sha(p)
    coding_index = {}
    for m in manifest:
        if m['key'] not in codes:
            continue
        method = 'plain' if m['side'] == 'plain' else m['method']
        coding_index[m['model'], m['id'], method] = (m, inp[m['key']], codes[m['key']])
    dump(out / 'semantic_snapshot.json', dict(codes=codes, sources=code_sources,
         input_sha256=audit.sha(audit.OUT/'inputs.json'),
         manifest_sha256=audit.sha(audit.OUT/'manifest.json')))
    extension_count=0
    extension=ROOT/'results/audits/answer_behavior_extension_20260930_v1'
    if (extension/'inputs.json').exists():
        einputs=src.read(extension/'inputs.json'); emeta=src.read(extension/'manifest.json')
        em={r['key']:r for r in emeta}; ei={r['key']:r for r in einputs}
        ecodes={}; esources={}
        for path in sorted((extension/'batches').glob('*/result.json')):
            packet=json.loads(path.read_text()); index=int(path.parent.name)
            batch=einputs[index*3:index*3+3]
            assert packet['prompt_sha256']==audit.digest(audit.PROMPT+json.dumps(batch,ensure_ascii=False))
            checked,repairs=audit.validate(packet['items'],batch); assert not repairs
            esources[str(path)]=audit.sha(path)
            for c in checked:
                m=em[c['key']];identity=(m['model'],m['id'],m['method'])
                assert identity not in coding_index and c['key'] not in ecodes
                coding_index[identity]=(m,ei[c['key']],c);ecodes[c['key']]=c
        extension_count=len(ecodes)
        dump(out/'extension_snapshot.json',dict(codes=ecodes,sources=esources))

    answers, pairs, inventory, summaries = {}, [], [], []
    with (out / 'individual_answers.jsonl').open('w') as full:
        for model, methods in config.items():
            for method, folder in methods.items():
                loaded = src.load(folder, method, questions, rubric)
                for qid, answer in loaded.items():
                    r = questions[qid]
                    item = dict(model=model, method=method, id=qid, dataset=r['set'],
                                question=r['question'], reference={k:r[k] for k in ['premise','correction','tpq_premises']},
                                **answer, answer_sha256=digest(answer['answer']), coding=None)
                    key = (model, qid, method)
                    if key in coding_index:
                        meta, original, code = coding_index[key]
                        assert original['answer'] == item['answer']
                        assert original['question'] == item['question']
                        assert meta['score'] == item['score']
                        item['coding'] = code
                    answers[key] = item
                    full.write(json.dumps(item, ensure_ascii=False)+'\n')
                for ds in ['fpq', 'nfp']:
                    ids = {q for q, r in questions.items() if r['set'] == ds}
                    rr = [answers[model,q,method] for q in sorted(ids & loaded.keys())]
                    inventory.append(dict(model=model,method=method,dataset=ds,expected=len(ids),
                        valid=len(rr),well_high=sum(r['score']>=4 for r in rr),
                        semantic_coded=sum(r['coding'] is not None for r in rr),
                        missing=';'.join(sorted(ids-loaded.keys()))))
            for method in methods:
                if method == 'plain': continue
                for qid, question in sorted(questions.items()):
                    if (model,qid,'plain') not in answers or (model,qid,method) not in answers: continue
                    p, a = answers[model,qid,'plain'], answers[model,qid,method]
                    old = ix[model,qid]
                    assert p['answer'] == old['plain'] and p['score'] == old['plain_score']
                    pairs.append(dict(model=model,method=method,id=qid,dataset=question['set'],
                        plain_score=p['score'],answer_score=a['score'],
                        transition=f'{int(p["score"]>=4)}->{int(a["score"]>=4)}',
                        plain_sha256=p['answer_sha256'],answer_sha256=a['answer_sha256'],
                        semantic_complete=p['coding'] is not None and a['coding'] is not None))
                for ds in ['fpq','nfp']:
                    rr=[r for r in pairs if (r['model'],r['method'],r['dataset'])==(model,method,ds)]
                    s=tally(rr)
                    assert sum(s[k] for k in ['gains','losses','both_low','both_high']) == s['n']
                    summaries.append(dict(model=model,method=method,dataset=ds,**s))
    csvout(out/'answer_inventory.csv',inventory)
    csvout(out/'all_pairs.csv',pairs)
    csvout(out/'all_transitions.csv',summaries)

    semantic=[]
    for s in summaries:
        group=[r for r in pairs if (r['model'],r['method'],r['dataset'])==(s['model'],s['method'],s['dataset'])]
        for trans in ['0->0','0->1','1->0','1->1']:
            rr=[r for r in group if r['transition']==trans]
            complete=[r for r in rr if r['semantic_complete']]
            transitions={f:Counter() for f in ['stance','target','response','grounding','flags']}
            for r in complete:
                p=answers[r['model'],r['id'],'plain']['coding']
                a=answers[r['model'],r['id'],r['method']]['coding']
                for field,col in [('stance','stance'),('response','response'),('target_grounding','grounding')]:
                    transitions[col][p[field]+' -> '+a[field]]+=1
                transitions['target'][p['reference_alignment']['status']+' -> '+a['reference_alignment']['status']]+=1
                pf={x['tag'] for x in p['flags']}; af={x['tag'] for x in a['flags']}
                for flag in pf|af: transitions['flags'][flag+':'+str(int(flag in pf))+'->'+str(int(flag in af))]+=1
            semantic.append(dict(model=s['model'],method=s['method'],dataset=s['dataset'],
                score_transition=trans,expected=len(rr),coded_pairs=len(complete),
                missing_ids=[r['id'] for r in rr if not r['semantic_complete']],**transitions))
    dump(out/'semantic_strata.json',semantic)

    native=[]
    for method in config['qwen38_off']:
        for ds in ['fpq','nfp']:
            rr=[]
            for qid,q in questions.items():
                if q['set'] != ds: continue
                p,a=[answers[m,qid,method] for m in ['qwen38_off','qwen38_on']]
                rr.append(dict(plain_score=p['score'],answer_score=a['score']))
            native.append(dict(method=method,dataset=ds,**tally(rr)))
    csvout(out/'thinking_off_to_on.csv',native)

    old=ROOT/'results/frontier_cli/luna_direct_cot_20260923_v1'
    new=ROOT/'results/frontier_cli/luna_context_direct_20260928_v1'
    nfp_audit=src.read(ROOT/'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/semantic_audit/annotations.json')
    reviews={r['id']:r for r in nfp_audit if r['set']=='nfp'}
    assert len(reviews)==149
    lost={r['id']:r for r in src.read(ROOT/'docs/reviews/luna_nfp_tpq_2026-09-28/lost_fpq/annotations.json')}
    context=[]
    for q in src.read(new/'evaluation_questions.json'):
        qid=q['id']; assert q['question']==questions[qid]['question']
        p=src.read(old/'direct'/qid/'record.json'); a=src.read(new/'context_rule'/qid/'record.json')
        assert p['status']==a['status']=='complete'
        response=json.loads(a['response']); assert response['has_false_premise']==a['gate_on']
        row=dict(id=qid,dataset=q['set'],category=q.get('category','not_annotated'),partition=q['partition'],
            question=q['question'],reference={k:q.get(k) for k in ['premise_text','correction']},
            before=p['result']['has_false_premise'],after=a['gate_on'],
            transition=f'{int(p["result"]["has_false_premise"])}->{int(a["gate_on"])}',
            direct_rationale_available=False,separate_review_code=None,lost_fpq_code=lost.get(qid))
        if qid in reviews:
            r=reviews[qid]; assert r['input']['question']==q['question']
            slot=next(s for s,m in r['slots'].items() if m=='luna')
            code=next(c for c in r['audit']['reviews'] if c['slot']==slot)
            review=src.read(old/'review'/qid/'record.json')['result']['review']
            assert review==next(v['text'] for v in r['input']['reviews'] if v['slot']==slot)
            assert code['question_quote'] in q['question'] and code['review_quote'] in review
            row['separate_review_code']=code
        context.append(row)
    assert len(context)==732 and len({r['id'] for r in context})==732
    context_summary=[]
    for ds in ['fpq','nfp']:
        for trans in ['0->0','0->1','1->0','1->1']:
            rr=[r for r in context if r['dataset']==ds and r['transition']==trans]
            context_summary.append(dict(dataset=ds,transition=trans,n=len(rr),
                ids=[r['id'] for r in rr],categories=Counter(r['category'] for r in rr),
                separate_review_coded=sum(r['separate_review_code'] is not None for r in rr),
                review_reading=Counter(r['separate_review_code']['reading'] for r in rr if r['separate_review_code']),
                review_reason=Counter(r['separate_review_code']['reason_type'] for r in rr if r['separate_review_code']),
                lost_fpq_scope=Counter(r['lost_fpq_code']['reference_scope'] for r in rr if r['lost_fpq_code'])))
    dump(out/'context_all_items.json',context)
    dump(out/'context_summary.json',context_summary)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),unique_questions=len(questions),
        individual_answers=len(answers),paired_comparisons=len(pairs),
        semantic_coded_answers=len(codes)+extension_count,primary_audit_coded=len(codes),
        extension_audit_coded=extension_count,primary_audit_expected=len(inputs),
        context_decisions=len(context),transitions=summaries,
        limits=['Well transitions are complete for available saved answers, not semantic causes.',
                'Semantic audit remains incomplete; report denominators and missing IDs by transition.',
                'Existing primary audit uses different methods across models; never pool as one intervention.',
                'Separate Review is not the reasoning used by Luna Direct; neither Direct condition saved rationale.',
                '147 lost-FPQ scope codes cannot estimate selectivity without retained-FPQ controls.',
                'Well score and AI target coding do not certify clinical truth; original scores/labels unchanged.'],
        source_sha256=src.SOURCES,script_sha256=audit.sha(Path(__file__)))
    dump(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['individual_answers','paired_comparisons','semantic_coded_answers','context_decisions']},indent=2))
    for r in summaries: print(json.dumps(r))
    print(json.dumps(context_summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
