"""Audit fixed additional cases and enumerate Gemma's stored-answer opportunities."""
import csv
import hashlib
import json
from pathlib import Path

from src import well_eval as we
from scripts.analyze_target_audit_batch1 import table

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'target_audit_batch2'
GEMMA=ROOT/'results/gemma4/full_20260926_v2'
MODELS={'qwen25':'Qwen2.5','gemma':'Gemma','luna':'Luna','qwen38_off':'Qwen3.8 OFF','qwen38_on':'Qwen3.8 ON'}
SOURCES={}


def read(path,lines=False):
    data=path.read_bytes();SOURCES[str(path)]=hashlib.sha256(data).hexdigest()
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()] if lines else json.loads(data)


def main():
    sel=read(OUT/'selection.json');old=read(BASE/'target_audit_batch1/selection.json')
    rows=read(Path(sel['source']),True)
    assert SOURCES[sel['source']]==sel['source_sha256']==old['source_sha256']
    ix={(r['model'],r['id']):r for r in rows};assert len(ix)==len(rows)==3660
    ids=[]
    for a,b in zip(sel['strata'],old['strata']):
        eligible=[r['id'] for r in rows if r['model']=='gemma' and r['set']==a['set']
                  and r['review_gate']==a['gemma_review_gate'] and r['id'] not in sel['excluded_ids']]
        eligible.sort(key=lambda q:hashlib.sha256((sel['seed']+q).encode()).hexdigest())
        assert eligible==b['ordered_eligible_ids'] and eligible[1:3]==a['selected_ids']
        ids+=a['selected_ids']
    assert len(ids)==len(set(ids))==8
    notes=read(OUT/'notes.json');assert set(notes['interpretations'])==set(ids)
    for q in notes['quotes']:
        assert q['text'] in ix[q['model'],q['id']][q['field']],q
    path=GEMMA/'analysis/question_links.csv'
    SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    links=list(csv.DictReader(path.open()));byid={r['id']:r for r in links}
    assert len(links)==len(byid)==732
    judges=[];records={};rubric=None
    for method in ['plain','premise_review_answer','fp_unconditional']:
        well=GEMMA/'well'/method;read(well/'judge_plan.json')
        events=read(well/'attempts.jsonl',True)
        plan,ph=we.load_plan(well);_,scores=we.read_ledger(well,plan,ph)
        cur=plan['sources']['templates']['derived_templates']
        if rubric is None:rubric=cur
        assert rubric==cur and plan['model']=='claude-sonnet-5'
        answers={}
        for entry in plan['sources']['answers']:
            path=Path(entry['path'])
            if path.name==method+'.jsonl':
                rr=read(path,True);assert SOURCES[str(path)]==entry['sha256']
                answers.update({r['id']:r for r in rr})
        assert len(answers)==732
        for r in links:
            q=r['id'];job=plan['mapping'][method][q]
            assert scores[job]==int(r[method+'_well'])
            assert we.digest(answers[q])==plan['answer_hashes'][method][q]
            if q in ids:
                rec=read(GEMMA/'records'/method/(q+'.json'));records[q,method]=rec
                assert rec['response']==answers[q]['response']
                assert rec['status']=='complete' and not rec.get('truncated',False)
                valid=[e for e in events if e.get('id')==job and e.get('valid') and e.get('score')==scores[job]]
                assert valid
                judges.append(dict(id=q,method=method,score=scores[job],raw=valid[-1]['raw']))
    for q in ids:
        review=read(GEMMA/'records/review'/(q+'.json'));gate=read(GEMMA/'records/cot'/(q+'.json'))
        assert review['response']==ix['gemma',q]['review']
        for rec in [gate,records[q,'premise_review_answer']]:
            assert rec['review_sha256']==we.digest(review['response']) and review['response'] in rec['prompt']
    for r in links:
        gate=r['cot_gate']=='True'
        assert gate==ix['gemma',r['id']]['review_gate']
        assert int(r['cot_routed_well'])==int(r['fp_unconditional_well' if gate else 'plain_well'])
    cells={}
    for subset in ['fpq','nfp']:
        for gate in [False,True]:
            for p in [False,True]:
                for c in [False,True]:
                    key=f'{subset}:gate={int(gate)},plain={int(p)},corrected={int(c)}'
                    cells[key]=[r['id'] for r in links if r['set']==subset and (r['cot_gate']=='True')==gate
                        and (int(r['plain_well'])>=4)==p and (int(r['fp_unconditional_well'])>=4)==c]
    assert sum(map(len,cells.values()))==732
    important={
        'fpq_missed_but_correctable':cells['fpq:gate=0,plain=0,corrected=1'],
        'fpq_selected_both_fail':cells['fpq:gate=1,plain=0,corrected=0'],
        'fpq_selected_harmed':cells['fpq:gate=1,plain=1,corrected=0'],
        'nfp_false_gate_harmed':cells['nfp:gate=1,plain=1,corrected=0'],
        'fpq_missed_both_fail':cells['fpq:gate=0,plain=0,corrected=0']}
    supplemental_judges=[]
    judge_cache={}
    for model,q in [('qwen38_on','fpq_739'),('qwen38_on','fpq_376'),('qwen38_off','nfp_1149')]:
        if model not in judge_cache:
            mode='thinking_on' if model.endswith('_on') else 'thinking_off'
            well=ROOT/'results/qwen38/full_transformers_20260926_v1'/mode/'well/plain'
            read(well/'judge_plan.json');events=read(well/'attempts.jsonl',True)
            plan,ph=we.load_plan(well);_,scores=we.read_ledger(well,plan,ph)
            judge_cache[model]=(plan,scores,events)
        plan,scores,events=judge_cache[model];job=plan['mapping']['plain'][q]
        assert scores[job]==ix[model,q]['plain_score']
        valid=[e for e in events if e.get('id')==job and e.get('valid') and e.get('score')==scores[job]]
        assert valid
        supplemental_judges.append(dict(model=model,id=q,score=scores[job],job=job,raw=valid[-1]['raw']))
    md=['# 추가 고정 표본과 답변 선택 손실', '',
        'batch1과 같은 사전 고정 정렬에서 각 층의 2·3번째 문항을 읽었다. 8문항×5조건 검토문 40개, '
        'Gemma Plain 8개와 추가 Plain 6개, Gemma 후속 답변 16개를 직접 읽었다. '
        '다른 Plain은 이번에 새로 독해하지 않고 저장 점수만 연결했다. 전체 감사 빈도 추정이 아니다.', '',
        '## 새 표본: Review 판정 / 독립 Plain 점수', '',
        table(['문항']+list(MODELS.values()),[[q]+[f"{'Yes' if ix[m,q]['review_gate'] else 'No'} / {ix[m,q]['plain_score']}" for m in MODELS] for q in ids]), '',
        '## Gemma 후속 답변', '',
        table(['문항','Plain','검토문→답변','무조건 교정','gate 선택'],[[q]+[byid[q][k] for k in
              ['plain_well','premise_review_answer_well','fp_unconditional_well','cot_routed_well']] for q in ids]), '',
        '## 전체 732문항 중 주요 손실 집단', '',
        table(['집단','문항 수'],[[k,len(v)] for k,v in important.items()]), '',
        '위 수는 원인별 의미 분류가 아니라 기존 Well ≥4 경계와 저장 답변 선택의 전수 집계다. '
        '167개·49개 모두를 의미 감사했다는 뜻이 아니다. 새로운 문항에서 얻을 수 있는 개선 예측도 아니다.', '',
        '[해석](findings.md) · [독해 근거](notes.json) · [원문](cases.md) · [전체 집계와 ID](summary.json)', '']
    evidence=['# 추가 표본 원문과 독해 범위', '']
    read_plain={tuple(x) for x in notes['plain_read_scope']}
    for q in ids:
        r0=ix['gemma',q]
        evidence += [f'## {q}', '',r0['question'],'','참조: '+json.dumps({k:r0[k] for k in
                     ['premise','correction','tpq_premises']},ensure_ascii=False),'']
        for m,label in MODELS.items():
            r=ix[m,q];evidence += [f'### {label}', '',f"Review={r['review_gate']}; Plain Well={r['plain_score']}",'',r['review'],'']
            if (m,q) in read_plain:evidence += ['독해한 Plain:', '',r['plain'],'']
        for method in ['premise_review_answer','fp_unconditional']:
            evidence += [f'### Gemma {method}', '',records[q,method]['response'],'']
    summary=dict(ids=ids,cells=cells,important_groups=important,quotes_checked=len(notes['quotes']),
                 sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name,value in [('summary',summary),('judge_explanations',judges),('supplemental_judge_explanations',supplemental_judges)]:
        (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('\n'.join(md));(OUT/'cases.md').write_text('\n'.join(evidence))
    print(json.dumps({k:len(v) for k,v in important.items()}))
    print(f'Validated fixed selection, {len(notes["quotes"])} quotes, 2196 saved scores/hashes, 732 routes.')


if __name__=='__main__':main()
