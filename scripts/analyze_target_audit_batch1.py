"""Validate a frozen, stratified four-question audit against saved outputs."""
import csv
import hashlib
import json
from pathlib import Path

from src import well_eval as we

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/detection_plain_link_2026-09-28/target_audit_batch1'
GEMMA = ROOT / 'results/gemma4/full_20260926_v2'
MODELS = {'qwen25':'Qwen2.5', 'gemma':'Gemma', 'luna':'Luna',
          'qwen38_off':'Qwen3.8 OFF', 'qwen38_on':'Qwen3.8 ON'}
SOURCES = {}


def read(path, lines=False):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return [json.loads(l) for l in data.decode().splitlines() if l.strip()] if lines else json.loads(data)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def main():
    selection = read(OUT/'selection.json')
    source = Path(selection['source'])
    rows = read(source, True)
    assert SOURCES[str(source)] == selection['source_sha256']
    index = {(r['model'],r['id']):r for r in rows}
    assert len(index) == len(rows) == 3660
    ids = []
    for s in selection['strata']:
        eligible = [r['id'] for r in rows if r['model']=='gemma' and r['set']==s['set']
                    and r['review_gate']==s['gemma_review_gate'] and r['id'] not in selection['excluded_ids']]
        eligible.sort(key=lambda q:hashlib.sha256((selection['seed']+q).encode()).hexdigest())
        assert eligible == s['ordered_eligible_ids']
        assert len(eligible)==s['eligible_n'] and eligible[0]==s['selected_id']
        ids.append(eligible[0])
    audit = read(OUT/'annotations.json')
    annotations = audit['rows']
    assert set(audit['cases'])==set(ids)
    assert {(a['model'],a['id']) for a in annotations} == {(m,q) for m in MODELS for q in ids}
    assert len(annotations)==20
    count=0
    for q, case in audit['cases'].items():
        for field in ['question','correction']:
            if field+'_quote' in case:
                assert case[field+'_quote'] in index['gemma',q][field]
                count+=1
    for a in annotations:
        r=index[a['model'],a['id']]
        for field in ['review','plain']:
            for suffix in ['_quote','_quote_2']:
                if field+suffix in a:
                    assert a[field+suffix] in r[field], (a['model'],a['id'],field)
                    count+=1
    path=GEMMA/'analysis/question_links.csv'
    SOURCES[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    links={r['id']:r for r in csv.DictReader(path.open())}
    records={}
    for q in ids:
        for stage in ['review','cot','plain','premise_review_answer','fp_unconditional']:
            r=read(GEMMA/'records'/stage/(q+'.json'))
            assert r['status']=='complete' and not r.get('truncated',False)
            records[q,stage]=r
        review=records[q,'review']['response']
        assert review==index['gemma',q]['review']
        assert records[q,'plain']['response']==index['gemma',q]['plain']
        for stage in ['cot','premise_review_answer']:
            r=records[q,stage]
            assert r['review_sha256']==we.digest(review) and review in r['prompt']
        gate=records[q,'cot']['result']['has_false_premise']
        assert gate==index['gemma',q]['review_gate']==(links[q]['cot_gate']=='True')
        assert links[q]['cot_routed_well']==links[q]['fp_unconditional_well' if gate else 'plain_well']
    judges=[]
    rubric=None
    for method in ['plain','premise_review_answer','fp_unconditional']:
        well=GEMMA/'well'/method
        read(well/'judge_plan.json')
        events=read(well/'attempts.jsonl',True)
        plan,ph=we.load_plan(well)
        _,scores=we.read_ledger(well,plan,ph)
        current=plan['sources']['templates']['derived_templates']
        if rubric is None:rubric=current
        assert current==rubric and plan['model']=='claude-sonnet-5'
        answers={}
        for entry in plan['sources']['answers']:
            path=Path(entry['path'])
            if path.name==method+'.jsonl':
                answer_rows=read(path,True)
                assert SOURCES[str(path)]==entry['sha256']
                answers.update({a['id']:a for a in answer_rows})
        for q in ids:
            job=plan['mapping'][method][q]
            score=scores[job]
            assert score==int(links[q][method+'_well'])
            assert we.digest(answers[q])==plan['answer_hashes'][method][q]
            assert answers[q]['response']==records[q,method]['response']
            valid=[e for e in events if e.get('id')==job and e.get('valid') and e.get('score')==score]
            assert valid
            judges.append(dict(id=q,method=method,score=score,job=job,explanation=valid[-1]['raw']))
    selected=[index[m,q] for q in ids for m in MODELS]
    md=['# 사전 고정 표본 1차: 대상·판정·답변', '',
        '기존 상세 사례 18개를 제외하고 Gemma Review 판정×원 FPQ/NFP 라벨의 4개 층에서 '
        '고정 seed의 SHA-256 정렬 첫 문항을 하나씩 선택했다. 선택 ID를 원문 독해 전에 파일로 저장했다. '
        '이후 네 모델(5조건)의 검토문 20개와 Plain 20개, Gemma 후속 답변 8개를 읽었다. '
        '표본은 4문항이며 20개 독립 문항이 아니다. 층별 1개로 원인 빈도나 모델 순위를 추정하지 않는다.', '',
        table(['문항','원 라벨','선정 층의 Gemma Review','후보 수'],
              [[s['selected_id'],s['set'],s['gemma_review_gate'],s['eligible_n']] for s in selection['strata']]), '',
        '## 독립 Plain 연결', '',
        '셀은 Review Yes/No · Plain Well 점수다. Plain은 Review를 입력받지 않았다.', '',
        table(['문항']+list(MODELS.values()),
              [[q]+[f"{'Yes' if index[m,q]['review_gate'] else 'No'} / {index[m,q]['plain_score']}" for m in MODELS] for q in ids]), '',
        '## Gemma 실제 후속 답변', '',
        table(['문항','Plain','검토문→답변','무조건 교정','gate 선택'],
              [[q]+[links[q][k] for k in ['plain_well','premise_review_answer_well','fp_unconditional_well','cot_routed_well']] for q in ids]), '',
        '검토문→답변과 gate는 같은 검토문을 받지만, 서로의 출력을 받지 않는다. gate 선택의 Yes 분기는 '
        '검토문 없이 원 질문+무조건 교정 지시로 생성한 답변을 사용한다. 지시·입력 차이가 있어 검토문 하나의 인과 효과가 아니다.', '',
        f'검증: 선택 규칙과 입력 해시, {count}개 정확 인용, Gemma 12개 답변의 점수·해시 및 4개 동일 검토문 입력. '
        '새 생성·재채점·GPU 작업 없음. [해석](findings.md), [원문](cases.md), '
        '[독해 코드](annotations.json), [선정 기록](selection.json), [기존 판정 설명](judge_explanations.json).', '']
    evidence=['# 고정 표본의 원문', '']
    for q in ids:
        r0=index['gemma',q]
        evidence += [f'## {q}', '',r0['question'],'',
                     '참조: '+json.dumps({k:r0[k] for k in ['premise','correction','tpq_premises']},ensure_ascii=False),'']
        for m,label in MODELS.items():
            r=index[m,q]
            evidence += [f'### {label}', '',f"Direct={r['direct']}; Review={r['review_gate']}; Plain={r['plain_score']}",
                         '', '검토문:', '',r['review'],'','독립 Plain:', '',r['plain'],'']
        for stage in ['premise_review_answer','fp_unconditional']:
            evidence += [f'### Gemma {stage}', '',records[q,stage]['response'],'']
    summary=dict(selected_ids=ids,question_n=len(ids),model_condition_rows=len(selected),
                 quotes_checked=count,gemma_scores_checked=12,
                 selected_scores={q:links[q] for q in ids},sources=SOURCES,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'README.md').write_text('\n'.join(md))
    (OUT/'cases.md').write_text('\n'.join(evidence))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'judge_explanations.json').write_text(json.dumps(judges,ensure_ascii=False,indent=2)+'\n')
    print(f'Checked fixed selection of {len(ids)} questions; {count} excerpts; 12 saved answer scores/hashes.')


if __name__=='__main__':
    main()
