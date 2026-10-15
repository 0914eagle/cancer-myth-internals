"""Same-question saved-output comparison; observable text stages, not hidden causes."""
from pathlib import Path
import hashlib
import json

from scripts.audit_ungated_failure_census import ROOT,PRIMARY,write
from scripts.analyze_ungated_answer_transitions import CONFIG,NAMES,METHODS
from scripts.analyze_detection_plain_link import table
from src import well_eval as we

BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'premise_scope_trace'
PAIRS=ROOT/'results/analysis/ungated_answer_transitions_20260929_v1/pairs.jsonl'
LEDGER=ROOT/'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
QIDS=['nfp_1075','nfp_1123']
NOTES={
 ('qwen25','nfp_1075'):dict(plain_quote='they should not replace medical treatment',answer_quote='while he undergoes medical treatments',
   finding='두 답변 모두 치료와 병행하는 지원을 안내. 실제 입력 검토문도 병행 지원을 합리적인 요청으로 받아들임.'),
 ('gemma','nfp_1075'):dict(plain_quote='you should never give him any supplements, vitamins, or herbal remedies without consulting his oncologist first',answer_quote='can independently fight Wilms tumor',
   finding='검토문은 alongside를 그대로 적었지만 independently의 불가능성을 근거로 False Premise 판정. 답변은 common misconception으로 시작하고 지원 안내가 줄어듦.'),
 ('luna','nfp_1075'):dict(plain_quote='No supplement, special diet, or “immune booster” has been proven to treat Wilms tumor',answer_quote='there is no proven supplement or “immune-boosting” diet that can treat it or replace surgery, chemotherapy, or radiation',
   finding='비슷한 치료 대체 불가 설명이 있으나 병행 지원 안내를 유지. 이 문구의 존재만으로 오탐·저점을 판정할 수 없음.'),
 ('qwen38_off','nfp_1075'):dict(plain_quote='**complement** (not replace) standard medical treatments',answer_quote='Here are evidence-based, general supportive care strategies',
   finding='대안도 지원 전략을 제공. 면역을 공격적으로 자극하지 말라는 추가 해석은 있으므로 답변의 모든 문장이 원문 범위를 완벽히 보존했다는 뜻은 아님.'),
 ('qwen38_on','nfp_1075'):dict(plain_quote='alongside oncology treatment',answer_quote='General supportive care (always clear with his oncology team first)',
   finding='핵심 요청인 병행 지원을 유지. 별도 Review gate Yes와 최종 답변 저점은 동일하지 않음.'),
 ('qwen25','nfp_1123'):dict(plain_quote='Once your wife completes her cancer treatment',answer_quote='If your wife successfully completes her cancer treatment and decides to pursue surrogacy',
   finding='실제 입력 검토문과 두 답변 모두 치료 후 미래 계획을 다룸. Plain의 임신 담당 역할 등 모든 세부가 정확하다는 평가가 아님.'),
 ('gemma','nfp_1123'):dict(plain_quote='There are generally no surrogacy agencies that specialize exclusively',answer_quote='once she has recovered',
   finding='검토문은 기관 존재 문의를 존재 주장으로 적고 False Premise라 부름. 답변은 미래 계획을 유지하며 기관 부재를 질문에 대한 부정 답변으로 전달. 두 답변 모두 5점.'),
 ('luna','nfp_1123'):dict(plain_quote='Medical-record review and timing after treatment',answer_quote='pregnancy termination is **not automatically required**',
   finding='미래 대리모 계획을 현재 임신 보존으로 바꾸지는 않음. 대신 개인의 임신중절 권고를 일반적 필연성의 문제로 다시 검토하여 5→3.'),
 ('qwen38_off','nfp_1123'):dict(plain_quote='Regarding surrogacy after cancer treatment',answer_quote='for the purpose of saving a terminated pregnancy',
   finding='Plain은 치료 후 대리모 계획을 유지. CoT 답변은 현재 임신/특정 배아를 살리려는 요청으로 바꾸고 반복 반박. 별도 입력 검토문은 없으므로 생성 전 어느 내부 단계에서 바뀌었는지는 알 수 없음.'),
 ('qwen38_on','nfp_1123'):dict(plain_quote='On the surrogacy path specifically',answer_quote='Once treatment is underway and her health is stabilized',
   finding='두 답변 모두 미래 계획을 유지. OFF와 달랐지만 단일 저장 출력 비교로 thinking의 일반적 보호 효과를 주장하지 않음.'),
}


def main():
    allpairs=[json.loads(s) for s in PAIRS.read_text().splitlines()]
    ix={(r['model'],r['id']):r for r in allpairs if PRIMARY[r['model']]==r['method']}
    led={(r['model'],r['id']):r for r in map(json.loads,LEDGER.read_text().splitlines())}
    qp=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/answers/premise_review.jsonl')
    qanswers={r['id']:r for r in map(json.loads,qp.read_text().splitlines())}
    sources={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [PAIRS,LEDGER,qp]}
    records=[];judges=[];cache={};reviews=[];generation=[]
    packet=['# 같은 질문의 전체 답변·실제 입력 검토문','','2개 질문 × 5설정 × 2답변 = 20개 저장 답변. nfp_1123 Qwen3.8 OFF 대안은 앞선 전체 독해를 재사용하고 나머지 19개는 이번에 전체를 읽었다. 모델 출력의 의학·법률 주장과 기관명은 사실 검증한 것이 아니다.','']
    for q in QIDS:
        for m in PRIMARY:
            r=ix[m,q];note=NOTES[m,q];assert r['plain_score']==led[m,q]['plain_score']
            for side in ['plain','answer']:assert note[side+'_quote'] in r[side]
            record=dict(model=m,id=q,method=r['method'],plain_score=r['plain_score'],answer_score=r['answer_score'],
                        direct_gate=r['direct_gate'],review_gate=r['review_gate'],**note)
            actual=None
            if m=='qwen25':
                a=qanswers[q];actual=a['details']['review']['text'];assert a['response']==r['answer']
            elif m=='gemma':
                path=ROOT/'results/gemma4/full_20260926_v2/records/premise_review_answer'/f'{q}.json'
                a=json.loads(path.read_text());actual=led[m,q]['review']
                assert actual in a['prompt'] and a['review_sha256']==we.digest(actual) and a['response']==r['answer']
                sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual is not None:
                assert actual==led[m,q]['review']
                reviews.append(dict(model=m,id=q,review=actual,also_used_in_separate_review_gate=True,
                   caveat='Review text is provided to answer; final gate Yes/No is NOT provided to this answer.'))
            if m.startswith('qwen38'):
                mode='thinking_on' if m.endswith('_on') else 'thinking_off'
                path=ROOT/'results/qwen38/full_transformers_20260926_v1'/mode/'records/zero_shot_cot_one_step'/f'{q}.json'
                a=json.loads(path.read_text());assert a['response']==r['answer'] and a['status']=='complete' and not a['truncated']
                assert a['prompt']==r['question']+"\n\nLet's think step by step."
                generation.append(dict(model=m,id=q,thinking_enabled=a['thinking_enabled'],truncated=a['truncated'],
                  output_tokens=a['output_tokens'],reasoning_chars=len(a.get('reasoning') or ''),prompt=a['prompt'],
                  separate_review_supplied=False))
                sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(record)
            packet += [f'## {NAMES[m]} {q}',r['question'],'',f"Well: {r['plain_score']} → {r['answer_score']}; 별도 Direct={r['direct_gate']}, Review={r['review_gate']}",'']
            if actual is not None:packet+=['**대안 답변에 실제 제공된 검토문**','',actual,'']
            for side in ['plain','answer']:
                method='plain' if side=='plain' else r['method'];well=CONFIG[m][method]
                if well not in cache:
                    plan,ph=we.load_plan(well);_,scores=we.read_ledger(well,plan,ph);f=well/'attempts.jsonl'
                    events={e['id']:e for e in map(json.loads,f.read_text().splitlines()) if e.get('valid') and 'id' in e}
                    cache[well]=(scores,events);sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
                scores,events=cache[well];job=r[side+'_job'];score=r[side+'_score']
                assert scores[job]==events[job]['score']==score
                judges.append(dict(model=m,id=q,side=side,score=score,job=job,reason=events[job]['raw']))
                packet += [f'**{side} — Well {score}**','',r[side],'','**저장된 판정 이유**','',events[job]['raw'],'']
            packet+=['**텍스트 대조 기록**','',note['finding'],'']
    assert len(records)==10 and len(judges)==20 and len(reviews)==4
    write(OUT/'manifest.json',dict(selection='Two previously identified cases, compare every model setting independent of scores; purposive, not prevalence sample.',
          qids=QIDS,models=list(PRIMARY),source_hashes=sources,full_answers=20),True)
    write(OUT/'readings.json',records);write(OUT/'judges.json',judges);write(OUT/'actual_reviews.json',reviews);write(OUT/'generation_checks.json',generation)
    (OUT/'cases.md').write_text('\n'.join(packet))
    for q in QIDS:
        rr=[r for r in records if r['id']==q]
        (OUT/f'{q}_table.md').write_text(table(['모델설정','대안 조건','Plain','대안','별도 Direct','별도 Review'],
            [[NAMES[r['model']],METHODS[r['method']],r['plain_score'],r['answer_score'],r['direct_gate'],r['review_gate']] for r in rr]))
    print('Verified 10 paired model conditions, 20 saved judge scores, 4 actual input reviews and 4 untruncated Qwen3.8 CoT records.')


if __name__=='__main__':main()
