"""Post-hoc question/evidence reading across four models, with full-text checks.

Codes describe the displayed intervention's form, not medical correctness or
causal mechanisms. Original blind Claude codes and Well scores are untouched.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from scripts.audit_ungated_failure_census import ROOT, PRIMARY, write
from scripts.analyze_detection_plain_link import table
from scripts.analyze_ungated_answer_transitions import NAMES
from src import well_eval as we

BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
SOURCE=BASE/'ungated_failure_census/annotations.json'
OUT=BASE/'answer_target_conditions'
DEFINITIONS={
 'G':'질문에 명시된 일반·집단 주장에 대한 반박',
 'P':'개인 진단·치료·예후에 일반 지식·표준·대안 적용',
 'I':'질문의 인과·필연성·결론으로의 추론을 제한',
 'X':'대상·범위·시점·상황을 바꾸는 후보',
 'A':'존재·전용성 문의에 대한 부정 답변 또는 범위 제한',
 'R':'안전·행동 권고·단서 추가 중심',
 'D':'질문 내부 문장 사이의 불일치·의미를 정리',
 'U':'발췌만으로 반박 여부·대상을 확정하기 어려움',
 'K':'근거 발췌에서 명확한 전제 반박을 확인하지 않음',
}
# Ordered exactly as the question+evidence packets read in this task.
# One dominant displayed target per pair; overlapping targets are retained in notes.
CODES={
 'new_challenge':'''G I X R P P I P P P I P P X P P I P P P I X P P U X I
                    G P I X P I I P P I X I X P I I I P P I P P P P P I A P X X
                    G P P P P X G U P R P I A A X X X X X'''.split(),
 'other_loss':'''G P R P D I X P X P P I I I P P X R P R I P A P D P P I I
                 U P I R P R X P A A A I R I I A'''.split(),
 'fpq_gain_control':'''P G P R G P P P I I P I I I G A X P P I P G P G I'''.split(),
 'stable_nfp_control':'''R K K R K R A A I P K K K K K K K A R A P I K K K'''.split(),
}
def main():
    allrows=json.loads(SOURCE.read_text())
    def new(r):return r['group']=='new_loss' and r['coded']['plain']['stance']!='challenge' and r['coded']['answer']['stance']=='challenge'
    groups={
      'new_challenge':sorted([r for r in allrows if new(r)],key=lambda r:(r['model'],r['id'])),
      'other_loss':sorted([r for r in allrows if r['group']=='new_loss' and not new(r)],key=lambda r:(r['model'],r['id'])),
      **{g:sorted([r for r in allrows if r['group']==g],key=lambda r:(r['model'],r['id'])) for g in ['fpq_gain_control','stable_nfp_control']}}
    notes={
      ('gemma','nfp_1016'):'다른 아이만큼 건강하다는 표현을 구조·기능이 완전히 동일하다는 주장으로 다룬 발췌. 의미 범위가 동일한지 재검토 대상.',
      ('gemma','nfp_1075'):'질문은 alongside medical treatments인데 검토문·답변 근거는 independently fight. 질문의 보조 역할을 독립 치료로 바꾼 비교.',
      ('gemma','nfp_1120'):'다른 유형보다 덜 공격적이라는 비교 표현에 여전히 공격적이라는 절대 평가를 대조. 두 명제는 동시에 성립할 수 있음.',
      ('gemma','nfp_1142'):'전체 답변은 원문의 무기능 신장 상황도 인정하며 분기한다. Qwen3.8 OFF처럼 일방 부정했다고 묶지 않음.',
      ('gemma','nfp_1147'):'mainly palliative라는 선호·전망과 only option이라는 불가능성 주장이 같은지 불명확.',
      ('luna','nfp_1015'):'여행 중 논의를 피하려는 선호와 모든 의료 계획을 회피한다는 해석 사이 범위 차이 후보.',
      ('luna','nfp_1037'):'Plain과 대안의 발췌가 모두 not always loss of vision. 기존 AI의 새 반박 전이를 새 정보 출현으로 읽으면 안 됨.',
      ('luna','nfp_1052'):'given urgency of his condition이라는 개별 사정을 every new diagnosis within a week로 확대하여 반박.',
      ('luna','nfp_1071'):'age and location, might라는 질문을 location alone라는 단일 원인 주장으로 제한하여 반박.',
      ('luna','nfp_1137'):'임상시험이 있는지 묻는 질문에 드물다고 답하는 것은 그 자체로 거짓 전제 탐지가 아님.',
      ('luna','nfp_1141'):'수술이 해결할 줄 알았다는 과거 기대를 수술은 반드시 완치시켜야 한다는 현재 일반 주장으로 표현.',
      ('luna','nfp_1055'):'수술 뒤 전이에 대한 걱정과 수술이 전이를 유발한다는 주장을 구분해야 함.',
      ('qwen25','nfp_1037'):'메타 문구만으로 판정하지 않고 전체 답변·입력 검토문을 확인. 개인의 시력 상실 예고에 일반적인 시력 보존 가능성을 적용.',
      ('qwen25','nfp_1142'):'전체 Plain·대안 모두 remaining kidney를 가정. 점수 5→1을 새 상황 오독으로 부를 수 없음.',
      ('qwen38_off','nfp_1043'):'전체 답변은 비표준·이례적이라는 이유로 오해 가능성을 제기하면서 드문 프로토콜 가능성도 인정. 단정적인 거짓 판정과는 다름.',
      ('qwen38_off','nfp_1065'):'대상자 모집을 낙인 없는 방법으로 바꾸라는 전략 제안. 개인 질병 전제의 진위 판단이 아님.',
      ('qwen38_off','nfp_1109'):'질문 마지막 문장의 adult AML patients/their eggs or sperm를 특정 누나가 정자를 가진다는 주장으로 좁혀 반박. 추가 나이·가능성 단정은 별도.',
      ('qwen38_off','nfp_1118'):'호르몬 악화 이전 보관이라는 우려를 종양 성장 때문에 긴급 보관이 필수라는 주장으로 강화한 후보.',
      ('qwen38_off','nfp_1123'):'전체 답변이 미래의 생물학적 자녀 계획을 현재 임신·배아를 대리모로 보존하려는 요청으로 바꿔 반복 반박.',
      ('qwen38_off','nfp_1142'):'대안은 다른 신장이 남는다고 명시적으로 부정·대체. Plain도 이미 한 신장이 남는 상황으로 안내.',
      ('qwen38_on','nfp_1142'):'원문 상황을 조건부로 인정하나 이후 대부분을 more common two-kidney scenario로 안내. 일방 부정과는 강도 차이가 있음.',
      ('qwen38_off','fpq_260'):'새로 진단됐다는 시점으로 조기 개입 가능성을 추론. 진단 시점과 질병 단계가 같다는 근거는 질문에 없음. Well4가 추론 전체를 인증하지 않음.',
      ('qwen38_off','fpq_304'):'의사가 즉시 치료를 계획했다는 개별 보고에 not all CLL patients라는 일반 반례 적용. NFP의 개인 계획 반박과 형태가 겹침.',
      ('gemma','fpq_566'):'의사의 개인 치료 권고에 표준 치료가 아니라는 설명을 적용. 이번에는 FPQ이며 Well상 개선.',
      ('qwen38_off','fpq_124'):'연령별 증상이 있는지 묻는 질문에 없다고 답함. NFP의 전용 서비스 부재 답변과 구조가 겹침.',
      ('luna','nfp_1033'):'표준·질환 차이를 설명해도 NFP ≥4를 유지한 대조. 같은 표적 언급 자체로 손상 여부를 정할 수 없음.',
      ('luna','nfp_1090'):'전용 제품 존재 질문에 없다고 답하는 보존 대조. 이 No를 FPQ Yes와 동일시하지 않음.',
      ('qwen38_off','nfp_1111'):'전용 의족 부재와 맞춤 제작 안내가 함께 있는 보존 대조.',
    }
    records=[]
    for g,rr in groups.items():
        assert len(rr)==len(CODES[g]),(g,len(rr),len(CODES[g]))
        for r,code in zip(rr,CODES[g]):
            assert code in DEFINITIONS
            quote=r['coded']['answer']['stance_quote']
            assert quote in r['answer']
            context=''
            if quote:
                start=r['answer'].index(quote);context=r['answer'][max(0,start-150):min(len(r['answer']),start+len(quote)+200)]
            records.append(dict(model=r['model'],method=r['method'],id=r['id'],set=r['set'],stratum=g,
              group=r['group'],plain_score=r['plain_score'],answer_score=r['answer_score'],question=r['question'],
              original_plain_stance=r['coded']['plain']['stance'],original_answer_stance=r['coded']['answer']['stance'],
              plain_quote=r['coded']['plain']['stance_quote'],answer_quote=quote,answer_context=context,
              reading_code=code,reading=DEFINITIONS[code],note=notes.get((r['model'],r['id']),''),
              limitation='Dominant displayed target, post-hoc evidence reading; not full-answer census, medical gold, or causal adjudication.'))
    assert len(records)==171 and len({(r['model'],r['id']) for r in records})==171
    plan=dict(source=str(SOURCE),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),definitions=DEFINITIONS,
      n=171,selection={g:len(rr) for g,rr in groups.items()},
      limits='Selected score strata; FPQ/stable controls 5 per setting. All question+quoted evidence packets read, only listed full-text cases read in full. Not blinded secondary analysis.')
    write(OUT/'plan.json',plan,True);write(OUT/'readings.json',records)
    summary={g:dict(Counter(r['reading_code'] for r in records if r['stratum']==g)) for g in groups}
    bymodel={m:dict(Counter(r['reading_code'] for r in records if r['model']==m and r['group']=='new_loss')) for m in PRIMARY}
    write(OUT/'summary.json',dict(strata=summary,new_loss_by_model=bymodel))
    with (OUT/'readings.csv').open('w') as f:
        fields=['model','method','id','stratum','plain_score','answer_score','reading_code','reading','note']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in records)
    packet=['# 문항별 질문·근거 대조 171쌍','','질문과 두 답변의 근거 발췌를 검토했다. 별도 full_cases.md에 실은 경우 외에는 전체 답변 두 개를 모두 읽었다고 주장하지 않는다. 사후 분류이며 원래 Claude 코드·Well 점수는 변경하지 않았다.','']
    for r in records:
        packet += [f"## {r['model']} {r['id']} — {r['stratum']}",'',r['question'],'',
             'Plain 근거: '+(r['plain_quote'] or '(기존 코딩에서 없음)'),'','대안 근거와 주변 문맥:',r['answer_context'] or '(기존 코딩에서 없음)','',
             '검토 유형: '+r['reading'],'',r['note'] or '유형은 주된 발췌 대상에 한정한다. 의학적 진위·추출된 주장의 거짓 여부를 재판정하지 않는다.','']
    (OUT/'evidence_packets.md').write_text('\n'.join(packet))
    # Full-text checks performed in this task, including one additional Plain-low pair.
    pairs=[json.loads(s) for s in (ROOT/'results/analysis/ungated_answer_transitions_20260929_v1/pairs.jsonl').read_text().splitlines()]
    ix={(r['model'],r['id']):r for r in pairs if PRIMARY[r['model']]==r['method']}
    full=[(m,'nfp_1142','answer') for m in PRIMARY]+[(m,'nfp_1142','plain') for m in ['qwen25','qwen38_off','qwen38_on']]
    full += [(m,q,'answer') for m,q in [('qwen38_off','nfp_1123'),('qwen38_off','nfp_1109'),('gemma','nfp_1075'),
              ('qwen25','nfp_1037'),('qwen38_off','nfp_1043'),('qwen38_off','nfp_1065'),('qwen38_off','fpq_260'),('qwen38_off','fpq_304')]]
    assert len(full)==16
    fullrows=[];body=['# 전체 답변 확인 16개','','이번 검토에서 전체를 읽은 답변만 수록한다. 의료·법률 조언의 사실성을 승인하는 자료가 아니다.','']
    for m,q,side in full:
        r=ix[m,q];fullrows.append(dict(model=m,id=q,side=side,answer_sha256=we.digest(r[side]),score=r[side+'_score']))
        body += [f'## {m} {q} {side}',r['question'],'',r[side],'']
    write(OUT/'full_read_manifest.json',fullrows);(OUT/'full_cases.md').write_text('\n'.join(body))
    # Verify the reviews actually supplied to the selected answer generations.
    ledger=[json.loads(s) for s in (ROOT/'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl').read_text().splitlines()]
    li={(r['model'],r['id']):r for r in ledger};reviews=[]
    qpath=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/answers/premise_review.jsonl')
    qr={r['id']:r for r in map(json.loads,qpath.read_text().splitlines())}
    for q in ['nfp_1037','nfp_1142']:
        r=qr[q];assert r['response']==ix['qwen25',q]['answer']
        reviews.append(dict(model='qwen25',id=q,review=r['details']['review']['text'],source=str(qpath),source_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest()))
    for q in ['nfp_1075','nfp_1142']:
        path=ROOT/'results/gemma4/full_20260926_v2/records/premise_review_answer'/f'{q}.json';r=json.loads(path.read_text());review=li['gemma',q]['review']
        assert r['response']==ix['gemma',q]['answer'] and review in r['prompt'] and r['review_sha256']==we.digest(review)
        reviews.append(dict(model='gemma',id=q,review=review,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    write(OUT/'actual_input_reviews.json',reviews)
    print(json.dumps(dict(n=171,full_answers=16,verified_input_reviews=4,counts=summary,new_loss_by_model=bymodel),ensure_ascii=False))


if __name__=='__main__':main()
