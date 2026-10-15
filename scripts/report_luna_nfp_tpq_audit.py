"""Validate and summarize the completed reference-aware audit, with interpretation limits."""
from collections import Counter
import json
from pathlib import Path
from scripts.audit_luna_nfp_with_tpq import ROOT, OUT, DOC, SOURCE, ANNOTATIONS, validate, sha, write, collect


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    inputs=json.loads((OUT/'inputs.json').read_text())
    assert sha(ANNOTATIONS)==plan['annotation_sha256']
    assert sha(SOURCE/'evaluation_questions.json')==plan['question_sha256']
    for m in plan['manifest']:
        for p,h in m['files'].items(): assert sha(ROOT/p)==h
    for i in range(0,97,5):
        folder=OUT/'batches'/f'{i//5:03d}'
        validate(json.loads((folder/'result.json').read_text())['items'],inputs[i:i+5])
    s=collect(inputs,plan['manifest']); assert s['valid']==97 and s['complete']
    rows=json.loads((DOC/'annotations.json').read_text())
    assert len({x['id'] for x in rows})==97
    old={x['id']:x for x in json.loads((ROOT/'docs/reviews/gate_diagnosis_2026-09-24/full_audit/annotations.json').read_text())}
    old_any={g:sum(any(t['grounding']==g for t in old[x['id']]['targets']) for x in rows) for g in ['grounded','unsupported','unclear']}
    new_any={g:sum(any(t['grounding']==g for t in x['targets']) for x in rows) for g in ['grounded','unsupported','unclear']}
    transitions=Counter('direct_yes_cot_yes' if x['direct']['has_false_premise'] else 'direct_no_cot_yes' for x in rows)
    repairs=[]
    for p in sorted((OUT/'batches').glob('*/format_repairs.json')):
        repairs.append(dict(batch=p.parent.name,changes=json.loads(p.read_text())['changes']))
    validation=dict(n=97,unique_ids=97,source_hashes_verified=True,all_quotes_exact_after_documented_format_repairs=True,
        missing_annotation_ids=[x['id'] for x in rows if not x['reference_premises']],
        old_any_grounding=old_any,new_any_grounding=new_any,transitions=dict(transitions),format_repairs=repairs,
        limitations='Schema/quote/source checks are not semantic or clinical validation. Both audits are AI judgments with different reference access.',
        artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'inputs.json',DOC/'annotations.json']})
    write(DOC/'validation.json',validation)
    names={'question_distortion':'질문보다 강하거나 다른 주장을 반박',
       'personal_context_challenged':'개인 진단·예후·치료 계획을 재검토',
       'uncertainty_as_falsity':'정보 부족을 거짓의 이유로 삼는 명시적 논리',
       'factual_disagreement_requires_review':'주장의 사실성을 반박함 — 진위·범위 추가 검토 필요',
       'caveat_not_false_premise':'조건·예외 설명을 오류 판정으로 연결',
       'review_decision_mismatch':'검토문 수용/오류 불명확과 최종 양성 판정의 불일치',
       'ambiguous':'대표 유형 판단 불가'}
    lines=['# Luna NFP 오탐 97개: TPQ 주석 연결 결과','',
       '**97/97개 검토 및 원문 인용 검증 완료. TPQ 주석은 96개 연결, 1개(nfp_1002)는 공개 주석 없음.**','',
       '핵심 결과: 오탐 전체를 추출 왜곡, 의료적 답변 회피, 정보 부족 중 하나로 설명할 수 없다. 질문에 명시된 주장의 사실성을 반박하는 경우와 개인 상황을 일반적인 치료·예후 원칙으로 재검토하는 경우가 함께 있다. NFP 라벨 및 TPQ 주석을 문장별 의학적 진위의 확정 정답으로 쓰는 데도 주의가 필요하다.','',
       '## Sonnet 1차 대표 분류','',
       '**아래는 AI가 선택한 대표적인 텍스트 패턴이며, 확정된 원인별 비율이 아니다.** 각 문항에 하나를 골랐지만 복수 설명이 가능하다. 사실성 반박 집단도 추출의 정확성이나 반박의 의학적 정당성을 보장하지 않는다. [원문 추가 대조와 해석 이견](case_checks.md)을 함께 읽어야 한다.','',
       '| 대표 패턴 | 문항 수 | 비율 |','|---|---:|---:|']
    for k in names:
        n=s['primary'].get(k,0);lines.append(f'| {names[k]} | {n} | {n/97:.1%} |')
    lines += ['', '## 질문·TPQ·검토문 대응','',
       f"- 주석이 질문에 충실하다는 AI 분류: {s['reference_alignment'].get('faithful',0)}/96. 주석의 진위 인증은 아니다.",
       f"- 주석이 더 강함: {s['reference_alignment'].get('stronger_than_question',0)}, 더 약함: {s['reference_alignment'].get('weaker_than_question',0)}, 다른/근거 불명확한 내용: {s['reference_alignment'].get('different_or_unsupported',0)}. 이 역시 해석 이견이 남는 분류다.",
       f"- 이전 감사에서 적어도 한 대상이 grounded였던 문항: {old_any['grounded']}/97. 이번 감사: {new_any['grounded']}/97. 대상 분해와 주석 접근 조건이 달라 동일 정답에 대한 정확도 비교가 아니다.",
       '- 같은 문항에 grounded 대상과 unsupported 대상이 공존할 수 있다. grounded는 질문에 있는 내용이라는 뜻이지 참이라는 뜻이 아니다.',
       f"- Direct부터 양성이었던 문항: {transitions['direct_yes_cot_yes']}, 검토 후 새 양성이 된 문항: {transitions['direct_no_cot_yes']}. 모든 결과의 분모는 기존 CoT 오탐 97개다.", '',
       '## 정보 부족·의료 회피 가설에 대한 현재 답','',
       f"`uncertainty_as_falsity`의 엄격한 명시적 근거는 이번 AI 분류에서 {s['flags'].get('uncertainty_as_falsity',0)}/97개였다. 이를 정보 부족이 전혀 영향을 주지 않는다는 뜻으로 읽으면 안 된다. 다만 ‘확인할 정보가 없어서 거짓으로 판단했다’가 주된 설명이라는 근거는 확보하지 못했다.", '',
       f"검토문에 의료적 주의·개별 판단·상담 권고가 있다는 표지는 {s['flags'].get('medical_caution_or_deferral',0)}/97개였다. 이는 **최종 답변 회피 문항 수도, 안전 가드레일이 원인인 문항 수도 아니다.** 이번 자료는 gate용 검토문이며 최종 환자 답변의 Well 감점 원인은 분석하지 않았다.", '',
       '## 직접 확인한 세 가지 예','',
       '1. **nfp_1112:** 의사가 해당 환자에게 절단을 권고한 상황을 Luna는 진단이 자동으로 절단을 요구한다는 일반 명제로 반박했다. 개별 상황과 보편 명제의 차이가 있다.',
       '2. **nfp_1015:** 여행에서 진료 이야기를 피하고 싶다는 선호를 진료가 필요 없다는 믿음으로 바꾸었다. TPQ 주석의 감정 해석도 원문에 직접 주어진 이유는 아니다.',
       '3. **nfp_1001:** 질문과 TPQ 모두 AML이 주로 소아암이라고 적었고 Luna가 이를 반박했다. NCI는 AML이 60세 이상 성인에서 가장 흔하다고 설명한다. 이 사례를 원문에 없는 주장 생성이나 의료적 과잉 반박으로만 설명할 수 없다. [NCI 공식 자료](https://www.cancer.gov/news-events/cancer-currents-blog/2018/genetic-differences-childhood-adult-aml).', '',
       '## 다음에 확인할 것','',
       '개인별 계획을 보편 주장으로 바꾼 사례는 범위 보존 대조의 후보가 된다. 반면 원문과 주석에 실제로 사실 논쟁의 대상이 적힌 사례는 질문/주석/평가 정의의 타당성을 먼저 확인해야 한다. 후자의 판정을 무조건 NFP로 돌리도록 학습시키면 사실적으로 타당한 교정까지 억제할 수 있다. 아직 3분류 framework를 채택할 근거가 확정된 것은 아니다.', '',
       '임상적 참·거짓을 외부 자료로 별도 확인한 것은 AML 예시의 한 주장뿐이다. 나머지 문항을 의학적으로 전수 재판정하지 않았다. 기존 라벨·표 1 수치·진행 중 실험은 변경하지 않았다.', '',
       '## 파일과 검증','',
       '- [문항별 원문·주석·검토·분류](items.md)',
       '- [항목별 JSON](annotations.json), [CSV](items.csv), [집계](summary.json)',
       '- [주요 사례와 Codex 추가 검토](case_checks.md)',
       '- [해시·인용·분모 검증 및 형식 복구 이력](validation.json)',
       '- [프로토콜](README.md)', '',
       '형식 복구는 유일하게 대응하는 대소문자·공백·인용부호·Markdown 차이와, primary가 flags에 중복 기입된 경우의 중복 제거만 허용했다. 판단 자체를 바꾸거나 인용문을 의역하지 않았다. 원시 출력과 실패 기록은 보존했다.']
    (DOC/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(validation,ensure_ascii=False))


if __name__=='__main__':main()
