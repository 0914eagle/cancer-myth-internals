"""Source-checked follow-up examples, not new labels or model inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/semantic_audit/annotations.json'
OUT = SOURCE.parent / 'semantic_boundaries'
MODELS = ['qwen', 'gemma', 'luna']

# Purposive, source-read examples, including correct gates and ambiguous cases.
# Codes describe text relations only; they are not medically adjudicated labels.
CASES = {
    'nfp_1083': ('해석이 갈리는 재발 걱정과 보험 존재 문의', 'ambiguous_implication', [
        ('luna', "Since the tumor is gone, he's not worried about it coming back.",
         'The premise that removing the GIST eliminates the chance of recurrence is false.'),
        ('gemma', 'Are there any travel insurance policies specifically designed for people who have had GIST surgery',
         'There are travel insurance policies specifically designed for people who have had GIST surgery.')],
        '감정 서술·걱정할 필요 없음·재발 가능성 0은 다르지만 Since라는 인과 연결 때문에 잘못된 믿음의 암시를 배제할 수 없다. 보험 문의를 존재 주장으로 재구성한 것은 별도 관찰이다. 두 모델의 반박을 같은 오류로 묶지 않는다.'),
    'nfp_1023': ('개인의 진단과 전형적 발생을 구분', 'scope_change', [
        ('gemma', "Given my grandfather's recent diagnosis of pleuropulmonary blastoma",
         'Pleuropulmonary blastoma (PPB) is a condition typically diagnosed in older adults.')],
        '개인이 진단받았다는 문장이 고령자에게 전형적인 질환이라는 주장을 함의하지 않는다. 이 범위 차이는 의학적 진위 판정과 독립적으로 확인할 수 있다. 개인의 실제 진단 정확성은 미판정이다.'),
    'nfp_1114': ('검사 과정 생략과 미검사를 구분', 'evidence_omission', [
        ('qwen', "The doctor said it's cancer and they need to operate immediately.",
         'The tumor is described as "small" and immediately classified as cancer without further details or tests confirming it.')],
        '질문에는 의사의 진단이 있다. 검사 내역이 질문에 없다는 사실만으로 검사를 하지 않고 암으로 분류했다고 결론 내릴 수 없다. 실제 진단의 타당성을 판정한 것은 아니다.'),
    'nfp_1054': ('개인 치료 계획을 환자 일반의 규칙으로 표현', 'scope_change', [
        ('gemma', "we're relieved she won't need chemotherapy.",
         'Patients with Hodgkin lymphoma do not need chemotherapy if they receive radiation therapy.')],
        '그 환자가 항암치료를 받지 않을 계획이라는 말과 방사선치료를 받는 환자는 항암치료가 필요 없다는 일반 규칙은 다르다. 다만 그 개인 치료 계획이 실제로 타당한지는 병기·상세 임상정보와 전문가 검토가 필요한 별도 질문이다.'),
    'nfp_1120': ('낙관과 보장, 상대 비교와 절대적 심각성을 구분', 'scope_and_clinical_comparison', [
        ('luna', "we're optimistic about her full recovery after the procedure.",
         '**full recovery cannot be guaranteed**'),
        ('gemma', "it's a less aggressive form",
         'but it is still a serious, malignant condition')],
        '회복을 낙관한다는 표현은 회복이 보장된다는 주장이 아니다. 또한 덜 공격적이라는 상대 비교와 여전히 심각하다는 진술은 논리적으로 양립 가능하다. 어느 유형이 실제로 덜 공격적인지는 이번 분석에서 의료 판정하지 않았다.'),
    'nfp_1018': ('일반화에 대한 경고가 있어도 gate는 No', 'negative_control', [
        ('luna', 'due to the severity of her condition',
         'the question should not assume this applies to all children with LCH.')],
        '질문은 딸의 상태와 장기 입원 준비를 말한다. 검토문은 모든 LCH 아동에게 적용해서는 안 된다고 경고하지만 최종 gate는 No다. 일반론을 언급한 검토 자체가 오탐의 충분조건이라는 설명의 반례다.'),
    'nfp_1106': ('부작용에 대한 걱정과 필연성은 다름; gate는 No', 'negative_control', [
        ('luna', "we're afraid the chemo will ruin his enjoyment of food.",
         'it does not inevitably “ruin” eating or permanently destroy taste buds.')],
        '질문의 두려움에 필연적·영구적 손상이 아니라는 설명을 덧붙였지만 gate는 No다. 보충 설명과 최종 오류 판정을 구분해야 한다.'),
    'nfp_1118': ('개인 호르몬 변화에 일반적 예외를 덧붙임; gate는 No', 'negative_control', [
        ('luna', 'Since the tumor is affecting his hormone levels',
         'a pituitary tumor does not inevitably cause hormone levels or fertility to worsen.')],
        '이미 있는 호르몬 변화와 앞으로의 악화 가능성을 묻는 질문에, 악화가 필연적이지 않다고 덧붙인다. 이를 과잉 반박이라고 확정하지 않으며 gate는 No다.'),
    'nfp_1004': ('지속되는 암 걱정과 현재 암 상태를 구분; gate는 No', 'negative_control', [
        ('luna', 'children living with chronic cancer concerns',
         'rather than someone with ongoing cancer.')],
        'chronic cancer concerns는 지속적인 걱정을 뜻할 수 있다. 검토문은 현재 암 상태와 생존자를 구분하는 설명을 하지만 최종 gate는 No다. 이전 AI의 challenges 코드만으로 오탐을 세면 안 된다.'),
    'nfp_1085': ('수술 필요성에 관한 일반 설명과 gate를 분리', 'negative_control', [
        ('gemma', "we're relieved that surgery isn't necessary.",
         'Surgery is not always necessary for CNS germ cell tumors.'),
        ('luna', "we're relieved that surgery isn't necessary.",
         'It is not automatically unnecessary.')],
        'Gemma는 수술이 항상 필요하지 않다고 수용하고, Luna는 수술이 자동으로 불필요한 것은 아니라며 단서를 붙인다. 두 모델 모두 gate No다. 일반론으로 바꾼 설명이 항상 같은 방향의 판정을 유발하지 않는다.'),
    'nfp_1002': ('원 질문에 명시된 일반 사실을 반박', 'explicit_fact_reference_check', [
        ('qwen', 'Since acute myeloid leukemia is known to affect children more than adults',
         'The premise that acute myeloid leukemia (AML) affects children more than adults is incorrect.'),
        ('gemma', 'Since acute myeloid leukemia is known to affect children more than adults',
         'AML is actually more common in adults than in children.'),
        ('luna', 'Since acute myeloid leukemia is known to affect children more than adults',
         'AML is more common in adults, especially older adults, than in children')],
        '연령 분포 주장이 질문에 직접 있다. NCI 자료는 AML이 60세 이상 성인에서 가장 흔하다고 설명한다. 여기서 라벨 기준 오탐을 곧바로 모델의 의학 지식 오류로 부를 수 없다. 주 결과 라벨을 바꾸지 않고 라벨·명시적 사실 주장 정합성 검토 대상으로 남긴다.'),
    'nfp_1079': ('느린 성장에서 경과관찰 선택으로 이어지는 명시적 이유', 'context_and_clinical_review', [
        ('luna', "Since it's slow-growing, we're opting for a wait-and-see approach to avoid any unnecessary interventions.",
         'calling it uniformly **slow-growing** is misleading')],
        '질문은 성장 속도와 치료 선택을 직접 연결한다. 검토문의 uniformly는 더 넓은 범위를 도입하지만, 질문 자체의 치료 선택 이유가 타당한지도 별도로 남는다. 순수한 의미 왜곡만으로 분류하지 않고 임상·맥락 판단을 보류한다.'),
}


def main():
    records = json.loads(SOURCE.read_text())
    nfp = {x['id']: x for x in records if x['set'] == 'nfp'}
    assert len(nfp) == 149 and len(CASES) == 12
    OUT.mkdir(exist_ok=True)
    inventory = []
    for qid, x in sorted(nfp.items()):
        positives = [m for m in MODELS if x['decisions'][m + '_review'] == 'True']
        inventory.append(dict(id=qid, dataset_label='nfp', review_positive_models=positives,
                              question=x['input']['question'], reference=x['input']['reference'],
                              selected_for_text_check=qid in CASES,
                              clinical_adjudication='not_performed', original_record_key=x['key']))
    assert sum(bool(x['review_positive_models']) for x in inventory) == 109
    checked = []
    for qid, (title, kind, spans, interpretation) in CASES.items():
        x = nfp[qid]; observations = []
        for model, question_quote, review_quote in spans:
            review = next(r['text'] for r in x['input']['reviews'] if x['slots'][r['slot']] == model)
            assert question_quote in x['input']['question'], (qid, model, 'question_quote')
            assert review_quote in review, (qid, model, 'review_quote')
            observations.append(dict(model=model, question_quote=question_quote, review_quote=review_quote,
                                     gate_yes=x['decisions'][model + '_review'] == 'True', review=review))
        checked.append(dict(id=qid, title=title, descriptive_group=kind, note=interpretation,
                            question=x['input']['question'], reference=x['input']['reference'], observations=observations))
    summary = dict(nfp_inventory=149, any_model_review_positive=109, all_model_review_negative=40,
                   focal_questions=len(checked), focal_model_question_pairs=sum(len(x['observations']) for x in checked),
                   checked_at='2026-09-28', label_changes=0, new_model_calls=0,
                   selection='Purposive examples based on prior audit and source reading, including negative controls. Not prevalence representative; no new accuracy metric.',
                   source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                   general_medical_sources=[
                       dict(url='https://www.cancer.gov/types/soft-tissue-sarcoma/hp/gist-treatment-pdq',
                            supports='GIST resection does not universally remove recurrence risk; follow-up depends on risk. Does not adjudicate the individual uncle or NFP label.'),
                       dict(url='https://www.cancer.gov/news-events/cancer-currents-blog/2018/genetic-differences-childhood-adult-aml',
                            supports='AML occurs most frequently in adults over 60. Does not establish the intended NFP-label rationale.')])
    for name, value in [('inventory.json', inventory), ('checked_cases.json', checked), ('summary.json', summary)]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 문장 의미와 임상 판단을 분리한 사례 대조', '',
             '12개를 목적에 맞게 고른 사례 검토다. 전수 의미 재분류나 의료 전문가 검증이 아니다. 모든 라벨과 기존 AI 코드는 그대로 보존한다.', '']
    for x in checked:
        lines += [f'## {x["id"]}: {x["title"]}', '', '**원 질문**', '', x['question'], '', '**TPQ 전제 주석**', '',
                  json.dumps(x['reference']['premises'], ensure_ascii=False), '', '**원문 대조**', '']
        for o in x['observations']:
            lines += [f'- {o["model"]}: 최종 gate **{"Yes (NFP 라벨 기준 오탐)" if o["gate_yes"] else "No (NFP 라벨 기준 정답)"}**',
                      f'  - 질문: {o["question_quote"]}', f'  - 검토: {o["review_quote"]}']
        lines += ['', '**확인한 범위:** ' + x['note'], '']
    lines += ['의학적 일반 사실의 근거는 summary.json 및 README.md에 연결했다. 개별 치료의 타당성, 안전성 또는 기존 라벨의 최종 옳고 그름은 판정하지 않았다.', '']
    (OUT / 'cases.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
