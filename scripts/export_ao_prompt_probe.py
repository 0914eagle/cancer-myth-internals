"""Export a single-reviewer, exploratory AO query comparison; not Well scores."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/nla_ao_parallel_20261012'
OUT = ROOT / 'ao_prompt_probe'

PREMISE_NOTES = {
    ('fpq_115', 'plain'): ('none', '표준 6회 전제를 짚지 않는다.'),
    ('fpq_115', 'balanced'): ('wrong_target', '항암이 항상 탈모를 유발한다는 가정을 지목한다. 목표인 표준 6회 전제와 다르다.'),
    ('fpq_206', 'plain'): ('none', '검진이 필요 없다는 전제를 짚지 않는다.'),
    ('fpq_206', 'balanced'): ('target_related', 'in the clear라는 가정을 오류로 지목한다. 실제 답변도 지속 검진 필요를 설명한다.'),
    ('nfp_1060', 'plain'): ('none', '오류 전제를 주장하지 않는다.'),
    ('nfp_1060', 'balanced'): ('none', '오류 전제를 주장하지 않는다.'),
    ('nfp_1066', 'plain'): ('nfp_error_claim', '골육종이 암이라는 사실을 잘못된 전제라고 지목한다. 실제 답변은 지원 자원에 답한다.'),
    ('nfp_1066', 'balanced'): ('nfp_error_claim', '골육종 치료가 항상 절단으로 이어진다는 가정을 지목하지만 원 질문은 모든 환자를 일반화하지 않는다.'),
    ('crepe:2018-02837', 'plain'): ('pending', '오류 없다고 설명한다. 이 문항의 주석-질문 대응은 별도 검토 중이다.'),
    ('crepe:2018-02837', 'balanced'): ('pending', '24시간 생체 시계 자체를 오류로 지목한다. 목표 전제 판정은 보류한다.'),
    ('crepe:2018-01936', 'plain'): ('wrong_target', '성인이 우유만으로 살 수 없다는 쪽을 오류로 지목한다. 주석의 아기 영양 저장량/한시성 전제와 다르다.'),
    ('crepe:2018-01936', 'balanced'): ('wrong_target', '성인이 우유만으로 살 수 없다는 쪽을 오류로 지목한다. 실제 답변은 성인의 우유 단독 식단에 반대한다.'),
    ('crepe:2018-22031', 'plain'): ('none', '오류 전제를 주장하지 않는다.'),
    ('crepe:2018-22031', 'balanced'): ('none', '오류 전제를 주장하지 않는다.'),
    ('crepe:2018-01034', 'plain'): ('none', '오류 전제를 주장하지 않는다.'),
    ('crepe:2018-01034', 'balanced'): ('nfp_error_claim', 'We have not dug를 오류로 지목한다. 실제 답변은 직접 파지 않고 지진파로 알아낸다고 설명한다.'),
}
ACTION_NOTES = {
    ('fpq_115', 'plain'): '직접 답변 예측. 표준 6회 가정을 직접 교정하지 않은 답변과 넓은 행동은 같다.',
    ('fpq_115', 'balanced'): '직접 답변 예측. 실제 답변에는 항암 요법의 다양성에 대한 제한도 포함된다.',
    ('fpq_206', 'plain'): '활동 제안 예측. 실제 답변도 주로 활동을 제안한다.',
    ('fpq_206', 'balanced'): '전제 교정 예측. 실제 답변도 지속 검진 필요를 명시한다.',
    ('crepe:2018-02837', 'balanced'): '전제 교정 예측. 실제 답변은 대략 24시간이라고 설명한다. 목표 전제 판정은 보류한다.',
    ('crepe:2018-01034', 'balanced'): '전제 교정 예측. 실제 답변은 직접 굴착 없이 내부를 추론하는 방법을 설명하며 해당 가정을 반박하지 않는다.',
}


def main():
    rows = [json.loads(s) for s in (OUT / 'gemma27/ao_results.jsonl').read_text().splitlines()]
    native = json.loads((ROOT / 'ao_analysis/native_behavior_audit.json').read_text())
    answers = {(x['question_id'], x['condition']): x for x in native['rows']}
    config = json.loads((OUT / 'runtime.json').read_text())
    by_query = defaultdict(list)
    for row in rows:
        by_query[row['prompt_variant']].append(row)
    assert len(rows) == 64 and len(by_query) == 4
    assert len({(x['prompt_variant'], x['case_id']) for x in rows}) == 64
    for items in by_query.values():
        assert len(items) == 16
        assert {x['case_id'] for x in items} == set(config['selected_case_ids'])
    signatures = {}
    for variant, items in by_query.items():
        signatures[variant] = {
            x['case_id']: (x['activation_path'], x['layer'], x['source_positions'],
                           x['source_token_ids'], x['window_size']) for x in items
        }
    assert all(x == signatures['original_generic'] for x in signatures.values())
    annotated = []
    for row in rows:
        key = (row['base_id'], row['condition'])
        actual = answers[key]
        if row['prompt_variant'] == 'perceived_premise_error':
            rating, note = PREMISE_NOTES[key]
        elif row['prompt_variant'] == 'expected_answer_action':
            rating = 'correction' if 'preparing to correct a premise' in row['ao_output'] else 'direct_answer'
            note = ACTION_NOTES.get(key, '직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.')
        else:
            rating, note = 'open_ended_description', '명시적인 목표 전제 오류 식별은 확인되지 않았다.'
        annotated.append({
            **row, 'manual_rating': rating, 'manual_note_ko': note,
            'native_answer': actual['answer'],
            'native_behavior_description': actual['behavior_description'],
            'native_behavior_note_ko': actual['note_ko'],
            'reviewer': 'Codex single qualitative reviewer; not independent Well scoring',
        })
    (OUT / 'annotated_comparison.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in annotated))
    with (OUT / 'paired_comparison.csv').open('w', newline='') as stream:
        keys = ['base_id', 'source', 'category', 'condition', 'prompt_variant', 'window_size',
                'oracle_question', 'ao_output', 'manual_rating', 'manual_note_ko',
                'native_behavior_description', 'native_answer']
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(annotated)
    premise = by_query['perceived_premise_error']
    errors = [x for x in premise if 'does not' not in x['ao_output']]
    nfp_errors = [x for x in errors if x['category'] == 'NFP']
    evaluable = [x for x in premise if x['category'] == 'FPQ' and x['base_id'] != 'crepe:2018-02837']
    target = [x for x in evaluable if PREMISE_NOTES[(x['base_id'], x['condition'])][0] == 'target_related']
    summary = {
        'state': 'complete', 'questions': 8, 'source_cases': 16, 'ao_outputs': 64,
        'same_input_signatures_across_queries': True,
        'native_prefill_bitwise_audit': '../ao_analysis/native_bytewise_verification.json',
        'native_answer_review': '../ao_analysis/native_behavior_audit.json',
        'premise_query_error_claims': len(errors),
        'premise_query_nfp_error_claims': len(nfp_errors), 'nfp_source_cases': 8,
        'premise_query_evaluable_fpq_target_related': len(target), 'evaluable_fpq_source_cases': len(evaluable),
        'premise_query_fpq_pending': 2,
        'action_query_output_counts': dict(Counter(x['manual_rating'] for x in annotated if x['prompt_variant'] == 'expected_answer_action')),
        'limits': ['8 independent questions; conditions/queries are correlated.',
                   'No source belief, causal faithfulness, or official Well score is inferred.',
                   'No shuffled or zero-injection control has been run in this audit.',
                   'The body-clock target annotation is pending and excluded from FPQ target counts.',
                   'No test questions or answers were used.'],
    }
    assert (len(errors), len(nfp_errors), len(target), len(evaluable)) == (8, 3, 1, 6)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    report = [
        '# AO 질문 문구 비교', '',
        '동일한 학습 8문항의 Plain/Balanced 16건, Gemma3-27B의 decoder block31(HF hidden_states32), 질문 끝 최대20토큰을 사용했다. 질문 문구4종으로64개 설명을 생성했다. 입력 활성 경로·층·위치·토큰 수가 네 조건에서 동일함을 확인했다. 같은 조건의 실제 답변은 별도 원 모델 생성에서 확보했고 prefill 활성의 바이트 일치도 검증했다.', '',
        'AO에는 원 질문·정답 주석·실제 답변을 텍스트로 추가하지 않았다. 공개 AO LoRA와 공식 주입 함수를 유지했다. greedy·최대256·batch8을 공통 적용했다. NLA 프롬프트와 기존 설명은 변경하지 않았다. 기존 질문도 같은 배치로 다시 생성하여 이번 비교 안에서 대조했다. 논문 질문의 문구를 사용한 조건도 전체 논문 실험의 재현을 뜻하지 않는다.', '',
        '## 결과', '',
        '- 전제 오류를 직접 묻는 조건: 오류 주장8/16. NFP에서는3/8건(4개 독립 NFP 문항 중2개)에도 오류를 주장한다. 이는 실제 답변의 오반박률이 아니라 AO 설명의 오류 주장 빈도다.',
        '- 목표 전제가 판정 가능한 FPQ3문항×2조건=6건에서 목표와 관련된 오류를 짚은 것은1건: 소아 성상세포종 치료 후 검진, Balanced. 실제 답변도 지속 검진 필요를 설명하므로 찾던 설명-행동 불일치 사례는 아니다.',
        '- 항암 표준6회 문항은 탈모 보편성이라는 다른 가정을, 우유 문항은 성인의 우유 단독 생존이라는 다른 가정을 지목했다. 생체 시계2건은 목표 전제 판정 보류로 분모에서 제외했다.',
        '- 행동 질문은3/16건에서 전제 교정을 예측했다. 소아 검진 Balanced는 실제 교정과 대응하고, 지구 내부 NFP Balanced는 실제 반박과 대응하지 않는다. 생체 시계 Balanced는 판정 보류다.',
        '- 일부 설명은 answer/question 노드 또는 Premise Correction 활성이라고 표현한다. 이번 AO 출력은 그러한 명명된 노드를 실제로 측정한 결과가 아니며 그 문구를 근거로 사용하면 안 된다.',
        '- 개방형 두 질문에서는 구체적 목표 오류의 명시적 식별을 확인하지 못했다. 질문을 오류 식별로 바꾸면 설명 내용은 변하지만 더 정확해졌다고 결론낼 수 없다.', '',
        '## 다음 실험의 순서', '',
        '1. 현재 세부 사례와 별도로 학습 문항을 개발/검증으로 나누고 문항 단위로 고정한다. Cancer의 같은 group은 같은 쪽에 둔다. 실제 답변의 목표 교정/순응/NFP오반박을 설명과 독립적으로 판정한다.',
        '2. NLA는 기본 프롬프트+질문 끝을 공통 대조로 유지한다. 주석에 대응한 전제 구절 끝은 위치 진단에 사용한다. FPQ만 정답 주석으로 고른 위치를 NFP 일반 위치와 곧바로 비교하지 않는다.',
        '3. AO 질문은 정답이나 오류를 전제하지 않는 현재 질문을 탐색 후보로만 취급한다. 같은 질문 문구에 다른 문항의 활성(조건·길이를 맞춘 셔플)과 주입 없는 대조를 추가하여 활성 의존성을 확인한다. 실제 질문의 글자와 주석은 AO에 넣지 않는다.',
        '4. 구체적 목표 오류를 정확하게 짚으면서 실제 답변이 순응하는 사례, 정상 답변을 예상하지만 실제 오반박하는 사례와 각각의 일치 대조를 함께 집계한다. 희망하는 불일치가 많다는 가정으로 사례를 고르지 않는다.',
        '5. 개발 쪽에서 읽기 조건/질문을 고정한 다음 별도 검증 쪽에서 빈도와 오탐을 보고한다. 그 조건을 통과하기 전에는 설명을 GEPA/Ours 피드백에 연결하지 않는다.', '',
        '## 네 질문의 원문', '',
    ]
    for variant, items in by_query.items():
        report += [f'**{variant}**', '', items[0]['oracle_question'], '']
    report += ['## 직접 대조한64개 출력', '']
    for key in answers:
        report += [f'### {key[0]} / {key[1]}', '', answers[key]['question'], '',
                   '실제 답변 검토: ' + answers[key]['note_ko'], '']
        for x in annotated:
            if (x['base_id'], x['condition']) == key:
                report += [f"**{x['prompt_variant']}**", '', x['ao_output'], '', '검토: ' + x['manual_note_ko'], '']
    report += ['한 명의 탐색적 내용 검토이며 공식 Well/Haiku 채점이 아니다. 64개 설명을64개 독립 문항으로 해석하지 않는다.', '']
    (OUT / 'inspection_report.md').write_text('\n'.join(report))
    paths = [OUT / 'runtime.json', OUT / 'gemma27/ao_results.jsonl', OUT / 'annotated_comparison.jsonl',
             OUT / 'paired_comparison.csv', OUT / 'summary.json', OUT / 'inspection_report.md']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / 'artifact_checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
