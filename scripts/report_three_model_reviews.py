"""Report blinded, descriptive review coding; do not run any model calls."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/semantic_audit'
RUN = ROOT / 'results/audits/three_model_reviews_20260928_v1'
MODELS = ['qwen', 'gemma', 'luna']
NAMES = {'qwen': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna'}
LABELS = {
    'challenges': '오류를 명시적으로 지적', 'accepts': '전제를 수용',
    'mixed_or_conditional': '혼합·조건부 판단', 'no_clear_verdict': '명확한 결론 없음',
    'faithful': '질문에 근거한 대상·범위', 'stronger_or_different': '더 강하거나 다른 주장으로 해석',
    'individual_context_reopened': '개인 진단·예후·계획을 재검토', 'unclear': '불명확',
    'similar_review': '주된 검토 내용 유사', 'different_target': '검토 대상이 다름',
    'different_scope': '같은 소재의 범위·강도가 다름',
    'different_factual_assessment': '같은 주장에 대한 사실 판단이 다름',
    'different_explicit_stance': '내용은 비슷하나 명시적 결론이 다름',
    'mixed_or_unclear': '복합·불명확',
}


def reviewed(case, model):
    return next(r for r in case['audit']['reviews'] if case['slots'][r['slot']] == model)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('\n', ' ') for v in row) + ' |' for row in rows])


def main():
    records = json.loads((DOC / 'annotations.json').read_text())
    summary = json.loads((DOC / 'summary.json').read_text())
    assert summary['complete'] and len(records) == 208
    assert Counter(x['set'] for x in records) == {'nfp': 149, 'fpq': 59}
    assert len({x['id'] for x in records}) == 208
    plan = json.loads((RUN / 'plan.json').read_text())
    # Make frozen source verification part of reporting, not a claim based on filenames.
    from scripts.audit_luna_nfp_with_tpq import sha
    for path, expected in plan['sources'].items():
        assert sha(Path(path)) == expected, path
    for x in records:
        for path, expected in x['sources'].items():
            assert sha(Path(path)) == expected, path
    question_paths = [ROOT / 'results/gemma4/full_20260926_v2/evaluation_questions.json',
                      ROOT / 'results/frontier_cli/luna_direct_cot_20260923_v1/evaluation_questions.json']
    a, b = [{x['id']: x for x in json.loads(p.read_text())} for p in question_paths]
    assert a.keys() == b.keys() and len(a) == 732
    for key in a:
        assert a[key]['question'] == b[key]['question'] and a[key]['set'] == b[key]['set'], key
    for x in records:
        gemma_path = ROOT / 'results/gemma4/full_20260926_v2/records/review' / (x['id'] + '.json')
        assert json.loads(gemma_path.read_text())['prompt'].endswith('Question: ' + x['input']['question'])
    (DOC / 'question_alignment.json').write_text(json.dumps(dict(
        gemma_luna_exact_questions_and_labels=732, selected_gemma_prompt_question_matches=208,
        sources={str(p): sha(p) for p in question_paths}), indent=2) + '\n')

    rows = []
    for x in records:
        for model in MODELS:
            r = reviewed(x, model)
            rows.append(dict(id=x['id'], split=x['set'], selection=x['selection'], model=model,
                             direct=x['decisions'][model + '_direct'],
                             review_gate=x['decisions'][model + '_review'], **r))
    with (DOC / 'review_codes.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)

    derived = {}
    baseline = json.loads((DOC.parent / 'summary.json').read_text())
    transition_rows = []
    for m in MODELS:
        t = baseline['transitions'][m]
        transition_rows.append([NAMES[m], t['fpq']['No→Yes']['n'], t['fpq']['Yes→No']['n'],
                                t['nfp']['No→Yes']['n'], t['nfp']['Yes→No']['n']])
    lines = ['# 같은 질문에 대한 세 모델의 전제 검토 비교', '',
             '원 질문·저장된 검토문·기존 gate 결정을 연결한 행동 분석이다. 새 답변 생성, GPU 추론, Well 채점은 하지 않았다.', '',
             '## 먼저 확인된 점', '',
             '전체 732문항의 결정에서 Review는 새 FPQ 탐지뿐 아니라 탐지 손실·새 NFP 오탐·오탐 해소를 함께 일으킨다. 아래 값은 AI 의미 분류가 아닌 저장된 결정의 직접 집계다.', '',
             table(['모델', 'FPQ 새 탐지', 'FPQ 기존 탐지 손실', 'NFP 새 오탐', 'NFP 기존 오탐 해소'], transition_rows), '',
             '각각 Direct→Review의 No→Yes / Yes→No 변화다. FPQ 583개, NFP 149개에서 계산했으며, 출발점의 Direct 성능이 달라 변화 개수만으로 모델 순위를 정하지 않는다.', '',
             '원문 대조에서는 **검토 대상·범위의 차이**와 **검토문에 쓴 결론을 후속 gate에 연결하는 차이**를 모두 확인했다. 예를 들어 같은 NFP를 세 모델이 모두 오탐해도 어떤 모델은 재발 가능성을, 다른 모델은 보험 상품의 존재를 반박했다. Qwen이 명시적으로 거짓 전제를 지적한 뒤 Yes/No 점수에서는 No가 우세한 사례도 있다. 구체 사례와 반례는 [사례 점검](case_checks.md)을 참고한다.', '',
             '## 범위와 해석', '',
             '- NFP 149개 전체와 FPQ 59개 표본을 비교했다. FPQ는 세 모델의 Review gate 정오 패턴 8종에서 최대 8개씩 고른 층화 표본이며, 전체 FPQ의 실패 비율을 추정하는 표본이 아니다.',
             '- NFP의 nfp_1002·nfp_1003은 현재 참조 파일에 전제 주석이 없다. 질문·검토문 비교에는 포함하되 주석 대응은 없는 상태로 처리했다.',
             '- Sonnet 5에 모델명·최종 gate 결정·직접적인 FPQ/NFP 라벨을 숨기고, 문항마다 모델 순서를 바꿔 검토문을 비교시켰다. 주석 내용·교정 주석의 유무로 클래스가 간접적으로 드러날 수 있어 완전한 라벨 맹검은 아니다.',
             '- 아래 의미 분류는 AI의 서술적 분류다. 임상적 진위 판정, 라벨 재검증, 내부 인과 분석이 아니다. 원문에 있는 주장을 반박했다고 해서 그 반박이 옳다는 뜻은 아니다.',
             '- 세 모델의 Review 사용자 지시는 동일하지만 시스템 지시·추론 설정·후속 판정 형식은 완전히 같지 않다. Qwen은 Yes/No 토큰 점수, Gemma·Luna는 JSON 판정이다. 모델 능력만의 차이로 해석할 수 없다.', '',
             '## 검토문에 어떤 결론을 썼는가', '']
    for split, caption in [('nfp', 'NFP 전체 149개'), ('fpq', 'FPQ 층화 표본 59개')]:
        subset = [x for x in records if x['set'] == split]
        derived[split] = {}
        lines += ['### ' + caption, '']
        categories = ['challenges', 'accepts', 'mixed_or_conditional', 'no_clear_verdict']
        counts = {m: Counter(reviewed(x, m)['stance'] for x in subset) for m in MODELS}
        lines += [table(['검토문의 표현'] + [NAMES[m] for m in MODELS],
                        [[LABELS[c]] + [counts[m][c] for m in MODELS] for c in categories]), '']
        consistency = []
        for m in MODELS:
            challenges_no = [x['id'] for x in subset if reviewed(x, m)['stance'] == 'challenges'
                             and x['decisions'][m + '_review'] == 'False']
            accepts_yes = [x['id'] for x in subset if reviewed(x, m)['stance'] == 'accepts'
                           and x['decisions'][m + '_review'] == 'True']
            positives = [x for x in subset if x['decisions'][m + '_review'] == 'True']
            newly_positive = [x for x in positives if x['decisions'][m + '_direct'] == 'False']
            derived[split][m] = dict(challenges_no=challenges_no, accepts_yes=accepts_yes,
                                     gate_yes=len(positives),
                                     gate_yes_reading=dict(Counter(reviewed(x, m)['reading'] for x in positives)),
                                     gate_yes_stance=dict(Counter(reviewed(x, m)['stance'] for x in positives)),
                                     newly_positive_ids=[x['id'] for x in newly_positive],
                                     newly_positive_reading=dict(Counter(reviewed(x, m)['reading'] for x in newly_positive)))
            consistency.append([NAMES[m], len(positives), len(challenges_no), len(accepts_yes)])
        lines += [table(['모델', '최종 gate Yes', '검토에서 오류 지적 → gate No', '검토에서 수용 → gate Yes'], consistency), '',
                  '혼합·조건부 검토는 위 두 불일치 수에 넣지 않았다. 이 수는 AI가 분류한 표현상 불일치 후보이며, 검토문이 의학적으로 맞다거나 내부 진위 판단이 형성됐음을 뜻하지 않는다. 특히 단순한 보충 설명을 AI가 challenges로 분류한 경계 사례가 있다(nfp_1044; 사례 점검 참조). 엄격한 불일치의 확정 건수로 읽지 않는다.', '']

    lines += ['## NFP 오탐 문항에서 무엇을 검토 대상으로 삼았는가', '',
              '각 모델이 실제로 오탐한 NFP만 분모로 삼는다. 하나의 주된 읽기 유형으로 분류했으므로 개인 정황 재검토와 범위 확대가 동시에 나타난 문항도 한 칸에만 들어간다. faithful에는 전제를 수용한 검토도 포함된다. 따라서 이 칸은 올바른 오류 추출의 성공률이나 정당한 반박의 개수가 아니다. 원문 검수에서 경계 분류가 확인됐으므로 범위 왜곡 가설을 배제하는 근거로 쓰지 않는다.', '']
    categories = ['faithful', 'stronger_or_different', 'individual_context_reopened', 'unclear']
    lines += [table(['주된 읽기 유형'] + [f'{NAMES[m]} ({derived["nfp"][m]["gate_yes"]}개)' for m in MODELS],
                    [[LABELS[c]] + [derived['nfp'][m]['gate_yes_reading'].get(c, 0) for m in MODELS] for c in categories]), '',
              '### Direct에서는 정답이었으나 Review에서 새로 오탐한 NFP', '',
              table(['Review 검토문의 주된 읽기 유형'] + [f'{NAMES[m]} ({len(derived["nfp"][m]["newly_positive_ids"])}개)' for m in MODELS],
                    [[LABELS[c]] + [derived['nfp'][m]['newly_positive_reading'].get(c, 0) for m in MODELS] for c in categories]), '',
              '이는 새 오탐 집단에 어떤 Review 표현이 있는지를 기술한다. 원래 Direct의 이유가 저장되어 있지 않으므로 Review에서 읽기가 이렇게 바뀌었다는 인과 설명으로 해석하지 않는다.', '',
              '## 모델 간 검토문 차이', '',
              '아래는 NFP 149개 전체의 쌍별 비교다. 같은 Yes/No 결정을 했는지와 별개로, 검토문이 같은 내용을 지적했는지를 분류했다.', '']
    pairs = ['gemma/qwen', 'luna/qwen', 'gemma/luna']
    pcounts = summary['pair_differences']['nfp']
    lines += [table(['주된 차이'] + pairs,
                    [[LABELS[c]] + [pcounts[p].get(c, 0) for p in pairs]
                     for c in ['similar_review', 'different_target', 'different_scope', 'different_factual_assessment',
                               'different_explicit_stance', 'mixed_or_unclear']]), '',
              '## 원문으로 확인한 사례와 해석', '',
              '별도 [사례 점검](case_checks.md)에서 AI 분류와 별개로 원 질문·검토문·최종 판정을 함께 확인했다. 전체 문항의 원문과 분류는 [items.md](items.md)에 있다.', '',
              '이 분석에서 분리할 것은 ① 검토 대상으로 삼는 내용·범위, ② 그 대상에 관해 검토문에 쓰는 결론, ③ 그 뒤 gate의 Yes/No 결정이다. 서로 다른 단계에서 판정이 갈릴 수 있으므로 모든 오탐을 추출 실패 하나로 묶지 않는다.', '',
              '## 기존 여섯 분석 항목과의 관계', '',
              table(['항목', '이번 자료로 확인한 범위'], [
                  ['전제 대상을 놓치거나 바꿈', '같은 질문에서 서로 다른 내용을 오류로 지적한 사례 확인. 대상 선택의 임상적 정답을 전수 판정한 것은 아님.'],
                  ['의미·범위를 달리 읽음', '개인 진단을 전형적 발병 주장으로 바꾸거나, 존재 여부 질문을 존재 주장으로 읽는 사례 확인.'],
                  ['사실 지식·검증 문제', '모델 간 설명 차이는 관찰되지만 어느 모델의 의학적 판단이 맞는지는 미확정.'],
                  ['오류를 찾는 쪽으로 판정 성향 이동', 'Review 후 순 Yes 증가 확인. 진위 구분 개선과 단순 Yes 성향의 기여를 분리한 것은 아님.'],
                  ['검토와 최종 gate의 불일치', '명시적 오류 지적 뒤 No, 명시적 수용 뒤 Yes인 원문 사례 확인. 내부 원인·빈도 확정은 별도.'],
                  ['과제·출력 형식 의존', '조건 차이를 기록했으며 모델별 성능 차이에 섞일 수 있음. 통제된 원인 검증은 아직 아님.'],
              ]), '',
              '## 재현', '',
              '- 원본 검토문·질문·결정·코딩: [annotations.json](annotations.json)',
              '- 문항×모델 단위 코드: [review_codes.csv](review_codes.csv)',
              '- AI 코딩 집계: [summary.json](summary.json)',
              '- 불일치 문항 ID와 오탐 부분집합 집계: [derived.json](derived.json)',
              '- 원래 전체 데이터 탐지·전환 집계: [상위 보고서](../README.md)',
              '- 실행: `python3 -m scripts.audit_three_model_reviews collect` 후 `python3 -m scripts.report_three_model_reviews`.',
              '- 실행 계획·프롬프트·배치 원출력·인용 형식 수정 기록: `results/audits/three_model_reviews_20260928_v1/`.', '']
    lines += ['- 원문과 어긋난 인용 5곳은 출처를 확인해 교체했고 `manual_quote_repairs.json`에 남겼다. 이는 단순 서체 수정과 구분된다. 원래 분류는 유지했다. 배치 064에서 빠진 reading 코드 2개는 별도 Sonnet 호출로만 보완하고 원출력·`schema_repairs.json`을 보존했다.', '']
    if (DOC / 'semantic_boundaries/README.md').exists():
        lines += ['## 후속: 라벨 기준 오탐과 의학적 오류를 구분', '',
                  '[12개 문항의 추가 대조](semantic_boundaries/README.md)에서 원 질문의 의미 변화, 의학적 사실 확인, 정상 gate를 유지하는 보충 설명을 분리했다. 특히 GIST 사례의 Qwen·Luna 반박을 단순한 의미 왜곡이나 의학 지식 실패로 확정하지 않는다. 원 라벨과 위 AI 코딩은 변경하지 않았다.', '']
    (DOC / 'README.md').write_text('\n'.join(lines))
    (DOC / 'derived.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2) + '\n')
    items = ['# 문항별 저장된 검토문과 비교 코딩', '',
             '질문과 검토문은 모델이 받은/생성한 원문이다. 주석과 모델의 의학적 주장은 이 문서에서 검증·승인한 사실이 아니다.', '']
    for x in sorted(records, key=lambda x: (x['set'], int(x['id'].split('_')[-1]))):
        items += [f'## {x["id"]} · {x["selection"]}', '', '**질문**', '', x['input']['question'], '',
                  '**참고 주석**', '', json.dumps(x['input']['reference'], ensure_ascii=False), '']
        for m in MODELS:
            r = reviewed(x, m)
            original = next(s['text'] for s in x['input']['reviews'] if s['slot'] == r['slot'])
            items += [f'### {NAMES[m]}', '',
                      f'Direct {x["decisions"][m + "_direct"]} → Review gate {x["decisions"][m + "_review"]}', '',
                      f'AI 분류: `{r["stance"]}` / `{r["reading"]}` / `{r["reason_type"]}` / 주석 대응 `{r["reference_match"]}`', '',
                      r['target_ko'] + ' — ' + r['explanation_ko'], '', '<details><summary>저장된 검토문 원문</summary>', '', original, '', '</details>', '']
        items += ['**쌍별 차이(AI 분류)**', '']
        for p in x['audit']['pairs']:
            items += [f'- {NAMES[x["slots"][p["a"]]]} / {NAMES[x["slots"][p["b"]]]}: {LABELS[p["difference"]]} — {p["explanation_ko"]}']
        items += ['', '**해석 유의:** ' + x['audit']['caveat_ko'], '']
    (DOC / 'items.md').write_text('\n'.join(items))
    print(json.dumps({'cases': len(records), 'rows': len(rows), 'derived': derived}, ensure_ascii=False))


if __name__ == '__main__':
    main()
