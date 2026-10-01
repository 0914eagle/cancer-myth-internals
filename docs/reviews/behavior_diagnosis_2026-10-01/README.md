# 행동 분석 근거 묶음

[41번 문서](../../41_behavior_diagnosis_and_classification_2026-10-01.md)의 동결 근거다. 원 점수·라벨·AI 코드는 보존하고 재분류를 별도로 추가했다. 의료 정답 주석이나 새로운 성능 측정 결과로 취급하지 않는다.

## 읽는 순서

1. [모델별 상황 변경 후보 재분류](candidate_classification_by_model.csv): 새 저점에서 기존 X 후보로 선정된 19건이다. 0은 모델 전체에서 해당 오류가 없다는 뜻이 아니다.
2. [19건의 원 질문·답변·기존 코드·재분류](all_19_X_candidates_reviewed.json): 모든 후보의 근거를 보존했다. 전체 답변 재독해 여부는 별도 필드다.
3. [같은 신장 질문의 다섯 설정, 10개 전체 답변](same_question_10_answers.json): Plain부터 상황을 바꾸는지, 단정적 대체인지 조건부 대안인지 확인한 기록이다.
4. [개인 상황 지시의 특징별 대조군](context_feature_controls.csv): 분모가 전체 질문인지 기존 Yes인지 구분한다.
5. [732문항의 원 질문·주석·전후 판정](context_732_items.jsonl): 별도 Review와 기존 감사 코드는 Direct의 이유가 아니다.
6. [NFP 새 저점 121건 목록](nfp_121_loss_index.csv), [질문·행동 근거·원래 채점 설명](nfp_121_evidence.jsonl): 79개 질문의 모델·문항 조합이다. JSONL은 한 줄당 한 문항이다.
7. [채점 기준 적용 경계 6건의 두 답변 전체](six_rating_boundary_full_cases.json): 2점 이유가 ‘반박이 지배하지 않는다’고 설명한다. 이 확인만으로 4점 이상으로 재분류하지 않는다.

## 분류와 범위

- [수동 검토 메모](context_manual_review_notes.json): 기존 오탐 NFP 59개 전체의 원 질문·별도 검토 인용구, FPQ 주석 범위 경계, 채점 설명 재검토를 구분한다.
- [모델별 재분류 요약](model_analysis_summary.json), [전수 연결 요약](context_analysis_summary.json).
- [FPQ 570건 등 주요 비교 집계](primary_key_comparisons.json): 각 설정당 비교 방법 하나만 포함한다. 모든 방법을 합한 수치가 아니다.
- [앞선 분석의 동결 요약](primary_findings_snapshot.json): 포함된 실행 상태 항목은 당시 기록이며 현재 작업 상태가 아니다.
- [기존 넓은 AI 태그 전수 집계](broad_AI_tags_NOT_error_rates.csv): 태그가 타당한 부연·대안까지 포함하므로 hallucination 발생률로 사용하지 않는다.

## 검증·재현

[스냅샷 출처·해시](snapshot_manifest.json), [전수 연결 검증](context_validation.json), [후보 분류 검증](model_validation.json)을 보존했다. 원본 저장 경로는 서버 추적용이며 다른 머신에서 그대로 존재한다는 뜻이 아니다.

재현 스크립트:

- [Direct 변화 및 NFP 점수 경계](../../../scripts/analyze_context_and_nfp_score_boundaries.py)
- [모델별 상황 변경 후보 재분류](../../../scripts/summarize_context_shift_by_model.py)

전체 원시 실행 결과는 서버의 `results/`에 남아 있다. 이 묶음에는 문서 수치와 사례 대응에 필요한 동결 자료를 포함했다. 임의 재채점, 새 임계값 설정, 임상 라벨 수정은 수행하지 않았다.
