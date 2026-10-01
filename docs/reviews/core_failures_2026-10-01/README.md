# 핵심 실패 전수 재집계 근거

설명은 [46번 문서](../../46_core_failure_analysis_2026-10-01.md)에 있다. 기존 모델 출력·Well 점수·Sonnet 행동 코드를 변경하지 않았다.

- `all_3659_pair_transitions.csv`: 개선·악화·유지 모두 포함한 답변 쌍 원장. 모델과 실제 대안 지시 명시.
- `well_movements.csv`: 모델·데이터별 양방향 점수 전이.
- `behavior_transitions.json`: 전체 및 점수 전이별 목표 대응·반박·태그 집계.
- `flag_controls.csv`: 고점과 저점 각각의 태그 분모. 태그는 오류 정답이 아니다.
- `fpq_178_evidence.csv`: FPQ 평가 불일치 178쌍 전부의 질문, 주석, 전체 두 답변, 코드 인용, 원래 두 Well 설명.
- `nfp_121_evidence.csv`: NFP 새 저점 121쌍 전부의 동일 근거.
- `disagreement178_summary.json`: 모델별 Plain 점수 분포와 고유 문항 수.
- `luna_repeat_items.csv`: 732문항 각각의 2조건×2반복 판정과 기존 AI 질문 특징.
- `luna_repeat_summary.json`: 반복성, 안정적 전환, 출처 그룹 bootstrap 구간.
- `luna_repeat_feature_associations.csv`: 기존 AI 특징별 탐색적 연관. 수동 보정하지 않은 원 코드이며 인과 효과가 아니다.
- `luna_stable_switch_questions.csv`: 두 반복 모두 조건 간 동일 방향으로 바뀐 129문항(Yes→No 128개, No→Yes 1개)의 원 질문.
- `luna_repeat_protocol.json`: 정확한 시스템/사용자 지시, 집계 프로토콜, 2,928개 원 기록 해시.
- `manifest.json`: 입력 파일·스크립트 해시와 한계.

`0->1`은 Well <4→≥4이며 gate의 No→Yes와 다르다. Luna 실제 답변 대안은 균형 지시다. Luna 반복 실험은 답변을 만들지 않는 Direct JSON gate다. 네 모델 계열·다섯 설정의 반복 문항을 독립 문항으로 세지 않는다.

행동 코딩과 Well 채점은 둘 다 기존 Sonnet 판단이다. 이번에 전체 근거를 연결한 것과 전체를 독립적으로 재판정한 것은 다르다. 후자는 아직 완료하지 않았다. 원문/채점 설명의 주장을 이 파일의 분석자가 의학적 사실로 승인한 뜻도 아니다.
