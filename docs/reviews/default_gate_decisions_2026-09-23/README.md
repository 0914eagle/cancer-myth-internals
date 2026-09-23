# 기본 판정 및 선택적 교정 재집계 — 2026-09-23

발표 문서 Table 1은 FPQ/NFP 중 기본 판정으로 선택한 결과, Table 2의 Text·Hidden은 같은 판정에 따른 답변 선택 결과로 갱신했다. 기존 calibration 목표 FPR 5% 조건의 원장은 advisor_presentation_2026-09-23에 보존한다.

- Text·Style·Masked·Hidden: 기존 held-out logistic score > 0이면 FPQ. 동점은 NFP.
- Qwen Direct·Review: Yes logit > No logit이면 FPQ. 동점은 NFP. 자유 응답 생성이 아니라 두 선택지 점수 비교.
- DiM: 기존 각 fold의 학습 집단 평균 중 유클리드 거리가 가까운 쪽. 기존 투영 점수의 학습 평균 중간점 경계와 동일함을 전 문항에서 검증. 층과 방향은 유지한다. 이 규칙은 이번 재집계에서 명시했으며 cosine 판정이 아니다.
- Luna: 저장된 boolean 출력 그대로 사용. 원 프롬프트는 보고 확률 >= 0.5일 때 true를 요구한다. AUROC는 발표하지 않는다.
- Table 1: 732개(583 FPQ, 149 NFP). Table 2: 두 답변 점수가 있는 731개(582 FPQ, 149 NFP); fpq_389 제외.
- 기존 Self-gated FP Identification의 답변 결과는 >= 0.5이며 Table 1과 동점 처리만 다르다. 다른 기존 답변 baseline은 별도 방법으로 유지한다.

```bash
python3 scripts/export_default_gate_decisions.py \
  --suite /data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1 \
  --luna results/frontier_cli/luna_direct_cot_20260923_v1 \
  --out docs/reviews/default_gate_decisions_2026-09-23 \
  --routing-scores docs/reviews/advisor_presentation_2026-09-23/routing_scores.csv
```

모델 재추론·답변 재생성·Well 재채점은 하지 않았다. 학습/평가 source group 분리, 원 Qwen logits와 결정 일치, 원 DiM 투영 점수 재현, DiM의 거리/중간점 결정 일치, Luna boolean 일관성 및 완료 여부를 확인했다. 원 점수와 새 선택 정책을 분리해 보존한다.
