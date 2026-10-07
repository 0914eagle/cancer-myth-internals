# Paper tables 수치 원장 — 2026-10-07

**새 실험 없이 저장된 결과 90칸을 표에 표시했다. 학습 비교용 6칸은 `excluded_training_results.json`에 보존하고 표에서는 제외했다. 개인 상황 보존 변형 2칸도 `excluded_scope_results.json`에 보존하고 메인 표에서 제외했다. 중복 Direct gate 실행 2칸은 `excluded_duplicate_gate_results.json`에 보존했다.** 아래 분자/분모로 비율을 재계산하며, 표에는 소수 첫째 자리까지 표시한다. 탐지는 양성 판정 수, 답변은 Well 4·5점 답변 수다. 점수 보존과 gate 오탐은 다른 지표다.

## 비교 범위와 중요한 차이

- H: 전체 Cancer-Myth 대상의 과거 결과. Qwen Plain/CoT 및 routing은 FPQ 유효 582개인 행이 있다. Qwen 범위 보존 추출 검증은 FPQ 577/NFP 148개다. 누락을 실패로 재분류하지 않았다.
- L: Luna는 GPT-6이며 task/judge 모두 medium이다. Cancer-Myth는 99 FPQ/100 NFP로 맞췄다. `fpq_291`은 공식 few-shot과 중복되어 네 방법 모두에서 제외했다. Plain/GEPA는 zero-shot, 파이프라인은 Well four-shot·no-RAG이므로 구조 하나만의 효과 비교는 아니다.
- R: CREPE의 751 FPQ/2253 정상 질문은 전체 test이다. 평균 점수나 GEPA 개발 점수를 성공률로 옮기지 않았다.
- 제외 자료(T): 예전 balanced SFT/DPO는 121 FPQ와 121 합성 쌍둥이로 학습했다. 자연 NFP를 섞은 새 DPO 결과로 표시하지 않는다. 평가 FPQ 234개 및 자연 NFP 149개에서 Plain도 맞춰 집계했으나, 사용자 요청으로 세 조건 모두 현재 표에서 제외했다. 합성 쌍둥이의 잔류 전제 문제에 대한 이전 감사 단서는 유지한다.
- H와 L/R은 채점 모델·템플릿 revision·프롬프트·문항 집단이 다르다. Gemma/Qwen3.8의 기존 judge는 claude-sonnet-5, Luna는 gpt-6-luna다. 최종 통일 실험이나 순수 backbone 순위로 해석하지 않는다.
- 모든 표시값은 기존 한 번 실행의 결과다. 부분 재채점/사후 감사나 반복 중 최고값으로 교체하지 않았다.

## 넣지 않은 값과 이유

| 대상 | 처리 이유 |
|---|---|
| GPT-5.6 Luna의 기존 탐지·전제 검토·균형 지시·gate 답변 | 현재 GPT-6 칸과 모델 버전이 다름. 기존 docs/45 결과는 유지 |
| Qwen 기존 `extract_verify` | 범위 보존 지시 및 질문 문맥 사용이 Well baseline과 다름. 메인 표에서 제외하고 `excluded_scope_results.json`에 별도로 보존 |
| Gemma `verify_extracted_routed` | 추출 검증의 판정으로 저장 답변을 선택하는 조건. 새 답변을 생성하는 Table 2 Extract+Verify와 달라 미대입 |
| 초소규모 smoke / 부분 채점 / 오류로 중단한 검증기 GEPA | 본표의 완료된 시험 평가나 답변 프롬프트 GEPA로 간주하지 않음 |
| B/C/D/E 및 의료 문구 제거 ablation | 본표에 정의한 방법과 다름. 기존 분석 문서를 유지하고 이번 표의 다른 방법 칸에 대입하지 않음 |
| MedQA / PubMedQA / Medbullets / MMLU / GSM8K | 이 모델·방법·고정 평가 조건에 맞는 완료 결과를 찾지 못함. — 유지 |
| GEPA + text / SAE / NLA / AO, feature-curated DPO | 아직 해당 방법의 완료 성능이 없음 |

## 출처 파일

| ID | 고정 스냅샷 | 원래 파일 |
|---|---|---|
| qwen_answers | [sources/qwen_answers.json](sources/qwen_answers.json) | `docs/reviews/advisor_presentation_2026-09-23/answer_summary.json` |
| default_gates | [sources/default_gates.json](sources/default_gates.json) | `docs/reviews/default_gate_decisions_2026-09-23/summary.json` |
| routing | [sources/routing.json](sources/routing.json) | `docs/reviews/table2_completion_2026-10-01/routing_summary.json` |
| gemma | [sources/gemma.json](sources/gemma.json) | `results/gemma4/full_20260926_v2/analysis/summary.json` |
| qwen38_off | [sources/qwen38_off.json](sources/qwen38_off.json) | `results/qwen38/full_transformers_20260926_v1/thinking_off/analysis/summary.json` |
| qwen38_on | [sources/qwen38_on.json](sources/qwen38_on.json) | `results/qwen38/full_transformers_20260926_v1/thinking_on/analysis/summary.json` |
| luna_pipelines | [sources/luna_pipelines.json](sources/luna_pipelines.json) | `results/fpqa_prompting/well_pipelines_luna6_20261004/metrics.json` |
| luna_test | [sources/luna_test.json](sources/luna_test.json) | `results/fpqa_prompting/well_upstream_test_v1_20261003_run/metrics.json` |
| lora | [sources/lora.json](sources/lora.json) | `docs/reviews/advisor_presentation_2026-09-23/lora_summary.json` |
| qwen_plain_lora_matched | [sources/qwen_plain_lora_matched.json](sources/qwen_plain_lora_matched.json) | `derived from archived CSVs; hashes and matched rows in snapshot` |

전체 SHA-256은 [measured_results.json](measured_results.json)에, 실행 manifest의 모델·분할·프로토콜 정보는 [runtime_provenance.json](sources/runtime_provenance.json)에 있다. 후자는 원 manifest의 선택 필드와 원 파일 hash이며 전체 원문 복사본은 아니다.

## 셀별 값과 집계 경로

| 표 | 모델 | 방법 | 지표 | 분자/분모 | 비율 % | 집단 | 출처 / JSON 경로 |
|---|---|---|---|---:|---:|---|---|
| table1 | Qwen2.5 7B | direct_gate | cancer_fpq_tpr_pct | 41/583 | 7.0 | H | default_gates / `direct` |
| table1 | Qwen2.5 7B | direct_gate | cancer_nfp_fpr_pct | 16/149 | 10.7 | H | default_gates / `direct` |
| table1 | Qwen2.5 7B | premise_review_cot | cancer_fpq_tpr_pct | 172/583 | 29.5 | H | default_gates / `review` |
| table1 | Qwen2.5 7B | premise_review_cot | cancer_nfp_fpr_pct | 24/149 | 16.1 | H | default_gates / `review` |
| table1 | Gemma 4 12B | direct_gate | cancer_fpq_tpr_pct | 236/583 | 40.5 | H | gemma / `gate → direct → fpq` |
| table1 | Gemma 4 12B | direct_gate | cancer_nfp_fpr_pct | 24/149 | 16.1 | H | gemma / `gate → direct → nfp` |
| table1 | Gemma 4 12B | premise_review_cot | cancer_fpq_tpr_pct | 299/583 | 51.3 | H | gemma / `gate → cot → fpq` |
| table1 | Gemma 4 12B | premise_review_cot | cancer_nfp_fpr_pct | 49/149 | 32.9 | H | gemma / `gate → cot → nfp` |
| table2 | Gemma 4 12B | plain | cancer_fpq_well_ge4_pct | 25/583 | 4.3 | H | gemma / `answers → plain → fpq` |
| table2 | Gemma 4 12B | plain | cancer_nfp_well_ge4_pct | 149/149 | 100.0 | H | gemma / `answers → plain → nfp` |
| table2 | Gemma 4 12B | premise_review_cot | cancer_fpq_well_ge4_pct | 240/583 | 41.2 | H | gemma / `answers → premise_review_answer → fpq` |
| table2 | Gemma 4 12B | premise_review_cot | cancer_nfp_well_ge4_pct | 117/149 | 78.5 | H | gemma / `answers → premise_review_answer → nfp` |
| table2 | Gemma 4 12B | cot | cancer_fpq_well_ge4_pct | 20/583 | 3.4 | H | gemma / `answers → zero_shot_cot_one_step → fpq` |
| table2 | Gemma 4 12B | cot | cancer_nfp_well_ge4_pct | 148/149 | 99.3 | H | gemma / `answers → zero_shot_cot_one_step → nfp` |
| table2 | Gemma 4 12B | always_correct | cancer_fpq_well_ge4_pct | 420/583 | 72.0 | H | gemma / `answers → fp_unconditional → fpq` |
| table2 | Gemma 4 12B | always_correct | cancer_nfp_well_ge4_pct | 3/149 | 2.0 | H | gemma / `answers → fp_unconditional → nfp` |
| table2 | Gemma 4 12B | direct_gate_response | cancer_fpq_well_ge4_pct | 217/583 | 37.2 | H | gemma / `answers → direct_routed → fpq` |
| table2 | Gemma 4 12B | direct_gate_response | cancer_nfp_well_ge4_pct | 125/149 | 83.9 | H | gemma / `answers → direct_routed → nfp` |
| table2 | Gemma 4 12B | cot_gate_response | cancer_fpq_well_ge4_pct | 253/583 | 43.4 | H | gemma / `answers → cot_routed → fpq` |
| table2 | Gemma 4 12B | cot_gate_response | cancer_nfp_well_ge4_pct | 100/149 | 67.1 | H | gemma / `answers → cot_routed → nfp` |
| table1 | Qwen3.8 27B OFF | direct_gate | cancer_fpq_tpr_pct | 152/583 | 26.1 | H | qwen38_off / `gate → direct → fpq` |
| table1 | Qwen3.8 27B OFF | direct_gate | cancer_nfp_fpr_pct | 24/149 | 16.1 | H | qwen38_off / `gate → direct → nfp` |
| table1 | Qwen3.8 27B OFF | premise_review_cot | cancer_fpq_tpr_pct | 335/583 | 57.5 | H | qwen38_off / `gate → cot → fpq` |
| table1 | Qwen3.8 27B OFF | premise_review_cot | cancer_nfp_fpr_pct | 53/149 | 35.6 | H | qwen38_off / `gate → cot → nfp` |
| table2 | Qwen3.8 27B OFF | plain | cancer_fpq_well_ge4_pct | 290/583 | 49.7 | H | qwen38_off / `answers → plain → fpq` |
| table2 | Qwen3.8 27B OFF | plain | cancer_nfp_well_ge4_pct | 119/149 | 79.9 | H | qwen38_off / `answers → plain → nfp` |
| table2 | Qwen3.8 27B OFF | premise_review_cot | cancer_fpq_well_ge4_pct | 358/583 | 61.4 | H | qwen38_off / `answers → premise_review_answer → fpq` |
| table2 | Qwen3.8 27B OFF | premise_review_cot | cancer_nfp_well_ge4_pct | 82/149 | 55.0 | H | qwen38_off / `answers → premise_review_answer → nfp` |
| table2 | Qwen3.8 27B OFF | cot | cancer_fpq_well_ge4_pct | 319/583 | 54.7 | H | qwen38_off / `answers → zero_shot_cot_one_step → fpq` |
| table2 | Qwen3.8 27B OFF | cot | cancer_nfp_well_ge4_pct | 106/149 | 71.1 | H | qwen38_off / `answers → zero_shot_cot_one_step → nfp` |
| table2 | Qwen3.8 27B OFF | always_correct | cancer_fpq_well_ge4_pct | 414/583 | 71.0 | H | qwen38_off / `answers → fp_unconditional → fpq` |
| table2 | Qwen3.8 27B OFF | always_correct | cancer_nfp_well_ge4_pct | 43/149 | 28.9 | H | qwen38_off / `answers → fp_unconditional → nfp` |
| table2 | Qwen3.8 27B OFF | direct_gate_response | cancer_fpq_well_ge4_pct | 309/583 | 53.0 | H | qwen38_off / `answers → direct_routed → fpq` |
| table2 | Qwen3.8 27B OFF | direct_gate_response | cancer_nfp_well_ge4_pct | 108/149 | 72.5 | H | qwen38_off / `answers → direct_routed → nfp` |
| table2 | Qwen3.8 27B OFF | cot_gate_response | cancer_fpq_well_ge4_pct | 362/583 | 62.1 | H | qwen38_off / `answers → cot_routed → fpq` |
| table2 | Qwen3.8 27B OFF | cot_gate_response | cancer_nfp_well_ge4_pct | 89/149 | 59.7 | H | qwen38_off / `answers → cot_routed → nfp` |
| table1 | Qwen3.8 27B ON | direct_gate | cancer_fpq_tpr_pct | 257/583 | 44.1 | H | qwen38_on / `gate → direct → fpq` |
| table1 | Qwen3.8 27B ON | direct_gate | cancer_nfp_fpr_pct | 32/149 | 21.5 | H | qwen38_on / `gate → direct → nfp` |
| table1 | Qwen3.8 27B ON | premise_review_cot | cancer_fpq_tpr_pct | 358/583 | 61.4 | H | qwen38_on / `gate → cot → fpq` |
| table1 | Qwen3.8 27B ON | premise_review_cot | cancer_nfp_fpr_pct | 57/149 | 38.3 | H | qwen38_on / `gate → cot → nfp` |
| table2 | Qwen3.8 27B ON | plain | cancer_fpq_well_ge4_pct | 237/583 | 40.7 | H | qwen38_on / `answers → plain → fpq` |
| table2 | Qwen3.8 27B ON | plain | cancer_nfp_well_ge4_pct | 132/149 | 88.6 | H | qwen38_on / `answers → plain → nfp` |
| table2 | Qwen3.8 27B ON | premise_review_cot | cancer_fpq_well_ge4_pct | 389/583 | 66.7 | H | qwen38_on / `answers → premise_review_answer → fpq` |
| table2 | Qwen3.8 27B ON | premise_review_cot | cancer_nfp_well_ge4_pct | 85/149 | 57.0 | H | qwen38_on / `answers → premise_review_answer → nfp` |
| table2 | Qwen3.8 27B ON | cot | cancer_fpq_well_ge4_pct | 251/583 | 43.1 | H | qwen38_on / `answers → zero_shot_cot_one_step → fpq` |
| table2 | Qwen3.8 27B ON | cot | cancer_nfp_well_ge4_pct | 131/149 | 87.9 | H | qwen38_on / `answers → zero_shot_cot_one_step → nfp` |
| table2 | Qwen3.8 27B ON | always_correct | cancer_fpq_well_ge4_pct | 486/583 | 83.4 | H | qwen38_on / `answers → fp_unconditional → fpq` |
| table2 | Qwen3.8 27B ON | always_correct | cancer_nfp_well_ge4_pct | 8/149 | 5.4 | H | qwen38_on / `answers → fp_unconditional → nfp` |
| table2 | Qwen3.8 27B ON | direct_gate_response | cancer_fpq_well_ge4_pct | 312/583 | 53.5 | H | qwen38_on / `answers → direct_routed → fpq` |
| table2 | Qwen3.8 27B ON | direct_gate_response | cancer_nfp_well_ge4_pct | 113/149 | 75.8 | H | qwen38_on / `answers → direct_routed → nfp` |
| table2 | Qwen3.8 27B ON | cot_gate_response | cancer_fpq_well_ge4_pct | 362/583 | 62.1 | H | qwen38_on / `answers → cot_routed → fpq` |
| table2 | Qwen3.8 27B ON | cot_gate_response | cancer_nfp_well_ge4_pct | 90/149 | 60.4 | H | qwen38_on / `answers → cot_routed → nfp` |
| table2 | Qwen2.5 7B | plain | cancer_fpq_well_ge4_pct | 14/582 | 2.4 | H | qwen_answers / `qwen/plain → fpq` |
| table2 | Qwen2.5 7B | plain | cancer_nfp_well_ge4_pct | 149/149 | 100.0 | H | qwen_answers / `qwen/plain → nfp` |
| table2 | Qwen2.5 7B | premise_review_cot | cancer_fpq_well_ge4_pct | 61/583 | 10.5 | H | qwen_answers / `qwen/premise_review → fpq` |
| table2 | Qwen2.5 7B | premise_review_cot | cancer_nfp_well_ge4_pct | 141/149 | 94.6 | H | qwen_answers / `qwen/premise_review → nfp` |
| table2 | Qwen2.5 7B | cot | cancer_fpq_well_ge4_pct | 4/582 | 0.7 | H | qwen_answers / `qwen/zero_shot_cot → fpq` |
| table2 | Qwen2.5 7B | cot | cancer_nfp_well_ge4_pct | 148/149 | 99.3 | H | qwen_answers / `qwen/zero_shot_cot → nfp` |
| table2 | Qwen2.5 7B | always_correct | cancer_fpq_well_ge4_pct | 332/583 | 56.9 | H | qwen_answers / `qwen/fp_unconditional → fpq` |
| table2 | Qwen2.5 7B | always_correct | cancer_nfp_well_ge4_pct | 32/149 | 21.5 | H | qwen_answers / `qwen/fp_unconditional → nfp` |
| table2 | Qwen2.5 7B | text_gate_response | cancer_fpq_well_ge4_pct | 295/582 | 50.7 | H | routing / `results → qwen/text → fpq` |
| table2 | Qwen2.5 7B | text_gate_response | cancer_nfp_well_ge4_pct | 90/149 | 60.4 | H | routing / `results → qwen/text → nfp` |
| table2 | Gemma 4 12B | text_gate_response | cancer_fpq_well_ge4_pct | 372/583 | 63.8 | H | routing / `results → gemma/text → fpq` |
| table2 | Gemma 4 12B | text_gate_response | cancer_nfp_well_ge4_pct | 75/149 | 50.3 | H | routing / `results → gemma/text → nfp` |
| table2 | Qwen3.8 27B OFF | text_gate_response | cancer_fpq_well_ge4_pct | 398/583 | 68.3 | H | routing / `results → qwen38_off/text → fpq` |
| table2 | Qwen3.8 27B OFF | text_gate_response | cancer_nfp_well_ge4_pct | 73/149 | 49.0 | H | routing / `results → qwen38_off/text → nfp` |
| table2 | Qwen3.8 27B ON | text_gate_response | cancer_fpq_well_ge4_pct | 456/583 | 78.2 | H | routing / `results → qwen38_on/text → fpq` |
| table2 | Qwen3.8 27B ON | text_gate_response | cancer_nfp_well_ge4_pct | 72/149 | 48.3 | H | routing / `results → qwen38_on/text → nfp` |
| table2 | Qwen2.5 7B | direct_gate_response | cancer_fpq_well_ge4_pct | 40/582 | 6.9 | H | routing / `results → qwen/direct → fpq` |
| table2 | Qwen2.5 7B | direct_gate_response | cancer_nfp_well_ge4_pct | 135/149 | 90.6 | H | routing / `results → qwen/direct → nfp` |
| table2 | Qwen2.5 7B | cot_gate_response | cancer_fpq_well_ge4_pct | 124/582 | 21.3 | H | routing / `results → qwen/review → fpq` |
| table2 | Qwen2.5 7B | cot_gate_response | cancer_nfp_well_ge4_pct | 128/149 | 85.9 | H | routing / `results → qwen/review → nfp` |
| table1 | Qwen2.5 7B | hidden_probe | cancer_fpq_tpr_pct | 437/583 | 75.0 | H | default_gates / `hidden` |
| table1 | Qwen2.5 7B | hidden_probe | cancer_nfp_fpr_pct | 71/149 | 47.7 | H | default_gates / `hidden` |
| table1 | Model- independent | tfidf | cancer_fpq_tpr_pct | 502/583 | 86.1 | H | default_gates / `text` |
| table1 | Model- independent | tfidf | cancer_nfp_fpr_pct | 76/149 | 51.0 | H | default_gates / `text` |
| table2 | GPT-6 Luna | prewome_style | cancer_fpq_well_ge4_pct | 88/99 | 88.9 | L | luna_pipelines / `methods → prewome → primary_excluding_overlap → fpq` |
| table2 | GPT-6 Luna | prewome_style | cancer_nfp_well_ge4_pct | 59/100 | 59.0 | L | luna_pipelines / `methods → prewome → primary_excluding_overlap → nfp` |
| table2 | GPT-6 Luna | extract_verify | cancer_fpq_well_ge4_pct | 87/99 | 87.9 | L | luna_pipelines / `methods → atomic → primary_excluding_overlap → fpq` |
| table2 | GPT-6 Luna | extract_verify | cancer_nfp_well_ge4_pct | 48/100 | 48.0 | L | luna_pipelines / `methods → atomic → primary_excluding_overlap → nfp` |
| table2 | GPT-6 Luna | plain | cancer_fpq_well_ge4_pct | 54/99 | 54.5 | L | luna_pipelines / `methods → prewome → matched_reference_on_completed_primary_ids → plain → fpq` |
| table2 | GPT-6 Luna | plain | crepe_fpq_well_ge4_pct | 546/751 | 72.7 | R | luna_test / `conditions → crepe/plain → metrics → fpq` |
| table2 | GPT-6 Luna | plain | cancer_nfp_well_ge4_pct | 91/100 | 91.0 | L | luna_pipelines / `methods → prewome → matched_reference_on_completed_primary_ids → plain → nfp` |
| table2 | GPT-6 Luna | plain | crepe_normal_well_ge4_pct | 2228/2253 | 98.9 | R | luna_test / `conditions → crepe/plain → metrics → nfp` |
| table2 | GPT-6 Luna | gepa | cancer_fpq_well_ge4_pct | 78/99 | 78.8 | L | luna_pipelines / `methods → prewome → matched_reference_on_completed_primary_ids → gepa → fpq` |
| table2 | GPT-6 Luna | gepa | crepe_fpq_well_ge4_pct | 561/751 | 74.7 | R | luna_test / `conditions → crepe/gepa → metrics → fpq` |
| table2 | GPT-6 Luna | gepa | cancer_nfp_well_ge4_pct | 72/100 | 72.0 | L | luna_pipelines / `methods → prewome → matched_reference_on_completed_primary_ids → gepa → nfp` |
| table2 | GPT-6 Luna | gepa | crepe_normal_well_ge4_pct | 2209/2253 | 98.0 | R | luna_test / `conditions → crepe/gepa → metrics → nfp` |
| table2 | Qwen2.5 7B | hidden_gate_response | cancer_fpq_well_ge4_pct | 261/582 | 44.8 | H | routing / `results → qwen/hidden → fpq` |
| table2 | Qwen2.5 7B | hidden_gate_response | cancer_nfp_well_ge4_pct | 96/149 | 64.4 | H | routing / `results → qwen/hidden → nfp` |
