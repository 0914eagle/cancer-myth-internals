# E1. 사전 진단 — 배경화된 전제가 표상에서 읽히는가

## 질문

Cancer-Myth 585(FPQ) vs NFP 150 + TPQ(참 전제)에서, 전제 구간·질문 끝·마지막 프롬프트 토큰의 hidden state가 "전제가 거짓이다"를 갈라내는가. 갈라낸다면 그 방향(A)과, 응답이 전제를 고칠지 따를지를 가르는 방향(C)은 얼마나 겹치는가.

## 모델·GPU (server 125)

| 모델 | GPU | 이유 |
|---|---|---|
| Llama-3.1-8B-Instruct | 0 | Table 3 행(4.8), Llama Scope, AO |
| Qwen2.5-7B-Instruct | 1 | Table 3 행(6.3), NLA L20, Well-Actually baseline 동일 데이터 |
| Gemma-2-9B-it | 2 | Gemma Scope, AO, 한 장에 들어가는 Gemma |
| Gemma-2-27B-it | 1,2,3 (phase 2) | Table 3 1위(17.3). bf16 54GB, 양자화 금지 |
| Gemma-3-12B-it | 2,3 (선택) | medical_nla 백본, NLA L32 그대로 |

## 위치

| 코드 | 문서(03) | 정의 | row |
|---|---|---|---|
| `A_premise` | A | 전제 구간 마지막 토큰 (`last_subtoken`) + 구간 평균 (`span_mean`) | `__prem` |
| `B_question_end` | B | 질문 텍스트 마지막 토큰 | `__qend` |
| `D_last_prompt_token` | D | assistant 턴 직전 마지막 프롬프트 토큰 | `__last` |
| `E_response_first5` | E | Plain 응답 첫 5토큰 평균, teacher-forced | `__resp5` |

전제 구간은 GPT-4o가 질문에서 **verbatim substring**을 뽑고 exact match로 검증한다(`align_method=llm`). 실패하면 content-word 겹침 heuristic, 그것도 실패하면 A 없이 B·D만. `alignment_audit.md`에 방법별 수가 나온다.

## 라벨

- A: `label_false_premise` — fpq 1, nfp/tpq 0. nfp와 tpq는 같은 질문 텍스트라 B·D 위치에서 중복 제거.
- C: `pcr` — Plain 응답의 GPT-4o Sharpness (+1 고침 / 0 / −1 따라감). C 방향은 +1 vs −1.
- 판정기는 Cancer-Myth `validate.py` 프롬프트 그대로, temperature 0 (논문은 0.7; `--paper-protocol`).

## 산출물

1. `probe_sweep/heatmap.png` — 층 × (위치, 축약) AUROC, logistic·diff-of-means 두 패널. 기준선 0.70 (Two Axes CREPE).
2. `direction_c/table.md` — 층별 `pcr_auroc_D_*` (생성 전 PCR 예측), `cos_A_C_at_D`, `cos_Aprem_Ce`.
3. `plain_summary.json` — Plain PCR/PCS/NFP/TPQ, 카테고리별. Table 3 재현치.
4. E2 `results/e2/<model>/*_summary.json` — 무조건 vs 게이트, alpha별 PCR·NFP·TPQ.

## 갈림길 (06 요약)

| A AUROC | cos(A,C) | E2 게이트 | 장르 |
|---|---|---|---|
| ≥0.70 | <0.5 | PCR↑, NFP≈Plain | framework — Table 1 행 |
| ≥0.70 | <0.5 | 안 움직임 | 진단 논문 + head 수준 개입 시도 |
| ≥0.70 | >0.8 | — | A≈C. "부분 분해"로 낮추고 NFP·게이트 신호로 차별 |
| <0.65 | — | — | 음성 결과 — H2 논문 |

## 판정기 보정 결과 (2026-09-07)

`scripts/calibrate_judge.py --backend codex`, all_data.json의 GPT-4o 채점과 대조, 답변 모델 3종 × 150문항 = 450건. codex가 서빙한 모델은 `gpt-5.6-sol`.

| 답변 모델 | 3-way 일치 | GPT-4o의 +1 중 codex도 +1 | PCR GPT-4o → codex | PCS GPT-4o → codex |
|---|---:|---:|---|---|
| Claude-3.5-Sonnet | 62.7% | | 22.0 → 16.0 | −0.23 → −0.51 |
| DeepSeek-R1 | 48.7% | | 15.3 → 10.7 | −0.26 → −0.57 |
| GPT-4o | 63.3% | | 6.0 → 4.7 | −0.49 → −0.75 |
| 전체 | 58.2%, κ 0.28 | 38/65 = 58% | | |

confusion(행 GPT-4o, 열 codex): 0→−1이 132/173, +1→0이 26/65. 위로 올린 경우는 29/450. **일관되게 한 방향으로 엄격한 판정기**이지 무작위 노이즈가 아니다. 논문은 GPT-4o의 PCR이 의사와 100% 일치한다고 했으므로, codex는 사람보다 엄격하다.

**결정.**
- 상대 비교(같은 판정기로 채점한 조건끼리: Plain vs 무조건 vs 게이트, alpha·층 스윕)는 codex로 한다. 순위가 보존되고(22>15.3>6 → 16>10.7>4.7) 비용이 없다.
- **표에 들어가는 숫자는 GPT-4o API(`JUDGE_BACKEND=openai`)로 채점한다.** Table 1·3과 같은 판정기여야 비교가 된다. codex 채점 PCR을 Table 3 옆에 놓지 않는다.
- 두 판정기의 결과 파일은 이름으로 구분한다: `*_judge.jsonl`(codex) / `*_judge_gpt4o.jsonl`.

## 상태

- [ ] E0 rows (alignment audit 확인)
- [ ] E1 phase 1 (llama / qwen / gemma-2-9b)
- [ ] E1 phase 2 (gemma-2-27b)
- [ ] E1 phase 3 (judge → E → sweep → C)
- [ ] E2 (한 모델)
- [ ] AO 통제 4종 (gemma-2-9b, 별도 스크립트 예정)
- [ ] NLA 프롬프트 수정 테스트 (gemma-3-12b, medical_nla `src.run_nla`)
