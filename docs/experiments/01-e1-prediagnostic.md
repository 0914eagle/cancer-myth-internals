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

## 중간 결과 (2026-09-08, 판정기 codex/gpt-5.6-sol)

llama와 qwen의 E1이 끝났다. gemma-2-9b, gemma-2-27b는 진행 중.

| | Llama-3.1-8B | Qwen2.5-7B |
|---|---|---|
| Plain PCR (585) | 3.4 % (+1 20개) | 5.6 % (+1 33개) |
| Plain PCS | −0.85 | −0.75 |
| NFP (150) | 45.3 % | 48.7 % |
| TPQ 보존 (148) | 50.7 % | 45.9 % |
| A 읽기, 최고 셀 (logistic) | 0.815 [0.77, 0.85], L11 A_premise span_mean | 0.812 [0.77, 0.85], L28 A_premise last_subtoken |
| A 읽기, B / D 위치 | 0.73 / 0.72 | 0.74 / 0.71 |
| 생성 전 D에서 PCR 예측 (diff-means, CV) | 0.847 (L15) | 0.823 (L20) |
| E(응답 첫 5토큰)에서 PCR 읽기 | 0.62–0.64 | 0.65–0.68 |
| E→D 전이 (c_e를 D에, CV) | 재계산 필요 | 0.71–0.78 (L19–22) |
| cos(A, C) at D | L12–13 ≈ 0.05, L17–21 0.4–0.5 | L3–13 0.5–0.6, L17–28 0.2–0.4 |

읽는 법.

- **A는 읽힌다, 그러나 약하게.** 전제 구간 위 중간층에서 0.80–0.82, 질문 끝·마지막 토큰으로 갈수록 0.72로 떨어진다. Two Axes의 CREPE 읽기(0.69–0.78)와 같은 대역. 0.8이 진위인지 fpq/nfp의 문체 차이인지는 아직 통제하지 않았다(같은 질문의 거짓/참 전제 짝으로 확인해야 함).
- **C의 재료가 없다.** 8B급은 585개 중 20–33개만 고친다. C = (+1 평균 − −1 평균)인데 +1이 20개면 방향이 아니라 20개 문항의 평균이다. `table.md`에 50개 미만 경고를 넣었다. 이 숫자 자체가 결과다: 오픈 8B는 거의 전부 동조하므로 개입 여지가 크다.
- **그래도 성향은 내부에 있다.** 답변을 내기 전 마지막 프롬프트 토큰(D)에서 "이 모델이 고칠지"가 0.82–0.85로 읽힌다. 양성이 적어 폭이 크지만 우연(0.5)과는 멀다.
- **E 위치 정의가 틀렸다.** 응답 첫 5토큰에서는 0.62–0.68밖에 안 나온다. 두 모델 다 응답을 "I'm not a doctor, but…" 류의 상투구로 시작해 첫 5토큰에 판단이 안 실린다. 32토큰 평균을 함께 뽑기로 했다(stage 8).
- **카테고리 편중.** 고치는 문항이 "no symptoms means no disease"(13–15 %)와 "causal misattribution"(qwen 13 %)에 몰리고 "inevitable side effect"는 0–2 %. C가 이 유형의 내용 방향과 섞일 위험이 있어 짝지은 구성이 필요했다.
- **판정기.** codex는 GPT-4o의 +1 중 58 %만 인정하므로 절대값은 낮게 나온다. NFP/TPQ 루브릭은 보정하지 않았으니 45–51 %는 모델 간 비교에만 쓴다. 표에 넣을 값은 gpt-4o API로 다시 판정한다.

### 그래서 stage 8: 짝지은 C

C를 모델 자신의 응답으로 못 만드니 Cancer-Myth `all_data.json`의 참조 응답을 쓴다. 232개 질문에 GPT-4o가 +1로 판정한 답과 −1로 판정한 답이 둘 다 있다. 같은 Plain 프롬프트 뒤에 두 답을 teacher-forcing하고 응답 첫 5/32토큰 평균의 **문항별 차이**를 평균한 것이 `c_pair`. 질문 내용은 뺄셈에서 사라지고 고치기/동조 차이만 남는다. 저자 스타일이 섞이지 않도록 +1 쪽과 −1 쪽의 저자 분포를 맞췄다. 확인 열:

| 열 | 뜻 | 기준 |
|---|---|---|
| `pair_sign_agreement` | 문항별 차이가 평균 방향과 같은 부호인 비율 | ≥ 0.9 |
| `own_pcr_auroc_D_via_cpair` | 그 문항을 빼고 만든 c_pair로 모델 **자신의** D 벡터에서 자기 PCR을 맞히는가 | ≥ 0.75면 조향 방향으로 쓸 근거 |
| `cos_cpair_Ad`, `cos_cpair_Ce` | 진위 방향 A, 자연 응답 방향 c_e와의 겹침 | A와 < 0.5 |

`directions.npz`에 `L{layer}_c_pair5/32`가 추가되고 `run_steer.py --direction-key c_pair32`가 그것을 민다.

### 개입 단위에 대해

지금 A와 C는 둘 다 **residual stream의 한 방향**(층 하나의 hidden state에 더하는 벡터)이다. 그게 가장 단순하고 CAA·refusal direction·persona vector와 같은 단위라 무조건 조향 대조군과 바로 비교된다. 06의 후보 (ii) Tripathi식 head별 방향, (iii) Pandey식 공유 head 증폭은 같은 게이트(A probe) 아래에서 **C를 무엇으로 미느냐**만 바꾼 변형이다. 순서는 residual 방향으로 E2를 먼저 돌리고, PCR이 안 움직이면 head 단위로 내려간다. head 단위로 가면 activation 추출을 head 출력별로 다시 해야 하므로 별도 stage가 된다.

## 상태

- [x] E0 rows (codex 정렬, 2026-09-07)
- [x] E1 phase 1 (llama / qwen / gemma-2-9b)
- [x] E1 phase 2 (gemma-2-27b 생성·추출)
- [x] E1 phase 3 llama, qwen (judge → E → sweep → C)
- [ ] E1 phase 3 gemma-2-9b, gemma-2-27b
- [ ] E1 stage 8 짝지은 C (4모델)
- [ ] A 읽기의 문체 통제 (같은 질문의 거짓/참 전제 짝)
- [ ] E2 (한 모델; c_pair32 우선)
- [ ] gpt-4o API 재판정 (표 숫자)
- [ ] AO 통제 4종 (gemma-2-9b, 별도 스크립트 예정)
- [ ] NLA 프롬프트 수정 테스트 (gemma-3-12b, medical_nla `src.run_nla`)
