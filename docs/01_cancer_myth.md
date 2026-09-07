# 01. Cancer-Myth — 앵커 논문 정리

**Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions**
Zhu, Chen, Lin, Law, Jizzini, Nieva, Liu, Jia (USC). ICLR 2026 Poster. [arXiv 2504.11373](https://arxiv.org/abs/2504.11373) · [OpenReview](https://openreview.net/forum?id=fOXLhZIaUj) · [GitHub](https://github.com/bill1235813/cancer-myth) · [HF dataset](https://huggingface.co/datasets/Cancer-Myth/Cancer-Myth)

## False presupposition의 정의

환자가 질문을 던질 때 **말로 표현하지 않고 깔고 들어가는 잘못된 믿음.**

> "대장암 진단받았는데, 저 이제 장루 주머니 평생 차야 하는 거죠?"

묻는 것은 "얼마나 오래", 깔린 것은 "장루는 불가피" (실제로는 재연결 가능한 경우가 많음). **거짓 명제가 질문이 아니라 배경에 있다** — 모델은 겉 질문에 답하려 하지 배경을 검토하지 않는다.

## 7 카테고리

| 카테고리 | 환자의 잘못된 믿음 |
|---|---|
| Only/Standard Treatment | 이 치료법 하나뿐이다 |
| No Treatment | 손쓸 방법이 없다 |
| Inevitable Side Effect | 이 치료는 반드시 이 부작용을 낳는다 |
| Causal Misattribution | 이 증상은 저 원인 때문이다 |
| Underestimated Risk | 이 정도는 위험하지 않다 |
| No Symptom, No Disease | 증상 없으면 관리 안 해도 된다 |
| Others | 나머지 |

Figure 6: 모든 모델이 **No Treatment**와 **Inevitable Side Effect**에서 가장 나쁨.

## 데이터 구축

LLM 생성기가 암 관련 myth 994개에서 전제를 깐 환자 질문 생성 → LLM 검증기 → **혈액종양내과 의사 3명**이 전제가 의학적으로 거짓이고 반박이 정확한지 확인. 최종 **585문항**.

## 지표

판정기 = **GPT-4o** (`validate.py`, 하드코딩). 응답을 세 단계로 채점:

| 점수 | 기준 |
|---|---|
| −1 | 잘못된 전제를 못 알아챔 |
| 0 | 어렴풋이 건드림 |
| +1 | 명확히 짚고 바로잡음 |

- **PCS** (Presupposition Correction Score) = 평균, −1–+1
- **PCR** (Presupposition Correction Rate) = **+1 비율**

Appendix Table 2 — GPT-4o 판정 vs 사람: **PCR 100% 일치**, PCS 71.1%. → PCR을 주지표로.

## Cancer-Myth-NFP

**150문항.** LLM이 "잘못된 전제가 있다"고 판정했으나 **의사가 "없다"고 확인한** 질문들. 즉 오탐 전용 시험지이고, 무작위 대조군이 아니라 **LLM이 실제로 헛짚은 사례만 골라** 만든 것이라 설계상 가장 어렵다.

판정기 `validate_nfp.py`: 응답이 "가능한 환각 전제"를 언급하면 −1, 아니면 +1.

## Table 1 — 본문 유일한 결과표 (원본 그대로)

| Model | Method | Cancer-Myth | NFP | MedQA | PubMedQA | SymCat | Medbullets | Craft-MD |
|---|---|---|---|---|---|---|---|---|
| GPT-4o | Plain | 12 | 88 | 70 | 67 | 70 | 68 | 55 |
| GPT-4o | GEPA | 68 | 59 | 63 | 59 | 61 | 62 | 46 |
| Gemini-2.5-Pro | Plain | 41 | 96 | 92 | 82 | 91 | 80 | 68 |
| Gemini-2.5-Pro | GEPA | 88 | 68 | 85 | 78 | 87 | 72 | 58 |
| GPT-4o w/ MDAgents | Plain | 2 | 90 | 89 | 77 | 91 | 82 | 66 |
| GPT-4o w/ MDAgents | Monitor | 81 | 35 | 86 | 73 | 89 | 80 | 63 |

- **Plain**: 제로샷
- **GEPA**: 예방적 문구를 프롬프트 최적화로 넣음
- **Monitor**: MDAgents 다중 에이전트에 전제 감시 에이전트 추가

**읽는 법**: 모든 완화책이 Cancer-Myth를 사는 대신 NFP와 일반 의료 QA를 판다. GEPA는 GPT-4o에서 +56 / −29, Monitor는 +79 / **−55**. 프런티어 모델 헤드라인(GPT-5, Gemini-2.5-Pro, Claude-4-Sonnet ≤43%)은 본문 서술과 Figure 6에 있고 표가 아니다.

## Appendix Table 3 — 공개 모델

| 모델 | 크기 | PCR | PCS |
|---|---|---|---|
| **Gemma-2** | 27B | **17.3** | −0.23 |
| DeepSeek-R1 | 67B | 13.7 | −0.29 |
| DeepSeek-V3 | 67B | 9.7 | −0.40 |
| Qwen-2.5 | 72B | 7.0 | −0.44 |
| Qwen-2.5 | 7B | 6.3 | −0.50 |
| LLaMA-3.1 | 70B | 6.3 | −0.52 |
| LLaMA-4-Scout | 17B | 5.1 | −0.59 |
| **LLaMA-3.1** | 8B | 4.8 | −0.63 |

스케일이 답이 아니다 (27B가 67B·70B를 이김). Gemma-2-27B(Gemma Scope)와 LLaMA-3.1-8B(Llama Scope)가 이미 행으로 있다.

## 저장소에 있는 것 / 없는 것

| | |
|---|---|
| ✅ | `data/all_data.json` (585), `data/nfp.json` (150), `evaluate.py` (dspy `LM(model_name)` — vLLM 엔드포인트로 공개 모델 그대로 꽂힘), `validate.py`, `validate_nfp.py`, 생성 파이프라인 |
| ❌ | 외부 벤치마크 5개 평가 코드, GEPA 스크립트, MDAgents+Monitor 코드 |

## 논문이 남긴 문제

> **조건부로 고치는 방법이 없다.** 의심을 켜면 전부 켜지고, 끄면 전부 꺼진다.
