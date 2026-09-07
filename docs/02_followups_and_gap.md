# 02. 후속 연구와 남은 틈

## 논문 약칭

이 저장소의 문서는 아래 약칭으로 인용한다.

| 약칭 | 논문 | arXiv |
|---|---|---|
| **Cancer-Myth** | Cancer-Myth: Evaluating LLMs on Patient Questions with False Presuppositions (Zhu et al., ICLR 2026) | 2504.11373 |
| **Verbalizing-Assumptions** | Verbalizing LLMs' assumptions to explain and control sycophancy | 2604.03058 |
| **Well-Actually** | Don't 'Well, Actually' Me Unless You Know What You're Talking About (Wang, Shwartz, Gonen) | 2608.06539 |
| **MedMisBench** | Untangling the Mechanisms of Misleading Context in Medical QA | 2609.02754 |
| **Contextual-Truth** | Language Models Encode the Contextual Truth of Propositions | 2608.03035 |
| **Two Axes** | Two Axes of LLM Abstention: Answer Correctness and Question Answerability (Wagner) | 2607.08456 |
| **Tripathi** | Gated Activation Steering for Reducing Sycophancy & Hallucination in Medical QA (Tripathi et al.) | 2608.23666 |
| **Pandey** | LLMs Know They're Wrong and Agree Anyway: The Shared Sycophancy-Lying Circuit (Pandey) | 2604.19117 |

2026-09-04 기준, Cancer-Myth를 인용한 논문 14편 (Semantic Scholar). 그중 이 문제를 직접 다룬 4편을 본문까지 확인했다.

## Verbalizing-Assumptions — Verbalizing LLMs' assumptions to explain and control sycophancy
[arXiv 2604.03058](https://arxiv.org/abs/2604.03058) · 2026-04

**한 일**: 모델에게 "사용자의 심적 모델을 추론해봐"라고 **직접 물어** 가정을 뽑고(open-ended 또는 9개 고정 축에 0–1 점수), 그 점수를 라벨로 **내부 표상(mean-pooled)에 linear probe 63개** 학습, probe 방향으로 steering `h + αv` (α ∈ [−4, 4]).
**축 9개**: validation seeking, emotional support, user rightness, user info advantage, social companionship, belonging support (S+) / objectivity seeking, information seeking, tangible support (S−).
**결과**: probe R² 0.64 (Llama-70B) / 0.50 (8B). ELEPHANT 등 사회적 sycophancy에서 steering 효과 ρ > 0.89.
**Cancer-Myth에서**: 데이터셋으로 씀. 그러나 *"Assumption steering has limited effects on factual sycophancy and Cancer-Myth… weaker and noisier."* 수치는 Appendix 그림뿐. 저자 해석: *"cannot compensate for missing knowledge; Cancer-Myth requires precise oncological corrections."* objectivity-seeking 방향 steering은 오히려 sharpness를 **떨어뜨림**.
**NFP**: 안 씀.
**Table 4의 예시** — Gemini가 대장암·장루 질문(거짓 전제 문항)에 *"user rightness 1.0: The user's premise of planning travel after a colostomy surgery is completely reasonable"*. 전제의 진위를 사용자 태도로 읽었다.

→ **읽은 것**: 사용자에 대한 모델의 사회적 가정. **못 본 것**: 전제 P 자체의 자격·진위. "지식 부족" 결론은 Well-Actually의 Table 1과 어긋난다.

## Well-Actually — Don't 'Well, Actually' Me Unless You Know What You're Talking About
[arXiv 2608.06539](https://arxiv.org/abs/2608.06539) · 2026-08 · [GitHub ShenranTomWang/Well](https://github.com/ShenranTomWang/Well) · [HF TPQ split](https://huggingface.co/datasets/shenranw/CancerMyth-TPQ)

**주장**: FPQ(거짓 전제)에 강한 방법일수록 TPQ(참 전제)에서 나빠진다. 원인은 "참인 전제까지 기각하는 약한 fact-checking". 4개 벤치마크(Cancer-Myth, QA², Syn-QA², CREPE), 5개 모델 계열.
**Cancer-Myth 분할**: FPQ 383/100/100, TPQ 30/18/100 (TPQ = NFP에 저자들이 전제를 수동 주석, Appendix I).
**판정**: gemini-3-flash, 0–5점. S5 = FPQ는 "거짓임을 명시하고 근거까지 정확", TPQ는 "의심 없이 정상 답변".

**Table 1 — 전제를 뽑아 직접 진위를 물었을 때** (RAG 없음):

| 모델 | 거짓 전제를 거짓이라 | **참 전제를 참이라** |
|---|---|---|
| Qwen2.5-7B-Instruct | 96% | **15.5%** |
| Llama-3-8B-Instruct | 97% | **22.4%** |
| gemma-4-E4B | 98% | **23.3%** |
| gemini-3-flash | 93% | **32.8%** |
| Llama3-Med42-8B | 95% | **31%** |

**이 표는 "모델이 안다"의 증거가 아니다.** 첫 열은 적중률, 둘째 열의 여집합은 오탐률이다. 모든 전제에 "거짓"이라 답하는 모델은 100 / 0이 나온다. Qwen2.5-7B의 96 / 15.5는 거기서 조금 벗어난 것뿐이다. "안다"는 참과 거짓을 **가르는 능력**이고, 그건 두 열의 차이에서만 나온다. 그런데 이 표는 **운영점 하나**(문턱 하나에서의 적중·오탐)라, 연속 점수의 ROC 곡선을 식별하지 못한다.

*보조 분석 (모형 의존).* 등분산 정규 SDT를 가정하면 d′ = Φ⁻¹(TPR) − Φ⁻¹(FPR), 잠재 AUROC = Φ(d′/√2). 이진 출력 자체의 AUC는 (TPR + TNR)/2. 두 열은 다른 대상이고 어느 쪽도 실측 변별력이 아니다.

| 모델 | 적중 | 오탐 | d′ | 잠재 AUROC (SDT 가정) | 이진 출력 AUC |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 96 | 84.5 | 0.74 | 0.70 | 0.56 |
| Llama-3-8B-Instruct | 97 | 77.6 | 1.12 | 0.79 | 0.60 |
| gemma-4-E4B | 98 | 76.7 | 1.32 | 0.83 | 0.61 |
| gemini-3-flash | 93 | 67.2 | 1.03 | 0.77 | 0.63 |
| Llama3-Med42-8B | 95 | 69.0 | 1.15 | 0.79 | 0.63 |

읽는 법: 판정 기준이 "거짓" 쪽으로 쏠려 있다 — **negative bias** (서울대 NAS 논문의 현상). 언어 판정은 지식(변별력)과 편향(기준)을 한 숫자로 뭉개기 때문에 이 표로는 둘을 분리할 수 없고, 따라서 **H1(모른다)은 이 표로 기각되지 않는다.** 분모도 다르다 — 이 표는 gold 전제 단위(FPQ 100 / TPQ 116)이고 우리 E1은 질문 단위다. 실측은 Two Axes가 CREPE에서 했다: 같은 모델·같은 문항에서 출력 readout(직접 질문 대부분 0.64–0.67, 학습된 출력 readout 0.49–0.58) < hidden-state probe 0.69–0.78 ([07 A1](07_related_work_2026.md)). 의료 배경 전제에서의 실측은 우리 실험 1이 같은 held-out 문항에서 연속 점수(log-odds)로 한다.

단서 둘. (a) TPQ는 NFP에서 왔고 NFP는 한 LLM이 헛짚은 문항만 골라 만든 집합이다. **어려운 통제군으로 설계됐다**는 동기이지, 다른 모델의 자연 분포 오탐률에 대한 확률적 상·하한은 아니다. (b) 언어 판정 밖의 증거는 있다 — Gemini-2.5-Pro Plain은 PCR 41 / NFP 96으로 생성 모드에서는 실제로 가른다. MedMisBench의 trace, Contextual-Truth의 선형 진위 표상도 있다. 그러나 전부 프런티어 또는 단언문 이야기이고, 공개 모델의 **배경화된** 전제에 대해서는 아직 증거가 없다.

**Table 8–10 — 방법별 S5 비율** (Qwen2.5-7B / gemma-4 / Llama-3-8B):

| 방법 | FPQ | TPQ |
|---|---|---|
| Direct QA | 1 / 3 / 3 | 100 / 100 / 100 |
| **FAITH** (head 차단) | **0 / 2 / 2** | 100 / 100 / 100 |
| FP Identification 프롬프트 | 75 / 70 / 75 | **0 / 0 / 0** |
| GEPA (FPQ만) — gemma | 62 | 64 |
| GEPA (FPQ+TPQ) — gemma | 19 | 96 |
| Fine-tuning — Qwen | 62 | **0** |

**FAITH는 무효**: `run_identify_heads.sh`가 네 모델 모두 head를 `wikidata_movies.json`에서 찾아 이식. 템플릿 단일 사실 질문용 head라 환자 질문에선 아무것도 안 함. 내부 개입이 안 통한다는 증거가 아니라 타깃 분포에 맞춘 적이 없다는 증거.
**해법 제안**: 없음. *"we hope our findings will help guide future work."*
**현실 가중**: WildChat 100문항 기준 FPQ 비율 약 13%로 가중하면 Direct QA가 1등. 단 이것은 평가한 방법들이 TPQ를 너무 많이 깎았다는 결과이지 불가능성 정리가 아니다 — 같은 척도에서 ΔU = 0.13·ΔFPQ + 0.87·ΔTPQ이므로 FPQ +10, TPQ −1이면 +0.43으로 이긴다. 13%는 암 환자 질문의 유병률이 아니므로 여러 비율에서 민감도 분석.

→ **읽은 것**: 강제 언어 판정(출력 수준). 내부는 안 봄. **남긴 것**: 해법 자리 전체, 그리고 FPQ/TPQ 분할·evaluator·6개 방법군 코드.

## MedMisBench — Untangling the Mechanisms of Misleading Context in Medical QA
[arXiv 2609.02754](https://arxiv.org/abs/2609.02754) · 2026-09-02

MedMisBench 8,627문항. 오도 맥락 두 종류(날조 근거 / 맨주장). 추론 모델 3개의 **reasoning trace** 분석.
- 오도 단서가 **trace의 81–98%에 등장하지만 응답에는 7–90%만** 반영
- 맨주장이 날조 근거보다 10–27%p 더 잘 먹힘
- 날조 근거는 추론 초반부터 오염·누적, 맨주장은 막판에 결론을 틂
- trace 모니터가 오염된 판단의 78%를 오탐 5%에서 포착

→ **읽은 것**: CoT 텍스트. "모델은 감지하지만 출력으로 안 꺼낸다"의 trace 수준 증거. 활성값 아님, Cancer-Myth 아님.

## Contextual-Truth — Language Models Encode the Contextual Truth of Propositions
[arXiv 2608.03035](https://arxiv.org/abs/2608.03035) · 2026-08

맥락 속 명제의 참/거짓이 **선형 표상**으로 존재. ExploreToM 스토리 + "Statement: … TRUE or FALSE" 형식. Llama-2-13B/70B, Qwen3-14B/32B/VL-32B. probe 정확도 74.4–89.4% (Llama-2-13B는 14층, Qwen3-32B는 49/64층).
**두 종류의 sycophancy**:
- *performative* — 거짓 명제를 내부에선 거짓으로 유지하며 출력만 동조. 암묵적 동조 시 **17/75 (22.7%)**
- *representational* — 표상이 참 쪽으로 넘어간 뒤 동조. **명시적으로 되뇔 때 58.7%** (104건 중), 암묵적일 때보다 2.59배
코드 없음.

→ **읽은 것**: **단언문**의 진위 방향. **못 본 것**: 질문 속에 *배경화된* 전제. 의료 아님.

## 그 외 인용 논문

MedRedFlag (2601.09853) · Evaluating LLMs on Misconceptions in Multi-Turn Medical Conversations (2607.12884) · Safer in Translation? Presupposition Robustness in Indic Languages (2511.01360) · How RLHF Amplifies Sycophancy (2602.01002) · EUDAIMONIA (2605.30654) · Morphological Shortcuts in Pharmacology (2606.05616). 인접: MedPRESS (2608.02520), Why LLMs Give In (2608.01017), SYCON-Bench (EMNLP 2025 Findings), PARROT (2511.17220).

## 점유된 것 / 비어 있는 것

| 제안 요소 | 상태 |
|---|---|
| 모델은 내부적으로 안다, 안 꺼낸다 | MedMisBench + Contextual-Truth가 프런티어·단언문에서 시사. Well-Actually Table 1은 언어 판정이라 지식과 편향을 못 가른다 (AUROC 0.7–0.8). **공개 모델의 배경화된 전제에서는 미결** — 실험 1이 첫 측정 |
| 무조건 개입은 NFP를 무너뜨린다 | Well-Actually가 6개 방법군에서 일반화 |
| 사회적 가정 → probe → steer | Verbalizing-Assumptions이 했고 Cancer-Myth에서 약함 |
| 영화 데이터 head 차단 | Well-Actually가 했고 무효 |
| 판정 probe | Contextual-Truth가 단언문에서 |
| 배경 전제가 hidden state에서 읽힌다 (일반 도메인) | **Two Axes**가 CREPE 마지막 토큰에서 AUROC 0.69–0.78 ([07 A1](07_related_work_2026.md)). 의료·전제 구간 위치·NFP는 남음 |
| probe 게이트 × 전제 검사 **프롬프트** | **Two Axes**가 함 — 항상 켜면 멀쩡한 질문 57% 오지적, probe 게이트 시 14% ([07 A1](07_related_work_2026.md)). 실험 3 baseline으로 |
| 의료 + 게이트 + ITI steering | **Tripathi et al.**이 함 — EHR 기반 질문·압력, 공개 runtime의 gate도 hidden-state probe ([11](11_tripathi_review.md)) |
| 알면서 따라간다 — head 회로 | **Pandey**가 12모델에서 — 단 전경화된 단언 ([07 A3](07_related_work_2026.md)) |
| **P가 어떤 자격으로(사실/믿음) 표상되는가** | 비어 있음 |
| **어떤 태도로 답하려 하는가 (disposition)** | 비어 있음 — 인접: refusal direction, CAA, persona vectors (전역 방향, 게이트 없음) |
| **진위 × 태도 분해와 진위로 게이트한 태도 개입** | 비어 있음. 단 Pandey: 진위·동조 방향 cos 0.4–0.8 → 분해 가능성을 먼저 재야 함 |
| **같은 held-out 문항에서 출력 readout vs 내부 readout** | Two Axes가 CREPE에서 함 (출력 < probe). 의료 배경 전제·NFP에서는 비어 있음. **측정이지 증명이 아님** — readout 수준 결과 |
| **Cancer-Myth+NFP를 verbalizer의 통제 벤치마크로** | 비어 있음 — ICML 2026 비판 논문이 요구한 것 |

인용 그래프 바깥의 2025–2026 연구는 [07](07_related_work_2026.md).
