# 07. 2025~2026 관련 연구 — 일반 도메인과 직접 경쟁자

2026-09-07 기준. 02는 Cancer-Myth **인용 그래프 안쪽**만 다뤘다. 여기는 그 바깥 — 일반 도메인 거짓 전제 QA, sycophancy 내부 표상, 조건부 steering, 그리고 "탐지는 되는데 제어는 안 된다"는 경고 논문들.

A절 3편은 원문 전체를 읽었다. B절은 검색 스니펫 기준이라 수치는 원문 재확인이 필요하다.

## A. 원문 확인 — 우리 설계와 정면으로 겹치는 3편

### A1. Two Axes of LLM Abstention — Wagner, [2607.08456](https://arxiv.org/abs/2607.08456) (2026-07)

**한 것.** 기권을 두 축으로 분해: 답이 맞는가(correctness) vs 질문에 답해도 되는가(answerability). SelfAware + **CREPE**(Reddit ELI5 실제 질문, 1/4이 거짓 전제). Gemma-2-2B-it, Qwen2.5-3B/7B/14B, Llama-3.1-8B.

**CREPE에서 거짓 전제 탐지 AUROC (Table 2, 500/500 균형, 공식 test split):**

| 신호 | 범위 |
|---|---|
| raw answer-confidence / 학습된 output readout | 0.49~0.58 |
| P(IK) / P(True) | 0.54~0.66 / 0.54~0.58 |
| **"이 질문은 거짓 전제 위에 있나?" 직접 질문** | 0.43~0.67 (Qwen2.5-3B는 거의 전부 "멀쩡하다"고 답함) |
| bag-of-words 상한 | 0.59 |
| **hidden-state logistic probe** | **0.69~0.73** |
| **difference-of-means 방향** | **0.74~0.78** |
| SelfAware→CREPE 전이 | 0.65~0.70 |

**확인 항목 답.**
- *위치·층*: **마지막 프롬프트 토큰** hidden state (chat template 적용, 깊이 0.4/0.6/0.8 중 nested calibration으로 선택). 전제 구간 위치는 안 봄. 프롬프트는 "Answer the question in a few words."
- *라벨*: CREPE 공식 주석 그대로. 참 전제 질문 = 같은 분포의 일반 질문 (NFP처럼 "LLM이 헛짚은 것만 고른" 어려운 통제군이 아님).
- *오탐률*: §6 프롬프트 라우팅 파일럿 (Llama-3.1-8B, Qwen2.5-7B, 각 120+120). "거짓 가정이 있으면 지적하라" 프롬프트를 **항상** 켜면 FP 74% 지적 / **멀쩡한 질문 57%도 지적** (Qwen 76/78 — 판별 0). **probe로 게이트**(35% 라우팅)하면 42/14 (Qwen 41/16), 무작위 게이트 26/20. → **"probe 게이트 × 전제 검사 프롬프트"가 일반 도메인에서 이미 됐다.**
- *교정 품질*: 지적한 것 중 NLI 검증 통과 47% (Llama) / 61% (Qwen). 나머지는 자신 있게 틀린 교정 — 게이트를 통과해도 프롬프트 개입은 confabulate.
- *인과 개입*: 없음 (limitations에 명시). 게이트는 두 프롬프트 사이 라우팅.

**H6 재해석.** Well-Actually(Well)는 *추출한 전제 문장*의 진위를 물으면 참 전제의 58~93%를 거짓이라 한다(negative bias). 이 논문은 *질문*이 거짓 전제 위에 있냐 물으면 거의 전부 멀쩡하다고 한다(accommodation). **판정을 시키면 framing에 따라 어느 한쪽으로 쏠리고, 어느 쪽이든 판별은 chance.** "판정 대신 표상"의 근거가 양방향에서 확정됐다.

**우리와의 거리.**

| | Two Axes | 우리 |
|---|---|---|
| 도메인 | Reddit ELI5 | 암 환자 질문, 종양내과 의사 검증 |
| 참 전제 통제군 | 같은 분포 일반 질문 | **NFP — LLM이 헛짚은 것만 150** |
| 읽는 위치 | 마지막 토큰만 | 전제 구간 / 질문 끝 / 응답 시작 |
| 읽는 축 | 답변 가능성 1축 | **진위(A) × 태도(C)** |
| 개입 | 프롬프트 라우팅 | 내부 방향 (C) |
| 개입 후 교정 품질 | 47~61% | 측정 필요 |

**가져올 것.**
1. **실험 3 baseline에 "probe-gated FP Identification" 추가.** Well-Actually의 FP Identification 프롬프트를 우리 A probe로 게이트한 것. 싸고 강하다. 우리 C 개입은 GEPA가 아니라 **이것**을 이겨야 한다 — 특히 교정 품질(PCS)에서.
2. 실험 1 기대치: 마지막 토큰 probe **0.70 안팎**이 일반 도메인 기준선. 의료 전제는 의사 검증이라 더 깨끗할 수 있고, 전제 구간 위치는 더 높을 수 있다. 0.70 아래면 CREPE보다 못 읽는 것.
3. difference-of-means 방향이 logistic probe보다 높았다 (0.74~0.78 vs 0.69~0.73). 우리도 둘 다.
4. 통제 프로토콜: 4-way split (probe-train / tune / certify / test), bag-of-words 상한, 100 resplit 감사. 그대로 쓴다.

### A2. Gated Activation Steering for Medical QA — Tripathi et al., [2608.23666](https://arxiv.org/abs/2608.23666) (2026-08)

**한 것.** EHR 기반 임상 QA에서 환각(맥락에 없는 주장)과 sycophancy(압력에 답 번복)를 ITI로 따로 steering, **게이트**로 필요할 때만. Gemma-3-12B-it, MedGemma-1.5-4B-it. [HF 체크포인트 공개](https://huggingface.co/himanshu5trpth/medgemma-sycophancy-hallucination-gated-steering).

**확인 항목 답.**
- *게이트*: "trigger detector" — 라벨된 턴으로 학습한 작은 분류기 g_H, g_S ∈ [0,1]. **최신 사용자 턴을 읽어 "거짓 주장이 있나 / 압력이 있나"를 판정.** AUROC 0.81/0.84 (Gemma-12B), 0.74/0.94 (MedGemma). 즉 게이트 신호 = **"도전이 일어나고 있나"**. 환자 질문의 배경 전제엔 압력 표지가 없으므로 이 게이트는 Cancer-Myth/NFP를 못 가른다 (A1의 output readout이 chance인 것과 같은 이유).
- *게이트 신호 vs 개입 개념*: 같은 행동. 게이트 = 압력 감지, 방향 = grounded−caving. 진위×태도 분해 아님.
- *양성 질문 오탐*: **측정함.** 압력·거짓 주장 없는 질문 600 + 정당한 교정 요청 25. 게이트 발화율 H 4% / S 2%, steered vs base BERTScore F1 0.997. 단, "정상 질문"은 표면부터 압력 턴과 다르므로 쉬운 통제군. 정당한 교정 100건 테스트에서 MedGemma는 H 게이트가 0.993으로 켜지고 EHR 정보를 거부한 사례 2건 — **게이트는 정당한 교정도 잡는다.**
- *Cancer-Myth 언급*: 없음. 자체 벤치마크 — MIMIC-IV 재구성 EHR 200개, 압력 단계 P1~P5 하네스, 대조쌍 행동당 200 (140/40/20). 판정 GPT-OSS-20B.
- *설정*: **멀티턴 압력.** 모델이 먼저 맞게 답한 뒤 사용자가 "틀렸어, 6.2야, 동의해"라고 밀어붙임. 거짓의 근거는 **맥락(EHR)**에 있음 — 파라미터 지식 아님. 배경화된 전제 아님.

**방법 세부 (재사용 가능).** head별 logistic probe로 grounded/caving 분리 정확도 → 상위 H 48 / S 24 head → 제거 시 행동 점수가 떨어지는 것만 유지 → head별 방향 v = (ḡ − ū)/‖·‖ → 생성 시 h̃ = h + α·s·ρ(t)·σ·v (s = 게이트 값, ρ(t) = 첫 몇 토큰 후 감쇠, σ = head 표준편차). 강도 α는 4개 verifier(RoBERTa 변화 / BiomedBERT 의미 보존 / Phi·Mistral 행동 판정 / perplexity)로 "행동을 뒤집는 최소값".

**결과.** MedGemma-4B: 600 궤적 중 rescue 551 / harm 12. Gemma-12B: 487 / 48. 자체 하네스에서 steered 4B ≈ 120B+ 모델. Gemma-12B는 H/S 인과 분리 50% (새는 중).

**우리와의 거리.** "의료 + 게이트 + ITI"는 선점. 남는 차이: (1) 단일 질문 속 **배경 전제** vs 멀티턴 압력, (2) 게이트 신호가 **진위**(내부에서만 읽힘) vs 압력 표지(표면), (3) 통제군이 **NFP**(표면상 동일) vs 표면부터 다른 정상 질문, (4) Cancer-Myth Table 1.

**가져올 것.** head 선택 → head별 방향 → 감쇠 스케줄 → verifier 기반 강도 튜닝 파이프라인. C 개입을 residual 방향이 아닌 **head 출력**에 걸 때의 레시피. 그리고 "게이트가 정당한 교정도 잡는다"는 관찰 — 우리 A 게이트도 NFP 외에 "전제가 참인데 모델이 틀리게 답하던 문항"에서 어떻게 행동하는지 봐야 한다.

### A3. LLMs Know They're Wrong and Agree Anyway — Pandey, [2604.19117](https://arxiv.org/abs/2604.19117) (2026-04)

**한 것.** 12개 공개 모델(1.5B~72B, 5개 랩)에서 **"이 문장은 틀렸다" 신호를 나르는 attention head 집합이 sycophancy(사용자 주장에 동조)와 factual lying(단언문 진위 판정)에서 같다.** Gemma-2-2B에서 그 head를 0으로 하면 sycophancy 28→81%, 사실 정확도 69→70%. → 회로는 지식이 아니라 **deference**를 제어. RLHF(Llama-3.1→3.3)는 행동을 10배 줄이지만 회로는 남고 projection ablation 효과는 커짐. 의견 동의는 같은 head 위치를 쓰되 **직교 방향**(|cos|<0.14).

**확인 항목 답.**
- *head 식별*: 마지막 프롬프트 토큰에서 head별 write-norm 차이(DLA식), 상위 K=⌈√N⌉, 두 과제 교집합. 검증은 activation patching(≤8B), projection ablation, path patching(Gemma-2-2B, Phi-4), head→unembed(70B).
- *우리 모델*: **Gemma-2-27B-IT** 패널 포함 — 26/39 head 공유, projection ablation으로 sycophancy 10.5→100%, faithfulness K=8. **Llama-3.1-8B** — 21/32 공유이나 per-head patching 교집합 3/15로 패널 중 가장 약함, head별 cos 0.44. **Gemma-3-27B-IT는 per-head 분석에서 제외** — layer-0 head 출력 norm이 다른 층의 100배라 중요도 순위가 망가짐 (residual cos 0.494는 유지). 
- *"사실 정확도 유지" 통제*: 단언문 true/false 판정 과제 정확도 (Gemma-2-2B 69→70, Qwen3-8B 50→50, Qwen2.5-32B 68.5→67.5). **참 의견 문항에서 동의율이 어떻게 변하는지(NFP 상당)는 보고 안 함.**
- *설정*: 단일 턴, "I believe the answer to 'q' is w. Am I correct?" — **전경화된 단언 + 판정 요청.** TriviaQA/NQ. 배경 전제 아님, 의료 아님.
- *방향*: 마지막 프롬프트 토큰 mean-diff. **sycophancy 방향과 진위 방향의 residual cos 0.44~0.73, head별 0.43~0.81** — 정렬돼 있으나 동일하진 않음. probe 층 ⌊0.85L⌋. syc-probe→lie 전이 AUROC 0.83~0.85.
- *인과 방향*: 실험은 전부 **회로 제거 → sycophancy 증가**. 즉 이 head들은 동조를 *억제*하는 쪽. 동조를 *줄이는* 개입(방향 추가·head 증폭)과 그것의 참 전제 부작용은 안 했다.
- *logit-lens*: 2B·7B는 중간층에서 정답 쪽 피크 후 후기층에서 뒤집힘(detect-then-override), **70B는 단조** — 분산 실행. H4 검증 시 70B에서는 층별 신호가 안 보일 수 있다.

**우리와의 거리.** H3(알면서 따라간다)의 일반 도메인 회로 수준 증거. 다만 전제가 *전경*에 있는 설정이라, **배경화된 전제에서 같은 head가 켜지는가**는 그대로 우리 H2 질문이다.

**가져올 것.**
1. **cos(A, C)를 재야 한다.** 이 논문에서 진위 방향과 동조 방향은 cos 0.4~0.8이다. 우리 A(진위 probe)와 C(PCR+1 vs −1 응답 시작 대조 방향)가 이만큼 겹치면 "A로 게이트하고 C를 민다"는 부분적으로 A를 미는 것이고, 지식을 안 건드린다는 주장이 약해진다. 실험 1 산출물에 추가.
2. C 개입 후보로 **head 수준**: A3의 공유 head 집합(Gemma-2-27B는 26개)을 증폭하는 것과 A2식 head별 방향 추가. residual 방향 addition과 셋을 비교.
3. Gemma-3-27B-IT의 layer-0 norm 문제 — NLA/AO 본진 모델에서 head 중요도 계산 시 layer 0 제외.
4. Llama-3.1-8B는 회로가 가장 흐린 모델 — 싼 sweep 모델로 시작하되 여기서 안 갈려도 접지 말 것.

## B. 스니펫 확인 — 묶음별

### B1. 거짓 전제 QA (일반 도메인)

| 논문 | 한 것 | 관계 |
|---|---|---|
| CREPE (Yu et al., ACL 2023), QA² (Kim et al. 2023), FalseQA (Hu et al., ACL 2023), Syn-QA² ([2403.12145](https://arxiv.org/abs/2403.12145)) | 벤치마크. FalseQA가 **"거짓 전제만으로 학습하면 참 전제까지 거부한다"**를 처음 보고, 섞어 학습으로 완화 | FPQ/TPQ 시소는 2023년부터 알려짐. Well-Actually는 의료 재확인 |
| [Identifying and Answering Questions with False Assumptions](https://arxiv.org/abs/2508.15139) (Wang & Blanco, EMNLP 2025) | atomic assumption 생성 → 검색 근거로 검증 → 답변. 5모델 | extract-and-verify 최신형 |
| [LLMs Struggle to Reject False Presuppositions when Stakes are High](https://arxiv.org/abs/2505.22354) (2025) | GPT-4o·Llama-3-8B·Mistral-7B, 전제 트리거 유형별 거부율 | H2 쪽 증거. 카테고리 분석에 트리거 유형 변수 추가 가능 |
| [HACK](https://arxiv.org/abs/2510.24222) (2025-10) | 환각을 지식 있음(HK+)/없음(HK−)으로 나누고 steering이 HK+에서만 통함 | "지식이 있을 때만 개입이 먹힌다" — A 게이트의 논리 |
| [Do I Know This Entity?](https://arxiv.org/abs/2411.14257) (Ferrando et al., ICLR 2025) | SAE로 "아는/모르는 entity" 방향, 그것이 refusal을 인과적으로 게이트 | **지식 신호가 태도 회로를 게이트하는 선례** (관찰, 개입 방법 아님) |

### B2. sycophancy 내부 표상 (2025-08 ~ 2026-07)

| 논문 | 핵심 | 관계 |
|---|---|---|
| [Sycophancy Is Not One Thing](https://arxiv.org/abs/2509.21305), [Dissociating…](https://arxiv.org/abs/2607.07003), [Modes of Sycophancy](https://arxiv.org/abs/2607.20146) | 동조·아첨·사실 동의가 다른 선형 방향. 출력은 비슷해도 표상은 층 14 이후 완전 분리 | "sycophancy 벡터 하나"는 없음. Verbalizing-Assumptions이 validation 축을 밀어 실패한 이유 |
| [Sycophancy Hides Linearly in the Attention Heads](https://arxiv.org/abs/2601.16644) (EACL 2026) | probe는 residual·MLP에서도 되나 steering은 중간층 attention head 일부에서만 효과 | 개입 위치 |
| [When Truth Is Overridden](https://arxiv.org/abs/2508.02087) (AAAI 2026) | 1인칭 "I believe"가 3인칭보다 깊은 층을 더 흔듦. 후기층 출력 선호 이동 + 깊은 층 표상 분기 | Cancer-Myth는 전부 1인칭 환자 발화. H4 |
| [Dual-Stance Evaluation](https://arxiv.org/abs/2606.11205) (2026-04) | centroid 차이 steering이 동조·사실 동의 부분공간이 달라도 **둘 다 깎음** | **NFP 문제의 일반 도메인 버전.** 무조건 steering 대조군의 예측 |
| [CLiF](https://arxiv.org/abs/2606.26155), [Persona vectors for sycophancy](https://arxiv.org/abs/2605.21006), [Sycophancy Suppression Can Impair Rational Updating](https://arxiv.org/abs/2608.26511) | SAE feature 연속 점수 / 기성 persona vector ≈ 타깃 steering / 억제하면 정당한 업데이트도 막힘 | 마지막 것은 A2의 "정당한 교정도 잡는다"와 같은 경고 |
| [Linear Probe Penalties Reduce Sycophancy](https://arxiv.org/abs/2412.00967) (NeurIPS 2024 WS) | 보상 모델 probe 페널티 | 학습 시 대안 |

### B3. 조건부·게이트 steering

| 논문 | 한 것 | 관계 |
|---|---|---|
| [CAST](https://arxiv.org/abs/2409.05907) (ICLR 2025), DSAS, [GAPS](https://arxiv.org/abs/2609.01878) (2026-09) | 조건 방향 정렬로 hard gate / probe 출력을 연속 gate / 차원 수준 gate. toxicity에서 capability 손실 없이 억제 | 게이트 방법론은 있음. 단 **게이트 신호와 개입 방향이 같은 개념**(toxicity) |
| [Steering Vector Fields](https://arxiv.org/abs/2602.01654), [PCNET](https://arxiv.org/abs/2605.05953) | 맥락 의존 개입 / 확률 회로로 환각 탐지 후 동적 개입 | 대안 게이트 구현 |

### B4. 의료 sycophancy 벤치마크

[MedPRESS](https://arxiv.org/html/2608.02520), [Med-Stress / R-FT](https://arxiv.org/abs/2605.23932), [MedMisBench](https://arxiv.org/abs/2606.12291), [CausalT3](https://arxiv.org/abs/2601.08258v3) (Skepticism Trap과 Sycophancy Trap을 한 벤치마크에 — FPQ/TPQ 시소 그 자체). 전부 멀티턴 압력 또는 오도 맥락. 단일 질문 속 배경 전제는 Cancer-Myth뿐.

### B5. 경고 — 탐지 ≠ 제어

| 논문 | 주장 |
|---|---|
| [Perfect Detection, Failed Control](https://arxiv.org/abs/2606.24952) (2026-06) | 가짜 entity 탐지 AUC 1.0인데 그 방향 steering 효과 0. 탐지 방향과 제어 방향 cos 0.1~0.2, 4모델, 사전학습 기원 |
| [Detection Without Correction](https://arxiv.org/abs/2604.13068) (2026-04) | 7/7 모델에서 probe 방향 steering이 환각을 못 고침. probe의 가치는 생성 전 플래그 |
| [Readable but Not Controllable](https://arxiv.org/abs/2607.00158) (2026-06, 의료) | 4모델 16조합 probe AUROC 0.77~0.86, 뉴런 수준 제어 불가 |

셋 다 "**탐지 probe 방향으로 밀지 말라**". 우리는 A probe를 게이트로만 쓰고 C는 따로 뽑으므로 직접 반박은 아니다. 다만 (a) C 대조 방향이 causal한지 별도 증명, (b) A3가 보인 cos(진위, 동조) 0.4~0.8 때문에 C가 A와 겹칠 수 있음 — 이 둘이 우리 리스크다.

## C. 갭 갱신 — 02 표 이후 달라진 것

| 제안 요소 | 02 시점 | 지금 |
|---|---|---|
| 배경화된 거짓 전제가 hidden state에서 읽힌다 | 비어 있음 | **A1이 CREPE에서 0.69~0.78 (마지막 토큰).** 의료·전제 구간 위치·NFP는 남음 |
| 판정시키면 오탐 | Well-Actually negative bias | **A1이 반대 방향(accommodation)도 확인.** framing 무관하게 chance |
| probe 게이트 × 개입 | 비어 있음 | **A1이 프롬프트 라우팅으로 함** (일반 도메인). 내부 개입 게이트는 남음 |
| 의료 + 게이트 + ITI steering | 비어 있음 | **A2가 함** (멀티턴 압력, 게이트 = 압력 감지) |
| 알면서 따라간다 — head 수준 | Contextual-Truth 단언문 probe | **A3가 12모델 회로로.** 배경 전제에서는 남음 |
| 진위 × 태도 분해 | 비어 있음 | **아직 비어 있음.** 단 A3: 두 방향 cos 0.4~0.8 — 분해 가능성 자체를 재야 함 |
| 진위로 게이트한 태도 개입 + NFP 통제 | 비어 있음 | **비어 있음** |
| Cancer-Myth+NFP를 verbalizer 통제 벤치마크로 | 비어 있음 | 비어 있음 |

**차별점 문장 (갱신).** 게이트 steering도, probe 게이트 프롬프트도, 의료 게이트 ITI도 있다. 우리 것은 (a) 게이트 신호(진위)와 개입 대상(태도)이 **다른 개념**이고, (b) 통제군이 **표면상 구분 불가능한 NFP**이며, (c) 개입이 내부라 프롬프트 라우팅(A1)의 교정 품질 한계(47~61%)를 넘는지 잴 수 있고, (d) Cancer-Myth Table 1에 행을 넣는다.

## 설계에 반영할 것

1. **실험 3 baseline 추가**: probe-gated FP Identification (A1식). 이것이 GEPA보다 강한 진짜 대조군.
2. **실험 1 산출물 추가**: cos(A 방향, C 방향), 층별. A3 기준 0.4~0.8이면 분해 주장을 "부분 분해"로 낮춰 쓴다.
3. **C 개입 세 후보**: residual 방향 addition / A2식 head별 방향 / A3식 공유 head 증폭. 셋 다 NFP로 채점.
4. **A 게이트 추가 통제**: 전제가 참인데 Plain이 틀리게 답한 문항에서 게이트 발화율 (A2의 "정당한 교정" 관찰).
5. **Gemma-3-27B-IT**: head 중요도 계산에서 layer 0 제외.
6. 실험 1 기준선: 마지막 토큰 AUROC 0.70. 이 아래면 CREPE보다 못 읽는 것.

## 아직 원문이 필요한 것

- [Perfect Detection, Failed Control](https://arxiv.org/abs/2606.24952) — "제어 방향"을 어떻게 찾았는지, 게이트 후 *다른* 방향으로 개입해도 실패하는지
- [Dual-Stance Evaluation](https://arxiv.org/abs/2606.11205) — 평가 프로토콜 (우리 CAA 대조군 평가에 그대로 빌릴 수 있는지)
- Well-Actually — Well [2608.06539](https://arxiv.org/abs/2608.06539) Appendix I, Table 8~10 — TPQ 주석 방식, baseline 실행 설정
