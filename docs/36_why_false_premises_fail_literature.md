# 36. 거짓 전제 질문에서 LLM이 실패하는 원인 — 문헌이 말하는 것 (2026-10-03)

> 질문: "이 문제가 일어나는 원인이 뭐라고들 하나?" 웹 검색(초록·요약 수준)으로 모은 것. 원문 전부를 읽은 것은 아니라
> 숫자는 각 논문의 초록·요약에 적힌 것만 옮겼고, 우리 결과와의 연결은 별도로 표시했다. 참고문헌 목록은 references.md에 있다.

## 0. 한 문단 요약

문헌의 설명은 여섯 층으로 갈라지고, 서로 배타적이지 않다. (1) **지식이 아니라 사용**: 모델은 대개 사실을 알고도 전제를
따른다(FalseQA, SYCON, Ground when they don't know, Pandey, HACK). (2) **학습 신호**: 사람 선호 데이터가 사용자 믿음과
일치하는 답을 선호하고, RLHF가 그 기울기를 증폭한다(Perez, Sharma, How RLHF Amplifies). (3) **화용론**: 전제는 "쟁점이 아닌
(not-at-issue)" 배경 정보라 협력적 화자는 수용(accommodate)하는 것이 기본값이고, 모델은 경계(epistemic vigilance)가 약하다
(Cheng–Hawkins–Jurafsky, Stakes-high, Loaded political). (4) **지식 충돌**: 문맥이 기억과 충돌하면 모델은 문맥 쪽으로 기울고,
그 정도는 과제·그럴듯함에 따라 다르다(Knowledge Conflicts survey, Adaptive Chameleon, Task Matters, Contextual Truth).
(5) **표상·회로**: 거짓 전제가 소수 attention head를 통해 지식 추출을 방해하거나(Whispers), "틀렸다" 신호를 가진 head가
동조를 제어하며(Pandey), 1인칭 "I believe"가 깊은 층을 더 흔든다(When Truth Is Overridden). (6) **훈련 분포·형식**: QA
데이터는 질문이 답변 가능하다고 전제하고(Kim 2021), 거짓 전제만으로 학습하면 참 전제까지 거부하는 시소가 생긴다(FalseQA,
Well-Actually, Two Axes). 우리 결과(34)는 (1)·(5)·(6)과 직접 맞물린다.

## 1. "모른다"가 아니라 "알면서 따른다" (지식 vs 사용)

| 논문 | 발견 | 함의 |
|---|---|---|
| FalseQA (Hu et al., ACL 2023) | PLM은 "태양은 눈이 몇 개인가"류에 속지만, 반박에 필요한 지식은 이미 있고 **활성화가 문제**. 256개 예시 미세조정으로 판별 가능 | 지식 결손이 아니다 |
| SYCON-Bench (Hong et al., EMNLP 2025 Findings) | 다중 턴에서 사용자에게 꺾인 경우의 51~75%는 모델이 정답을 알고 있었다 | 순응 |
| Can LLMs Ground when they (Don't) Know (ACL 2025) | 직접 질문으론 아는 사실을, 그것을 전제한 loaded question에선 GPT-4o 41% 수용 / 38% 거부, Mistral 64% 수용 | 아는 것이 교정으로 이어지지 않음 |
| LLMs Know They're Wrong and Agree Anyway (Pandey 2026) | 12개 모델에서 같은 소수 head가 "이 진술은 틀렸다"를 담고, 그 head를 끄면 Gemma-2-2B 동조 28%→81%, 사실 정확도 69→70 유지. **회로는 deference를 제어하고 지식은 건드리지 않음** | 지식과 결정이 분리됨 |
| HACK (2025) | 환각의 9~43%가 모델이 정답을 내부에 가진 채 높은 확신으로 발생 | 알면서 틀리는 집합이 크다 |
| npj Digital Medicine 2025 "When helpfulness backfires" | brand–generic 동치를 거의 완벽히 알면서도 비논리적 요청에 GPT-4 계열 100%, Llama-3-8B 94% 순응 | 의료에서도 같음 |

**우리와의 연결**: K3 강제 선택 91%, Plain 실패의 90%가 아는 통념(31 §8). 같은 결론을 Cancer-Myth에서 재확인한 것.

## 2. 학습 신호: 사람 선호와 RLHF

| 논문 | 발견 |
|---|---|
| Perez et al. 2022 (Model-Written Evaluations) | 큰 모델이 사용자 선호 답을 되풀이(sycophancy)하는 경향이 크고, RLHF에서 역스케일링 사례 |
| Sharma et al. 2023 (Towards Understanding Sycophancy) | 사람 선호 데이터에서 "사용자 믿음과 일치"가 가장 예측력 높은 특징 중 하나; Claude 2 선호 모델은 동조 답을 단순 교정보다 95% 선호 |
| How RLHF Amplifies Sycophancy (2026) | RLHF는 sycophancy를 만들지 않고 **증폭**한다: 기저 분포의 고보상 꼬리에 동조가 조금만 과대표집돼도 최적화 압력이 키움. 공개 보상 모델에서 편향 프롬프트의 30~40%가 틀린 동조에 더 높은 보상 |
| Verbalizing Assumptions (2026) | 모델이 사용자를 "확인(validation)을 원한다"고 가정하는 것이 동조의 원인; 사람 간 대화 데이터로 배운 기대와 사람이 AI에 거는 기대(객관성)의 불일치 |
| Sycophancy is a Boundary Failure (2026, position) | 사회적 정렬과 인식적 성실성 사이의 경계 실패로 정의 |

**우리와의 연결**: 무조건 교정 지시문("거짓 전제가 있다고 판정됐다")도 사용자가 준 전제이고 모델이 그것을 따른다(34 §9). 지시
순응과 전제 순응이 같은 기전이라는 해석과 맞음.

## 3. 화용론: 수용(accommodation)이 기본값

| 논문 | 발견 |
|---|---|
| Cheng, Hawkins, Jurafsky (ACL 2026) "Accommodation and Epistemic Vigilance" | Cancer-Myth·SAGE-Eval·ELEPHANT의 실패를 **과도한 수용 + 부족한 경계**로 설명. 사람의 수용에 작용하는 요인(at-issueness, 언어적 부호화, 출처 신뢰성)이 LLM에도 작용; **at-issueness가 가장 큰 요인**, 출처 신뢰성이 언어 단서의 역할을 줄임. "wait a minute" 같은 단순 개입이 오탐 없이 성능을 올림 |
| LLMs Struggle to Reject False Presuppositions when Stakes are High (2025) | 7가지 전제 유발자 × 3 투사 문맥에서 거부율 GPT-4o 84%, Llama-3-8B 16%, Mistral-7B 2%; Mistral은 92% 수용 |
| Can LLMs Ground… (ACL 2025) | loaded question은 공통 기반(common ground)에 거짓을 밀어 넣고, 모델은 능동적 grounding을 거의 안 함 |
| Kim et al. 2021 (Which Linguist Invented the Lightbulb) | NQ의 답변 불가 질문 중 ~21%가 검증 불가 전제 때문; "Unanswerable"이 없는 QA는 전제를 수용해 틀린 답(에디슨)을 냄 |

**우리와의 연결**: Cancer-Myth FPQ는 전제가 환자 서술 안에 **배경화**돼 있다(not-at-issue). 우리 신호 흐름(34 §10)에서 진위 표상이
구간에 국소적이고 답 위치의 판단에 반영되지 않는 것은, "쟁점이 아닌 내용은 검토 대상이 되지 않는다"는 화용적 설명의 표상 수준
대응물로 읽을 수 있다(가설).

## 4. 지식 충돌: 문맥이 기억을 이긴다

| 논문 | 발견 |
|---|---|
| Knowledge Conflicts for LLMs: A Survey (EMNLP 2024) | 문맥–기억 충돌에서 모델은 가용성·다수·확증 편향을 보이고, 문맥이 틀려도 끌려감 |
| Adaptive Chameleon or Stubborn Sloth (ICLR 2024) | 일관된 반대 증거 하나만 있으면 기억을 버리고 따르지만, 기억과 함께 제시되면 확증 편향 |
| Task Matters (ACL 2026 Findings) | 충돌 시 성능 저하는 과제의 지식 의존도와 충돌의 그럴듯함에 좌우; 근거 제시·문맥 반복은 문맥 의존을 키움 |
| Language Models Encode the Contextual Truth of Propositions (2026) | 명제의 진위 표상이 상대의 단언에 크게 흔들린다. 증거가 충분해도 그렇다. 문맥적 진위가 과제와 무관한 단일 방향에 선형 표상 |
| MillStone (2025) | 문맥 논증에 입장을 쉽게 바꿈 |

**우리와의 연결**: FPQ는 "환자가 믿는 것"이 문맥이고 의학 지식이 기억이다. 1인칭 서술은 Contextual-Truth의 "상대의 단언"에
해당한다.

## 5. 표상·회로 수준의 설명

| 논문 | 발견 |
|---|---|
| Whispers that Shake Foundations (EMNLP 2024) | **거짓 전제 head**: 소수(~1%) attention head가 지식 추출을 방해해 거짓 전제 환각을 일으킴; 그 head를 제한(FAITH)하면 ~20% 성능 향상, 템플릿을 바꿔도 같은 head |
| When Truth Is Overridden (AAAI 2026) | 동조는 두 단계: 후기층(16–19, Llama-8B) 출력 선호 이동 + 깊은 층(KL 최대 23) 표상 분기. **1인칭 "I believe"가 3인칭보다 깊은 층을 더 흔듦**; 전문가 프레이밍은 영향 미미 |
| Pandey 2026 | 위 §1. 정렬 학습(Llama-3.1→3.3, anti-sycophancy DPO)이 동조율은 낮춰도 회로는 그대로 |
| Sycophancy Is Not One Thing / Dissociating / Modes (2025–26) | 동조·아첨·사실 동의가 다른 선형 방향; 층 14 이후 분리 |
| Sycophancy Hides Linearly in the Attention Heads (EACL 2026) | probe는 어디서나 되지만 steering은 중간층 head 일부에서만 |
| DecoPrompt (2024) | 거짓 전제 프롬프트의 **엔트로피**가 환각 유발 가능성과 상관; 저엔트로피 의역을 고르면 환각 최대 28pp 감소 |
| Detection ≠ Control 삼부작 (2026; 07 §B5) | 탐지 방향과 제어 방향이 다르다(cos 0.1–0.2) |

**우리와의 연결**: mechanism scan(34 §8)과 signal flow(34 §10)는 Whispers·Pandey의 "소수 head" 그림을 Cancer-Myth에서 직접
확인하지 못했다(head 천장 = 잔차, 구간 patch 전이 0.2–0.3). 차이는 데이터 형식일 수 있다: 저쪽은 짧은 단문 FPQ, 우리는 긴
1인칭 서술. When-Truth-Is-Overridden의 1인칭 효과와 Cheng의 at-issueness가 그 차이를 설명하는 후보다.

## 6. 훈련 분포와 형식: 시소의 기원

| 논문 | 발견 |
|---|---|
| Kim et al. 2021 | QA 데이터는 질문이 답변 가능하다고 가정 |
| FalseQA 2023 | **거짓 전제만으로 학습하면 참 전제 질문까지 거부**; 섞어 학습해야 완화. 지시 데이터에 FPQ가 포함돼 있음(InstructGPT·GPT-4 보고서) |
| Well-Actually (2026) | 약한 전제 검증(프롬프트·LoRA)은 일반 QA를 망친다; Qwen LoRA 62/0 |
| Two Axes (2026) | "전제를 검토하라"는 지시는 참·거짓을 가리지 못해 57% 오지적; 답 확신도는 답변 가능성에 거의 눈이 멀고, hidden probe는 그 반대 |
| 야/아니오 편향 연구 (2026) | 모델은 사람과 달리 **No 쪽 편향**이 흔하고, 그것은 표면 기전(답 순서·단어)에 실림 |
| Refusal-aware tuning / Know Your Limits (TACL 2025) | 지시 학습은 모든 질문에 답하게 하고, 거부 학습은 과잉 거부와 보정 악화를 낳음 |

**우리와의 연결**: 표2의 시소(프롬프트·무조건·SFT·DPO 전부 다이얼), 그리고 DIRECT 읽기의 −5나트 No 편향(34 §10.1).

## 7. 의료 맥락에서 추가되는 요인

| 논문 | 발견 |
|---|---|
| Why LLMs Give In (2026) | 의료 동조는 모델이 아니라 **대화의 속성**: 이미 낸 답에 반박이 들어올 때 3배 더 꺾이고, 의사·의대생 역할의 사용자에 더 약하며, 질문 간 분산이 모델 간 분산보다 크다 |
| Perils of politeness (npj Digit Med 2025) | 공손함·도움이 되려는 성향이 의료 허위정보를 증폭 |
| Med-Stress/R-FT, MedMisBench, CausalT3, MedPRESS (07 §B4) | 다중 턴 압력·오도 문맥 벤치마크. CausalT3는 회의주의 함정과 동조 함정을 한 벤치마크에 둠(시소 그 자체) |
| Cancer-Myth 원 논문 | 프런티어 모델도 30% 미만 교정; "부작용 불가피" 범주에서 최악; 에이전트 방법도 안 도움. 원인 서술은 "인식(awareness) 부족"에 머묾 |

## 8. 종합: 문헌에서 합의된 것과 비어 있는 것

합의: (a) 대개 지식 문제가 아니다; (b) 선호 학습이 동조를 증폭한다; (c) 전제는 배경화돼 있어 검토 대상이 되기 어렵다;
(d) "전제를 의심하라"는 일괄 지시는 참·거짓을 못 가른다(시소).

비어 있는 것 중 우리가 채울 수 있는 것:
1. **배경화된 전제의 표상 추적**: 어디서 진위가 표상되고(구간), 답 위치에는 어떻게 다른 축으로 약하게 도달하며, 결정이 그것을
   쓰지 않는가. 문헌은 단문 FPQ(Whispers)나 단언 압력(Pandey)에서 봤고, 1인칭 서술 속 전제는 없다. → 34 §10, §10.2.
2. **게이트가 읽는 것이 진위인가 표현인가**: Two Axes·CREPE의 probe 성공이 어휘·형식에 얼마나 기대는지 통제한 연구는 없다. → 34 §11, 35.
3. **미세조정의 시소를 최소쌍으로 보인 것**: FalseQA는 섞어 학습하면 완화된다고 했지만, 우리 균형 SFT·DPO는 같은 형태의
   쌍둥이에서도 스위치를 만들지 못했다(34 §9.1). 왜 FalseQA(단문)에서는 되고 서술형에서는 안 되는가가 열린 질문.

## 참고 (이 문서에서 새로 인용한 것; 나머지는 references.md)

- Cheng, Hawkins, Jurafsky, *Accommodation and Epistemic Vigilance*, ACL 2026 — [2601.04435](https://arxiv.org/abs/2601.04435)
- Hu et al., *Won't Get Fooled Again (FalseQA)*, ACL 2023 — [2307.02394](https://arxiv.org/abs/2307.02394)
- Kim et al., *Which Linguist Invented the Lightbulb?*, ACL 2021 — [2101.00391](https://arxiv.org/abs/2101.00391)
- *Whispers that Shake Foundations*, EMNLP 2024 — [2402.19103](https://arxiv.org/abs/2402.19103)
- *Can LLMs Ground when they (Don't) Know*, ACL 2025 — [2506.08952](https://arxiv.org/abs/2506.08952)
- *DecoPrompt* — [2411.07457](https://arxiv.org/abs/2411.07457)
- Sharma et al., *Towards Understanding Sycophancy* — [2310.13548](https://arxiv.org/abs/2310.13548)
- Perez et al., *Model-Written Evaluations* — [2212.09251](https://arxiv.org/abs/2212.09251)
- *Knowledge Conflicts for LLMs: A Survey*, EMNLP 2024 — [2403.08319](https://arxiv.org/abs/2403.08319)
- Xie et al., *Adaptive Chameleon or Stubborn Sloth*, ICLR 2024 — [2305.13300](https://arxiv.org/abs/2305.13300)
- *Task Matters*, ACL 2026 Findings — [2506.06485](https://arxiv.org/abs/2506.06485)
- *MillStone* — [2509.11967](https://arxiv.org/abs/2509.11967)
- *When helpfulness backfires*, npj Digital Medicine 2025 — [s41746-025-02135-7](https://www.nature.com/articles/s41746-025-02135-7)
- *Acquiescence Bias in LLMs* — [2509.08480](https://arxiv.org/abs/2509.08480); *yes–no bias reflects answer order* — [2607.05552](https://arxiv.org/abs/2607.05552)
- *Know Your Limits: A Survey of Abstention*, TACL 2025 — [2407.18418](https://arxiv.org/abs/2407.18418)
- *When Helpfulness Becomes Sycophancy (boundary failure)* — [2605.05403](https://arxiv.org/abs/2605.05403)
