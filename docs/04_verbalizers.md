# 04. Verbalizer — 바닥부터

## 한 문장

> 타깃 모델의 hidden state 벡터 하나(또는 몇 개)를 받아, 그 안에 뭐가 들었는지 영어로 써주는 **별도의 LM.**

## 물리적으로 벌어지는 일

**Verbalizing-Assumptions 추출.** 타깃 모델에 입력을 넣고 forward. 층 ℓ, 토큰 위치 p의 residual stream 벡터 v ∈ ℝ^d. (Qwen2.5-7B d=3584, Gemma-3-12B d=3840)

**Well-Actually 주입.** verbalizer도 LM이라 입력이 토큰 임베딩 열. 프롬프트 안의 **자리표시 토큰**(NLA `㈎`, AO `<ACT>`)의 임베딩을 v로 **덮어쓴다.** 나머지 토큰은 평범한 텍스트. v는 학습 때 norm으로 맞춘다 (NLA-Qwen: L2=150).

**MedMisBench 생성.** verbalizer가 평소처럼 autoregressive 생성. 그 텍스트를 읽는다.

가능한 이유: **덮어쓴 자리를 읽도록 파인튜닝됐기 때문.** 파인튜닝 안 된 모델에 꽂으면(Patchscopes) 읽긴 하나 흐릿하다.

## 왜 모델마다 따로 있나

d가 맞아야 하고, 무엇보다 **residual stream의 "언어"가 모델마다 다르다.** Gemma 벡터를 Llama verbalizer에 꽂으면 아무 뜻도 없다. 그래서 verbalizer는 **타깃 모델의 복사본을 파인튜닝**해 만들고, 체크포인트가 타깃 단위로 나온다.

## 네 종류 — 무엇으로 학습됐나

| | 학습 | 입력 | 프롬프트 | 출력 | 층 |
|---|---|---|---|---|---|
| **Patchscopes / SelfIE** (2024) | 없음 | 벡터 1 | 자유 | 흐릿한 묘사 | 아무 층 |
| **LatentQA** (ICLR 2026) | 지도. (활성값, 질문, 답) | 벡터들 | 질문 | 답 | 학습한 층 |
| **Activation Oracle** (ICML 2026 Oral) | LatentQA + 이진 분류 + 맥락 예측(past lens)을 다양하게 섞어 범용화 | **구간** `<ACT>`×n | **임의 질문** + `Layer N:` | 답 | 25/50/75% 깊이 |
| **NLA** (Anthropic 2026) | **비지도**. AV(벡터→글)·AR(글→벡터)를 GRPO로 — AV 글로 AR이 벡터를 복원하면 보상. Opus 요약으로 SFT 워밍업 | 벡터 **1** | **고정** | 스니펫 3개 | 체크포인트당 **1층** |

핵심 구분 두 축:
- **지도 vs 비지도.** AO는 "질문에 답하기"를, NLA는 "복원 가능하게 묘사하기"를 배웠다. 복원을 최대화하려면 **분산이 큰 방향**(형식·주제·다음 토큰)을 말해야 하므로 NLA 출력이 거기로 수렴한다. Contextual-Truth의 참/거짓은 선형 방향 하나 — 분산이 작아 NLA가 말할 유인이 없다.
- **질문 가능 vs 불가.** AO엔 묻고, NLA는 못 묻는다 (프롬프트를 바꿔 물을 수는 있으나 학습 분포 밖이고 AR 게이트를 잃는다).

## NLA 실제 출력 — Qwen2.5-7B L20, "What are you hiding?"

> **' hiding'** (cos 0.91): *"Formal chat format with informal tone ('Hey, I'm a cat') suggests a dialogue or mystery prompt about a person's hidden information or secrets. / The phrase 'What are you hiding' signals a question prompt structure… / Final token 'hiding'… expecting…"*
> **'What'** (cos 0.92): *"…likely a list or article about Tesla… related to the title 'The Future of Space Exploration.'"*

패턴: 스니펫 1 = 장르/형식, 2 = 방금 읽은 구절과 예고, 3 = 마지막 토큰과 다음 토큰 예상. cos 0.9에서도 맥락을 지어낸다. 주제("숨김·비밀")는 잡지만 "화자가 X라 믿고 그건 틀렸다"는 없다.
→ **L20 NLA는 주제·형식 판독기.** 진위 탐지기 아님.

## NLA 추론 계약 (kitft/nla-inference)

- AV 프롬프트는 `nla_meta.yaml`에서 로드. *"You are a meticulous AI researcher… describe the semantic content of that activation vector… `<concept>㈎</concept>`… explanation enclosed in `<explanation>` tags, 2-3 text snippets."*
- 주입 위치는 토큰 ID + 양옆 이웃(`>`, `<`)으로 검출. 이 부분만 유지하면 주변 텍스트는 바꿀 수 있다 (단 OOD).
- `injection_scale` 필수 (Qwen 150, Gemma-3-12B 80000). `embed_scale` Gemma는 √d.
- SGLang `input_embeds`, `--disable-radix-cache` 필수.
- **AR 왕복 `mse_nrm` = 2(1−cos)**: <0.5 좋음, 약 2 무작위. 샘플별 신뢰도 게이트.
- 층 20 = HF `hidden_states[21]`. temp 0.

## AO 인터페이스

`"Layer 18: <ACT> <ACT> <ACT> Is this a positive sentiment?"` — 층 번호, 활성값 구간, 자유 질문. 학습 깊이 25/50/75%, 평가는 주로 50%. 학습 과제: Taboo(숨긴 단어), User Gender, Secret Side Constraint(숨긴 지시), Emergent Misalignment — 전부 **"입력 텍스트엔 없고 활성값에만 있는 정보 꺼내기."** 보정 연구 별도 ([2605.26045](https://arxiv.org/html/2605.26045)).

## 공통 함정

ICML 2026 *Do Activation Verbalization Methods Convey Privileged Information?* — 기존 벤치마크는 **활성값 없이 텍스트만 봐도 풀린다.** verbalizer가 벡터를 읽은 건지 맥락으로 추론한 건지 구분이 안 됐다. NLA·AO 모두 해당. NLA 저자 자신의 limitations: confabulation, excessive expressivity, verbalizer 자기 지식 반영, 비용, 퇴화 목적.

**통제 4종**:
1. **NFP** — 없는 전제를 지어내나
2. **활성값 없이** — 텍스트만 주고 얼마나 맞히나 (그 논문의 핵심 검사)
3. **무작위/섞은 활성값** — 엉뚱한 벡터에도 같은 답인가
4. **그 사실을 모르는 모델의 활성값** — 그래도 "거짓"이라 하나

## 우리 세 층에 각각

| 층 | 주 도구 | 보조 | 이유 |
|---|---|---|---|
| A 판정 | **linear probe** | Contextual-Truth식 | 판정을 verbalizer에 안 맡김 — H6 회피, 게이트 신호 |
| B 내용 | **AO 서술 질문** | NLA (AR 게이트 통과분) | LatentQA의 "숨은 시스템 프롬프트 복원"이 "흡수된 전제 복원"과 같은 구조 |
| C 태도 | **대조 방향** (CAA식) + AO 질문 | persona-vector식 | refusal direction·persona vector 선례 |

## 왜 AO가 주력이고 NLA가 보조인가

"무엇이 들었나"(NLA)가 아니라 "X가 들었나"(AO)를 물어야 하기 때문. 가설 지도의 질문들(P의 자격, 계획, 태도)은 전부 "X가 들었나"다. NLA의 자리: 유도 질문 없는 서술(AO의 통제), AR 게이트, 그리고 "비지도 묘사기는 형식을 읽고 지도 질의기는 진위를 읽는다"는 대비 자체가 결과.

**NLA 프롬프트 수정 실험** (1시간, 학습 없음): 원 템플릿 vs "…Also describe what the model has taken as given and what it plans to address"를 Cancer-Myth 20 + NFP 20에 돌려 답이 일관되고 갈리는지. 갈리면 쓰고, 아니면 AO로.
