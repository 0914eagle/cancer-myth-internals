# 19. Method 실행 명세 — 질문 probe와 선택적 residual steering

2026-09-09. 교수님 발표의 Method 14–17장을 구체화한 **다음 본 실험의 실행안**이다. 아래 수치의 탐색 범위는 설계 선택이며 성공한 실험 설정이 아니다. 이번 변경은 문서 수정이며 코드를 실행·수정하거나 학습 완료를 의미하지 않는다. 기존 후보 나열보다 이 명세의 주 방법을 우선한다.

## 1. 주 방법과 데이터 흐름

주 방법은 **질문의 마지막 프롬프트 토큰에서 선형 probe로 FPQ를 판별하고, 같은 질문의 교정/비교정 참조 응답을 대조해 만든 방향을 선택된 질문의 생성에 적용**하는 것이다. 기반 LLM의 가중치는 학습하지 않는다. probe의 회귀 계수를 학습하고, steering 방향은 활성값의 평균 차로 계산한다.

| 부품 | 학습 입력 | 목표·산출물 | 추론 입력 |
|---|---|---|---|
| A probe | FPQ/정상 질문의 마지막 프롬프트 토큰 활성값 | 질문 라벨 1/0; scaler·w·b·층·문턱 | 새 질문만 |
| 교정 방향 C | 같은 학습 FPQ의 교정/비교정 참조 답변 쌍을 타깃 모델에 teacher-forcing | 응답 첫 32토큰의 평균 활성값 차; 단위 방향·층·scale | 답변 생성 중 활성값 |
| 정책 | 내부 dev에서 생성한 실제 답변 | 문턱·강도 및 허용 손실 조건 | 고정한 gate와 C |

A의 양성은 ‘거짓 전제 질문’이지 ‘Plain이 틀린 질문’이 아니다. C의 +/−는 참/거짓 **질문**이 아니라 교정/비교정 **응답**이다. 정상 질문을 거짓 질문에서 뺀 벡터를 곧바로 C로 쓰지 않는다. 이름 A와 코드의 위치 A_premise도 다른 개념이며, 주 probe의 코드 위치 이름은 D_last_prompt_token이다.

## 2. 분할과 입력 고정

- 모델: `Qwen/Qwen2.5-7B-Instruct`(현재 config d=3584, L=28), `meta-llama/Llama-3.1-8B-Instruct`(d=4096, L=32). 모델별로 probe·방향을 각각 만든다. revision과 실제 config를 저장한다.
- 원 질문 기준 FPQ 585, 정상 NFP/TPQ 150. 같은 원 질문·myth·파생 질문은 하나의 그룹이다. 정상 NFP/TPQ는 한 번만 fit하고, 주 방법에는 합성 true twin을 추가하지 않는다.
- outer는 grouped 5-fold, seed 17이다. 각 outer-train 안에서 그룹 단위 약 80% fit / 20% dev를 고정한다. 가능한 범위에서 두 라벨 비율을 유지하며 실제 문항 수를 보고한다. fit만 scaler·probe·방향·scale 계산에 사용하고, dev는 선택에만 쓴다. 선택 뒤 fit+dev로 다시 학습하여 문턱 척도를 바꾸지 않는다.
- 질문 입력은 같은 타깃 모델의 chat template에 user 질문과 assistant generation prefix를 붙인다. Plain과 gate의 원 입력은 같고, 참조 전제·라벨·교정 답변은 추론 입력에 넣지 않는다.
- 양쪽 클래스가 없어 probe/평가가 성립하지 않는 분할은 실행 전 그룹·라벨만 보고 수정한다. test 성능을 보고 분할 seed를 바꾸지 않는다. artifact에는 실제 split ID 목록과 hash를 저장한다.

## 3. A: 어디서 무엇을 추출해 probe를 학습하는가

### 3.1 질문 한 개당 한 벡터

개입 없이 질문을 한 번 forward한다. 각 후보 층에서 **padding을 제외한 마지막 프롬프트 토큰**의 residual 벡터 h를 얻는다. 질문 본문의 마지막 단어가 아니라 chat template의 assistant 시작 표지까지 처리한 위치다. 정답 생성이나 gold 전제 구간이 필요하지 않다.

층 인덱스 k는 `hidden_states[k]`를 뜻한다. 0은 embedding이고, 마지막 L은 모델의 final norm 이후일 수 있으므로 주 방법은 **k=1,…,L−1**만 사용한다. 이 범위는 decoder block k−1의 출력 hook과 대응한다. 추출 위치와 hook 위치의 수치적 일치 확인이 본 실행의 선행 조건이다. 마지막 정규화 이후 방향을 정규화 이전 hook에 그대로 넣지 않는다.

### 3.2 학습 목표

fit의 n개 질문으로 `X ∈ R^(n×d)`와 `y ∈ {0,1}^n`를 만든다. `y=1`은 FPQ, `y=0`은 중복 제거한 NFP/TPQ다. fit에서만 feature별 평균 μ와 표준편차 σ를 구하고 상수 feature의 scale은 1로 처리한다.

```text
z_i = (h_i − μ) / σ
s_i = wᵀz_i + b
p_i = sigmoid(s_i)
학습: class-weighted binary cross-entropy + L2 penalty
class weight: n / (2 × 해당 클래스 표본 수)
```

현 코드의 StandardScaler + LogisticRegression(class_weight='balanced', max_iter=2000)를 출발점으로 한다. 회귀의 inverse regularization `C_reg ∈ {0.01, 0.1, 1, 10}`을 dev에서 비교한다. `C_reg`는 교정 방향 C와 다른 기호다. 수렴하지 않은 후보는 실패로 기록하며 test로 넘어가 선택하지 않는다. class-balanced sigmoid 출력은 보정된 임상 확률이라고 부르지 않고 분류 점수로 사용한다.

### 3.3 층과 문턱 선택

층·C_reg는 dev AUROC가 가장 큰 조합으로 선택한다. 동률이면 작은 C_reg(강한 규제), 작은 k 순으로 정한다. 그 probe의 dev 점수와 모든 질문을 제외하는 문턱을 후보로 두고, **dev FPR ≤ 0.05에서 TPR 최대**인 문턱 τ_det를 선택한다. 동률은 낮은 FPR, 높은 문턱 순이다. 코드가 logit을 반환하므로 저장·비교는 logit 단위로 통일한다. sigmoid 표시는 가능하지만 logit 문턱과 혼용하지 않는다. 판정은 `s ≥ τ`로 통일하고 기존 `>` 비교와 동점 규칙을 구현에서 맞춘다.

이 5%는 본 실험의 탐지 operating-point 목표이며 임상 안전 보장이 아니다. 정상 dev가 18개라면 1개 오탐이 5.6%이므로, 달성 가능한 FPR과 분모를 함께 보고한다. Table 1에는 τ_det를 고정한 test TPR과 실제 test FPR을 별도 열로 쓴다. test FPR을 보고 τ를 다시 맞추지 않는다.

## 4. C: 어떤 응답에서 교정 방향을 추출하는가

### 4.1 같은 질문의 응답 쌍

주 방법은 현재 paired pipeline에 맞춰 Cancer-Myth `all_data.json`의 **fit FPQ에 대한 기존 참조 응답**을 사용한다. 같은 질문에 대해 유효한 기존 GPT-4o 판정이 +1인 교정 응답 r+와 −1인 교정 누락 응답 r−가 모두 있는 질문만 방향 학습에 쓴다. 0점·미판정 응답은 사용하지 않는다. 이 값은 응답별 judge 점수이지 집계 PCR이 아니다. 모든 FPQ에 쌍이 있다고 가정하지 않으며 fold별 사용/제외 수를 보고한다.

출처 모델의 말투가 방향을 지배할 수 있으므로, **fit에 한정하여** 현 코드의 출처 분포 균형 선택을 수행하고 양쪽 출처 모델·길이 분포를 기록한다. 균형 선택이 문체 교란을 완전히 없애는 것은 아니다. 현 스크립트는 전체 all_data에서 먼저 선택하므로 fit 필터를 선택 이전에 적용하도록 구현해야 한다. 한 쌍이라도 두 텍스트를 서로 다른 타깃 모델의 활성값으로 뺄 수는 없다. 모두 같은 타깃 모델로 다시 읽는다.

### 4.2 Teacher-forcing과 평균 차

fit 질문 x와 r+를 user/assistant 대화로 넣고, 같은 x와 r−도 동일한 chat template로 넣는다. Teacher-forcing은 모델이 답을 새로 만들게 하는 것이 아니라, **주어진 답변을 토큰 순서대로 읽을 때 활성값을 기록하는 것**이다. 교정 지시문을 한쪽 system prompt에만 붙이지 않는다.

assistant 답변 본문의 첫 m=min(32, 답변 토큰 수)개 residual을 평균한다. 역할 표지·padding·EOS는 평균에서 제외한다. 사용자 질문까지 함께 평균하지 않는다. 응답마다 평균을 먼저 구해 긴 답변이 더 큰 가중치를 갖지 않게 한다.

```text
e⁺_i,k = mean_t h_k(x_i, r⁺_i)[답변 첫 m⁺_i 토큰]
e⁻_i,k = mean_t h_k(x_i, r⁻_i)[답변 첫 m⁻_i 토큰]
d_k = mean_i (e⁺_i,k − e⁻_i,k)
c_k = d_k / ||d_k||₂
```

C는 raw residual 공간에서 계산하고 probe의 feature별 표준화를 적용하지 않는다. 동일 질문 대조는 질문 차이의 교란을 줄이지만 질문 내용이 정확히 상쇄되거나 순수 교정 회로만 남는다는 보장은 없다. 첫 5토큰은 부록 위치 ablation으로 두고, 주 방법은 첫 32토큰을 사용한다. 자연 응답의 비짝 대조와 시스템 지시 대조도 부록 후보이며, test에서 가장 잘 된 후보로 주 방법을 교체하지 않는다.

## 5. 개입 위치·강도와 최종 정책 선택

교정 층 후보는 `k_C ∈ {round(L/4), round(L/2), round(3L/4)}`로 제한한다. 현재 모델 설정에서 Qwen은 {7,14,21}, Llama는 {8,16,24}다. gate 층 k_A와 같을 필요는 없다. 각 C 후보 층에서 fit 질문의 마지막 프롬프트 토큰 residual norm 중앙값 ρ_k를 한 번 구해 저장한다.

```text
α ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.5}
개입 벡터 = α × ρ_k × c_k
```

이는 검증할 유한 탐색 범위다. α=0은 개입 없는 기준이다. test 첫 batch나 resume 후 남은 질문에서 ρ_k를 다시 추정하지 않는다. C의 평균 차 norm이 0이면 해당 후보는 무효다.

**Table 1과 Table 2의 문턱 구분:** Table 1은 앞의 탐지 전용 τ_det를 보고한다. Table 2의 최종 정책은 같은 probe를 고정한 상태에서 dev FPR≤5%를 만족하는 문턱 후보와 k_C·α를 함께 비교한다. dev 실제 응답의 NFP/TPQ 손실 제약을 만족하는 후보 중 PCR, PCS 순으로 선택하고, 동률이면 α가 작고 개입률이 낮은 쪽을 선택한다. 남은 동률 규칙도 config에 저장한다. 두 문턱이 다르면 `τ_det`, `τ_policy`로 구분하여 보고한다. dev 기준의 선별은 test 비열등성 입증과 다르다.

ε_NFP·ε_TPQ·QA별 허용폭은 기존 실행 명세처럼 **실험 전 확정해야 할 남은 연구 판단**이다. 표본 수가 적다는 이유로 허용폭을 넓혀 성공을 만들지 않는다. 허용폭을 만족하는 설정이 없거나 C 쌍이 부족해 추정이 불안정하면 그 사실을 보고한다. α=0만 남으면 교정 개선 가설을 지지한 결과가 아니다. QA test는 어떤 선택에도 쓰지 않는다.

## 6. 새 질문에서 실제로 steering하는 순서

1. 원 질문을 **개입 없는 prefill**로 읽고, k_A의 마지막 프롬프트 벡터를 저장한 scaler·probe에 넣는다.
2. `g(x)=1[s(x)≥τ_policy]`를 한 번 계산하여 저장한다. 생성 도중 재판정하지 않는다.
3. g=0이면 같은 Plain 생성 설정으로 답한다. g=1이면 새 생성 pass에 hook을 설치한다.
4. hook은 decoder block k_C−1의 출력에서 **마지막 프롬프트 토큰부터** 고정 벡터를 더한다. 이 위치가 첫 답변 토큰의 logits에 영향을 주기 때문이다. 그보다 앞의 질문 토큰은 변경하지 않는다.
5. 이어지는 autoregressive decode의 각 토큰 위치에도 같은 벡터를 더한다. 첫 주 실험은 시간 감쇠 없이 EOS 또는 최대 512개 새 토큰까지, greedy 생성으로 고정한다. gate를 읽은 pass 자체에는 steering이 없다.

```text
학습된 고정값: k_A, μ, σ, w, b, τ_policy, k_C, c, ρ, α
s = wᵀ((unsteered_h[k_A, last_prompt] − μ) / σ) + b
g = 1[s ≥ τ_policy]
답변 pass의 해당 위치에서: h' = h + g × α × ρ × c
```

현 구현은 gate forward와 답변 생성을 별도로 수행한다. 따라서 ‘추가 forward 없이 무료’라고 설명하지 않고 실제 지연을 측정한다. 선택된 질문도 먼저 읽은 KV cache를 무심코 재사용하면 첫 토큰 개입 정의와 달라질 수 있으므로, 초기 구현은 명시적으로 새 생성 pass를 사용한다.

## 7. Table 3와의 연결 및 저장 파일

Table 3에서는 내부 dev에서 고정한 동일 C·scale·α를 모든 steering 행에 공유한다. 개입 비율 q는 0<q<1인 사전 후보 {0.1,0.2,0.35,0.5}에서 dev의 같은 보존/교정 규칙으로 정한다. q를 만족하는 설정이 없으면 ‘해당 제약 아래 선택 불가’로 보고한다. 평가 fold에서는 라벨 없이 K개를 골라 3(a)를 채우고, 내부 H의 문항 목록을 그대로 프롬프트와 steering에 공유하여 3(b)를 채운다. 이 q는 Table 2의 질문별 문턱을 대체하지 않는다.

fold/model별 저장할 것은 split manifest, chat template/revision, probe(μ·σ·w·b·k_A), τ_det/τ_policy, pair manifest(질문·응답·출처·판정·토큰 span), c·k_C·ρ·α, dev 선택 기록, 평가 문항별 score·gate·응답·judge provenance다. run/config hash로 캐시를 구분한다. 학습된 숫자를 다른 모델에 그대로 복사하지 않는다.

## 8. 현재 코드와 추가 구현의 경계

| 항목 | 이미 있는 출발점 | 본 평가 전에 필요한 작업 |
|---|---|---|
| 질문 위치 D 추출 | [rows.py](../src/rows.py), [extract_activations.py](../src/extract_activations.py) | 고정 분할·padding·추출/hook 위치 일치 확인 |
| 표준화+L2 logistic | [probes.py](../src/probes.py) | fit/dev 선택, 두 문턱·동점 규칙 및 artifact 저장 |
| 참조 응답 쌍·C | [make_paired_rows.py](../scripts/make_paired_rows.py), [run_direction_pair.py](../scripts/run_direction_pair.py) | 참조 선택 전 fit 필터, fold별 방향, span·출처 기록 |
| gate와 residual hook | [steering.py](../src/steering.py) | ≥ 판정 규약 통일, score 기록, 고정 scale 로드 |
| 실행기 | [run_steer.py](../scripts/run_steer.py) | 전체 fold orchestration, dev 정책 선택, random/top-K·프롬프트 경로 |
| 보존 판정 | [13](13_task_spec.md), [16](16_code_overview.md) | ε·CI·TPQ 이산화 규칙 확정, judge 파싱/캐시와 QA 평가 구현 |

현재 코드 주석의 ‘같은 질문이라 내용이 정확히 상쇄된다’는 표현은 이 명세의 인과적 주장으로 채택하지 않는다. `steering.py` 상단의 ‘generated positions only’ 요약보다 실제 마지막 프롬프트 토큰부터 시작하는 hook 동작이 정확한 실행 기준이다. 현재 `run_steer.py`의 첫 batch norm 계산도 본 실험에는 그대로 사용하지 않는다.
