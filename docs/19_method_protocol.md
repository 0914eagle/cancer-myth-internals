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

### 1.1 기존 부품과 초기 설정을 구분한다

선형 logistic probe, 조건/행동 분리, 대조 응답 평균 차, 같은 질문의 응답 쌍은 기존 기법이다. Two Axes의 기여는 새 probe 공식보다 두 오류 축의 분석·정책이고, Gated의 기여는 기존 ITI·probe·평균 차를 행동별 gate·head 선택·보존 조정으로 결합한 구성이다. 현재의 residual 선택은 새로운 알고리즘으로 확정하지 않는다. [20의 출처·기여 비교](20_baseline_transfer_and_novelty.md)

마지막 프롬프트는 생성 전 판정 가능성, logistic은 단순한 출발점, paired C는 질문 차이의 교란 감소와 현 pipeline 사용 가능성을 이유로 선택했다. 첫 32토큰·층 후보·α grid·5% 운영점·split seed는 제안한 초기 실행 설정이다. 문헌에서 최적이라고 입증한 값이 아니며 dev에서 정해 test에 잠근다. 핵심 비교 없이 ‘내부/steering이 필요하다’는 결론을 내리지 않는다.

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

**Table 1과 Table 2의 문턱 구분:** Table 1은 앞의 탐지 전용 τ_det를 보고한다. Table 2의 최종 정책은 같은 probe를 고정한 상태에서 dev FPR≤5%를 만족하는 문턱 후보와 k_C·α를 함께 비교한다. dev 실제 응답의 NFP 손실 제약을 만족하는 후보 중 PCR, PCS 순으로 선택하고, 동률이면 α가 작고 개입률이 낮은 쪽을 선택한다. 남은 동률 규칙도 config에 저장한다. 두 문턱이 다르면 `τ_det`, `τ_policy`로 구분하여 보고한다. dev 기준의 선별은 test 비열등성 입증과 다르다.

ε_NFP·QA별 허용폭은 기존 실행 명세처럼 **실험 전 확정해야 할 남은 연구 판단**이다. 표본 수가 적다는 이유로 허용폭을 넓혀 성공을 만들지 않는다. 허용폭을 만족하는 설정이 없거나 C 쌍이 부족해 추정이 불안정하면 그 사실을 보고한다. α=0만 남으면 교정 개선 가설을 지지한 결과가 아니다. QA test는 어떤 선택에도 쓰지 않는다.

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

## 7. Table 2·부록 A1과의 연결 및 저장 파일

부록 Table A1에서는 내부 dev에서 고정한 동일 C·scale·α를 모든 steering 행에 공유한다. 개입 비율 q는 사전 후보 {0.1,0.2,0.35,0.5}에서 dev의 보존/교정 규칙으로 정한다. 제약을 만족하는 설정이 없으면 선택 불가로 보고한다. 평가 fold에서는 라벨 없이 K=round(qN)개를 골라 선택 신호를 비교한다. 이 배치 진단은 Table 2의 질문별 문턱을 대체하지 않는다.

**Table 2의 동일 선택 비교.** Hidden gate + FP prompt와 Hidden gate + C steering은 probe·문턱·문항별 gate mask를 정확히 공유한다. 현재 주 정책의 probe와 τ_policy·C를 내부 dev에서 선택한 뒤 FP prompt 행에도 그 gate를 적용한다. 프롬프트 버전은 dev에서 정하되 gate를 다시 고르지 않는다. 선택되지 않은 질문에는 같은 Plain 응답을 사용하고, 선택된 질문에서만 교정 연산을 바꾼다. 따라서 이 두 행은 같은 대상에서의 교정 방식 비교이며, 각 교정 방식에 독립적으로 최적화된 gate끼리의 비교가 아니다. 프롬프트에 맞춰 gate까지 별도 최적화한 Two Axes식 정책은 필요하면 부록에 구분해 보고한다. 이 두 행의 paired ΔPCR·ΔPCS·ΔNFP와 95% CI를 함께 해석하고 별도 Table 3에서 같은 결과를 반복하지 않는다.

fold/model별 저장할 것은 split manifest, chat template/revision, probe(μ·σ·w·b·k_A), τ_det/τ_policy, pair manifest(질문·응답·출처·판정·토큰 span), c·k_C·ρ·α, dev 선택 기록, 평가 문항별 score·gate·응답·judge provenance다. run/config hash로 캐시를 구분한다. 학습된 숫자를 다른 모델에 그대로 복사하지 않는다.

## 8. 현재 코드와 추가 구현의 경계

| 항목 | 이미 있는 출발점 | 본 평가 전에 필요한 작업 |
|---|---|---|
| 질문 위치 D 추출 | [rows.py](../src/rows.py), [extract_activations.py](../src/extract_activations.py) | 고정 분할·padding·추출/hook 위치 일치 확인 |
| 표준화+L2 logistic | [probes.py](../src/probes.py) | fit/dev 선택, 두 문턱·동점 규칙 및 artifact 저장 |
| 참조 응답 쌍·C | [make_paired_rows.py](../scripts/make_paired_rows.py), [run_direction_pair.py](../scripts/run_direction_pair.py) | 참조 선택 전 fit 필터, fold별 방향, span·출처 기록 |
| gate와 residual hook | [steering.py](../src/steering.py) | ≥ 판정 규약 통일, score 기록, 고정 scale 로드 |
| 실행기 | [run_steer.py](../scripts/run_steer.py) | 전체 fold orchestration, dev 정책 선택, random/top-K·프롬프트 경로 |
| 보존 판정 | [13](13_task_spec.md), [16](16_code_overview.md) | NFP·QA의 ε·CI 확정, judge 파싱/캐시와 QA 평가 구현, Table 3 gold 전제 사실 확인과 최종 응답 점수 부록 경로 |

현재 코드 주석의 ‘같은 질문이라 내용이 정확히 상쇄된다’는 표현은 이 명세의 인과적 주장으로 채택하지 않는다. `steering.py` 상단의 ‘generated positions only’ 요약보다 실제 마지막 프롬프트 토큰부터 시작하는 hook 동작이 정확한 실행 기준이다. 현재 `run_steer.py`의 첫 batch norm 계산도 본 실험에는 그대로 사용하지 않는다.

## 9. 직접 baseline의 구현 경계

Table 1은 [13 §1](13_task_spec.md#1-task-1--판별-신호-비교--table-1)의 열 가지 readout이다. 새 detector Ours 행은 없다. 출력 readout은 생성 이후 정보이며 CoT monitor의 모델·템플릿·점수 추출 규칙은 실행 전에 추가로 확정한다. 최종 응답이나 judge 라벨을 입력에 유출하지 않는다.

Table 2의 Gated head steering은 원 artifacts를 다른 모델에 꽂는 조건이 아니다. 동일 의료 fit 질문과 교정/비교정 쌍으로 head를 탐색하고, FPQ/정상 gate를 학습하여 의료 전제 목표의 head 개입을 평가한다. 원 H/S 이중 제어에서 사용자 압력 축은 주 Cancer-Myth 라벨에 없으므로 FP 교정 adaptation으로 명명한다. 사용한 원 head ranking·ablation·방향·강도 규칙과 변경한 입력·라벨을 기록한다. head 선택의 ablation 점수·토큰 pooling·연속 gate 강도 매핑 및 원 학습 코드와의 대응은 아직 구현 명세를 보완해야 한다. 현재 residual C 실행안과 동일한 구현이라고 간주하지 않는다.

CAST도 원 PCA 조건/행동 벡터와 유사도 문턱의 의료 adaptation을 별도로 명세한다. logistic gate를 넣고 CAST 원 방법이라고 부르지 않는다. 이러한 재학습 baseline의 성능이 없으면 비교 우위를 주장할 수 없다. 미구현 행은 미측정 상태로 남긴다.


## 10. Table 3 후보: Well Table 1을 계승한 사실 확인 진단

**현재 위치와 확정 범위.** 가져올 선행표는 Well-Actually Table 1로 확정한다. 이 표는 교수님 발표의 기존 연구 근거로 사용할 수 있다. 아래의 우리 Table 3 및 추가 probe 행은 사실 확인 진단을 독립 연구 질문으로 채택할 경우의 후보이며 본 실험으로 확정한 것은 아니다. 기존 결과를 인용하는 것만으로 우리 실험 결과표가 되지 않는다. 현재 질문 gate의 검증이 목적이면 Table 1의 질문 단위 탐지 비교에 RAG 기반 전제 추출·검증을 추가하는 방안도 가능하며, gold 전제를 받는 원 Table 1의 숫자와 직접 섞지 않는다.

Table 3 후보는 최종 환자 답변을 평가하는 RAG 표가 아니다. 주석 전제를 직접 제공하고 근거 조건별 참·거짓 정확도를 평가한다. 원 보고값은 참고 패널에 두고, 같은 우리 전제·문서에서 재실행한 baseline과 추가 진단을 별도 비교한다. [21 — 원 표·공개 코드·재현 범위](21_well_rag_reproducibility.md)

실행은 `prompting/run_fact_check.py --check_gold --pipeline fact_check`이며 주석 전제별로 검색하고 판정한다. 주석 전제 자체는 이 진단의 허용 입력이고 진위 정답 라벨은 입력하지 않는다. 원 질문 기반 gate 평가와 배포 응답에는 gold 전제를 제공하지 않는다. FP/TP 정확도는 별도 분모로 계산한다. `run_check_gold_eval.py`는 오답률을 반환하므로 표의 정확도로 변환한다. response-level S5 judge는 이 경로가 아니다.

내부 readout 추가행은 전제·근거 입력으로 별도 학습하는 진단이다. 기존 질문 A gate의 숫자를 복사하지 않는다. 기본 학습기는 §3의 표준화+L2 logistic을 사용하되 입력은 원 사실 확인 템플릿의 마지막 프롬프트 토큰이고, 레이블은 해당 전제의 FP=1/TP=0이다. RAG 조건별 fit/dev/held-out 그룹 목록은 공유하며 readout artifact는 조건별로 분리한다. fit에서만 scaler·probe를 학습하고 dev에서 층·규제·문턱을 선택한다. 이 진단의 문턱은 dev balanced accuracy를 최대화하도록 정하고 동률이면 참 전제 오탐이 낮은 쪽으로 고정한다. 기존 질문 gate의 τ_det/τ_policy와 다른 문턱이다. 같은 질문·myth의 전제는 같은 fold에 묶는다.

출력과 내부의 비교 입력은 같은 전제·문서·few-shot 및 허용 문맥이다. CoT 행은 동일 전제를 검토한 뒤 참·거짓을 판정하며 추가 토큰·시간을 기록한다. 원 지원 reasoning 설정과 별도 CoT 프롬프트를 혼동하지 않는다. doctor_suggestion 문맥을 사용하면 양쪽에 동일하게 제공하고 해당 shortcut 가능성은 별도 분석한다. 새 정확도에는 전체 전제 분모·95% CI·무효율을 기록하고, 같은 질문의 전제를 함께 재표집한다.

Table 3는 사실 확인 단계의 진단이며 실제 질문의 교정 효과는 Table 2에서만 주장한다. 같은 gate prompt/C 및 부록 A1 규칙은 유지한다. 사실 확인 재실행과 추가 probe 경로는 아직 실행되지 않았다.
