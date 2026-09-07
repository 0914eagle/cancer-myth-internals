# 06. 실험 계획

## 실험 1 — 표상이 배경화된 전제에서 읽히는가

이 프로젝트의 성패가 갈리는 실험. 학습 없음. 모델당 하루.

### 입력

- Cancer-Myth 585 (`data/all_data.json`, `wrong_assumption` 필드로 전제 텍스트)
- NFP 150 (`data/nfp.json`) + Well-Actually의 참 전제 수동 주석 116개 (`Well/data_gen/CancerMythNFP/`, HF `shenranw/CancerMyth-TPQ`)
- 각 문항에 타깃 모델의 chat template 적용

### 위치 (prefill)

| 위치 | 정의 | 정렬 |
|---|---|---|
| A. 전제 구간 마지막 토큰 | `wrong_assumption`을 질문 안에서 fuzzy match | 필요. 못 찾으면 B만 |
| B. 질문 끝 | 사용자 턴 마지막 토큰 | 불필요 — **첫 실행은 B부터** |
| C. 전제 구간 평균 | Verbalizing-Assumptions이 쓴 방식 | 필요 |
| D. 마지막 프롬프트 토큰 | assistant 턴 시작 직전 | 불필요 — **태도(C층) 읽는 자리** |
| E. 응답 첫 5토큰 | 생성 시작 | H4·태도 |

### 캐시 (모델당 한 번 forward)

- 위치 A~E × **전 층** residual → probe용 (735 × 5 × 28–46층 × d)
- verbalizer 층(NLA 고정층, AO 50% 깊이)의 **전 토큰** → NLA/AO용

그 뒤는 전부 오프라인.

### 읽기

| 층 | 방법 | 출력 |
|---|---|---|
| **A 판정** | linear probe + difference-of-means, 층×위치 sweep. 개발 데이터 안에서 grouped CV, test는 잠금 | 거짓 vs 참 전제 **AUROC**와 **고정 FPR에서의 TPR**. 같은 held-out 문항에서 출력 readout과 비교: 직접 질문의 연속 점수, extract-and-verify의 log-odds, 같은 분할로 학습한 출력 readout, 텍스트만 쓴 단순 기준선(BoW). 02의 SDT 환산은 보조 |
| **B 내용** | AO: "what the model has taken as given / what it plans to address" (위치 A, B) | P의 자격(사실/믿음/주장) 분류, 계획에 '바로잡기' 유무 |
| **C 태도** | 대조 방향: PCR +1 응답 vs −1 응답의 응답 시작 활성값 평균 차 → 위치 D 투영. AO: "accept / correct / sidestep?" | 생성 전 태도 점수, AUROC vs 실제 PCR |
| NLA | Gemma-3-27B(L41), Llama-70B(L53), Qwen2.5-7B(L20). 원 템플릿 + 수정 템플릿. AR `mse_nrm` < 0.5 게이트 | 전제 언급률, NFP confabulation률 |

### 통제 (verbalizer 결과에 필수)

1. NFP에서 없는 전제를 지어내는 비율
2. 활성값 없이 텍스트만으로 같은 질문 → 얼마나 맞히나
3. 무작위/섞은 활성값 → 같은 답인가
4. 그 사실을 모르는 모델(작은 모델)의 활성값 → 그래도 "거짓"인가

### 산출물

- 층×위치 AUROC 히트맵 (모델별)
- **출력 readout vs 내부 readout**, 같은 held-out 문항, 모델별 — readout 수준 결과이지 지식 유무의 증명이 아님
- **2×2 분포표** (A 점수 × C 점수, 측정 조합으로 기술), 7 카테고리별
- P의 자격 분포 (사실/믿음/주장), Cancer-Myth vs NFP
- verbalizer 통제 4종 표
- Verbalizing-Assumptions의 validation 방향과 우리 교정-vs-동조 방향의 cos
- **cos(A 방향, C 방향), 층별** — Pandey([07 A3](07_related_work_2026.md))에서 진위·동조 방향 cos 0.4–0.8. 그만큼 겹치면 "지식은 안 건드린다"를 "부분 분해"로 낮춰 쓴다
- 기준선: 마지막 토큰 probe AUROC **0.70** (Two Axes의 CREPE). 이 아래면 CREPE보다 못 읽는 것

### 갈림길

| 결과 | 해석 | 다음 |
|---|---|---|
| A에서 분리 | 전제를 읽는 순간 판정이 나 있다 | Contextual-Truth의 빈칸 직접 채움 → 개입 |
| B에서만 분리 | 질문 전체를 본 뒤 판정 | 개입 위치도 B |
| 다른 층에서만 분리 | NLA 고정층은 못 씀, probe로만 | AO 층 지정으로 대응 |
| 내부 readout ≈ 출력 readout | 시험한 readout의 추가 이득을 검출 못 함. 지식 부재의 증명은 아님 | 층·위치·probe 종류·표본을 재검토. 그래도 같으면 "라우팅 가치 없음"을 결과로 보고 |
| **어디서도 안 갈림** | 표상에 신호 없음 | **접는다.** 음성 결과도 Contextual-Truth의 빈칸 |
| 2×2에서 "A 낮음" 칸 지배 | H2(화용론적 누락) 쪽 시사 | shortcut 통제 후 확정. 그 자체로 기술 결과 |
| 2×2에서 "A 높음 × C 따라감" 칸 지배 | H3 쪽 시사 | 게이트 개입의 대상 집합 |

### 사전 진단 (반나절)

20문항 골라 프롬프트 **전 토큰**에 probe 점수 히트맵. 신호가 전제 구간·질문 끝·엉뚱한 곳 중 어디서 켜지는지 눈으로 확인 후 A~E 확정.

## 실험 2 — 개입

> A probe 점수가 τ를 넘으면 C 방향을 α만큼 민다. 그 외 무개입. (A∧C 게이트는 확장 ablation.)

**검증할 질문.** 같은 개입 강도 α와 비슷한 개입 비율에서, A 게이트가 무작위 게이트·무조건 개입보다 나은 교정/부작용 균형을 만드는가. 무조건 개입이 NFP를 깎는다는 것은 예측이지 성공 조건이 아니다. 무조건이 NFP를 유지하거나 무작위 게이트가 PCR을 올리면 그대로 보고한다.

- **첫 실행 (한 모델).** Plain / 무조건 C / 무작위 게이트 + C (개입 비율을 A 게이트에 맞춤, seed 5개) / A 게이트 + C / 같은 A 게이트 + FP Identification 프롬프트 / 같은 데이터로 학습한 CAST식 조건·행동 벡터. 핵심 효과가 여기서 안 보이면 확장하지 않는다.
- **확장 (결과에 따라).** A∧C 게이트, 방향 후보 (ii) Tripathi식 head별 방향 + 감쇠 ([07 A2](07_related_work_2026.md)), (iii) Pandey식 공유 head 증폭 ([07 A3](07_related_work_2026.md)), 다중 모델, AO/NLA.
- 위치: D (마지막 프롬프트 토큰) + 응답 생성 전 구간
- **oracle 게이트** (의사 라벨로 라우팅): 게이트가 완벽할 때 개입 자체가 얼마나 고치는지 보는 **진단 조건**. 교정이 실패할 수 있으므로 성능의 엄밀한 상한은 아니다.
- α·τ는 validation에서 고른다. 재시작 시 α의 절대 크기가 바뀌지 않도록 scale은 고정 calibration 부분집합에서 한 번 재서 저장 (리뷰 12).
- 게이트 추가 통제: 전제가 참인데 Plain이 틀리게 답한 문항에서 게이트 발화율 (Tripathi의 "게이트가 정당한 교정도 잡는다" 관찰)

## 실험 3 — held-out 비교표

### 분할 (고정, 실험 1·2·3 공통)

Well-Actually의 Cancer-Myth 분할을 그대로 쓴다: FPQ **383 / 100 / 100**, TPQ **30 / 18 / 100** (train / dev / test), 별도로 FPQ·TPQ 각 2개가 few-shot 예약. TPQ 148 + 예약 2 = NFP 150이므로 NFP 전체가 이 분할 안에 있다.

| 데이터 | 용도 |
|---|---|
| train (FPQ 383, TPQ 30) | A probe·C 방향·CAST 벡터·GEPA·출력 readout 학습 |
| dev (FPQ 100, TPQ 18) | 층·α·τ·probe 종류 선택 |
| **test (FPQ 100, TPQ 100)** | **잠금.** 표에 들어가는 숫자만. 한 번 |

표본이 부족해 CV를 쓰면 **train+dev 안에서 grouped CV**로 하고 test는 건드리지 않는다. NFP와 TPQ는 같은 질문 텍스트이므로 항상 같은 그룹. 중복·의역 검사도 이 분할 기준. TPQ train이 30개뿐이라 A probe의 음성이 적다 — 부족하면 CREPE 참 전제로 보강하되 그 사실을 표에 적는다.

### 통계

- **허용폭은 먼저 정한다.** "Plain 대비 NFP 손실 얼마까지 받아들이는가"를 사전 등록하고, 표본이 그 주장을 뒷받침하는지 검토한다. 이것과 "0건 관찰 시 오류율 상한"은 다른 개념이다.
- 참고: 독립 n건에서 오류 0건일 때 단측 95% 상한은 n=100에서 약 2.95%, n=50에서 5.8%, n=18에서 **15.3%**. dev의 TPQ 18개로는 "FPR 5%를 인증"할 수 없다. dev에서는 목표 FPR에 맞춰 **τ를 고르는 것**까지이고, 위험의 통계적 보장은 test 100개에서 신뢰구간으로 따로 보고한다.
- NFP·TPQ는 문항별 paired harm/rescue (Plain에서 맞았는데 개입 후 틀림 / 그 반대)와 CI. PCR/PCS는 bootstrap CI. 무작위 게이트는 seed별 분산.

### 열과 판정기

**PCR, PCS** (`validate.py`, GPT-4o) + **NFP** (`validate_nfp.py`) + **TPQ** (참 전제를 부당하게 부정·교정했는가를 재는 별도 루브릭 — NFP 루브릭에 참 전제를 넣으면 뒤집힌다, 리뷰 5; Well-Actually의 S5와 같은 이름으로 부르지 않는다). 판정기는 표에 들어가는 숫자 전부 GPT-4o API. 외부 벤치마크 5개는 제외 (필요 시 후속).

### 원본 표와의 관계

| 결과 | 제시 방식 |
|---|---|
| Well test 100/100을 GPT-4o 판정기로 | Cancer-Myth 지표를 쓴 **새 held-out 비교표** (본 표) |
| 같은 test를 Well의 gemini 0–5 루브릭으로 | Well Table 8–10과 비교 (부록). 생성 설정·few-shot·RAG 조건을 맞췄는지 명시 |
| Cancer-Myth 원본 Table 1·3 수치 | 평가 범위(585 전체, 닫힌 모델)가 다르므로 **참고 결과로 분리**. 같은 행에 놓지 않는다 |

`GEPA(FPQ+TPQ)`는 Well의 설정이다. Cancer-Myth 원본 GEPA는 7개 벤치마크 각각 train 5·validation 5 설정이라 "원본 재현"이라 부르지 않는다.

### 행

| 묶음 | 방법 | 출처 | 실행 |
|---|---|---|---|
| Cancer-Myth 것 | Plain / GEPA / Monitor | Table 1 | 닫힌 모델 숫자는 참고 결과로 분리. 공개 모델 GEPA는 Well 설정(FPQ+TPQ)으로 명시, Monitor는 코드 없어 보류 |
| Well-Actually 것 | FP Identification / Extract+FactCheck (LLM·MiniCheck) / Self-Dual-Critique / PreWoMe / Question-to-Statement / FAITH / LoRA | `Well` | 코드로 실행, **채점만 Cancer-Myth 판정기로** (Well-Actually의 gemini 0–5 judge와 섞지 않음) |
| 해석가능성 것 | 같은 C를 게이트 없이 (무조건) / 무작위 게이트 (seed 5개) / CAST식 조건·행동 벡터 (같은 train으로 학습) | 우리가 실행 | 라우팅 가치와 조건부 steering 대비를 가르는 대조군 |
| **probe 게이트 × FP Identification 프롬프트** | Two Axes식 라우팅 ([07 A1](07_related_work_2026.md)) | Well-Actually의 프롬프트 + 우리 A probe | **가장 강한 대조군.** 우리 C 개입은 GEPA가 아니라 이것을 이겨야 함 — 특히 PCS |
| **우리** | A 게이트 × C 방향 (확장: A∧C 게이트) | — | |

**예상 (검증 대상이지 성공 조건이 아님)**, 한 모델 기준. Plain PCR은 test 100에서 다시 잰다 (Table 3의 17.3은 585 전체).

| Method | PCR | PCS | NFP / TPQ | 예상 |
|---|---|---|---|---|
| Plain | 기준 | | 기준 | |
| GEPA (Well 설정) | | | | PCR↑, NFP↓ 예상 |
| FP Identification | | | | PCR↑↑, NFP↓↓ 예상 |
| Extract + FactCheck | | | | |
| 무조건 C | | | | NFP↓ 예상. 유지되면 그대로 보고 |
| 무작위 게이트 + C | | | | PCR 일부 ↑ 가능. seed 분산 |
| CAST식 조건·행동 벡터 | | | | |
| A 게이트 + FP 프롬프트 | | | 교정 품질 (CREPE 47–61%) | |
| **A 게이트 + C** | | | | |

**성공 조건.** 사전에 정한 NFP/TPQ 허용폭 안에서 Plain 대비 PCR이 CI로 유의하게 오르는 것. 다른 행도 허용폭을 만족하면 그 행들과 같은 허용폭에서 PCR·PCS·비용을 비교한다 — "다른 행은 전부 실패해야" 한다는 조건은 없다. 크기는 6→20, 17→30이면 충분하다.

## 리스크

1. **표상이 배경화된 전제에서 안 읽힐 수 있다** — 실험 1이 답. 하루.
2. **NFP는 어려운 통제군으로 설계됐다** — 한 LLM이 헛짚은 것만 골라 만듦. 자연 분포(Well-Actually: WildChat 100문항 기준 FPQ 약 13%)에서 평가된 방법들은 TPQ를 너무 깎아 Direct QA에 졌다. 가중식은 ΔU = 0.13·ΔFPQ + 0.87·ΔTPQ이므로 TPQ 손실이 작으면 이길 수 있다. NFP 보존은 이 가중식의 귀결이 아니라 **독립적인 연구 제약**으로 정당화하고, 13%는 여러 비율에서 민감도 분석.
3. **verbalizer 자기 지식 혼입** — 통제 4종 없이는 B·C 결과를 못 믿는다.
4. **판정기** — 논문은 GPT-4o(사람과 PCR 100% 일치). 우리는 `codex exec`(또는 OpenAI API)로 같은 프롬프트를 쓰되, `all_data.json`에 있는 GPT-4o 채점 8모델×585과 먼저 보정한다(`scripts/calibrate_judge.py`). 판정기가 다르면 Table 1의 닫힌 모델 행은 GPT-4o 채점으로 표시하고, 우리 블록은 한 판정기로 전부 채점한다.
5. **Verbalizing-Assumptions의 반례** — objectivity 축 steering이 sharpness를 떨어뜨렸다. 축을 잘못 잡으면 해가 된다. cos 검사가 먼저.

## 순서

1. 사전 진단 히트맵 (반나절)
2. 실험 1, Llama-3.1-8B부터 (싸다) → Gemma-2-27B → Gemma-3-27B(NLA+AO) → 70B
3. Plain PCR/PCS/NFP, 4 모델
4. 갈림길 판단
5. 실험 2
6. 실험 3 baseline
7. Table 1 블록 + Table 3 새 행
