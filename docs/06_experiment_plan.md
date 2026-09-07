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
| **A 판정** | linear probe, 층×위치 sweep, 5-fold | 거짓 vs 참 전제 **AUROC**. 기준선 = 같은 모델의 언어 판정 AUROC (Well-Actually의 extract-and-verify를 같은 문항에 돌려 적중/오탐으로 계산; Table 1 환산값 0.7–0.8) |
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
- **언어 판정 AUROC vs probe AUROC**, 모델별 한 줄 — "판정 대신 표상"의 직접 증거
- **2×2 분포표** (A × C), 7 카테고리별
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
| probe AUROC ≈ 언어 판정 AUROC | 표상에도 그만큼밖에 없다. H1 부활 | 내부를 읽어도 얻을 게 없음. 지식 주입 쪽으로 방향 전환 |
| **어디서도 안 갈림** | 표상에 신호 없음 | **접는다.** 음성 결과도 Contextual-Truth의 빈칸 |
| 2×2에서 H2 지배 | sycophancy가 아니라 화용론적 누락 | 그 결론 자체가 논문 |
| 2×2에서 H3 지배 | 알지만 따라간다 | 게이트 개입으로 |

### 사전 진단 (반나절)

20문항 골라 프롬프트 **전 토큰**에 probe 점수 히트맵. 신호가 전제 구간·질문 끝·엉뚱한 곳 중 어디서 켜지는지 눈으로 확인 후 A~E 확정.

## 실험 2 — 개입

> A가 "거짓" ∧ C가 "따라간다" → C 방향을 α만큼 민다. 그 외 무개입.

- 방향 후보 셋: (i) residual 대조 방향 addition (또는 Gemma Scope feature), (ii) Tripathi식 head별 방향 + 감쇠 ([07 A2](07_related_work_2026.md)), (iii) Pandey식 공유 head 증폭 ([07 A3](07_related_work_2026.md)). 셋 다 NFP로 채점
- 위치: D (마지막 프롬프트 토큰) + 응답 생성 전 구간
- 비교: 같은 방향 **무조건** steering (CAA baseline) — NFP가 무너져야 함
- 게이트 추가 통제: 전제가 참인데 Plain이 틀리게 답한 문항에서 게이트 발화율 (Tripathi의 "게이트가 정당한 교정도 잡는다" 관찰)

## 실험 3 — baseline 비교, Cancer-Myth 지표만

열: **PCR, PCS** (`validate.py`, GPT-4o) + **NFP** (`validate_nfp.py`). 외부 벤치마크 5개는 제외 (필요 시 후속).

| 묶음 | 방법 | 출처 | 실행 |
|---|---|---|---|
| Cancer-Myth 것 | Plain / GEPA / Monitor | Table 1 | 닫힌 모델 숫자 인용. 공개 모델은 GEPA 재현(dspy), Monitor는 코드 없어 보류 |
| Well-Actually 것 | FP Identification / Extract+FactCheck (LLM·MiniCheck) / Self-Dual-Critique / PreWoMe / Question-to-Statement / FAITH / LoRA | `Well` | 코드로 실행, **채점만 Cancer-Myth 판정기로** (Well-Actually의 gemini 0–5 judge와 섞지 않음) |
| 해석가능성 것 | CAA 동조 벡터 / persona vector, 게이트 없음 | 기존 | 우리의 무조건부 대조군 |
| **probe 게이트 × FP Identification 프롬프트** | Two Axes식 라우팅 ([07 A1](07_related_work_2026.md)) | Well-Actually의 프롬프트 + 우리 A probe | **가장 강한 대조군.** 우리 C 개입은 GEPA가 아니라 이것을 이겨야 함 — 특히 PCS |
| **우리** | probe 게이트 × C 방향 | — | |

기대 표 (Gemma-2-27B):

| Method | PCR ↑ | PCS | NFP ↑ |
|---|---|---|---|
| Plain | 17.3 | −0.23 | ? |
| GEPA | ↑↑ | | ↓↓ |
| FP Identification | ↑↑ | | ↓↓↓ |
| Extract + FactCheck | ↑ | | ↓↓ |
| FAITH | ≈ | | ≈ |
| CAA 무조건 | ↑ | | ↓ |
| probe 게이트 × FP 프롬프트 | ↑ | 교정 품질 한계 (CREPE 47–61%) | ≈ Plain? |
| **Ours** | ↑ | | **≈ Plain** |

이기는 조건은 마지막 열. **PCR을 올리면서 NFP를 Plain 수준으로 유지하는 행이 지금 하나도 없다.** 6.3→20대, 17.3→30대면 충분하다. 80%는 필요 없다.

## 리스크

1. **표상이 배경화된 전제에서 안 읽힐 수 있다** — 실험 1이 답. 하루.
2. **NFP는 설계상 최악 사례** — LLM이 헛짚은 것만 골라 만듦. 자연 분포(Well-Actually: FPQ 약 13%)에서는 Direct QA가 1등. 어떤 방법이든 **NFP를 1점도 깎으면 진다.**
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
