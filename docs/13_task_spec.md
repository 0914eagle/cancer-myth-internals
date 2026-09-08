# 13. Task 명세와 표 뼈대

2026-09-08. [09](09_research_proposal.md)의 Task 1–3을 실행 수준으로 내린 것. 09가 주장·가설·기여를 정하고, 이 문서는 **무엇을 돌려 어떤 표를 채우는가**만 다룬다. 칸은 비어 있다. 결과가 아니다.

용어: 09의 gate를 여기서는 **개입 조건**이라 부른다. "이 질문에 손을 댈까 말까"를 정하는 점수와 문턱이다. 관련 연구 절에서는 원문 용어 gate를 병기한다.

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-7B-Instruct(세 논문 표에 모두 있음), Llama-3.1-8B-Instruct(Cancer-Myth·Two Axes; Well은 Llama-3-8B라 "다른 버전"으로 표기) 우선. Gemma-2-27B는 자원에 따라 |
| 데이터 | Cancer-Myth FPQ 585 (`all_data.json`), NFP/TPQ 150 (`nfp.json`, Well의 참 전제 주석) |
| 라벨 (Task 1·2 공통) | **질문에 거짓 전제가 있는가.** FPQ = 1, TPQ/NFP = 0. 의사 검증. 모델 행동과 무관 |
| 판정기 | GPT-4o API. `validate.py`(PCR/PCS), `validate_nfp.py`(NFP) 원문 그대로. **TPQ 오교정은 Well-Actually의 TPQ 루브릭(App. E: 1 = 참 전제를 잘못 고치려 함 … 5 = 교정 시도 없음)을 GPT-4o로 돌려 재사용** — 리뷰 1의 5번 해결. 부록에서는 Well 판정기(gemini-3-flash)로도 |
| 평가 단위 | **585 전체 + 150 전체.** 아래 cross-fitting으로 전 문항에 out-of-fold 결과를 낸다 |
| 허용폭 | ε_normal (NFP/TPQ), ε_QA (의료 QA)를 실행 전에 숫자로 적는다 |

### 왜 585 전체인가, 그리고 어떻게

probe·C 방향·CAST 벡터·GEPA 프롬프트를 Cancer-Myth 문항으로 학습하므로, 학습한 문항에서 평가하면 누수다(리뷰 1의 2번). 두 방식 중 **nested grouped 5-fold cross-fitting**을 택한다.

| 방식 | 장점 | 단점 |
|---|---|---|
| 잠근 test 100/100 (Well 분할) | 규율 명확. 저자 ID를 얻으면 Well 표와 같은 문항 | 분모 100, CI 넓음, TPQ train 30개. 저자 ID 미확보 ([12 §5](12_discussion_decisions.md)) |
| **5-fold cross-fitting, 735 전체** | 585/150 전체에 숫자. Cancer-Myth Table 1·3과 분모가 같음. 학습 음성 120개 | 층·α·문턱 선택도 fold 안에서(nested). 학습형 baseline을 5번 학습 |

규칙. (1) NFP와 TPQ는 같은 질문이므로 같은 그룹. 같은 myth에서 나온 문항도 같은 그룹. (2) 방향·probe·분류기·GEPA·CAST는 학습 fold에서만. (3) 층·α·문턱은 학습 fold 안의 내부 dev에서. (4) 평가 fold의 문항은 그 fold의 어떤 선택에도 안 쓴다. (5) Plain은 학습이 없으니 585 전체를 그대로 재고, 원 논문 Table 3 값(Qwen2.5-7B 6.3, Llama-3.1-8B 4.8)과 나란히 적어 판정기 일치를 보인다. (6) Well 분할 ID를 얻으면 부록에서 Well 판정기로 한 번 더.

### 새로 만드는 것과 가져오는 것

| 만드는 것 | 가져오는 것 (코드) |
|---|---|
| 개입 조건: hidden probe (A), 텍스트 분류기, CoT monitor | Well: Direct QA, FP Identification, Extract+FactCheck, PreWoMe, Q2S, Self-Dual-Critique, GEPA, FalseQA LoRA, FAITH |
| 개입: C 방향 (교정/비교정 응답 대조) | CAA (nrimsky/CAA), CAST (IBM/activation-steering), ITI (likenneth/honest_llama), Tripathi (HF artifacts + 추론 코드) |
| ablation: 무작위 조건 | Cancer-Myth: `evaluate.py`, 판정기 두 개 |
| TPQ 오교정 루브릭 | Two Axes §6 라우팅 (단순, 재구현) |

판정 비용 어림: 735문항 × 본 표 행 8 × 모델 2 ≈ 12,000회 + dev. GPT-4o API로 수만 원 단위.

## 1. Task 1 — 개입 조건 선발 → Table 2

**질문.** 같은 held-out 문항에서, 내부 표상을 읽는 점수가 질문 텍스트나 CoT를 읽는 점수보다 "거짓 전제가 있는 질문"을 더 잘 가르는가. 합치면 더 나은가. (RH1)

**왜 하나.** 논문의 논리는 "고를 수 있으면 고른 것에만 개입한다"이다. 무엇으로 고를지가 먼저 정해져야 Task 2가 성립한다. Task 1의 1등이 Task 2의 개입 조건이 된다.

**readout.** 전부 같은 out-of-fold 문항에.

| readout | 점수 | 비용 |
|---|---|---|
| 텍스트 분류기 (BoW 로지스틱) | 질문 단어 빈도 | 0 |
| 모델에게 직접 질문 | "거짓 전제 위에 있나"의 P(yes) log-odds | 생성 1회 |
| extract-and-verify (Well 프롬프트) | 추출한 전제에 대한 P(false) 최대값 | 생성 2회 |
| CoT monitor | 명시적 전제 검사 CoT를 생성시키고 별도 판정기가 그 텍스트를 읽어 낸 점수 | 생성 1회 + 판독 |
| **hidden probe** | 마지막 프롬프트 토큰 residual에 로지스틱. 층은 내부 dev에서 | forward 1회 |
| 텍스트 + hidden | 두 점수 로지스틱 | forward 1회 |

비용은 표 열이 아니라 본문 한 문장으로. CoT monitor가 probe만큼 잘 고르면 "왜 내부를 읽나"의 답이 비용과 지연이다.

**Table 2 뼈대** (모델당 하나. 라벨 = 거짓 전제 유무. 585 vs 150, out-of-fold)

| readout | AUROC [CI] | TPR @ FPR 5% |
|---|---|---|
| 텍스트 분류기 | | |
| 직접 질문 | | |
| extract-and-verify | | |
| CoT monitor | | |
| **hidden probe** | | |
| 텍스트 + hidden | | |

FPR 5%의 문턱은 내부 dev에서 정하고, 평가 fold에서 실제 FPR을 각주로.

**같이 내는 것 (부록).** 층×위치 AUROC 히트맵. shortcut 통제: 암종별 holdout, 텍스트 분류기 대비 이득. "Plain이 못 고칠 문항인가"(PCR −1 vs +1) 라벨의 AUROC는 별도 소표 — Task 2의 A∧C 조건용.

**해석.** hidden > 텍스트·CoT이고 결합이 더 높으면 RH1 지지. hidden ≈ 텍스트면 "이 readout으로는 추가 정보 검출 실패"이고, Task 2의 조건은 텍스트 분류기로도 되므로 기여가 "내부 신호"에서 "선택적 개입 자체"로 옮겨간다. 지식 유무의 증명은 아니다.

## 2. Task 2 — 선택적 교정 → Table 3, Figure 1

**질문.** 개입할 질문을 고르면 시소가 개선되는가. 어느 부품(조건 / 개입)이 기여하는가. (RH2, RH2b)

**요인 격자.** 조건은 원래 질문에서 한 번 계산한 문항별 mask이고, 개입을 바꿔도 같은 mask를 쓴다.

| 개입 조건 | 정의 | 출처 |
|---|---|---|
| 항상 | 전 문항 개입 | 무조건 steering = CAA |
| 무작위 | hidden 조건과 같은 개입 비율, seed 5개 | ablation |
| 텍스트 분류기 | Task 1 행 | |
| CoT monitor | Task 1 행 | |
| **hidden probe (A)** | Task 1 1등, 내부 dev 문턱 | |
| A ∧ C | A 열림 그리고 C 점수가 "못 고침" 쪽 | 확장 |
| oracle | 의사 라벨 FPQ 전부. **진단 조건**, 상한 아님 | |

| 개입 | 정의 | 출처 |
|---|---|---|
| FP Identification 프롬프트 | Well 원문 프롬프트 | Well |
| **C residual steering** | 교정(PCR +1)/비교정(−1) Plain 응답의 응답 시작 활성값 평균 차. **후보(층 × 위치) 선택은 AUROC가 아니라 Lavi et al.식 steering score(내부 dev에서 교정 행동을 얼마나 유발하는가)로** — 탐지≠제어 경고 대응. α도 내부 dev | 우리 (선택 절차는 Lavi et al. 2026) |
| head steering | Tripathi식 head별 방향 | 확장 |

**Table 3 뼈대** (한 모델. 셀 = PCR / NFP / TPQ 오교정률). 행 구조는 Two Axes Table 3(Never / Always / Random / Probe-gated)을 따르고, 파일럿 규율도 따른다: 가설과 kill-criteria를 스크립트에 먼저 고정, 무작위 조건은 같은 예산, 지적 여부는 판정기(Two Axes는 strict 템플릿 detector, 48토큰 greedy였음 — 우리는 전체 답변에 GPT-4o).

| 조건 \ 개입 | 없음 | FP 프롬프트 | C steering |
|---|---|---|---|
| 항상 | Plain | = FP Identification | = CAA |
| 무작위 (5 seed 평균±sd) | | | |
| 텍스트 분류기 | | | |
| CoT monitor | | | |
| **hidden probe (A)** | | = Two Axes 라우팅 | **= 우리** |
| oracle (진단) | | | |

세로로 읽으면 같은 개입에서 조건의 가치(RH2). 가로로 읽으면 같은 조건에서 개입의 가치(RH2b). 격자의 칸이 곧 선행 방법이다: 항상+프롬프트가 Well의 FP Identification, 항상+C가 CAA, hidden+프롬프트가 Two Axes 라우팅.

**Table 3 보조** (셀 = 발화율 FPQ / TPQ, harm / rescue)

| 조건 | 발화율 (FPQ / TPQ) | FPQ harm / rescue | TPQ harm / rescue |
|---|---|---|---|
| 무작위 | | | |
| hidden probe | | | |
| oracle | | | |

harm = Plain에서 맞았는데 개입 후 틀림. rescue = 그 반대. 순손실만 보고하지 않는다.

**Figure 1.** x = TPQ 보존(오교정 안 한 비율), y = FPQ PCR. Table 3의 모든 셀과 Table 1의 모든 행이 점 하나씩. Well-Actually Figure 1과 같은 모양. ε_normal을 수직선으로.

**절차.** fold마다 (1) 학습 fold에서 C 방향, A probe, 텍스트 분류기, CAST 벡터, GEPA 학습. (2) 내부 dev에서 층·α·문턱을 ε_normal 제약 아래 PCR 최대로. 무작위의 비율은 A의 dev 발화율. (3) 평가 fold에 적용. (4) 5 fold 합쳐 585/150 전체 집계, bootstrap CI.

**해석.** 같은 ε 안에서 hidden+C가 항상+C, 무작위+C, 텍스트+C보다 PCR이 CI로 높으면 RH2 지지. hidden+C가 hidden+프롬프트보다 PCS 높으면 RH2b 지지. 무작위나 텍스트가 같으면 그대로 보고하고 기여를 옮긴다. 무조건이 NFP를 유지해도 그대로 보고한다.

## 3. Task 3 — 일반 의료 QA 보존 → Table 1 완성

**질문.** Task 2에서 고정한 시스템(조건 + 문턱 + 개입 + α)을 다른 의료 질문에 그대로 두면 성능이 유지되는가. (RH3)

**벤치마크.** MedQA (USMLE 4지선다), PubMedQA (yes/no/maybe), Medbullets (선다). Cancer-Myth Table 1의 열 중 셋. 정확도만 세므로 판정기 불필요. SymCat·Craft-MD는 코드가 없어 제외한다고 적는다.

**절차.** 시스템을 그대로 적용한다. 평가셋 이름을 보고 조건을 끄지 않는다. 각 벤치마크의 원래 채점 규칙. Plain 대비 문항별 paired 차이와 CI. 조건 발화율 (선다형에서 얼마나 켜지는가 자체가 결과). ε_QA는 사전 등록. 밖이면 RH3 기각이고, 재조정하려면 QA dev를 따로 두고 밝힌다.

**Table 1 뼈대** (본 표. 모델당 블록. 행 8 = 선행 6 + ablation 1 + 우리 1)

| Method | 출처 | PCR ↑ | PCS | NFP ↑ | TPQ 오교정 ↓ | MedQA | PubMedQA | Medbullets | 발화율 |
|---|---|---|---|---|---|---|---|---|---|
| Plain | Cancer-Myth | | | | | | | | 0 |
| FP Identification | Well-Actually | | | | | | | | 100 |
| GEPA (FPQ+TPQ) | Cancer-Myth / Well 구현. **Well은 Gemini·Gemma-4에만 돌렸으므로 Qwen·Llama용은 우리가 실행** (예산 500, 검증 50) | | | | | | | | 100 |
| CAA (무조건 C) | Rimsky 2023 | | | | | | | | 100 |
| CAST (같은 train) | Lee, ICLR 2025 | | | | | | | | |
| Two Axes 라우팅 (hidden + FP 프롬프트) | Wagner 2026 | | | | | | | | |
| 무작위 조건 + C | ablation | | | | | | | | |
| **hidden 조건 + C** | 우리 | | | | | | | | |
| *참고: GPT-4o Plain (원 논문)* | Cancer-Myth Table 1 | 12 | | 88 | | 70 | 67 | 68 | |
| *참고: GPT-4o GEPA (원 논문)* | Cancer-Myth Table 1 | 68 | | 59 | | 63 | 59 | 62 | |
| *참고: Qwen2.5-7B Plain (원 논문)* | Cancer-Myth Table 3 | 6.3 | −0.50 | | | | | | |

참고 행은 선으로 분리하고 비교하지 않는다(닫힌 모델, 또는 판정 시점·생성 설정 차이). 각 셀에 CI. NFP·TPQ·QA 열은 ε 안이면 표시. 성공 조건은 사전에 적은 ε_normal·ε_QA 안에서 Plain 대비 PCR이 CI로 오른 행. 여럿이면 같은 ε 안에서 PCR·PCS·비용 비교.

**부록 행.** Extract+FactCheck, Self-Dual-Critique, PreWoMe, FalseQA LoRA (Well). ITI (head 무조건). Tripathi (artifacts 전이 / 재학습). A∧C + C. head steering.

**가져올 수 없는 것.** Cancer-Myth의 Monitor (MDAgents 코드는 있으나 감시 에이전트 프롬프트 없음). 참고 행에만.

## 4. 부록 표

- A×C 2×2 분포, 7 카테고리별. 점수 조합 이름, H2/H3 라벨 아님.
- 층별 cos(A, C).
- AO/NLA 통제 4종 (텍스트만 / 활성값 없음 / 섞은 활성값 / NFP).
- Well 판정기로 다시 채점한 Table 3 (split ID 확보 시).
- "Plain이 못 고칠 문항" 라벨의 readout AUROC.

## 5. 실행 순서

1. 분할 manifest 고정 (myth·NFP-TPQ 그룹), ε 두 개 숫자로, 코드 리뷰 지적 1·2·5 해결.
2. 한 모델(Qwen2.5-7B)에서 Task 1 전부 → Table 2.
3. 같은 모델에서 Table 3의 3×3 (항상·무작위·hidden × 없음·프롬프트·C) → 핵심 효과 확인.
4. 효과가 있으면 Table 1의 8행 + Task 3. 없으면 그 결과로 보고하고 확장하지 않는다.
5. 두 번째 모델, 부록.
