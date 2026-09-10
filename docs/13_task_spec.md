# 13. Task 명세와 표 뼈대

2026-09-08 작성, 2026-09-10 갱신. [09](09_research_proposal.md)의 Task 1–3을 실행 수준으로 내린 것. 09가 주장·가설·기여를 정하고, 이 문서는 **무엇을 돌려 어떤 표를 채우는가**를 다룬다. 초록·서론부터 결론까지의 원고 구성은 [17](17_manuscript_storyline.md)에서 연결한다. 빈칸은 미측정 결과이며, 가설이 성립한다고 미리 가정하지 않는다.

**Task와 Table의 역할 (2026-09-10 갱신).** 일대일 대응이 아니다. Task 1 → Table 1 (판별 신호 비교), Task 2 → Table 2 (완성된 방법의 최종 효용) + 부록 Table A1 (선택 통제), Task 3 → Table 2의 의료 QA 3열(MedQA, PubMedQA, Medbullets). Table 1은 **무엇으로 고를 수 있는가**, Table 2는 **최종 방법이 유용한가**, Table 3는 **전제를 직접 주어도 참·거짓 판정을 잘 하는가**에 답한다. Figure 1은 완성된 시스템의 NFP–PCR 관계다. 실행 순서와 원고 설명 순서는 별개이며, 본 결과인 Table 2의 질문을 초록·서론에서 먼저 제시한다. 현재 표 명세는 이 문서를 기준으로 하고, [12](12_discussion_decisions.md)의 이전 표는 논의 이력으로 읽는다.

**현재 위치와 확정 범위.** 가져올 선행표는 Well-Actually Table 1로 확정한다. 이 표는 교수님 발표의 기존 연구 근거로 사용할 수 있다. 아래의 우리 Table 3 및 추가 probe 행은 사실 확인 진단을 독립 연구 질문으로 채택할 경우의 후보이며 본 실험으로 확정한 것은 아니다. 기존 결과를 인용하는 것만으로 우리 실험 결과표가 되지 않는다. 현재 질문 gate의 검증이 목적이면 Table 1의 질문 단위 탐지 비교에 RAG 기반 전제 추출·검증을 추가하는 방안도 가능하며, gold 전제를 받는 원 Table 1의 숫자와 직접 섞지 않는다.

**CoT의 두 역할.** (a) Table 2의 균형 전제 검토 CoT는 우리가 구성하는 직접 prompting 대조군이다. 전제를 검토하고 틀린 경우 교정하되 정상 전제를 부정하지 말도록 지시한다. Well의 원 FP Identification과 동일한 절차라고 부르지 않는다. (b) Table 1의 CoT monitor는 전제 검토 글을 생성한 뒤 별도 판정기가 원 질문과 그 글을 읽어 선택 점수만 낸다. 검토 글은 최종 답변 입력에 추가하지 않는다. 구체적 monitor 모델·템플릿·점수 규칙은 test 전에 잠근다.

용어: 09의 gate를 여기서는 **개입 조건**이라 부른다. "이 질문에 손을 댈까 말까"를 정하는 점수와 문턱이다. 관련 연구 절에서는 원문 용어 gate를 병기한다.

**Method 구체화:** [19](19_method_protocol.md)가 주 학습·추론 명세다. 주 C는 fit 질문의 +1/−1 참조 답변 첫 32토큰 평균 차다. A probe는 마지막 프롬프트 토큰의 표준화+L2 logistic이며, scaler·probe·C·scale은 fit에서만 만든다. 층·문턱·강도는 내부 dev에서 고정한다. 아래의 다른 방향 후보는 부록 ablation으로 읽는다.

## 0. 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | Qwen2.5-7B-Instruct(세 논문 표에 모두 있음), Llama-3.1-8B-Instruct(Cancer-Myth·Two Axes; Well은 Llama-3-8B라 "다른 버전"으로 표기) 우선. Gemma-2-27B는 자원에 따라 |
| 데이터 | Cancer-Myth FPQ 585 (`all_data.json`), NFP/TPQ 150 (`nfp.json`, Well의 참 전제 주석) |
| 라벨 (Task 1·2 공통) | **질문에 거짓 전제가 있는가.** FPQ = 1, TPQ/NFP = 0. 의사 검증. 모델 행동과 무관 |
| 판정기 | 본 표는 GPT-4o의 Cancer-Myth PCR/PCS·공식 NFP 루브릭. 원 템플릿과 판정기 버전을 기록하고 파싱 오류는 바로잡는다. Table 3는 Well 사실 확인 정확도, 최종 응답 S5·분포는 부록; 원 조건 변경 시 adaptation 표시 |
| 평가 단위 | **585 전체 + 150 전체.** 아래 cross-fitting으로 전 문항에 out-of-fold 결과를 낸다 |
| 허용폭 | ε_NFP, ε_QA를 지표 단위와 함께 실행 전에 숫자로 적는다. QA는 벤치마크별 허용폭·통과 규칙도 지정한다 |

### 왜 585 전체인가, 그리고 어떻게

probe·C 방향·CAST 벡터·GEPA 프롬프트를 Cancer-Myth 문항으로 학습하므로, 학습한 문항에서 평가하면 누수다(리뷰 1의 2번). 두 방식 중 **nested grouped 5-fold cross-fitting**을 택한다.

| 방식 | 장점 | 단점 |
|---|---|---|
| 잠근 test 100/100 (Well 분할) | 저자들이 실제 사용한 분할의 **문항 식별자 목록**을 확보하면 Well 표와 같은 test 질문으로 평가 가능 | 분모 100, CI 넓음, 보고된 TPQ train 30개. 원 실험의 정확한 분할 목록 미확보 ([12 §5](12_discussion_decisions.md)) |
| **5-fold cross-fitting, 735 전체** | 585/150 전체에 숫자. Cancer-Myth Table 1·3과 분모가 같음. 정상 질문 outer-train은 명목상 약 120개 | 그룹 크기에 따라 fold별 수가 다름. 내부 dev를 제외한 실제 fit은 더 적음. 층·α·문턱 선택도 fold 안에서(nested). 학습형 baseline을 5번 학습 |

**분할 문항 ID의 뜻.** 저자 계정이나 ORCID가 아니라, 원 논문 실행에서 어떤 원 질문을 train/dev/test 및 few-shot에 배정했는지 연결하는 목록(split manifest)이다. 원본 ID, 질문 원문 또는 데이터 버전과 대응 가능한 식별 정보를 사용한다. 개수만 같은 100/100은 같은 시험지가 아니다. 같은 문항을 확보한 뒤에도 모델 revision·프롬프트·RAG·생성·판정 조건을 맞춰야 원 표와 직접 비교할 수 있다. NFP 150개와 Well TPQ 주석은 동일 질문 집합이므로 별도로 합산하지 않는다.

규칙. (1) NFP와 TPQ는 같은 질문이므로 같은 그룹. 같은 myth에서 나온 문항도 같은 그룹. (2) 방향·probe·분류기·GEPA·CAST는 학습 fold에서만. (3) 게이트 종류·층·α·문턱·개입 예산은 학습 fold 안의 내부 dev에서만 선택한다. (4) 평가 fold의 라벨·출력 채점 결과는 그 fold의 어떤 선택에도 안 쓴다. 부록 Table A1의 사전 지정된 배치 예산 통제만 평가 fold의 라벨 없는 점수 순위를 이용한다. (5) Plain은 학습이 없으니 585 전체를 그대로 재고, 원 논문 Table 3 값(Qwen2.5-7B 6.3, Llama-3.1-8B 4.8)을 참고로 나란히 적는다. 숫자가 비슷하다는 것만으로 생성·판정 조건이 같다고 판단하지 않는다. (6) Well 분할 ID를 얻으면 부록에서 Well 판정기로 한 번 더.

### 구현·재학습할 부품과 가져오는 절차

| 구현·재학습할 것 (새 알고리즘이라는 뜻 아님) | 가져오는 것 (코드) |
|---|---|
| 개입 조건: hidden probe (A), 텍스트 분류기, CoT monitor | Well: Direct QA, FP Identification, Extract+FactCheck, PreWoMe, Q2S, Self-Dual-Critique, GEPA, FalseQA LoRA, FAITH |
| 개입: C 방향 (교정/비교정 응답 대조) | CAA (nrimsky/CAA), CAST (IBM/activation-steering), ITI (likenneth/honest_llama), Tripathi (HF artifacts + 추론 코드) |
| ablation: 무작위 조건 | Cancer-Myth: `evaluate.py`, 판정기 두 개 |
| Table 3 gold 전제 판정 및 부록 응답 점수 경로 | Two Axes §6 라우팅 (의료 재학습·재구현) |

본 표의 생성 대상 규모: 735문항 × 본 표 행 9 × 모델 2 = 13,230응답 + dev. Table 3의 RAG 조건·부록 A1의 무작위 seed·QA는 별도다. 생성 응답 수와 판정 API 호출 수는 루브릭 수에 따라 다르므로, 비용은 실제 토큰 사용량과 실행 시점 단가로 산출한다.

## 1. Task 1 — 판별 신호 비교 → Table 1

**질문.** 같은 held-out 문항에서, 내부 표상을 읽는 점수가 질문 텍스트나 CoT를 읽는 점수보다 "거짓 전제가 있는 질문"을 더 잘 가르는가. 합치면 더 나은가. (RH1)

**왜 하나.** 논문의 논리는 "고를 수 있으면 고른 것에만 개입한다"이다. Task 1은 각 판별 신호의 성능을 비교한다. **Task 1의 1등을 Task 2의 개입 조건으로 뽑지 않는다.** Task 2에서는 텍스트·CoT·내부 게이트를 모두 유지한 채 동일한 개입을 적용해, 판별 성능 차이가 실제 교정 성능과 정상 질문 보존으로 이어지는지 본다(부록 Table A1). AUROC 1등이 개입에서 최선이라는 보장이 없고(놓친 문항이 개입으로 고칠 수 없는 문항일 수 있음), 내부 게이트만 남기면 "텍스트나 CoT로 골라도 되지 않나"에 답할 수 없기 때문이다. 게이트 종류의 선택도 층·문턱과 마찬가지로 내부 개발 분할에서만 한다 (리뷰 3).

**출처와 역할.** Two Axes Table 2의 핵심 비교를 의료 데이터에서 재평가하고, 전제 추출·사실 확인과 CoT monitor를 추가한다. 원 표의 도메인 내 readout 8종을 모두 포함하고 새 비교 2종을 더한 10행이다. 원문의 SelfAware→CREPE 전이 행은 별도 전이 실험이므로 부록에 SelfAware/CREPE→의료 조건으로 명세한다. 새 탐지 알고리즘의 최초 제안이 아니다. 현재 hidden logistic은 우리 시스템에 채택한 기존 게이트이며 별도 Ours 탐지 행을 만들지 않는다. 전체 정책을 추가하는 자리는 Table 2다. [기여·적용 범위](20_baseline_transfer_and_novelty.md)

**readout.** 같은 out-of-fold 문항과 FPQ/정상 라벨을 사용한다. 모든 학습·점수 결합·층 선택은 평가 fold 밖에서 한다.

| 판별 신호 | 실제 입력과 점수 | 출처·정보 시점 |
|---|---|---|
| 질문 텍스트 분류기 (BoW) | fit 질문 단어 빈도에 L2 logistic 학습 | Two Axes의 표면 대조군; LLM 생성 없음 |
| 직접 전제 질문 | 원 질문에 FP 유무를 묻고 Yes/No의 log-odds | Two Axes; 판정 프롬프트 forward |
| 학습된 출력 readout | Plain 답변 평균 토큰 log-probability와 첫 토큰 분포 entropy에 L2 logistic | Two Axes; Plain 답변 생성 후 사용 가능 |
| Raw 답변 확신도 | Plain 답변의 평균 토큰 log-probability를 사전 고정 방향으로 점수화 | Two Axes; 답변 생성 후, 별도 분류기 없음 |
| P(IK): 답변 전 자기평가 | 답하기 전 스스로 답할 수 있는지 물은 확률 | Two Axes; 자기평가 판정 후 |
| P(True): 답변 후 자기평가 | 생성한 답변이 맞는지 재평가한 확률 | Two Axes; 답변 생성·자기평가 후 |
| 내부 평균 차 방향 (DiM) | fit FPQ와 정상 질문의 마지막 프롬프트 활성값 평균 차 방향에 투영 | Two Axes; 생성 전. 행동 방향 C와 다름 |
| 내부 logistic probe | 마지막 프롬프트 residual의 표준화+L2 logistic; 층은 dev 선택 | Two Axes와 같은 종류의 readout; 현재 채택한 게이트 |
| 전제 추출 + 사실 확인 | 전제 문장을 추출하고 각 전제의 false 점수 중 최대값 | Well 절차의 질문 단위 탐지 adaptation; 추출 후 전제별 판정 |
| 전제 검토 CoT + 별도 모니터 | 전제 검토 글 생성 후 별도 판정기가 원 질문+검토 글을 읽어 점수화 | 추가 비교; 생성+판독. 판정기 모델·프롬프트는 미확정 |

Raw·P(IK)·P(True)는 높은 확신도가 정상 쪽이라는 의미에 맞춰 FPQ 점수 방향을 사전 고정하고, test에서 AUROC가 높아지는 부호를 고르지 않는다. 자기평가의 학습·평가 정답도 질문의 FP 유무이며, QA 정답률과 동일시하지 않는다. 직접 질문의 토큰화·Yes/No 정규화, 추출 실패·빈 전제·다중 전제 집계, CoT 판정기의 점수 규칙을 test 전에 고정한다. 부품 실행 횟수를 일률적으로 ‘생성 1회/2회’라고 비용으로 쓰지 않는다. 학습량·생성 토큰·실제 forward·지연을 측정한다. 출력 readout과 CoT는 생성 전 probe보다 많은 정보를 소비하므로 정보 시점을 명시한다. 추가 텍스트+hidden 결합은 fit 내부의 out-of-fold 점수로 결합기를 학습하는 부록 분석이다.

**Table 1 뼈대** (모델별 블록; FPQ 585·정상 150의 out-of-fold 평가)

| 판별 신호 | AUROC ↑ | TPR (%) ↑ | FPR (%) ↓ |
|---|---|---|---|
| 질문 텍스트 분류기 (BoW) | — | — | — |
| 직접 전제 질문 | — | — | — |
| 학습된 출력 readout | — | — | — |
| Raw 답변 확신도 | — | — | — |
| P(IK): 답변 전 자기평가 | — | — | — |
| P(True): 답변 후 자기평가 | — | — | — |
| 내부 평균 차 방향 (DiM) | — | — | — |
| 내부 logistic probe (채택한 게이트) | — | — | — |
| 전제 추출 + 사실 확인 | — | — | — |
| 전제 검토 CoT + 별도 모니터 | — | — | — |

각 셀은 추정값 [95% CI]다. TPR·FPR은 방법별 dev FPR≤5%에서 TPR 최대인 탐지 문턱 τ_det를 잠근 test 결과다. 같은 숫자의 문턱을 모든 방법에 강제하지 않는다. AUROC는 점수 순위의 판별 성능이며 ACC가 아니다. 이진 출력만 있으면 고정 TPR/FPR만 보고하고 AUROC용 연속 점수 추출 버전을 구분한다. 작은 정상 dev의 분모와 실제 test FPR을 함께 보고한다. [probe 명세](19_method_protocol.md#3-a-어디서-무엇을-추출해-probe를-학습하는가)

**같이 내는 것 (부록).** 층×위치 AUROC 히트맵. shortcut 통제: 암종별 holdout, 텍스트 분류기 대비 이득. "Plain이 못 고칠 문항인가"(PCR −1 vs +1) 라벨의 AUROC는 별도 소표 — Task 2의 A∧C 조건용.

**해석.** 같은 입력 범위와 같은 평가 분할에서 텍스트 기준선 대비 추가 판별 정보가 있는지를 본다. hidden이 직접 질문·전제 검증·CoT와 낮은 오탐 조건에서 비교해 우세하면 RH1의 판별 부분을 지지한다. 결합의 보완성은 부록 분석이다. hidden ≈ 텍스트면 "이 readout으로는 추가 정보 검출 실패". 어느 쪽이든 Task 2는 세 게이트를 다 돌리고, 실제 선택의 이득이 내부 신호 때문인지는 부록 Table A1에서 확인한다. 지식 유무의 증명은 아니고 "전제 진위를 읽는다"의 증명도 아니다.

## 2. Task 2 — 선택적 교정 → Table 2 (본 표), 부록 Table A1 (선택 통제), Figure 1

**질문.** 정상 질문의 손실을 제한하면서 FPQ 교정을 개선하는가. 선택 여부, 선택에 쓰는 신호, 선택 후 개입 방식이 각각 어떤 효과를 내는가. (RH2, RH2b)

**실험 조건을 구분한다.** Table 2는 공통 개발 데이터·보존 허용폭·탐색 예산 아래의 완성 방법을 비교하되, 두 Hidden 행은 동일 gate를 공유하는 통제 비교로 둔다. Table 3는 전제 추출 단계를 제거한 사실 확인 진단이다. 같은 C·K에서 선택만 바꾸는 진단은 부록 Table A1이다.

### 2.1 방법의 부품

| 개입 조건 | 정의 | 역할 |
|---|---|---|
| 항상 | 전 문항 개입 | 선택하지 않는 대조군 |
| 무작위 | 사전에 정한 개입 예산만큼 무작위 선택, seed 5개 | 개입량 감소와 선택 정보의 효과를 구분 |
| 텍스트 분류기 | Task 1에서 학습한 텍스트 점수 | 외부 입력만으로 고르는 대조군 |
| CoT monitor | 별도 전제 검토 글에서 얻은 점수 | 추론 텍스트로 고르는 대조군 |
| **hidden probe (A)** | Task 1에서 학습한 내부 점수. 구성·문턱은 내부 dev에서 선택 | 제안 게이트. 전체 Task 1 평가 결과의 1등이라는 이유로 선정하지 않음 |
| A ∧ C | A가 열리고 행동 예측 C 점수가 지정된 조건을 만족 | 부록 확장. 제어 방향 C와 행동 예측 점수의 정의를 별도로 명시 |
| oracle | 의사 라벨 FPQ 전부에 개입 | 진단 조건. 개입 부작용이 있으므로 최종 효용의 상한으로 부르지 않음 |

| 개입 | 정의 | 출처·선택 규칙 |
|---|---|---|
| FP correction prompt (선택 후 개입) | 저장된 gate 선택에 따라 교정 지시를 추가하여 응답 생성 | 전체 FP Identification의 판별 단계를 다시 실행하지 않으며 정확한 템플릿 기록 |
| **C residual steering** | 교정/비교정 응답의 활성값 대조. 후보: (i) 다른 질문의 교정·비교정 Plain 응답 평균 차, (ii) 같은 질문에 시스템 지시만 바꾼 응답 쌍의 차, (iii) 같은 질문의 기준 교정·비교정 답변을 teacher-forcing한 활성값 차 | 짝 대조는 질문 차이의 교란을 줄이는 장치이며, 질문 내용의 정확한 상쇄나 순수 행동 방향을 보장하지 않는다. 주 방법은 (iii)의 첫 32토큰으로 고정하고, 층·α는 내부 dev의 실제 교정·정상 손실로 선택한다. (i)·(ii)는 부록 후보이며 [19](19_method_protocol.md)가 우선한다. AUROC만으로 제어력을 판정하지 않는다. CAA·Persona Vectors·RepE·Lavi 관련 근거는 [10](10_evidence_and_baselines.md), [12](12_discussion_decisions.md) 참조 |
| head steering | head별 방향을 적용하는 방식 | Gated 의료 재학습은 본 Table 2의 직접 baseline. 원 artifacts 전이·추가 head 변형은 부록 |

### 2.1a Table 2 — 전제 교정과 일반 의료 QA를 합친 본 결과표

Task 2의 교정·정상 질문 평가와 Task 3의 일반 의료 QA 평가를 **하나의 Table 2**에 넣는다. 아래가 논문에 들어갈 행·열이며, 모델마다 같은 블록을 반복한다. 문제별 기존 방법과 가까운 조건부 방법을 포함한 아홉 행을 사용한다. Unconditional C는 부록 Table A1의 U행으로 이동하고, 각 셀은 실험 후 채운다.

| Method | PCR (%) ↑ | PCS ↑ | NFP (%) ↑ | MedQA ACC (%) ↑ | PubMedQA ACC (%) ↑ | Medbullets ACC (%) ↑ |
|---|---|---|---|---|---|---|
| Plain | — | — | — | — | — | — |
| FP Identification | — | — | — | — | — | — |
| Extract + FactCheck | — | — | — | — | — | — |
| GEPA (FPQ+TPQ) | — | — | — | — | — | — |
| 균형 전제 검토 CoT prompting | — | — | — | — | — | — |
| CAST (의료 adaptation) | — | — | — | — | — | — |
| Gated head steering (의료 adaptation) | — | — | — | — | — | — |
| Hidden gate + FP prompt | — | — | — | — | — | — |
| Hidden gate + C steering | — | — | — | — | — | — |

**Table 2의 동일 선택 비교.** Hidden gate + FP prompt와 Hidden gate + C steering은 probe·문턱·문항별 gate mask를 정확히 공유한다. 현재 주 정책의 probe와 τ_policy·C를 내부 dev에서 선택한 뒤 FP prompt 행에도 그 gate를 적용한다. 프롬프트 버전은 dev에서 정하되 gate를 다시 고르지 않는다. 선택되지 않은 질문에는 같은 Plain 응답을 사용하고, 선택된 질문에서만 교정 연산을 바꾼다. 따라서 이 두 행은 같은 대상에서의 교정 방식 비교이며, 각 교정 방식에 독립적으로 최적화된 gate끼리의 비교가 아니다. 프롬프트에 맞춰 gate까지 별도 최적화한 Two Axes식 정책은 필요하면 부록에 구분해 보고한다. 이 두 행의 paired ΔPCR·ΔPCS·ΔNFP와 95% CI를 함께 해석하고 별도 Table 3에서 같은 결과를 반복하지 않는다.

**Table 2 캡션 초안.** *End-to-end premise correction and medical QA performance.* 동일 모델의 방법별 결과를 비교한다. Cancer-Myth의 PCR·PCS는 거짓 전제 교정, NFP는 정상 질문에 없는 거짓 전제를 지어내 지적하지 않은 비율이다. NFP는 답변 전체의 의학적 정확도를 뜻하지 않으며, 별도 QA 열은 일반 의료 문제의 정답 정확도를 평가한다. MedQA·PubMedQA·Medbullets는 정답 정확도(ACC)다. PCS를 제외한 성능은 백분율이며, 각 성능 셀은 `추정값 [95% CI]`로 채운다. Cancer-Myth 열은 grouped cross-fitting의 out-of-fold 결과, 의료 QA 열은 QA test를 보지 않고 별도 최종 train/calibration 규칙으로 고정한 시스템의 결과다. `—`는 미측정이며 0을 뜻하지 않는다.

**보고 규칙.** 모델마다 동일한 행 블록을 반복하고, 세 QA 열을 모든 주 비교 방법에 채운다. 정상 지표·QA의 보존 여부는 Plain 대비 paired 차이의 CI와 사전 허용폭으로 판정하며, 해당 차이와 판정은 표 주석 또는 동반 부록 표에 제시한다. NFP와 Well TPQ는 같은 정상 질문이다. 본 표는 공식 NFP 하나를 주 지표로 쓰고, Table 3는 주석 전제의 참·거짓 정확도이며 Well 최종 응답 S5와 0–5 분포는 부록에 둔다. 미확정 TPQ 이산화 지표는 주 평가에서 제외한다. 정확한 답 추출·무효 출력 처리 규칙도 QA별로 잠근다. 데이터셋별 개입률·harm/rescue·비용은 부록에 보고하여 QA를 하나의 불명확한 발화율로 합치지 않는다.

**행의 선정 기준과 절차:** [20 §5](20_baseline_transfer_and_novelty.md#5-table-2의-행을-고르는-기준). Plain은 기준, FP Identification·Extract+FactCheck·GEPA는 기존 전제 교정 방법, 균형 CoT는 직접 검토 대조군, CAST·Gated head steering·Hidden+prompt는 가까운 조건부 방법, Hidden+C는 평가 대상 전체 정책이다. CAST는 원 조건 벡터·유사도 문턱을 사용하는 의료 adaptation이며 logistic gate와 혼동하지 않는다. Gated adaptation은 의료 gate·head·방향을 다시 학습하며, 호환 모델의 원 artifacts 전이는 부록에서 별도 보고한다. 재현·재학습 완료를 의미하지 않는다. 프롬프트·추가 데이터·탐색 예산·유지/변경한 원 절차를 config에 기록한다. 출판 숫자는 별도 참고 자료다.

**본 표에서 NFP만 남기는 이유.** NFP와 Well TPQ는 같은 150개 질문에 서로 다른 루브릭을 적용한다. Cancer-Myth 표와의 연결을 위해 공식 NFP를 주 정상 지표로 채택한다. Well 사실 확인 정확도는 Table 3, 최종 FPQ/TPQ 응답 점수는 부록에서 제시하고, 미확정 ‘TPQ 오교정률’은 주 제약에서 제외한다. 두 지표는 독립 표본도 정확한 보수 관계도 아니다.

### 2.2 Table 3 후보 — 주석 전제의 사실 확인: Well Table 1 계승

**현재 위치와 확정 범위.** 가져올 선행표는 Well-Actually Table 1로 확정한다. 이 표는 교수님 발표의 기존 연구 근거로 사용할 수 있다. 아래의 우리 Table 3 및 추가 probe 행은 사실 확인 진단을 독립 연구 질문으로 채택할 경우의 후보이며 본 실험으로 확정한 것은 아니다. 기존 결과를 인용하는 것만으로 우리 실험 결과표가 되지 않는다. 현재 질문 gate의 검증이 목적이면 Table 1의 질문 단위 탐지 비교에 RAG 기반 전제 추출·검증을 추가하는 방안도 가능하며, gold 전제를 받는 원 Table 1의 숫자와 직접 섞지 않는다.

**계승하는 표는 Well-Actually의 본문 Table 1, “Fact-checking accuracy on CancerMyth”다.** 이전 문서의 Table 8 S5 표로 대체한 구성은 철회한다. 사람이 주석한 전제 문장을 입력으로 주고 참·거짓 판정 정확도를 잰다. 최종 환자 답변 생성이나 PCR·PCS·S5를 평가하는 표가 아니다.

**Table 3(a): Well-Actually Table 1의 원 보고값 — 참고 자료.** Model → RAG → FPQ Accuracy → TPQ Accuracy의 원 구조를 유지했다. 수치는 정확도 % (정답 수/전제 수)다. FPQ Accuracy는 거짓 전제를 거짓으로 판정한 비율, TPQ Accuracy는 참 전제를 참으로 판정한 비율이다. 원 표의 분모 100·116은 전제 문장 수이며 116개의 정상 질문을 뜻하지 않는다. [원문 §3.2·Table 1](https://arxiv.org/html/2608.06539v1#S3.SS2)

| Model | RAG | FPQ Accuracy | TPQ Accuracy |
|---|---|---|---|
| Llama3-Med42-8B | None | 95 (95/100) | 31 (36/116) |
| Llama3-Med42-8B | Top-4 | 96 (96/100) | 32.8 (38/116) |
| Llama3-Med42-8B | All | 98 (98/100) | 20.7 (24/116) |
| Llama-3-8B-Instruct | None | 97 (97/100) | 22.4 (26/116) |
| Llama-3-8B-Instruct | Top-4 | 94 (94/100) | 42.2 (49/116) |
| Llama-3-8B-Instruct | All | 100 (100/100) | 16.4 (19/116) |
| Olmo-3-7B-Instruct | None | 100 (100/100) | 17.2 (20/116) |
| Olmo-3-7B-Instruct | Top-4 | 93 (93/100) | 8.6 (10/116) |
| Olmo-3-7B-Instruct | All | 100 (100/100) | 11.2 (13/116) |
| Qwen2.5-7B-Instruct | None | 96 (96/100) | 15.5 (18/116) |
| Qwen2.5-7B-Instruct | Top-4 | 97 (97/100) | 12.9 (15/116) |
| Qwen2.5-7B-Instruct | All | 100 (100/100) | 6.9 (8/116) |
| Gemini-3-flash | None | 93 (93/100) | 32.8 (38/116) |
| Gemini-3-flash | None + reasoning | 98 (98/100) | 21.6 (25/116) |
| Gemini-3-flash | Top-4 | 95 (95/100) | 26.7 (31/116) |
| Gemini-3-flash | Top-4 + reasoning | 96 (96/100) | 17.2 (20/116) |
| Gemini-3-flash | All | 98 (98/100) | 25 (29/116) |
| Gemini-3-flash | All + reasoning | 96 (96/100) | 12.9 (15/116) |
| Gemini-3-flash | Web | 96 (96/100) | 31 (36/116) |
| Gemini-3-flash | Web + reasoning | 98 (98/100) | 20.7 (24/116) |
| Gemma-4-E4B-it | None | 98 (98/100) | 23.3 (27/116) |
| Gemma-4-E4B-it | Top-4 | 96 (96/100) | 16.4 (19/116) |
| Gemma-4-E4B-it | All | 61 (61/100) | 37.1 (43/116) |
| MiniCheck | Top-4 | 99 (99/100) | 10.3 (13/116)† |
| MiniCheck | All | 95 (95/100) | 11.2 (14/116)† |

† MiniCheck 두 TPQ 셀은 원문·사용자 첨부 표를 그대로 옮겼다. 다만 13/116≈11.2%, 14/116≈12.1%로 원문의 앞 백분율과 일치하지 않는다. 원 결과 파일 확인 전에는 어느 값을 정정값으로 채택할지 단정하지 않으며, 이 두 셀로 차이·순위를 계산하지 않는다.

**Table 3(b): 같은 전제·근거를 사용하는 재실행 및 내부 readout 진단 — 미측정 초안.** 독립 진단을 채택할 경우 Qwen에서 아래 조건을 비교할 수 있다. (a)의 기존 모델·RAG 조건은 보고값으로 소개하고, 직접 우위 비교는 같은 우리 분할에서 baseline까지 재실행한 (b) 안에서만 한다. 원 split·문서와 일치하지 않는 새 결과를 (a)에 그대로 붙이지 않는다.

| Model / 판별 방식 | RAG | FPQ Accuracy (%) ↑ | TPQ Accuracy (%) ↑ |
|---|---|---|---|
| Qwen2.5-7B / 원 사실 확인 프롬프트 | None | — | — |
| Qwen2.5-7B / 원 사실 확인 프롬프트 | Top-4 | — | — |
| Qwen2.5-7B / 원 사실 확인 프롬프트 | All | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | None | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | Top-4 | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | All | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | None | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | Top-4 | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | All | — | — |

**추가 행의 범위.** 내부 행은 주석 전제와 해당 근거를 입력받는 판별기의 진단이다. 원 질문만 읽도록 학습한 현재 A gate를 그대로 옮기거나 C steering의 최종 응답 점수를 붙이는 행이 아니다. 동일한 전제·근거·few-shot 문맥에서 출력 판정과 내부 readout을 비교한다. probe는 전제 단위 fit 자료에서 별도로 학습하고 층·문턱은 dev에서 고정한다. 같은 질문의 여러 전제와 같은 myth는 같은 fold에 묶는다. 추가 CoT 행은 전제를 검토한 뒤 참·거짓을 판정하는 비교이며 별도 모니터가 환자 답변을 채점하는 절차가 아니다. 원 모델이 지원하는 reasoning 설정과 추가 CoT 프롬프트도 구분한다.

**캡션 초안.** *Fact-checking accuracy on annotated medical presuppositions under external evidence conditions.* (a) Published results from Well-Actually Table 1. (b) New evaluations on shared held-out presuppositions and frozen evidence. Accuracy is computed separately for false and true presuppositions. New entries report estimates, 95% confidence intervals, and correct/total counts. The premise-input probe is a diagnostic readout, distinct from the question-level deployment gate. Published and new results are not pooled.

**세 표의 차이.** 우리 Table 1은 원 질문에서 잘못된 전제를 탐지하는 능력, Table 2는 최종 교정·정상 질문·의료 QA 성능, Table 3는 전제를 직접 제공해 추출 단계를 제거한 사실 확인 능력을 평가한다. Table 3에서만 주석 전제를 입력으로 제공하며 정답 진위 라벨은 입력하지 않는다. 이는 Task 1의 원인 진단 확장이다. 같은 개입량의 선택 비교는 부록 A1, 동일 gate 뒤의 prompt/C 효과는 Table 2에 남긴다.

**해석 범위.** RAG 후에도 참 전제를 부정한다면 그 근거 조건에서 사실 확인 문제가 남았다는 결과다. 문서의 관련성·충분성을 확인하지 않은 채 지식 부족을 배제하지 않는다. 전제 입력 probe가 더 좋아도 실제 환자 질문을 읽는 A gate의 우위나 steering의 교정 효과를 입증한 것은 아니다.

### 2.2a 부록 Table A1 — 선택 방식 통제

**부록 Table A1: 선택 신호가 실제 교정에 유용한가?** 이전 Table 3(a)를 옮긴 진단이다. Table 1의 탐지 성능만으로 실제 교정 성능을 알 수 없으므로 비교는 유지한다.

| ID | 교정 대상 선택 | 교정 방식 | PCR (%) ↑ | PCS ↑ | NFP (%) ↑ | 개입률 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| U | 모든 질문 | 고정 C steering | — | — | — | 100 |
| R | 무작위 K개, 5 seeds | 고정 C steering | — | — | — | 100K/N |
| T | 질문 텍스트 점수 상위 K개 | 고정 C steering | — | — | — | 100K/N |
| Q | CoT monitor 점수 상위 K개 | 고정 C steering | — | — | — | 100K/N |
| H | 내부 probe 점수 상위 K개 | 고정 C steering | — | — | — | 100K/N |

**캡션 초안.** *Controlled selection at a matched intervention budget.* C의 방향·층·강도·scale·토큰 구간을 고정한다. R/T/Q/H는 fold마다 같은 K개에 개입하며 U는 전부 개입한다. PCR·PCS는 전체 held-out FPQ, NFP는 전체 held-out 정상 질문을 분모로 한다. 각 성능 셀은 추정값 [95% CI]이며 미측정은 `—`다. 개입률은 `100 × 선택 수 / 전체 평가 질문 수`이고 성공률·출력 변경률이 아니다.

C와 개입 비율 q는 내부 dev에서 고정하고 K=round(qN)으로 정한다. test 라벨 없이 점수 순위로 상위 K개를 고르며 동점 규칙·무작위 5개 seed도 사전 고정한다. 이 배치 예산 진단은 Table 2의 개별 질문 고정 문턱 정책과 다르다. q를 test 결과로 조정하지 않는다. CoT 검토 글은 선택 점수 계산에만 사용하고 최종 답변 입력에는 넣지 않는다.

H−U는 개입량 감소까지 포함한 효과, H−R은 같은 개입량에서 정보를 보고 고르는 효과, H−T/Q는 선택 신호에 따른 실제 교정 차이를 평가한다. paired ΔPCR·ΔPCS·ΔNFP와 CI를 내며, 무작위 seed 변동과 문항 표본 불확실성을 구분한다. Table 2와 모델·fold·mask·C·생성·판정 조건이 모두 같을 때만 응답을 재사용하고 독립 증거로 다시 세지 않는다. harm/rescue·비용·Oracle·A∧C 등 추가 진단도 부록에 둔다.

이전 Table 3(b)의 프롬프트/C 비교는 Table 2의 두 Hidden 행으로 통합했다. A1의 상위 K 목록을 Table 2의 고정 문턱 목록과 같은 것으로 간주하지 않는다.

### 2.3 Figure 1과 실행·해석

**Table 3 후보의 데이터 범위.** 공개 TPQ 148질문에는 178개 주석 전제가 있지만 원 Table 1의 TP 전제 116개 목록은 확인되지 않았다. 진단을 채택하면 질문 ID와 전제 index를 매핑하고 같은 질문·myth를 같은 fold로 묶어 전제 단위 out-of-fold 판정을 낸다. few-shot 그룹은 해당 평가에서 제외하며 FP/TP 전제 분모를 각각 기록한다. 원 100/116 결과와 같은 평가라고 쓰지 않는다. [자료 감사](21_well_rag_reproducibility.md)

**Figure 1.** x = NFP(오교정 안 한 비율), y = FPQ PCR. Table 2의 **고정 문턱을 사용하는 완성 시스템**을 점으로 찍는다. Table 1의 판별기에는 이 좌표가 없다. 부록 Table A1의 예산 통제 점을 추가한다면 별도 기호나 패널로 진단임을 구분한다. 허용 경계는 `Plain의 NFP − ε_NFP`를 수직선으로 표시한다. Well-Actually Figure 1의 시각화 취지를 참고하되, 다른 지표·판정기로 낸 좌표를 원 논문의 숫자와 동일시하지 않는다.

**절차.** fold마다 (1) 학습 자료로 C 방향, A probe, 텍스트 분류기, CAST 벡터, GEPA를 학습한다. (2) 내부 dev에서 각 Table 2 방법의 층·α·문턱 등을 정상 지표별 허용폭 아래에서 조정하고, 허용폭을 만족하는 설정이 없으면 그 사실을 보고한다. (3) 부록 Table A1의 공통 C·q·프롬프트·seed를 고정한다. (4) 평가 fold에서 Table 2의 고정 문턱 평가와 부록 Table A1의 예산 통제를 구분해 실행한다. (5) 5 fold의 out-of-fold 응답을 합쳐 집계하고, 같은 myth와 문항쌍을 함께 재표집하는 paired CI를 낸다. test 결과를 보고 방법·문턱·층을 다시 선택하지 않는다.

**해석.** Table 2에서 Plain 대비 PCR 개선과 사전 지정한 정상·QA 보존 조건을 동시에 만족하는지 먼저 본다. Table 3는 전제 단위 사실 확인을 진단하고, 선택 정보의 기여는 부록 Table A1에서 판단한다. 내부 선택이 텍스트·CoT와 비슷하거나, 프롬프트 개입이 C보다 좋으면 해당 우위 가설을 지지하지 않는 결과로 보고한다. 무조건 개입도 정상 성능을 보존할 수 있으며, 모든 baseline이 실패해야만 연구가 성립하는 것은 아니다.

## 3. Task 3 — 일반 의료 QA 보존 → Table 2의 오른쪽 열 셋

**질문.** Task 2에서 고정한 시스템(조건 + 문턱 + 개입 + α)을 다른 의료 질문에 그대로 두면 성능이 유지되는가. (RH3)

**벤치마크.** MedQA (USMLE 4지선다), PubMedQA (yes/no/maybe), Medbullets (선다). Cancer-Myth Table 1의 열 중 셋. 정확도만 세므로 판정기 불필요. SymCat·Craft-MD는 코드가 없어 제외한다고 적는다.

**절차.** 아래 규칙으로 고정한 최종 시스템을 그대로 적용한다. 평가셋 이름을 보고 조건을 끄지 않는다. 각 벤치마크의 원래 채점 규칙을 따르고, Plain 대비 문항별 paired 차이와 CI 및 개입률을 보고한다. 허용폭 안임을 입증하지 못하면 RH3 지지 실패로 적고, CI가 넓어 불확실한 경우와 허용폭을 넘는 저하가 드러난 경우를 구분한다. QA dev로 재조정하는 실험을 한다면 별도 조건으로 명시한다.

### 판정 규칙 (표를 채우기 전에 고정, 리뷰 3)

- 정상 질문 주 지표는 공식 NFP 하나이며 ε_NFP를 정한다. Well 사실 확인 정확도는 Table 3의 별도 진단이고 별도 TPQ 이산화 제약은 사용하지 않는다.
- 보존 판정은 "차이가 유의하지 않다"가 아니라 **Plain 대비 차이의 신뢰구간이 허용폭 안에 들어오는가**(비열등성).
- Table 2의 각 방법(GEPA, CAST, 게이트 문턱·α 등)은 같은 개발 데이터·같은 허용폭 아래에서 조정한다. 손잡이가 없는 방법(FP Identification)은 그대로.
- QA에 적용할 최종 시스템: cross-fitting은 fold별 시스템을 만든다. **QA용 최종 train/calibration 그룹 분할과 inner-dev 결과의 집계 규칙을 사전 지정**하여, 각 방법당 시스템 하나를 다시 학습·고정한다. 구조·후보를 inner-dev 결과로 선택하고 새 probe의 문턱·개입 강도는 최종 calibration에서 보존 제약 아래 정한다. 다른 probe의 문턱을 단순 평균하지 않는다. QA test와 outer 평가 성능으로 선택하지 않는다. Table 2의 Cancer-Myth 열은 기존 out-of-fold 결과이며, QA 열은 이렇게 고정한 최종 시스템의 결과임을 캡션에 구분한다. Cancer-Myth 열을 최종 재학습 시스템의 학습 문항 성적으로 대체하지 않는다.
- 코드 선행 조건: 분할 누수, TPQ 루브릭, 파싱 실패, 판정 캐시([16 §6](16_code_overview.md))를 먼저 해결한다.

발화율은 지정된 **교정 개입 분기**가 실행된 비율이며 검사 절차 실행률과 다르다. FP Identification의 전제 검사는 전 문항에 적용되지만 교정 분기율은 별도 측정한다. GEPA처럼 명시적 분기가 없으면 해당 없음으로 표시하고, 프롬프트 적용률·검사 비용은 따로 보고한다.

출판 수치는 별도 참고 표에 분리하고 동일 조건의 순위 비교에 넣지 않는다(닫힌 모델, 또는 판정 시점·생성 설정 차이). 각 셀에 CI. NFP·QA 열은 각 지표의 사전 보존 판정을 통과하면 표시한다. 성공 조건은 ε_NFP·벤치마크별 ε_QA를 모두 만족하면서 Plain 대비 PCR의 paired 개선이 관측되는 것이다. 개선 CI의 판정 기준도 사전 지정한다. 여럿이면 같은 보존 조건에서 PCR·PCS·측정 비용을 비교하고, test에서 골라낸 우승자를 독립 검증된 새 방법처럼 제시하지 않는다.

### Table 2의 출판 수치 참고 자료 — 직접 비교 순위에 포함하지 않음

| 원 논문 보고 조건 | PCR (%) | PCS | NFP (%) | MedQA ACC (%) | PubMedQA ACC (%) | Medbullets ACC (%) |
|---|---|---|---|---|---|---|
| GPT-4o Plain, Cancer-Myth Table 1 | 12 | — | 88 | 70 | 67 | 68 |
| GPT-4o GEPA, Cancer-Myth Table 1 | 68 | — | 59 | 63 | 59 | 62 |
| Qwen2.5-7B Plain, Cancer-Myth Table 3 | 6.3 | −0.50 | — | — | — | — |

위 값은 이전 문서에 기록한 원 논문 수치이며 새 실험 결과가 아니다. `—`는 여기서 인용하지 않은 값이다. 출처·모델·생성·판정 조건이 다르므로 본 결과와 구분한다. Well의 GEPA 보고 모델은 Gemini·Gemma 계열이므로 Qwen·Llama용 GEPA는 우리가 실행하는 확장이다. 기존 예산 500 및 검증 표본 설정은 정상 dev 표본 제약에 맞춰 실행 전에 명시한다.

**부록 행.** Self-Dual-Critique, PreWoMe, FalseQA LoRA (Well). ITI (head 무조건), Tripathi 원 artifacts의 호환 모델 전이, A∧C + C. 의료 재학습 Gated와 Extract+FactCheck는 본 Table 2에 포함한다.

**가져올 수 없는 것.** Cancer-Myth의 Monitor (MDAgents 코드는 있으나 감시 에이전트 프롬프트 없음). 참고 행에만.

## 4. 부록 표

- A×C 2×2 분포, 7 카테고리별. 점수 조합 이름, H2/H3 라벨 아님.
- 층별 cos(A, C).
- AO/NLA 통제 4종 (텍스트만 / 활성값 없음 / 섞은 활성값 / NFP).
- Well 고정 split과 판정기를 확보하면 Table 2의 방법을 해당 프로토콜로 별도 평가. 원 논문의 Table 8–10에 추가할 행과 현재 cross-fitting 결과를 구분.
- "Plain이 못 고칠 문항" 라벨의 readout AUROC.

## 5. 실행 순서

1. 분할 manifest(myth·NFP-TPQ 그룹), QA용 최종 fit/calibration 규칙, ε_NFP·벤치마크별 ε_QA와 CI 판정 기준을 고정한다. 분할·TPQ 루브릭·파싱·캐시의 코드 선행 조건을 해결한다.
2. 한 모델(Qwen2.5-7B)에서 Task 1의 모든 readout을 실행해 Table 1을 채운다. 각 outer fold의 내부 dev에서 개입 후보와 설정을 선택한다.
3. 같은 모델에서 Table 2의 공유 gate prompt/C를 포함한 9개 방법과 Task 3의 QA 평가를 실행한다. 전체 outer 평가로 C·게이트의 우승자를 다시 고르지 않는다.
4. Table 3 사실 확인 진단을 채택하면 고정 전제·문서·분할에서 RAG 조건을 재실행해 참·거짓 정확도를 집계한다. 부록 A1의 같은 C·K 선택 통제도 수행한다. 실패·불확실한 결과도 보고한다. 파일럿 중단 규칙이 있다면 실행 전에 정하고, 중단한 평가와 이미 얻은 결과를 명시한다.
5. 두 번째 모델과 사전 지정 부록 평가를 진행한다. 자원에 따라 생략한 범위와 주장 한계를 적는다.
