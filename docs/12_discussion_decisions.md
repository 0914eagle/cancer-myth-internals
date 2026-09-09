# 12. 논의 결정 기록과 남은 질문

2026-09-07–08 대화의 연구 관련 논점을 통합했다. 축어록이 아니라 현재 결정, 변경 이유, 아직 열린 선택을 남긴 기록이다. [09](09_research_proposal.md)는 교수님 설명용 연구계획, [10](10_evidence_and_baselines.md)은 근거, [11](11_tripathi_review.md)은 직접 선행연구 검토다.

## 1. 논의에서 고정한 것

| 질문 | 현재 정리 |
|---|---|
| 우리의 목표가 무엇인가? | FPQ 교정 향상과 TPQ/NFP 보존, 별도 의료 QA 보존을 함께 평가한다 |
| 실험부터 하고 이야기를 만드는가? | 주장·가설·예상 기여를 먼저 세우고 이를 지지/반박할 실험을 설계한다 |
| 반드시 새 알고리즘이어야 하나? | 기존 선택적 개입을 적용하는 실증·분석 연구도 가능하다. 새로운 방법론 기여와는 구분한다 |
| 교수님이 원하는 기존 표+우리 행은 어디인가? | Task 2와 3을 합친 본 결과표가 Cancer-Myth Table 1의 축과 연결된다. Task 2의 FPQ–TPQ 그림은 Well-Actually와 연결된다 |
| Two Axes는 단순 참고인가? | 전체 문제는 다르지만 premise-check 라우팅은 직접적인 방법 baseline이다 |
| CAST는 무엇이 실패했나? | CAST가 의료 전제에서 실패했다고 확인한 적 없다. 무조건 steering의 부작용을 줄이는 선행 방법이므로 제대로 적용해 비교한다 |
| CoT로 안 된다고 확정했나? | 아니다. 명시적 전제 검사 CoT와 CoT monitor를 실제 비교해야 한다 |
| 내부가 더 좋다고 확정했나? | 아니다. 내부의 추가 가치는 RH1이다 |
| NLA로 반드시 가야 하나? | 아니다. linear probe를 기준으로 AO/NLA의 추가 가치를 측정한다 |
| 일반 의료 QA 저하를 새로 발견해야 하나? | 기존 근거가 있다. 우리 정책에서 보존되는지는 별도 검증한다 |
| contribution 세 개가 확정됐나? | 연구계획의 예상 기여는 명시한다. 결과 전의 성과 주장이나 독창성 보장은 아니다 |

## 2. 가설·Task·표 이름의 변경 내역

- 기존 **원인 H1–H6**: 지식 부족, 검사 누락, 진위를 읽으면서 따라감, 생성 중 변화, 답변 우선, 판정 편향 등 설명 후보. 확정 라벨이 아니다.
- 현재 **RH1–RH3**: 내부 신호 추가 가치 / 선택적 교정 효과 / 일반 의료 QA 보존. 논문을 검증하는 가설이다.
- 현재 **Task 1**: 탐지·신호 분석. **Task 2**: FPQ+TPQ 선택적 교정. **Task 3**: 별도 의료 QA 보존.
- 기존 06의 **실험 3=held-out 비교표**와 현재 Task 3는 다르다. 서버 E1/E2는 구현 단계 이름이므로 자동으로 번호를 바꾸지 않는다.
- AO/NLA의 PersonaQA 등 별도 도구 task는 핵심 Task 3로 추가하지 않는다.
- 표 번호는 논문 배치다. Table 1은 본 결과, Table 2는 readout, Table 3은 개입 분해로 계획한다. 실행 순서와 동일할 필요가 없다.

## 3. 두 축 A/C와 AO/NLA

**A**는 FPQ/TPQ 판별 점수 또는 그 방향이다. **C**는 교정/비교정 대조 응답에서 만든 행동 방향 또는 교정 실패 예측 점수다. 방향과 예측 점수는 구분하여 표기한다. 위치 이름 A/B/C/D/E와도 다르다.

“잘못된 전제라는 방향이 있으니 찾아서 행동 벡터와 합치면 된다”는 확정 명제가 아니다. 데이터에서 선형적으로 읽히는지, 문체·주제 shortcut이 아닌지, 실제 개입이 올바른 교정을 늘리는지 각각 검증한다. 성공/실패 응답의 차이는 문체나 길이, 지식 차이까지 담을 수 있으므로 matched contrast가 필요하다.

A×C의 2×2는 점수 분포를 기술하는 부록 분석이다. “거짓으로 안다/모른다”라는 확정적 심리 상태를 셀 이름으로 붙이지 않는다. cosine이 작아도 기능적으로 독립적이라고 보장되지 않는다.

AO/NLA는 이 표상을 언어로 설명하거나 gate 점수로 바꾸는 후보다. 진단 도구로만 쓰는 경우와 실제 라우팅에 쓰는 경우를 구분한다. AO/NLA 없이는 교정 실험을 할 수 없다는 의존성은 두지 않는다.

## 4. 같은 gate / 같은 개입의 정확한 뜻

같은 개입을 고정한다는 것은 행동 벡터, 크기 정규화, α, 층·head, 적용 토큰과 감쇠를 고정한다는 뜻이다. 이 상태에서 gate만 바꾸면 선택의 효과를 비교할 수 있다.

같은 gate를 고정한다는 것은 **원래 질문과 고정된 readout에서 얻은 문항별 선택 결과**를 프롬프트와 steering에 공통으로 쓴다는 뜻이다. 각 intervention용 프롬프트에서 별도로 gate를 재계산하면 입력 차이까지 섞인다.

CoT monitor가 선택을 위해 추가 생성을 한다면 그 비용과 상태 변화가 있다. 이때 모든 조건에 같은 CoT prefix를 제공하는 같은 시점 비교와, 각 방법을 원래 방식으로 실행하는 end-to-end 비교를 따로 보고한다.

## 5. 기존 표와 Figure 1을 가져오는 조건

현재 계획은 기존 평가를 버리고 새 benchmark를 만드는 것이 아니다. 원본 데이터를 사용하되 학습·선택·최종 평가가 겹치지 않는 프로토콜을 만든다.

Well-Actually 논문의 보고 분할은 FPQ 383/100/100, TPQ 30/18/100이며 별도 few-shot 예약이 있다. 그러나 **보고된 개수와 저자들의 실제 문항 ID는 다르다**. 대화 중 공개 코드에서 분할 생성의 무작위성이 확인되었으나 원 논문 실행의 정확한 ID manifest와 raw scored outputs는 확보하지 못했다. split을 그대로 확보했다는 기존 표현은 과도했다. [저자 저장소](https://github.com/ShenranTomWang/Well)

따라서 다음 중 어떤 조건인지 명시해야 한다.

1. 원 저자 test ID와 결과를 확보했다면 동일 조건으로 우리 행을 추가한다.
2. ID를 확보하지 못하면 공개 데이터로 재현 가능한 split manifest를 고정하고 주요 baseline을 함께 재실행한다. “원래 100/100과 같은 문항”이라 하지 않는다.
3. 원 논문 숫자는 reported reference로 분리한다. Figure 좌표를 표의 반올림 분포에서 계산했다면 approximate reconstruction으로 표시한다.

NFP와 TPQ가 동일 질문인 경우 하나의 그룹으로 분할한다. “NFP 150에서 TPQ test 100을 빼면 50개를 전부 train에 쓸 수 있다”는 계산에는 dev/few-shot 예약과 실제 중복이 빠져 있다. 실제 manifest를 기준으로 사용 가능 수를 센다.

**2026-09-08 추가 결정.** 본 표의 평가 단위는 585 전체와 150 전체로 하고, 학습형 부품은 nested grouped 5-fold cross-fitting으로 out-of-fold 결과를 낸다 ([13 §0](13_task_spec.md)). 이유는 Cancer-Myth Table 1·3과 분모를 맞추는 것과 학습 음성(TPQ) 표본을 늘리는 것이다. 층·α·문턱 선택도 fold 안에서 한다. Well의 잠근 분할은 저자 ID 확보 시 부록.

원 test 고정과 myth 단위 분리가 충돌할 수 있다. 이 경우 original-protocol 비교와 myth-disjoint 일반화 평가를 분리하고, 원 분할이 보장하지 않는 일반화를 주장하지 않는다. 외부 CREPE 보강은 별도 학습 조건으로 보고한다.

## 6. 성능 보존과 통계

- “안 떨어진다”는 기본적으로 사전에 정한 비열등성 허용폭 안이라는 뜻이다. 검정에서 유의한 차이가 안 나왔다는 의미와 다르다.
- NFP 공식 지표, TPQ의 부당한 전제 부정, TPQ 답변 정확도, 일반 QA 정확도는 각각 다른 측정이다.
- Plain 대비 **harm**은 원래 성공한 문항이 실패로 바뀜, **rescue**는 원래 실패한 문항이 성공으로 바뀜이다. 순손실만 보고 harm을 숨기지 않는다.
- 독립 100문항에서 오류 0건일 때 단측 95% 이항 상한은 약 2.95%다. 이것은 paired 성능 차이의 비열등성 허용폭이나 자동 성공 기준이 아니다.
- 표본이 적으면 허용폭을 통과하도록 사후 확대하지 않는다. 부족한 정밀도와 검정력을 보고한다.
- 모든 다른 방법이 실패해야 우리 결과가 유효한 것은 아니다. 같은 위험 예산에서 성능·비용을 비교한다.
- 무작위 gate도 FPQ를 일부 개선할 수 있고 무조건 개입도 정상 질문을 유지할 수 있다. 둘의 실패를 실험 성공 조건으로 정하지 않는다.
- 실제 의료 FPQ 비율은 이 대화에서 추정하지 않았다. WildChat 표본의 비율을 환자 질문 유병률처럼 사용하지 않는다.

## 7. faithfulness 등 용어

| 용어 | 이 연구에서의 의미 |
|---|---|
| factual correctness | 답변이나 교정 내용이 의학적 근거·평가 정답에 맞음 |
| premise correction | 거짓 전제를 명시적으로 바로잡음. 잘못된 반박은 교정 성공이 아님 |
| overcorrection | 정상 질문의 정당한 전제를 부당하게 부정하거나 없는 전제를 지어냄 |
| CoT faithfulness | 보이는 사고 텍스트가 답 생성에 실제로 기여한 요인을 얼마나 반영하는가. 유창함이나 정답률과 다름 |
| activation verbalization faithfulness | AO/NLA 설명이 타깃 활성값 정보를 반영하는가. verbalizer 자신의 지식만 답하는 것과 구분 |
| context faithfulness | 제공된 기록·문서와 일치하는가. 일반 의학적 진실성과 동일하지 않을 수 있음 |
| internal monitoring | 활성값에서 정의된 관측 목표를 예측함. 모델의 마음을 직접 관찰한다는 뜻이 아님 |

## 8. 철회하거나 약화한 주장

- “모델은 이미 정답을 알고 있다.” → 데이터·readout 결과만으로 확정하지 않는다.
- “CoT로는 못 한다 / prompting은 해결 불가능하다.” → 시험된 조건의 한계와 비교 가설로 제한한다.
- “조건과 행동의 분리는 최초다.” → CAST 등 선행연구가 있다.
- “의료 gated steering과 정상 응답 보존은 최초다.” → [Tripathi](11_tripathi_review.md)와 중복한다.
- “Tripathi는 표면 압력 감지기라 배경 전제에는 반응하지 않는다.” → 공개 runtime은 내부 probe이며 전이 실패를 단정할 근거가 없다.
- “게이트 때문에 설계상 NFP가 보장된다.” → 오탐과 실제 응답 오류를 held-out에서 측정한다.
- “TPQ 1점 손실이면 반드시 진다.” → 혼합비율·효용에 따라 다르다. 보존 제약은 별도 연구 목표다.
- “probe가 안 읽히면 모델에 지식이 없다.” → 해당 readout에서 증거를 못 찾은 것이다.
- “gold gate가 전체 성능 상한이다.” → 특정 개입을 진단하는 조건이다.

## 9. 미결정 — 결과처럼 쓰지 않는다

핵심 모델은 기존 표와 겹치는 Qwen2.5-7B-Instruct 및 Llama-3.1-8B-Instruct를 우선 계획한다. Gemma 계열 추가는 자원과 AO/NLA 호환성에 따라 결정한다. 모델 revision과 precision은 실행 전 고정한다.

아직 미확정인 것은 최종 gate, C 방향 추출 대조쌍, AO/NLA의 주력 여부, 신규 알고리즘 변경, 정확한 split manifest, 허용폭 수치, 최종 모델 수, API judge 비용이다. 이 항목이 열려 있어도 [09](09_research_proposal.md)의 연구 주장과 검증 구조는 먼저 제시할 수 있다.

## 10. 저장소 이력과 검증 범위

이번 문서는 `claude/document-session-o2temh`의 `db4caad1dd6f6ce14fd3121220387f717e33cb3f`를 기준으로 작성했다. 이 이력에 `7ec1591` 머지와 두 리뷰 반영이 실제로 들어 있음을 확인했다. 원래 링크의 실험 브랜치와 `main`은 다른 이력이며 main이 최신이라고 가정하지 않았다.

이 변경은 연구 문서 정리다. GPU 실험, 코드 리뷰 지적 수정, 새로운 모델 결과 생성, 기존 결과 수치의 재현은 수행하지 않았다. 기존 [리뷰 1](reviews/2026-09-07-review.md)과 [리뷰 2](reviews/2026-09-07-review-2.md)는 그대로 유지한다.

## 11. 2026-09-08 문서 세션 추가 논의

### 용어
- gate → 문서와 보고서에서는 **개입 조건** (또는 선택적 개입, 라우팅). 원문 용어 gate는 관련 연구 절에서 병기.
- "내부 hidden state 분류기" = linear probe. "질문 문장만 보는 단순 분류기" = text classifier (BoW). 개입 여부를 정하는 것 전체 = gate / router.
- "단언문" → 보고서에서는 "직접 진술된 문장".

### AUROC가 논문마다 무엇을 맞히는가

| 논문 | 양성 라벨 | 종류 |
|---|---|---|
| Two Axes Table 2 | 질문이 거짓 전제 위에 있는가 (CREPE 주석) | 입력 속성 |
| Well-Actually Table 1 (SDT 환산) | 전제 문장이 거짓인가 (gold, 전제 단위) | 입력 속성 |
| Tripathi | 이번 사용자 턴에 거짓 주장(H) / 압력(S)이 있는가 | 입력 속성, 하네스 라벨 |
| Pandey (전이 0.83–0.85) | 사용자 주장 / 단언문이 거짓인가 | 입력 속성 |
| Perfect Detection | entity가 가짜인가 | 입력 속성 |
| MedMisBench trace 모니터 | 최종 판단이 오도 맥락에 오염됐는가 | 모델 출력 예측 |
| Readable but Not Controllable | 답변이 환각인가 | 모델 출력 예측 |
| 우리 C 방향 AUROC vs PCR | Plain이 이 문항을 고칠 것인가 | 모델 출력 예측 |

우리 Task 1의 주 라벨은 "질문에 거짓 전제가 있는가"(입력 속성) 하나. "Plain이 못 고칠 문항인가"는 다른 목표이므로 부록과 A∧C 조건에서만. 개입 조건이 실제로 맞혀야 하는 것은 "개입하면 좋아지는 문항"이고 전제 진위는 그 대리 지표라는 점을 논문에 적는다. Tripathi의 0.81–0.94는 다른 과제(하네스 턴)라 우리 숫자와 나란히 놓지 않는다.

### 조건부 steering은 작은 분야다
CAST(ICLR 2025), DSAS, GAPS, Steering Vector Fields, PCNET, Tripathi, Two Axes 라우팅, HACK, Ferrando. 우리 기여는 이 틀일 수 없다. "같은 뼈대로 다른 문제"가 논문이 되는 조건 셋 중 둘: (1) 옮겨가는 게 자명하지 않은 문제 — 있음 (표면 단서 없음, 물으면 양방향 쏠림, NFP 표면상 구분 불가), (2) 아무도 못 낸 결과가 나오는 벤치마크 — 있음 (Cancer-Myth·Well 둘 다 시소 못 끊음), (3) 뼈대를 쓰면서 새로 배우는 것 — 실험에 달림 (고르는 것 × 고치는 것 분해). 텍스트 분류기로 골라도 같고 prompt로 고쳐도 같으면 (3)이 없어 워크숍 수준.

### 교수님 지침과 anchor 표
원 지침은 "논문 하나를 따라가되(스토리·데이터·baseline·task) 내 방법론 행을 추가". 새 알고리즘을 요구한 것이 아니라 **표에 넣을 수 있는 정의된 행**을 요구한 것으로 읽힌다. 문자 그대로 따르면 따라갈 논문은 Cancer-Myth(Table 1이 전부 닫힌 모델)보다 **Well-Actually**가 맞을 수 있다: Table 8–10(방법 × FPQ/TPQ, 공개 모델 3개, 코드 전부)에 우리 행을 넣고, CREPE·QA²·Syn-QA²까지 같은 행을 넣으면 일반성이 따라오며, Cancer-Myth는 동기와 의료 QA 열(Task 3)을 준다. 대가: Well 판정기(gemini 0–5)를 그대로 돌려야 하고 분할 ID 확보 또는 고정 재실행 필요. **목요일 결정 사항.** 이메일에는 이 대안을 넣지 않았다.

### Well ↔ Two Axes 불일치 해소 (PDF 확인)
Well-Actually는 Two Axes를 한 문장으로만 인용한다: *"LLM-based fact checking has a strong prior to reject presuppositions regardless of their truth value (Wagner, 2026)."* Two Axes 원문에는 두 결과가 다 있다. 중립적으로 물으면 거의 다 "멀쩡"(0.93 vs 0.84), 지적하라고 지시하면 멀쩡한 질문 57–78%까지 지적. Well은 후자를, 우리 07은 전자를 기록했던 것. 둘 다 맞고, "판정을 시키면 framing에 따라 양방향으로 쏠린다"는 이제 한 논문으로 근거가 닫힌다. Well-Actually 본문에 probe·hidden state·라우팅은 없다 — Table 8–10의 빈 행 확인. 세부는 [07 A1](07_related_work_2026.md), [10 §2](10_evidence_and_baselines.md).

추가로 확인된 것: Well의 "TPQ 0"은 S1(참 전제를 잘못 고치려 함)이 93–100%라는 뜻이라 실질적으로 "거의 전부 거짓이라 했다"가 맞다(앞선 답변 정정). Two Axes 라우팅 파일럿은 48토큰 greedy 생성이라 교정 품질 47–61%를 긴 답변과 직접 비교할 수 없다. Two Axes가 든 nearest prior **Lavi et al. (2026)**(CREPE에서 abstention 방향 steering)은 아직 미확인.

### Two Axes 라우팅은 TPQ를 지켰는가
측정했다: "sound contested" = TPQ 오교정률이고 probe-gated에서 14% (Llama) / 16% (Qwen). 무작위 선택(같은 예산 35%) 26/20 대 probe 42/14이므로 **고르는 것의 가치는 확인**(두 수치가 다 개선). 통제군은 무작위 정상 질문이라 NFP보다 쉽다. **(2026-09-09 삭제)** 이전 판의 "42% × 교정 정확도 55%를 Well 가중식에 넣으면 Direct QA에 진다"는 계산은 뺐다. Two Axes의 47%/61%는 거짓 전제 문항 전체가 분모라 게이트 탐지율에 곱할 조건부 성공률이 아니다(리뷰 3). probe 라우팅만으로 엄격한 허용폭을 맞출 수 있는지는 우리 데이터와 루브릭으로 직접 잰다.

### 2026-09-08 발송 이메일의 질문
"내부 신호로 골라 개입하는 방식은 이미 여러 논문에 있다. 이걸 Cancer-Myth에 적용해 아직 아무도 못 낸 'FPQ를 올리면서 TPQ를 지키는' 결과를 내는 것으로 논문이 되는지, 아니면 방법론 자체가 새로워야 하는지." 전문은 [14](14_advisor_report_2.md).

### 목요일 준비물
1. Table 1·2·3 뼈대 한 장 ([13](13_task_spec.md)).
2. Figure 1 개념도 (x = TPQ 보존, y = PCR, 기존 방법 점들과 빈 오른쪽 위).
3. 선행연구 한 줄씩: Well-Actually, Two Axes, Tripathi, CAST, Contextual-Truth, Pandey.
4. 결정 요청 목록: (a) 기여 성격 / 방법 변경 필요 여부, (b) anchor 표 (Cancer-Myth Table 1 vs Well Table 8), (c) NFP 허용폭 ε, (d) 모델 범위 (7–8B 둘 먼저), (e) 평가 단위 (585 전체 cross-fitting vs 잠근 100/100).
5. 답이 나오면 이 문서 §9 미결정 항목을 채우고 13의 실행 순서로 넘어간다.

### 설명용 문장 (교수님·발표용)
- WildChat 13%: "실제 사용자 질문 중 거짓 전제는 약 13%뿐이라, 총점 = 거짓 전제 질문 점수 × 0.13 + 정상 질문 점수 × 0.87. 기존 방법은 13%에서 벌고 87%에서 잃어 바닐라가 1등."
- Two Axes 47–61%: "challenge 출력에서 전체 FP 문항 중 NLI 검증 교정 비율이다. 선택된 문항 안의 조건부 성공률이 아니므로 gate 탐지율과 곱하지 않는다. 의료 데이터의 교정 품질은 직접 측정한다."
- Contextual-Truth: "사용자가 거짓 주장을 하면 모델이 동조하는데, 속으로는 거짓이라 판단하면서 말만 맞춰주는 경우가 대부분(77%). 그 거짓 주장을 자기 말로 되풀이하고 나면 내부 판단까지 참으로 바뀐 경우가 59%. 전제를 말로 꺼내 검토시키는 방식이 오염일 수 있다."
- Pandey: "'사용자 주장이 틀렸다' 신호를 나르는 head들의 출력을 0으로 만들면 동조가 28→81%인데 사실 판정 정확도는 그대로. 지식은 남고, 그 지식을 앞세울지 정하는 회로만 꺼진 것."
- MedMisBench: "잘못된 단서가 사고 과정에는 81–98% 등장하지만 최종 답을 바꾸는 비율은 설정에 따라 7–90%. 보는 것과 반영하는 것이 다르다."

### 논문용 표 요약 (2026-09-09 갱신; 실행 명세는 13, 원고 구성은 17)

**Table 1.** Detecting false-premise questions (Cancer-Myth FPQ vs. NFP/TPQ, AUROC, out-of-fold). 행: Bag-of-words / Premise-check elicitation / Extract-and-verify / CoT monitor / Hidden readout (logistic) / Difference-of-means / Text + hidden. 열: Qwen2.5-7B, Llama-3.1-8B.

**Table 2.** Main results on Cancer-Myth (585 FPQ, 150 NFP/TPQ) and general medical QA. Judge GPT-4o. 행: Plain / FP Identification / GEPA (FPQ+TPQ) / Balanced premise-check CoT prompting / Unconditional C steering / CAST / Hidden gate + FP prompt (Two Axes-inspired) / **Hidden gate + C steering (ours)**; 선 아래 *GPT-4o Plain, GEPA (reported)*. 열: PCR, PCS, NFP, TPQ 오교정률 (Well 루브릭의 변환 경계는 사전 지정), MedQA, PubMedQA, Medbullets. 모델당 블록. (리뷰 3: CoT 행 추가, CAA → Unconditional, 무작위는 Table 3으로)

**Table 3.** 구성 요소의 세 통제 비교: (A) 같은 C에서 항상/무작위/내부 선택, (B) 같은 C·선택 예산에서 텍스트/CoT/내부 신호, (C) 같은 문항별 mask에서 FP prompt/C steering. 기준 절대값과 ΔPCR·ΔPCS·ΔNFP·ΔTPQ 및 CI를 보고한다. Table 2의 방법별 조정 결과와 구분하고 QA 열을 반복하지 않는다. 같은 설정의 셀만 참조·재사용하며, 이미 본 표에 세 통제가 있으면 독립 표로 분리하지 않는다. oracle은 부록 진단이다. [13](13_task_spec.md) · [17 §6.3](17_manuscript_storyline.md)

**Figure 1.** TPQ 보존(x) vs. PCR(y). Table 2의 완성 시스템을 표시한다. Table 3의 배치 예산 진단을 추가하면 별도 기호·패널로 구분한다. Table 1 판별기에는 좌표가 없다. 점선 = Plain TPQ 보존율 − ε_TPQ. Well의 시각화 취지를 참고하지만 원문의 1–5점 평균 축과 동일 지표로 부르지 않는다.

주요 차이와 보존 판정의 CI는 본 결과와 함께 제시한다. 부록에는 상세 운영점·발화율·harm/rescue, Well 프로토콜 평가, 사전 지정한 추가 데이터, A×C 분포·cos(A,C)를 둔다.

### 리뷰 3 반영 (2026-09-09, [reviews/2026-09-09-review-3.md](reviews/2026-09-09-review-3.md))
- Task 3는 보존 평가. Task와 contribution을 일대일로 맞추지 않는다. 표 번호 유지, 서론에서 Table 2를 먼저 소개.
- "Task 1의 1등이 Task 2의 게이트"를 철회. 텍스트·CoT·내부 게이트를 Task 2에 모두 유지(Table 3). 게이트 종류 선택도 내부 개발 분할에서만.
- Table 2에 균형 전제 검토 CoT prompting 행 추가. "CAA" → "Unconditional C steering". "Hidden gate + FP prompt"는 Two Axes 착안 표기. 무작위는 Table 3에만.
- 과한 해석 교체: 짝 응답 대조는 "질문 차이 교란을 줄인다"까지, 교정 효과는 개입으로 검증. 텍스트 기준선은 "같은 입력 범위·분할에서 추가 판별 정보 평가". Two Axes 곱셈 계산 삭제. "내부 모니터가 필요하다" → "이점을 검증한다".
- 판정 규칙: ε_NFP·ε_TPQ 분리, 비열등성은 CI로 판정, 코드 수정 선행. QA 최종 정책 규칙은 아래 §12와 13에서 보완했다. 새 probe의 문턱을 fold 문턱의 단순 평균으로 정하지 않는다.
- Figure 1은 완성 시스템만, 경계는 Plain TPQ − ε_TPQ.
- 한 문장 주장: **의료 질문의 교정 필요성을 판별해 선택적으로 개입함으로써, 정상 질문과 일반 의료 QA의 성능 손실을 제한하면서 거짓 전제 교정 성능을 개선한다.** 내부 신호가 필요한지, 활성값 개입이 프롬프트보다 유리한지는 하위 가설.

## 12. Table 2·3의 구분과 전체 논문 구성 (2026-09-09 후속 논의)

- **Table 1은 판별 정보, Table 2는 최종 효용, Table 3은 부품의 효과**를 답한다. Task 3는 의료 QA 보존 평가이며 세 task와 세 contribution을 일대일로 맞추지 않는다.
- Table 3은 전체 격자를 다시 그리지 않고 선택 여부·선택 신호·개입 방식의 세 통제 비교로 줄인다. 같은 C와 저장한 동일 mask를 사용한다. 모든 통제가 Table 2에 있으면 분석 블록으로 합친다. 대화에서 사용한 100/100·75개 선택 예시의 숫자는 설명용 가상 값이며 실험 결과가 아니다.
- Table 2는 방법별로 공통 개발·보존 규칙 아래 조정한다. Table 3의 정확한 선택 예산 통제는 dev에서 잠근 비율에 따른 라벨 없는 상위 K 배치 진단이다. 고정 문턱 정책의 배포 성능과 구분한다. 실제 비용은 CoT 생성까지 측정한다.
- QA용 최종 train/calibration 그룹 분할·inner-dev 결과 집계 규칙을 사전에 정한다. 새 모델의 문턱·강도는 최종 calibration에서 고정하며 QA test와 outer 평가 성능으로 선택하지 않는다. Cancer-Myth OOF 수치를 재학습한 문항의 점수로 대체하지 않는다.
- [17](17_manuscript_storyline.md)에 Introduction 다섯 문단, Related Work 네 묶음, Methodology·Setup, 표별 결과 해석, Discussion·Conclusion을 작성했다. [09](09_research_proposal.md)와 [13](13_task_spec.md)을 함께 갱신했다.
- E1 탐색 결과 이후 정교화된 계획이라는 점을 공개한다. 모든 설계가 데이터 관찰 전에 사전 등록됐다는 표현은 사용하지 않는다. 방향의 판별력과 실제 제어 효과, TPQ 오교정과 설명 전체의 정확성도 구분한다.

## 2026-09-09 후속 결정 — 지표·baseline·기여 구분

- NFP/TPQ는 같은 150개 정상 질문이다. 본 Table 2·3은 공식 NFP만 유지하고 Well 원 FPQ/TPQ 점수는 부록으로 이동한다. 미확정 TPQ 오교정률과 ε_TPQ는 주 정책에서 제외한다.
- Table 1은 Two Axes의 BoW·직접 질문·학습된 출력·내부 DiM·내부 logistic을 계승하고 전제 추출/검증·CoT monitor를 추가한 일곱 행이다. 텍스트+hidden은 부록. 기존 logistic과 별도의 Ours 탐지 행을 만들지 않는다.
- Table 2는 아홉 방법. Extract+FactCheck와 의료 재학습 Gated head를 본 비교에 포함하고 Unconditional C를 Table 3 U행으로 이동한다. 동일 의료 train/dev와 판정 조건의 adaptation이며 재현 완료는 아니다.
- 개입률은 전체 중 교정 분기를 실행한 비율이다. TPR·FPR·교정 성공률과 다르며 Table 3의 K는 개입량 통제다.
- Two Axes는 새 probe 공식보다 두 축 분석·위험별 정책, Gated는 기존 ITI·probe·평균 차를 행동별 제어·head ablation·보존 확인과 결합한 구성이 중심이다. Gated의 MedQuAD 강도 선택 중 확인과 정상 보존 평가도 인정한다.
- 두 방법 모두 의료 데이터에 적용할 수 있다. 입력·목표·모델별 재학습 필요성과 방법의 실패를 구분한다. 현재 우리 계획은 기존 조건부 기법의 적용·분석이며, 세 표 자체가 세 독립 기여를 보장하지 않는다.

상세 근거와 교수님 설명은 [20](20_baseline_transfer_and_novelty.md). 과거 결정과 충돌하면 이 기록 및 13/19의 현재 명세를 우선한다. 이번 변경은 문서만 수정했고 학습·본 실험·코드 구현을 완료한 것은 아니다.
