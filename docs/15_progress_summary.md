# 15. 진행 현황 정리 (2026-09-15 갱신, 과거 기록 포함)

**교수님 보고 준비:** [28](28_advisor_progress_narrative.md)에 평가가 애매했던 실제 문장·변경 이유·기각/보류한 대안·text 대비 hidden 필요성의 조건을 연결했다. 최종 Qwen 답변 평가는 아직 대기다.

> **2026-09-15 갱신:** 현재 통합 현황은 [26 — 실험·의사결정 이력](26_experiment_history_and_decisions.md), 방법 후보는 [27](27_method_candidates_and_tests.md)를 우선한다.
> Gemma Quick45는 180/180 판정 완료, overnight는 1,144/1,144 작업 완료(49.4분, GPT 0회)다.
> Qwen 질문 판별은 text/hidden/DiM AUROC .756/.638/.731. 최종 답변 512 제한을 fit 12개에서 점검한 뒤 별도 1024 suite로 전환했다.
> 마지막 사용자 생성 로그는 extract_verify 319/732이며 최종 Well 점수·일반 의료 QA·선택적 C steering 효과는 아직 확보하지 않았다.
> 기존 결과·평가 버전을 보존하며 아래 과거 “미확보/다음 단계”를 현재 상태로 읽지 않는다. 다음 독립 GPU 작업부터 물리 0·1 병렬을 기본으로 한다.

> **최신 실행 상태:** [23](23_gemma_steering_pilot.md)의 Gemma-2-9B-it baseline과 L21 gate 파일럿을 실행했다.
> Plain/FP Identification/CoT 답변을 생성했고 FPQ는 Sol, NFP는 Terra v2 판정을 사용한다.
> 첫 hidden gate × FP Identification의 PCR/NFP는 11.1/100으로 CoT 34.2/100을 넘지 못했다.
> 이번 gate는 저장 답변의 프롬프트 라우팅이며 조건부 activation steering은 아직 실행하지 않았다.
> [결과·정정·오프라인 후속 명령](reviews/gate_l21_first_results_2026-09-14.md). 새 oracle/곡선/C 진단 수치는 아직 미확보다.
> 평가상의 모호함과 dev 반복 사용 한계가 남는다. 방법론은 미정이며 장기 Table 3는 CREPE 전이 방향이다.
> 아래 이전 계획의 모델 순서·표 번호·미결 상태보다 이 갱신과 23을 우선한다.

문서 세션(2026-09-07–10)과 실험 세션의 작업을 한 곳에 모은 것. 세부는 각 문서 링크. 이 문서는 "지금 어디까지 왔고 무엇이 열려 있나"만 답한다.

**2026-09-10 공개 RAG 감사 완료.** Well 코드 revision과 TPQ 파일을 내려받아 확인했다. TPQ 148문항·전제 주석 178개, 147문항에 passages가 있다. 당시 Top-4·명시적 split·최종 응답은 공개 파일에서 확인되지 않았다. 원 수치 복원은 미완료이며 embedding·생성·judge 실행은 하지 않았다. [재사용 범위·새 Table 3](21_well_rag_reproducibility.md). Table 1은 Raw·P(IK)·P(True)를 복원한 10행, Table 2는 두 Hidden 행의 gate를 공유한다.

## 1. 한 줄 현황

- **연구 방향은 정해졌다.** 모델 내부 표상으로 "이 질문에 잘못된 전제가 있는가"를 읽어, 넘는 질문에만 교정 개입을 걸고, 정상 질문의 손실을 사전 허용폭 안에 두면서 Cancer-Myth 교정률을 올릴 수 있는지 검증한다. 새 알고리즘이 아니라 조건부 개입(CAST·Two Axes·Tripathi)의 적용·요인 분석이다. ([09](09_research_proposal.md), [14](14_advisor_report_2.md))
- **교수님께 방향 확인 이메일을 보냈고** (2026-09-08), 목요일 미팅에서 "이 적용으로 논문이 되는지, 방법이 새로워야 하는지"를 결정한다. ([14](14_advisor_report_2.md), [12 §11](12_discussion_decisions.md))
- **E1(표상 읽기)은 네 모델에서 한 바퀴 돌았다.** 전제 구간 probe AUROC 0.80–0.82, 그러나 텍스트만 쓴 TF-IDF가 0.77이라 "내부가 텍스트보다 낫다"는 아직 미확인. 최소 짝(참 전제 쌍둥이) 통제가 필수가 됐다. ([experiments/01](experiments/01-e1-prediagnostic.md))

## 2. 결정된 것

| 항목 | 결정 | 근거 문서 |
|---|---|---|
| 중심 문장 | 고정 held-out 평가에서 정상 질문의 추가 오교정을 사전 허용폭 안으로 제한하면서 거짓 전제 교정률을 높일 수 있는지 검증. 내부 신호의 라우팅 가치와 활성값 개입의 추가 효과를 분리해 측정 | 09 §1, README |
| 가설 | RH1 내부 신호의 추가 가치 / RH2 선택적 개입의 효과 (+RH2b steering vs prompt) / RH3 일반 의료 QA 보존 | 09 §3 |
| Task ↔ 표 | Task 1 → Table 1 (판별) / Task 2 → Table 2 (최종 효용) + 부록 A1 (같은 C·K 선택 통제); Table 3는 Task 1의 주석 전제 진단 후보 / Task 3 → Table 2의 QA 열. 같은 결과의 반복을 독립 증거로 세지 않음 | 13, 17 §1 |
| 본 표 행 | Plain, FP Identification, Extract+FactCheck, GEPA, 균형 CoT, CAST, Gated head 의료 adaptation, Hidden+prompt, Hidden+C. Unconditional C·무작위는 부록 A1, 출판 수치는 별도 참고 | 13 §3 |
| 평가 단위 | 585 + 150 전체, nested grouped 5-fold cross-fitting. Well 잠근 분할은 ID 확보 시 부록 | 13 §0, 12 §5 |
| 판정기 | 본 표 GPT-4o의 PCR/PCS·공식 NFP. 같은 정상 질문의 중복 열 제거. Table 3는 Well 사실 확인 정확도, 최종 응답 S5·분포는 부록 | 13 §0, 10 §2 |
| 모델 | Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct 먼저. Gemma-2-27B는 E2 1순위(E1 결과) | 13 §0, experiments/01 |
| C 방향 | 주 실행안은 fit FPQ의 교정/비교정 참조 응답 첫 32토큰 평균 차. 후보·층·강도는 dev, 최종 test 고정. 다른 방향은 부록이며 과거 E1 관찰과 구분 | 19, 20 |
| 용어 | gate → 개입 조건 / 선택적 개입. 분류기 = linear probe(hidden), text classifier(BoW) | 12 §11 |
| 주장하지 않는 것 | 모델은 안다, CoT는 안 된다, 조건·행동 분리 최초, 의료 gated steering 최초, 설계상 NFP 보존, TPQ 1점 손실이면 진다, probe 안 읽히면 지식 없음, oracle이 상한 | 12 §8, 09 |

## 3. 근거 확인 상태

| 논문 | 확인 수준 | 핵심 사실 |
|---|---|---|
| Cancer-Myth | 저장소 코드 + 논문 | Table 1은 닫힌 모델. GPT-4o 판정기. NFP 150 |
| Well-Actually | **PDF 전문 + 저장소 코드** | 여러 전제 교정 방법에서 FPQ–TPQ 상충을 관찰. TPQ 루브릭은 정답 정확도와 다름. 파인튜닝은 Cancer-Myth FPQ + ARC-DA TPQ. FAITH는 Yuan et al.이 보고한 영화 head. 가중식 0.13/0.87. Two Axes를 한 문장으로만 인용, probe·라우팅 없음. 한계 절에 "의료는 FPQ 비율이 더 높을 수 있다" ([10 §2](10_evidence_and_baselines.md)) |
| Two Axes | **PDF 전문, 분모 재확인** | CREPE의 probe 라우팅 효과. 47%/61%는 challenge 출력에서 전체 FP 문항 중 NLI 검증 교정 비율이며 gated 조건부 성공률이 아님. 탐지율과 곱한 가중 비교는 철회. 의료 전제·activation 개입의 추가 효과는 직접 평가 ([12 §11](12_discussion_decisions.md), [17 §3](17_manuscript_storyline.md)) |
| Lavi et al. (EACL 2026) | **PDF 전문** | 추출형 QA 답변 불가능성. 방향 선택을 steering score로. CREPE는 분류 전이만(F1 59–62). 인과 개입은 기권 제어 ([07 B3-1](07_related_work_2026.md)) |
| Tripathi | 원문 + HF runtime (codex 세션) | 내부 probe gate + head steering, EHR 멀티턴 압력 ([11](11_tripathi_review.md)) |
| CAST | 원문 (codex 세션) | 조건 벡터·행동 벡터 분리 |
| Verbalizing-Assumptions, Contextual-Truth, Pandey, MedMisBench | 이전 세션 본문 | 기존 해법 아님(전자), 분석만(나머지) |

## 4. E1 중간 결과가 설계에 주는 영향 (experiments/01, 판정기 codex)

| 관찰 | 값 | 설계 변화 |
|---|---|---|
| Plain PCR이 매우 낮다 | Llama 3.4%, Qwen 5.6%, Gemma-9B 8.4%, Gemma-27B ≈ 9.4% (codex 판정, GPT-4o보다 엄격) | 자연 응답으로 C를 만들 재료(+1)가 20–55개뿐 → C 후보 (iii) 참조 답변 짝 대조 추가 |
| A는 읽히지만 약하다 | 전제 구간 probe 0.80–0.82, 질문 끝·마지막 토큰 0.70–0.74 | Two Axes CREPE 대역과 같음 |
| **텍스트만으로도 갈린다** | 질문 전체 TF-IDF 0.77 > B/D probe; 전제 구간 TF-IDF 0.73 < A probe 0.80 (CI 겹침) | RH1 위험 현실화. fpq/NFP는 출처가 달라 문체가 다름. **최소 짝(전제 구간만 참으로 바꾼 쌍둥이) 통제가 필수** (stage 9). Table 1의 BoW 행은 이미 그 자리 |
| 생성 전 D에서 "고칠지"가 읽힌다 | 0.82–0.88 | C의 신호는 있음. 양성 적어 폭 큼 |
| cos(A, C) | 0.5–0.6 (27B L22–35) | 기하적 유사도 진단. 인과적 분해의 증명은 아님. A 성분을 뺀 C는 별도 개발·개입 검증 후보 |
| 응답 첫 5토큰은 상투구 | E 위치 0.62–0.68 | 32토큰 평균 병행 |
| 27B가 E2 1순위 | +1 55개, D 0.86–0.88, L26–30 정점 | E2 층 L28 |

이 결과는 **RH1이 텍스트 분류기에 질 수 있다**는 리뷰 2·09의 경고가 실제 데이터에서 나타난 것이다. 쌍둥이 통제에서 probe가 텍스트를 뚜렷이 넘지 못하면 기여는 "내부 신호"에서 "선택적 개입 자체"로 옮겨간다 (12 §11의 "약한 경우").

## 5. 미결 (목요일 이후)

1. 기여의 성격: 적용·분석 연구로 가는지, 방법 변경을 넣는지. → 2026-09-10 답: 본체는 적용·분석, 방법 변형은 과교정 방향 직교화 C 하나를 사전 등록 ([22](22_method_variants_and_arr_plan.md) §1·§4).
2. anchor 표: Cancer-Myth Table 1 형식 vs Well Table 8 형식. 교수님 원 지침("논문 하나 따라가고 내 행 추가")에는 Well이 더 맞을 수 있음 (12 §11).
3. NFP/TPQ 허용폭 ε_normal, 의료 QA 허용폭 ε_QA 수치.
4. 데이터 범위: Cancer-Myth만인지, Well의 CREPE·QA²·Syn-QA²를 부 벤치마크로 붙이는지 (권고: 붙인다).
5. Well 분할 ID 확보 여부.
6. 원문 미확인: Cancer-Myth App. C.1.1의 76건 판정기 일치, 원본 GEPA 5/5 설정, Well App. A 분할 개수.

## 6. 문서 지도

| 번호 | 역할 | 상태 |
|---|---|---|
| 00–06 | 초기 노트 (선정, 앵커 정리, 후속 4편, 가설, verbalizer, 도구, 실험 계획) | 리뷰 반영 수정 완료. 09 이후 문서가 우선 |
| 07 | 인용 그래프 밖 관련 연구, 직접 경쟁자 | Two Axes·Lavi PDF 확인 반영 |
| 08 | 논문 서사 | 리뷰 반영. 09가 우선 |
| **09** | 연구계획: 주장·가설·Task·표·기여 | 최상위 |
| 10 | 근거와 baseline, "말하는 것/안 하는 것" | Well PDF 확인 반영 |
| 11 | Tripathi 원문·코드 검토 | |
| 12 | 논의 결정 기록, 철회 목록, 미결, 용어, 논문용 표 | §11에 문서 세션 논의 |
| **13** | Task 명세와 표 뼈대, 출처, cross-fitting | 실행 기준 |
| 14 | 교수님 보고 2 (발송본) | |
| 15 | 이 문서 | |
| 16 | 코드 지도 | |
| **17** | 논문 절별 스토리라인·문단 초안·표 해석·결론의 범위 | 현재 원고 작성 기준. 결과는 미확정 |
| 18–21 | 발표 원고, 방법 프로토콜, baseline 전이·독창성, Well RAG 재현성 | 다른 세션, 2026-09-10 |
| 22 | 방법 변형 후보, 모의 ARR 패널, 10월 12일 제출 판단, 문서 간 모순 | 2026-09-11 전면 개정 |
| experiments/01 | E1 설계·중간 결과 | 실험 세션이 갱신 |
| reviews/ | 리뷰 1 (코드+연구), 리뷰 2 (표·프로토콜) | 대응 상태는 16 §6 |

## 7. 브랜치

`main` = `claude/document-session-o2temh` (문서) ⊇ `claude/cancer-myth-internals-context-cccnkd` (실험) ⊇ `codex/*` (리뷰·논의). 실험 세션은 실험 브랜치에서 계속 작업하고 문서 세션이 주기적으로 main에 합친다. 실험 브랜치에는 12·13·14·reviews가 없으므로 실험 세션이 main을 한 번 받아야 한다.

## 8. 다음 순서

1. 목요일 미팅 → §5 미결 채우기 → 12 §9 갱신.
2. 실험 세션: E0 재정렬 → stage 9 쌍둥이 → A probe 대 텍스트 재측정 (RH1의 첫 답).
3. 코드 리뷰 잔여 항목(16 §6)과 13 대비 빠진 부품(16 §7) 구현.
4. E2 탐색 설정과 최종 평가를 구분하고, Table 2의 공유 gate prompt/C와 부록 A1 선택 통제를 실행. Well Table 1은 선행 근거로 소개하고 추가 사실 확인 진단의 필요성을 정한 뒤 Table 3 채택 여부를 결정.
5. 미측정 Table 1·2를 채우고 출판값과 새 결과를 분리. Table 3 후보는 최종 응답 S5가 아니라 주석 전제의 진위 판정 정확도다.
6. 10월 12일 ARR 제출. 31일은 측정 타당성에 쓴다. ε 고정 → PCR 분해 → paired 검정 순 (22 §9). 결정일 9월 15일·9월 25일.
