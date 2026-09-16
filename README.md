# Cancer-Myth Internals

**교수님 보고용 연결 원고:** [28 — 실제 애매한 문장, 평가 변경, 실패 실험, 선행 재검토, text/hidden의 역할](docs/28_advisor_progress_narrative.md).
당시 기대·관측·보류한 대안·현재 판단을 함께 기록했다. Qwen 최종 답변 수치는 평가 완료 후 채울 빈 표로 남겼다.

**2026-09-15 결정: 10월 12일 ARR 제출 확정, 분석 + 방법 한 개(효용 라벨·상보적 선택).** 범위·일정은 [12 §17](docs/12_discussion_decisions.md).

**현재까지 무엇을 했고 왜 바꿨는가 (2026-09-15):** [26 — 실험 수치·발견·중단·전환 이력](docs/26_experiment_history_and_decisions.md).
Gemma baseline/판정 감사/gate/Quick45/overnight와 Qwen 판별·512→1024 전환을 구분했다.
사용자 마지막 로그는 `extract_verify 319/732`이며 실시간 서버 상태 확인은 아니다.
**문체 교란 검정 (2026-09-16):** text 0.755 > hidden 0.693이 출처 문체인지 전제인지 — 하한선·의역·쌍둥이 세 검정과 판독 기준은 [29](docs/29_style_confound_tests.md).

**새 방법론 논의:** [27 — 확인된 대조쌍·학습 C·효과 예측·쌍둥이 증폭·선행과 반증 조건](docs/27_method_candidates_and_tests.md).
후보는 미실행이며 신규성·성능을 확보했다고 주장하지 않는다. 다음 독립 GPU 실험은 물리 0·1 병렬이 기본이다.
아래 과거 계획과 충돌하는 현재 상태·판정기·표 역할은 26/27과 실행 명세 25를 우선한다.

**다음 실행 — 판별·생성 baseline 분리 (2026-09-15):** [25 — 공통 질문, Qwen/Gemma 실행, Well 평가, grouped holdout/OOF](docs/25_baseline_suite.md).
`scripts/run_baselines.sh`의 `prepare → gpu → gate`는 외부 GPT 채점 없이 실행한다.
답변 평가는 별도 `judge-plan → judge → report`이며 새 호출 상한을 명시해야 한다.
Well의 당시 split은 공개되지 않아 우리 manifest를 사용한다. GEPA/finetuning은 이번 자동 실행에 포함하지 않는다.

**앞선 최대 9시간 C 방향 진단:** [GPU 0·1 overnight 실행·실험표·아침 보고서](docs/experiments/overnight_c_diagnostics.md).
`bash scripts/run_overnight_c.sh start`로 SSH와 독립 실행한다. 기존 C의 강도·층·위치, 무작위 방향,
fit 전용 새 방향, 소규모 생성 답변을 비교한다. GPT 채점 0회, test 미사용. 개별 실패·시간 초과도 보고한다.

**앞선 45문항 비교:** [원본 평가 프롬프트 + Terra, CoT 1024와 L21 C steering](docs/reviews/quick_next_experiment_2026-09-14.md).
기존 코드와 같으면 3조건 답변을 재사용하고 2조건만 생성한다. 코드가 달라졌으면 같은 45문항에서
5조건 모두 현재 코드로 생성한다. 최대 225회 채점, 자동 재시도 없음, test 미사용.
생성 완료 후 채점 재개: `python scripts/quick_pilot_readout.py resume --out-dir "$QUICK_DIR"`.
한 줄 JSON 3건은 재호출 없이 복구한다. 신규 준비·생성 명령은 연결 문서를 따른다.

**바로 실행:** [23 §0 서버 명령](docs/23_gemma_steering_pilot.md#0-바로-실행하기--우선-baseline-세-행).

**2026-09-14 CoT 성공 답변 감사:** 기존 +1 FPQ 40개의 [19/3/18 검토 초안·원문·SHA](docs/reviews/cot_positive40_2026-09-14/summary.md)와
[전제 귀속·세 방법 비교·추가 의학적 오류 감사 절차](docs/reviews/fpq_attribution_audit_protocol_2026-09-14.md)를 보존했다.
19개만 정답이라는 뜻이 아니며 기존 PCR 34.2%는 유지한다. 다음 단계는 질문·참조 확인과
동일 질문의 세 방법 비교다. 새 GPT 채점·GPU 생성·test 평가는 하지 않았다.

**외부 AI 검토 파일:** [18문항 공유 순서와 회신 점검 절차](docs/reviews/external_ai_attribution_18_v1/README.md).
[질문만 보는 1단계](docs/reviews/external_ai_attribution_18_v1/share/01_question_only.md)를 먼저 보내고,
응답을 저장한 뒤 [참조를 대조하는 2단계](docs/reviews/external_ai_attribution_18_v1/share/02_reference_comparison.md)를 보낸다.
사용자에게 임상 판정을 요구하지 않으며, AI 간 합의와 의학적 검증은 구분한다.

**외부 AI 회신 대조 완료:** [18개 대조·15개 1차 자료·내 기존 메모 정정](docs/reviews/external_ai_attribution_18_v1/followup_2026-09-14/report.md).
stage1 SHA는 일치하며 원 응답은 보존했다. 외부의 5/13/0을 확정 라벨로 채택하지 않고,
귀속·의학 근거·자료의 적용 범위를 분리했다. 새 PCR·oracle은 계산하지 않았다.

**2026-09-14 gate 첫 결과:** L21 hidden gate × FP Identification은 PCR/NFP 11.1/100,
text gate는 16.2/96.7, CoT는 34.2/100이었다. 이는 프롬프트 라우팅 파일럿이며 activation steering이 아니다.
[원 수치·해석·GPU/GPT 없이 oracle/구제/예산 곡선을 확인하는 명령](docs/reviews/gate_l21_first_results_2026-09-14.md).
`prepare → baselines → report-baselines`로 첫 dev 표를 만든 뒤 `fit → sweep → report`로 steering을 비교한다.
[24 — MISP-Bench와의 관계](docs/24_misp_relation.md): CoT/검증 지시 근거, 입력 요소별 외부 평가 후보, 현재 파일럿과의 구분.

> **현재 코드 실행 순서 (2026-09-10): [23 — Gemma baseline·기본 steering 파일럿](docs/23_gemma_steering_pilot.md).** 교수님 지시에 따라 Gemma-2-9B-it 한 모델에서 Plain → FP Identification 적용 → 전제 검토 CoT → 기본 paired-C steering을 비교한다. `scripts/run_gemma_pilot.sh`로 prepare/baselines/fit/sweep/select/test/report를 실행한다. fit/dev/test와 fit 전용 방향·강도 기준값을 고정했다. 방법론은 미정이며 SAE·직교화·27B 격자는 확정하지 않는다. 아래 이전 계획과 충돌하면 23의 파일럿 범위가 우선한다. 장기 Table 3의 최신 방향은 **Cancer-Myth→CREPE 전이 평가**이고, 아래 Well 사실 확인 표는 선행연구 참고 자료다. 구현 검증과 실제 9B 성능 측정은 구분한다.

> **2026-09-10 교수님 발표 준비:** [18 — 슬라이드별 상세 발표 원고](docs/18_professor_presentation_script.md). Introduction의 단계별 근거와 마지막 contribution 세 개, Related Work, Method, 결과표 개요를 23장으로 구성했다. Two Axes·Gated Activation Steering은 각각 2장, Method는 4장이다. 정상 질문은 NFP와 Well TPQ를 합친 300개가 아니라 **동일한 150개**이며, FPQ를 포함한 명목상 전체 질문 수는 **735개**다.

> **질문했던 설명도 같은 원고 안에서 읽기:** 18의 각 장에 환자 질문부터 판정·개입까지의 상세 노트를 넣었다. [출력 readout](docs/18_professor_presentation_script.md#learned-output-readout), [CoT와 별도 모니터](docs/18_professor_presentation_script.md#cot-monitor-details), [TPR·FPR·AUROC·문턱](docs/18_professor_presentation_script.md#threshold-details), [아홉 교정 방법](docs/18_professor_presentation_script.md#table2-method-details), [RAG Table 3](docs/18_professor_presentation_script.md#table3-details), [같은 선택·개입률](docs/18_professor_presentation_script.md#selection-control-details)을 예시와 함께 설명한다. 문서 앞부분의 질문별 바로가기를 이용할 수 있다.

> **Method 14–17장 상세화:** [질문 벡터와 학습 자료](docs/18_professor_presentation_script.md#method-input) → [probe 목적식·층·문턱](docs/18_professor_presentation_script.md#method-probe) → [응답 쌍의 교정 방향](docs/18_professor_presentation_script.md#method-direction) → [새 질문의 steering](docs/18_professor_presentation_script.md#method-inference). 각 장에 화면 구성, 단계별 발표 원고, 계산 예시, 저장 값, 설정 근거와 구현 상태를 넣었다. 예시 수치는 실제 결과가 아니다.

> **현재 표 정정:** **현재 위치와 확정 범위.** 가져올 선행표는 Well-Actually Table 1로 확정한다. 이 표는 교수님 발표의 기존 연구 근거로 사용할 수 있다. 아래의 우리 Table 3 및 추가 probe 행은 사실 확인 진단을 독립 연구 질문으로 채택할 경우의 후보이며 본 실험으로 확정한 것은 아니다. 기존 결과를 인용하는 것만으로 우리 실험 결과표가 되지 않는다. 현재 질문 gate의 검증이 목적이면 Table 1의 질문 단위 탐지 비교에 RAG 기반 전제 추출·검증을 추가하는 방안도 가능하며, gold 전제를 받는 원 Table 1의 숫자와 직접 섞지 않는다.

> **2026-09-10 현재 연구계획:** 논문 전체의 이야기는 [17 — Introduction부터 Conclusion까지](docs/17_manuscript_storyline.md), 주장·가설은 [09](docs/09_research_proposal.md), 실행 명세는 [13](docs/13_task_spec.md)을 읽는다. Table 1은 판별 신호, Table 2는 최종 교정·보존 성능과 일반 의료 QA ACC 세 열을 합친 본 결과표, Table 3는 Well 본문 Table 1을 계승한 주석 전제의 사실 확인 정확도 비교다. 같은 gate의 prompt/C 비교는 Table 2에 통합하고 같은 C·K의 선택 통제는 부록 A1에 둔다. [10 — 근거](docs/10_evidence_and_baselines.md), [11 — Tripathi 검토](docs/11_tripathi_review.md), [12 — 결정 기록](docs/12_discussion_decisions.md), [15 — 진행 현황](docs/15_progress_summary.md), [16 — 코드 지도](docs/16_code_overview.md)가 이를 뒷받침한다. 일반 의료 QA 보존은 Task 3이며 독립적인 새 방법 기여가 아니다. 아래 초기 요약과 충돌하면 이 문서들이 우선한다. 계획 갱신은 실험 성공을 뜻하지 않는다.

> **방법 후보 검토 기록:** [22](docs/22_method_variants_and_arr_plan.md)는 과거 후보·일정 초안이다. 직교화, 27B 실행 순서, 코사인 기각 기준은 현재 확정안으로 사용하지 않는다. 당장 실행할 코드는 [23](docs/23_gemma_steering_pilot.md)을 따른다.

의료 LLM이 환자 질문 속 **잘못된 전제(false presupposition)** 를 왜 못 고치는지를 **내부 표상**에서 읽고, 그 신호로 **조건부로** 고치는 연구.

앵커 논문: **Cancer-Myth** (ICLR 2026, Zhu et al., [arXiv 2504.11373](https://arxiv.org/abs/2504.11373)).
목표 산출물: Cancer-Myth 지표(PCR/PCS/NFP, GPT-4o 판정기)로 채점한 **공개 모델 held-out 비교표**. 그 표에서 **정상 질문의 추가 오교정을 사전에 정한 허용폭 안으로 제한하면서 PCR을 올리는 행**을 넣는 것. 원본 Table 1의 닫힌 모델 수치는 평가 범위가 달라 참고 결과로 분리한다 ([06](docs/06_experiment_plan.md#실험-3--held-out-비교표)).

> **중심 문장.** 고정된 held-out 평가에서, 정상 질문의 추가 오교정 위험을 사전에 정한 범위로 제한하면서 거짓 전제 교정률을 높일 수 있는지 검증한다. 그 과정에서 내부 신호의 라우팅 가치와 활성값 개입의 추가 효과를 분리해 측정한다.

> **표·기여 후속 정리:** [20 — 기존 방법의 기여와 의료 적용](docs/20_baseline_transfer_and_novelty.md). Table 1은 Two Axes의 핵심 readout 비교를 의료에서 재평가하고 검증 모니터를 추가한 10행이며 별도 새 Ours 탐지 행은 없다. Table 2는 NFP 주 지표 하나와 QA ACC 세 열, 의료 재학습 Gated·Extract+FactCheck를 포함한 아홉 방법이다. Well 최종 응답 S5는 부록이며 Table 3는 전제 참·거짓 정확도다. Unconditional C 선택 통제도 부록에 둔다. 두 선행 방법 모두 적용 가능하며 재학습 필요성을 실패 증거로 부르지 않는다.

> **RAG를 가져올 수 있는 범위 확인:** [21 — Well GitHub·공개 데이터 감사](docs/21_well_rag_reproducibility.md). 검색·생성·판정 코드와 TPQ 148문항의 파일을 확인했고 147문항에 근거 문서가 있다. 당시 split·Top-4·모델 응답은 확인되지 않아 원 수치 인용과 새 재실행을 구분한다. [버전·SHA256·개수 manifest](docs/audits/well_rag_inventory_2026-09-10.json). 모델 실험은 아직 실행하지 않았다.

## 한 줄 요약

| | |
|---|---|
| 현상 | 환자가 틀린 믿음을 깔고 질문하면 모델은 겉 질문에만 답한다 (프런티어 ≤43%, 공개 모델 ≤17.3% PCR) |
| 기존 완화책의 실패 | GEPA·Monitor는 Cancer-Myth를 올리는 대신 **전제 없는 질문에서 없는 전제를 지어낸다** (NFP −29, −55) |
| 왜 실패하나 | 전제의 진위를 모델에게 말로 판정시키면 framing에 따라 쏠린다. 전제 문장을 뽑아 물으면 참 전제의 58–93%를 거짓이라 하고(Well-Actually), 질문째 물으면 거의 다 멀쩡하다고 한다(Two Axes). 같은 문항에서 출력 readout의 판별력은 내부 readout보다 낮았다(Two Axes, CREPE: 직접 질문 대부분 0.64–0.67 vs probe 0.69–0.78). 이 결과를 의료에서 재평가하고 전제 검증·CoT monitor와 비교한다. 언어 판정의 한계를 모델 지식 부재나 모든 CoT의 실패로 일반화하지 않는다 |
| 우리 방법 | "전제가 거짓인가"를 prefill 표상에서 읽어(A) 게이트로 쓰고, 열렸을 때만 교정/비교정 응답에서 뽑은 대조 방향(C)을 민다. 분리해서 재는 것 둘: (1) A 게이트가 같은 개입 강도·비슷한 개입 비율의 무작위·무조건 개입보다 나은 교정/부작용 균형을 만드는가, (2) 같은 게이트 뒤에서 활성값 개입이 프롬프트 개입보다 나은가. NFP 보존은 게이트 오탐과 개입의 실제 효과를 함께 평가할 **경험적 제약**이다 |
| 선행연구와의 거리 | 사용자 태도(Verbalizing-Assumptions), 전역 동조 벡터(CAA), 단언문 진위(Contextual-Truth), 강제 판정(Well-Actually), CoT 언급(MedMisBench)은 있었다. 일반 도메인에선 probe 게이트 **프롬프트**(Two Axes), 의료에선 내부 probe 게이트 ITI(Tripathi, 공개 코드 확인은 11), 조건 벡터와 행동 벡터를 따로 뽑는 조건부 steering(CAST·GAPS)까지 있다. 우리는 조건부 steering의 **확장**이고, 다른 점은 게이트 신호가 **배경화된 의료 전제의 진위**, 개입 대상이 **교정/비교정 대조 방향**, 통제군이 **어려운 정상 질문 NFP**, 그리고 같은 게이트 아래 **프롬프트 대 활성값** 비교 ([07](docs/07_related_work_2026.md)) |

## 논문 약칭

문서 전체에서 Cancer-Myth, Verbalizing-Assumptions, Well-Actually, MedMisBench, Contextual-Truth, Two Axes, Tripathi, Pandey를 약칭으로 쓴다. 전체 제목과 arXiv 번호는 [docs/02](docs/02_followups_and_gap.md#논문-약칭) 상단 표.

## 문서

| 파일 | 내용 |
|---|---|
| [docs/29_style_confound_tests.md](docs/29_style_confound_tests.md) | 문체 교란 검정 셋(문체 하한선·의역·쌍둥이)의 절차, 누출 규칙, 판독 기준, 같은 오탐 TPR 부록, claude 백엔드 |
| [docs/28_advisor_progress_narrative.md](docs/28_advisor_progress_narrative.md) | 교수님 보고 원고, 원문 예시·판단 변경·미확정 가설·완료 후 채울 수치 표 |
| [docs/26_experiment_history_and_decisions.md](docs/26_experiment_history_and_decisions.md) | 실측 수치, 발견, 중단 이유, 변경 내용, 원본 보고서·출처 수준, 현재 진행 |
| [docs/27_method_candidates_and_tests.md](docs/27_method_candidates_and_tests.md) | 새 방법 후보 전체 지도, 수식·선행·한계, 우선순위와 병렬 의존성 |
| [docs/00_why_this_paper.md](docs/00_why_this_paper.md) | ICLR/ICML 2026 의료 NLP 106편 중 왜 Cancer-Myth인가. 지도 조건, 스크리닝, 탈락 후보 |
| [docs/01_cancer_myth.md](docs/01_cancer_myth.md) | 앵커 논문 정리 — 정의, 7 카테고리, PCR/PCS, NFP, Table 1·3 원본, 저장소 |
| [docs/02_followups_and_gap.md](docs/02_followups_and_gap.md) | 후속 연구 14편 중 핵심 4편이 무엇을 봤고 무엇을 남겼나 |
| [docs/03_hypothesis_map.md](docs/03_hypothesis_map.md) | 내부 원인 가설 H1~H6, 읽을 세 층 A·B·C, 2×2 분해 |
| [docs/04_verbalizers.md](docs/04_verbalizers.md) | verbalizer 바닥부터 — 추출·주입·생성, 네 종류, 함정과 통제 |
| [docs/05_tools_and_models.md](docs/05_tools_and_models.md) | NLA·AO 체크포인트, SAE 스위트, 모델별 배치 |
| [docs/06_experiment_plan.md](docs/06_experiment_plan.md) | 실험 1 설계, 캐시, 통제, baseline 비교표, 갈림길 |
| [docs/07_related_work_2026.md](docs/07_related_work_2026.md) | 인용 그래프 바깥 — 일반 도메인 거짓 전제 QA, sycophancy 회로, 조건부 steering, 직접 경쟁자 3편 원문 검토 |
| [docs/08_story_outline.md](docs/08_story_outline.md) | 초기 논문 서사와 근거 기록 — 현재 주장·가설·기여는 09를 우선 |
| [docs/09_research_proposal.md](docs/09_research_proposal.md) | 현재 연구계획 — 한 문장 주장, RH1–RH3, Task 1–3, Table 1–3, 예상 contribution |
| [docs/10_evidence_and_baselines.md](docs/10_evidence_and_baselines.md) | Cancer-Myth·Well·GEPA·CoT 근거, CAST·라우팅·AO/NLA의 역할 |
| [docs/11_tripathi_review.md](docs/11_tripathi_review.md) | 2608.23666 원문과 공개 산출물 검토, 직접 중복 및 적용 가능성 |
| [docs/12_discussion_decisions.md](docs/12_discussion_decisions.md) | 대화 결정 기록, split·통계·faithfulness, 철회한 주장과 미결정 사항 |
| [docs/13_task_spec.md](docs/13_task_spec.md) | Task 1–3 실행 명세, 최종 방법 비교와 구성 요소 통제 비교 |
| [docs/14_advisor_report_2.md](docs/14_advisor_report_2.md) | 교수님 보고 2의 당시 발송 기록 |
| [docs/15_progress_summary.md](docs/15_progress_summary.md) | 결정·탐색 결과·미결 사항의 현재 요약 |
| [docs/16_code_overview.md](docs/16_code_overview.md) | 실험 코드와 구현·리뷰 대응 상태 |
| [docs/17_manuscript_storyline.md](docs/17_manuscript_storyline.md) | 논문 절별 스토리라인·문단 초안·표 읽는 법·결과별 결론 범위 |
| [docs/18_professor_presentation_script.md](docs/18_professor_presentation_script.md) | 교수님 발표 23장 상세 원고 — 단계별 Intro·예상 기여 세 개·선행연구·방법·빈 결과표·질문 대비 |
| [docs/21_well_rag_reproducibility.md](docs/21_well_rag_reproducibility.md) | Well RAG 코드·공개 근거 문서 감사, 재현 가능 범위, 새 Table 3의 출판값과 재실행 계획 |
| [docs/experiments/01-e1-prediagnostic.md](docs/experiments/01-e1-prediagnostic.md) | E1 사전 진단 — 모델·GPU 배치, 위치 정의, 라벨, 산출물, 갈림길, 상태 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 서버 세팅(125번, medical_nla 관례)과 실행 명령 전부 |
| [docs/references.md](docs/references.md) | 링크 전부 |
| [docs/appendix/candidates_screening.md](docs/appendix/candidates_screening.md) | 후보 논문 22편 스크리닝 결과 |

[19 — Method 실행 명세](docs/19_method_protocol.md): 마지막 프롬프트 표상, probe 목적식·문턱, paired C 추출, 생성 hook·강도 및 구현 과제.

[20 — 기존 방법의 기여·의료 적용·표 구성](docs/20_baseline_transfer_and_novelty.md): NFP/TPQ 정리, 10종 탐지 신호, baseline 선정과 개입률, 현재 기여 범위.

## 상태

- [x] 앵커 논문 확정, 표·지표·저장소 확인
- [x] 후속 연구 전수 확인 (Semantic Scholar 인용 14편, 핵심 4편 본문 검토)
- [x] 가설 지도, 읽을 층, 도구 선정
- [x] 인용 그래프 바깥 관련 연구 (2025–2026), 직접 경쟁자 3편 원문 검토 → baseline·산출물 갱신
- [x] 코드 세팅 — medical_nla와 같은 레이아웃(`/home/eagle0914`, `/data1/heejae`, uv, env.sh), 4090 4장 E1 파이프라인, nohup 자체 분리, 27 tests
- [~] **실험 1** — 표상이 배경화된 전제에서 읽히는가 (A·B·C, 4 모델). llama·qwen 완료: A 0.81, Plain PCR 3–6 %라 자연 C 재료 부족 → stage 8 짝지은 C ([experiments/01 중간 결과](docs/experiments/01-e1-prediagnostic.md))
- [ ] 개입 설계 (A 게이트 × C 방향)
- [ ] baseline 재실행 (Well-Actually의 코드, Cancer-Myth 판정기로 채점)
- [ ] Table 1 판별 / Table 2 최종 효용·QA / 부록 A1 선택 통제 완성; 전제 사실 확인 Table 3 후보의 독립적 필요성 검토

## 코드

```
configs/     모델별 yaml (${CANCER_MYTH_DATA_ROOT} 치환, _base 상속)
src/         config · rows(전제 정렬) · extract_activations · probes · steering · judge_prompts
scripts/     env.sh · bootstrap_server.sh · make_rows · run_generate · run_judge · run_probe_sweep · run_direction_c · make_paired_rows · run_direction_pair · run_steer · run_e1_4gpu_125.sh · run_e1_stages_125.sh · jobs.sh
tests/       pytest -q (GPU 불필요)
```

## 관련

논문 목록 파이프라인(ICLR/ICML 2026 accepted 11,699편 → 의료 NLP 92편)은 별도 폴더 `~/openreview-med-nlp`.
