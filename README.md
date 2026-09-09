# Cancer-Myth Internals

> **2026-09-10 교수님 발표 준비:** [18 — 슬라이드별 상세 발표 원고](docs/18_professor_presentation_script.md). Introduction의 단계별 근거와 마지막 contribution 세 개, Related Work, Method, 결과표 개요를 23장으로 구성했다. Two Axes·Gated Activation Steering은 각각 2장, Method는 4장이다. 정상 질문은 NFP와 Well TPQ를 합친 300개가 아니라 **동일한 150개**이며, FPQ를 포함한 명목상 전체 질문 수는 **735개**다.

> **2026-09-09 현재 연구계획:** 논문 전체의 이야기는 [17 — Introduction부터 Conclusion까지](docs/17_manuscript_storyline.md), 주장·가설은 [09](docs/09_research_proposal.md), 실행 명세는 [13](docs/13_task_spec.md)을 읽는다. Table 1은 판별 신호, Table 2는 최종 교정·보존 성능과 일반 의료 QA ACC 세 열을 합친 본 결과표, Table 3은 (a) 같은 steering의 선택 방식과 (b) 같은 질문의 개입 방식을 비교하는 두 패널 결과표다. [10 — 근거](docs/10_evidence_and_baselines.md), [11 — Tripathi 검토](docs/11_tripathi_review.md), [12 — 결정 기록](docs/12_discussion_decisions.md), [15 — 진행 현황](docs/15_progress_summary.md), [16 — 코드 지도](docs/16_code_overview.md)가 이를 뒷받침한다. 일반 의료 QA 보존은 Task 3이며 독립적인 새 방법 기여가 아니다. 아래 초기 요약과 충돌하면 이 문서들이 우선한다. 계획 갱신은 실험 성공을 뜻하지 않는다.

의료 LLM이 환자 질문 속 **잘못된 전제(false presupposition)** 를 왜 못 고치는지를 **내부 표상**에서 읽고, 그 신호로 **조건부로** 고치는 연구.

앵커 논문: **Cancer-Myth** (ICLR 2026, Zhu et al., [arXiv 2504.11373](https://arxiv.org/abs/2504.11373)).
목표 산출물: Cancer-Myth 지표(PCR/PCS/NFP, GPT-4o 판정기)로 채점한 **공개 모델 held-out 비교표**. 그 표에서 **정상 질문의 추가 오교정을 사전에 정한 허용폭 안으로 제한하면서 PCR을 올리는 행**을 넣는 것. 원본 Table 1의 닫힌 모델 수치는 평가 범위가 달라 참고 결과로 분리한다 ([06](docs/06_experiment_plan.md#실험-3--held-out-비교표)).

> **중심 문장.** 고정된 held-out 평가에서, 정상 질문의 추가 오교정 위험을 사전에 정한 범위로 제한하면서 거짓 전제 교정률을 높일 수 있는지 검증한다. 그 과정에서 내부 신호의 라우팅 가치와 활성값 개입의 추가 효과를 분리해 측정한다.

> **표·기여 후속 정리:** [20 — 기존 방법의 기여와 의료 적용](docs/20_baseline_transfer_and_novelty.md). Table 1은 Two Axes의 핵심 readout 비교를 의료에서 재평가하는 일곱 행이며 별도 새 Ours 탐지 행은 없다. Table 2는 NFP 주 지표 하나와 QA ACC 세 열, 의료 재학습 Gated·Extract+FactCheck를 포함한 아홉 방법이다. Well 원 FPQ/TPQ 점수는 부록, Unconditional C는 Table 3으로 옮긴다. 두 선행 방법 모두 적용 가능하며 재학습 필요성을 실패 증거로 부르지 않는다.

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
| [docs/experiments/01-e1-prediagnostic.md](docs/experiments/01-e1-prediagnostic.md) | E1 사전 진단 — 모델·GPU 배치, 위치 정의, 라벨, 산출물, 갈림길, 상태 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 서버 세팅(125번, medical_nla 관례)과 실행 명령 전부 |
| [docs/references.md](docs/references.md) | 링크 전부 |
| [docs/appendix/candidates_screening.md](docs/appendix/candidates_screening.md) | 후보 논문 22편 스크리닝 결과 |

[19 — Method 실행 명세](docs/19_method_protocol.md): 마지막 프롬프트 표상, probe 목적식·문턱, paired C 추출, 생성 hook·강도 및 구현 과제.

[20 — 기존 방법의 기여·의료 적용·표 구성](docs/20_baseline_transfer_and_novelty.md): NFP/TPQ 정리, 일곱 탐지 신호, baseline 선정과 개입률, 현재 기여 범위.

## 상태

- [x] 앵커 논문 확정, 표·지표·저장소 확인
- [x] 후속 연구 전수 확인 (Semantic Scholar 인용 14편, 핵심 4편 본문 검토)
- [x] 가설 지도, 읽을 층, 도구 선정
- [x] 인용 그래프 바깥 관련 연구 (2025–2026), 직접 경쟁자 3편 원문 검토 → baseline·산출물 갱신
- [x] 코드 세팅 — medical_nla와 같은 레이아웃(`/home/eagle0914`, `/data1/heejae`, uv, env.sh), 4090 4장 E1 파이프라인, nohup 자체 분리, 27 tests
- [~] **실험 1** — 표상이 배경화된 전제에서 읽히는가 (A·B·C, 4 모델). llama·qwen 완료: A 0.81, Plain PCR 3–6 %라 자연 C 재료 부족 → stage 8 짝지은 C ([experiments/01 중간 결과](docs/experiments/01-e1-prediagnostic.md))
- [ ] 개입 설계 (A 게이트 × C 방향)
- [ ] baseline 재실행 (Well-Actually의 코드, Cancer-Myth 판정기로 채점)
- [ ] Table 1 판별 / Table 2 최종 효용·QA / Table 3 구성 요소 비교 완성

## 코드

```
configs/     모델별 yaml (${CANCER_MYTH_DATA_ROOT} 치환, _base 상속)
src/         config · rows(전제 정렬) · extract_activations · probes · steering · judge_prompts
scripts/     env.sh · bootstrap_server.sh · make_rows · run_generate · run_judge · run_probe_sweep · run_direction_c · make_paired_rows · run_direction_pair · run_steer · run_e1_4gpu_125.sh · run_e1_stages_125.sh · jobs.sh
tests/       pytest -q (GPU 불필요)
```

## 관련

논문 목록 파이프라인(ICLR/ICML 2026 accepted 11,699편 → 의료 NLP 92편)은 별도 폴더 `~/openreview-med-nlp`.
