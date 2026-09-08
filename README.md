# Cancer-Myth Internals

의료 LLM이 환자 질문 속 **잘못된 전제(false presupposition)** 를 왜 못 고치는지를 **내부 표상**에서 읽고, 그 신호로 **조건부로** 고치는 연구.

앵커 논문: **Cancer-Myth** (ICLR 2026, Zhu et al., [arXiv 2504.11373](https://arxiv.org/abs/2504.11373)).
목표 산출물: Cancer-Myth **Table 1**에 공개 모델 한 블록(Plain / 기존 완화책 / **Ours**)을 추가하되, **Cancer-Myth PCR을 올리면서 NFP를 Plain 수준으로 유지**하는 첫 행이 되는 것.

## 한 줄 요약

| | |
|---|---|
| 현상 | 환자가 틀린 믿음을 깔고 질문하면 모델은 겉 질문에만 답한다 (프런티어 ≤43%, 공개 모델 ≤17.3% PCR) |
| 기존 완화책의 실패 | GEPA·Monitor는 Cancer-Myth를 올리는 대신 **전제 없는 질문에서 없는 전제를 지어낸다** (NFP −29, −55) |
| 왜 실패하나 | 모델은 거짓 전제를 96–100% *알면서* 안 꺼낸다 (Well-Actually). 판정을 시키면 참 전제의 58–93%도 거짓이라 한다 (negative bias) |
| 우리 베팅 | 판정을 시키지 말고 **표상을 읽어라**. 진위(A)와 태도(C)를 같이 읽으면 sycophancy가 갈리고, A로 게이트한 C 개입은 설계상 NFP를 건드리지 않는다 |
| 선행연구와의 거리 | 사용자 태도(Verbalizing-Assumptions), 전역 동조 벡터(CAA), 단언문 진위(Contextual-Truth), 강제 판정(Well-Actually), CoT 언급(MedMisBench)은 있었다. 일반 도메인에선 probe 게이트 **프롬프트**(Two Axes), 의료에선 압력 감지 게이트 ITI(Tripathi)까지 있다. **"게이트 신호(진위)와 개입 대상(태도)이 다른 개념이고, 통제군이 표면상 구분 불가능한 NFP"** 인 것은 없다 ([07](docs/07_related_work_2026.md)) |

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
| [docs/08_story_outline.md](docs/08_story_outline.md) | 논문 서사 — 도메인 요구→모델은 안다→출력 개입의 문제→내부에서→왜 게이트 개입→기존 것을 못 쓰는 이유(데이터·구조·목적식)→TL;DR·contribution·정의 |
| [docs/experiments/01-e1-prediagnostic.md](docs/experiments/01-e1-prediagnostic.md) | E1 사전 진단 — 모델·GPU 배치, 위치 정의, 라벨, 산출물, 갈림길, 상태 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 서버 세팅(125번, medical_nla 관례)과 실행 명령 전부 |
| [docs/references.md](docs/references.md) | 링크 전부 |
| [docs/appendix/candidates_screening.md](docs/appendix/candidates_screening.md) | 후보 논문 22편 스크리닝 결과 |

## 상태

- [x] 앵커 논문 확정, 표·지표·저장소 확인
- [x] 후속 연구 전수 확인 (Semantic Scholar 인용 14편, 핵심 4편 본문 검토)
- [x] 가설 지도, 읽을 층, 도구 선정
- [x] 인용 그래프 바깥 관련 연구 (2025–2026), 직접 경쟁자 3편 원문 검토 → baseline·산출물 갱신
- [x] 코드 세팅 — medical_nla와 같은 레이아웃(`/home/eagle0914`, `/data1/heejae`, uv, env.sh), 4090 4장 E1 파이프라인, nohup 자체 분리, 27 tests
- [~] **실험 1** — 표상이 배경화된 전제에서 읽히는가 (A·B·C, 4 모델). llama·qwen 완료: A 0.81, Plain PCR 3–6 %라 자연 C 재료 부족 → stage 8 짝지은 C ([experiments/01 중간 결과](docs/experiments/01-e1-prediagnostic.md))
- [ ] 개입 설계 (A 게이트 × C 방향)
- [ ] baseline 재실행 (Well-Actually의 코드, Cancer-Myth 판정기로 채점)
- [ ] Table 1 블록 완성

## 코드

```
configs/     모델별 yaml (${CANCER_MYTH_DATA_ROOT} 치환, _base 상속)
src/         config · rows(전제 정렬) · extract_activations · probes · steering · judge_prompts
scripts/     env.sh · bootstrap_server.sh · make_rows · run_generate · run_judge · run_probe_sweep · run_direction_c · make_paired_rows · run_direction_pair · run_steer · run_e1_4gpu_125.sh · run_e1_stages_125.sh · jobs.sh
tests/       pytest -q (GPU 불필요)
```

## 관련

논문 목록 파이프라인(ICLR/ICML 2026 accepted 11,699편 → 의료 NLP 92편)은 별도 폴더 `~/openreview-med-nlp`.
