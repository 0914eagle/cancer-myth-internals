# 16. 코드 지도 (2026-09-09, 실험 브랜치 `fbf746d` 기준)

실험 세션이 만든 파이프라인을 문서 세션 관점에서 정리한 것. 실행 방법은 [EXPERIMENTS.md](../EXPERIMENTS.md), 결과는 [experiments/01](experiments/01-e1-prediagnostic.md). 여기서는 **무엇이 어디 있고, 데이터가 어떻게 흐르고, 리뷰 지적과 13의 설계 대비 무엇이 남았나**만 다룬다.

## 1. 배치

```
configs/     모델별 yaml. default.yaml을 _base로 두고 모델 파일은 다른 것만 적음. ${CANCER_MYTH_DATA_ROOT} 치환
src/         라이브러리. config · rows · extract_activations · manifests · probes · steering · judge_prompts · llm_backend · modeling · jsonl
scripts/     단계별 실행기(python)와 wrapper(bash). 긴 작업은 nohup으로 자체 분리 (lib/detach.sh)
tests/       pytest 28개, GPU 불필요
EXPERIMENTS.md  서버 125/62 레이아웃, 첫 설치, 판정기 보정, 분리 실행
```

서버: RTX 4090 4장(125), 코드 `/home/eagle0914/cancer-myth-internals`, 산출물 `${DATA_ROOT}/cancer_myth_internals/{data,external,activations,results,reports,logs}`. medical_nla 프로젝트와 같은 레이아웃이라 그쪽 `src.run_nla`(NLA) 스크립트를 매니페스트에 그대로 꽂을 수 있다.

## 2. 파이프라인 (한 모델 기준)

```
E0   make_rows.py            questions.jsonl (fpq 585 / nfp 150 / tpq), activation_rows.jsonl (위치 A·B·D)
                             전제 구간은 codex가 verbatim substring으로 정렬, exact match 검증, 재시도 1회
E0b  make_true_twins.py      각 fpq의 전제 구간만 참으로 바꾼 쌍둥이 (set=tpair). 스플라이스 + LLM 확인
E1   run_e1_model.sh  stage 1  run_generate.py         Plain 응답 (greedy; --paper-protocol이면 0.7)
                      stage 2  extract_activations     A/B/D, 전 층, span_mean·last_subtoken
                      stage 3  run_judge.py            Cancer-Myth 판정 (codex 또는 openai)
                      stage 4  make_response_rows.py   위치 E 행 + 라벨(pcr, nfp_score) 병합
                      stage 5  extract_activations     E (응답 첫 5/32토큰, teacher-forced); 라벨을 A/B/D 매니페스트에도 병합
                      stage 6  run_probe_sweep.py      A 읽기: 층 × 위치 AUROC, logistic + diff-of-means, 히트맵
                               run_text_baseline.py    TF-IDF 텍스트 상한 (같은 fold)
                      stage 7  run_direction_c.py      C 방향 (자연 응답 +1 − −1), cos(A,C), D에서 PCR 예측 (CV), directions.npz
                      stage 8  make_paired_rows.py + run_direction_pair.py
                                                       참조 답변(+1/−1) teacher-forcing 짝 대조 c_pair5/32, 자기 PCR 검증
                      stage 9  probe_sweep_twins       fpq vs 쌍둥이로 A probe와 텍스트 상한 재측정
E2   run_e2_steer_125.sh     run_steer.py (policy none / unconditional / gated, alpha, gate layer·threshold)
                             → run_judge.py → summarize_judge.py
보조  calibrate_judge.py     all_data.json의 GPT-4o 점수와 대조 (3-way 일치, κ, PCR 차이)
     run_e1_4gpu_125.sh      4장에 모델 분배 (phase 1 소형 3개, phase 2 27B, phase 3 후속 단계)
     run_e1_stages_125.sh    끝난 모델의 stage 3–8을 유휴 카드에서
     prune_manifests.py      재정렬 후 옛 A 행 제거
     jobs.sh                 detach된 작업 목록과 마지막 로그
     check_gpu_setup.py      GPU 가시성·메모리 확인, 오프로드 거부
```

산출물 위치: `data/e1_rows_v1/` (행), `activations/e1/<model>/{ad,e,pair}/layerNN/<position>/manifest.jsonl + .pt`, `results/e1/<model>/{plain_responses, plain_judge, probe_sweep/, direction_c/, direction_pair/}`, `results/e2/<model>/<tag>{,_judge,_summary}.jsonl`.

## 3. src 모듈

| 모듈 | 역할 | 비고 |
|---|---|---|
| `config.py` | yaml + `${VAR}` 치환, `_base` 상속 | 머신당 변수 하나 |
| `rows.py` | 세 질문 집합을 한 행 형식으로. 전제 정렬(llm / heuristic), 위치 행 생성, E 행(teacher-forced), 짝 행, 쌍둥이 | `PASSTHROUGH`에 pcr·judge_parsed 포함 (리뷰 1 수정) |
| `extract_activations.py` | 지정 층·위치의 hidden state 저장, medical_nla 매니페스트 호환 | `hidden_states[k]` = 블록 k−1 출력 |
| `manifests.py` | 매니페스트 → 행 목록·행렬 | |
| `probes.py` | logistic probe와 diff-of-means, grouped CV (base_id 기준) | Two Axes 두 readout |
| `steering.py` | residual hook으로 방향 더하기, A probe gate | 층 인덱스 규약은 extract와 동일 |
| `judge_prompts.py` | `validate.py`·`validate_nfp.py` 프롬프트·정규식·fallback 원문 그대로 | 정규식은 `{` 뒤 개행 필수 (원본 그대로) |
| `llm_backend.py` | OpenAI API 또는 `codex exec` 한 호출 | codex 배너에서 실제 모델명 기록 |
| `modeling.py` | 백본 로딩, CPU/meta 오프로드 거부 | |

## 4. 판정기

- 기본 `codex exec` (ChatGPT 로그인, API 키 없음). 2026-09-07 보정: gpt-5.6-sol이 GPT-4o보다 일관되게 엄격 (3-way 일치 58%, κ 0.28, GPT-4o의 +1 중 58%만 +1). codex는 gpt-4o를 서빙 못 함.
- **표에 들어가는 숫자는 `JUDGE_BACKEND=openai`(gpt-4o)** 로 다시 채점 ([experiments/01](experiments/01-e1-prediagnostic.md) 결정). 상대 비교(α·층 sweep)는 codex.
- 행마다 backend·model provenance 기록, id로 resume, lock 파일.

## 5. 테스트 (28)

config 치환·상속, 전제 정렬(verbatim·heuristic·재시도·탈락), 위치 계산(마지막 토큰·assistant prefix·target substring), E 행이 라벨을 실음, 짝 행이 프롬프트를 공유, 쌍둥이 스플라이스·검증, probe가 sklearn과 일치·무신호에서 0.5·grouped fold, 전이 AUROC가 CV, 판정 프롬프트 레이아웃·정규식·fallback, codex 배너 파싱.

## 6. 리뷰 지적 대응 상태 (2026-09-09)

| # | 지적 ([reviews/2026-09-07](reviews/2026-09-07-review.md)) | 상태 |
|---|---|---|
| 1 | E 행 PCR 라벨 유실 → C 추출 불가 | **수정** (`40136cc`: PASSTHROUGH에 pcr, stage 5가 E 매니페스트에도 병합) |
| 2 | E2 평가셋이 학습 문항과 겹침; `pcr_auroc_D_via_cE` in-sample | **절반 수정** (`58ced72`: E→D 전이를 CV로). **E2는 여전히 같은 `questions.jsonl`에서 표본 추출, 분할 필터 없음.** 13의 cross-fitting과 함께 구현 필요 |
| 3 | 파싱 실패가 +1로 집계 | **미수정.** `summarize_judge.py`는 unparsed 개수를 세지만 PCR 분모·분자에 포함. C 라벨도 제외 안 함. 원 정규식은 유지하되 미판정을 제외·보고 |
| 4 | 판정기 바꿔도 같은 `_judge.jsonl` 재사용 | **미수정.** wrapper 경로가 backend와 무관. 문서의 `_judge_gpt4o.jsonl` 규약이 코드에 없음 |
| 5 | TPQ 참 전제를 "언급하면 감점"으로 전달 | **미수정.** `run_judge.py`가 tpq를 NFP 루브릭으로 채점. **결정: Well의 TPQ 루브릭(App. E)을 GPT-4o로** ([13 §0](13_task_spec.md)) |
| 6 | CAST 서술 | 문서 수정 완료 (07·08) |
| 7 | SDT 환산 AUROC | 문서 수정 완료 (02). Table 1에 텍스트·출력 readout 동일 용량 |
| 8 | probe vs 언어 판정으로 H1/H3 판정 불가 | 문서 수정 완료 (03·06). **코드: stage 6 텍스트 상한, stage 9 쌍둥이가 shortcut 통제** |
| 9 | NFP 보존 자동 아님, A∧C 미구현 | 문서 수정 완료. **코드: gate는 A probe만. A∧C, 발화율·harm/rescue 집계 미구현** |
| 10 | 상·하한, 13% 논리 | 문서 수정 완료 |
| 11 | 판정기 보정 일반화 | 문서에 주의 기록. C를 codex 라벨로 만드는 위험은 stage 8(참조 답변은 GPT-4o 라벨)로 일부 회피 |
| 12 | resume 시 α 절대값 변동 | **미수정.** `run_steer.py`가 done 제외 후 첫 batch로 scale 계산 |

## 7. 13의 설계 대비 아직 없는 부품

| 부품 | 13 위치 | 상태 |
|---|---|---|
| 질문·myth·NFP–TPQ 그룹 manifest, nested 5-fold cross-fitting | §0 | 없음. probes.py의 grouped CV는 readout 평가용이고 E2·GEPA·CAST 학습 분할은 없음 |
| 무작위 조건 (예산 맞춤, seed 5) | Table 3 | 없음. `run_steer.py` policy에 `random` 추가 필요 |
| Two Axes 라우팅 행 (probe → 교정 프롬프트) | Table 2·3 | 없음. gate는 있으나 개입이 steering뿐. 프롬프트 개입 경로 추가 |
| CoT monitor, 직접 질문, extract-and-verify readout | Table 1 | 없음. Well 프롬프트 코드 재사용 |
| Well 프롬프트 baseline (FP Identification, Extract+FactCheck 등), GEPA (Qwen·Llama) | Table 2 | 없음. `external/Well` 클론은 bootstrap에 있음 |
| CAST (조건 벡터 + 행동 벡터) | Table 2 | 없음 |
| C 후보 (ii) 시스템 프롬프트 짝 대조 | §2 | 없음. (i)·(iii)는 있음 |
| C 후보 선택을 steering score로 | §2 | 없음. 현재 층은 E1 결과를 보고 수동(27B L28) |
| A 직교화 C (C − proj_A C) | experiments/01 | 계획만 |
| Well TPQ 루브릭 | §0 | 없음 (지적 5) |
| 의료 QA 3종 (MedQA, PubMedQA, Medbullets) 정확도 | Task 3 | 없음 |
| 일반 도메인 (CREPE, QA², Syn-QA²) | 부록 | 없음. Well 파이프라인 |
| GPT-4o 재채점 + 판정기별 파일명 | §0 | 없음 (지적 4) |
| 사전 등록 (가설·kill-criteria를 스크립트 헤더에) | Table 3 | 없음 |

## 8. 실험 세션에 넘길 우선순위

1. 지적 3·5·12 수정 (작음). 지적 4는 파일명 규약 한 줄.
2. 분할 manifest + cross-fitting 틀. 이게 없으면 E2 숫자를 표에 못 넣는다.
3. `run_steer.py`에 `random` policy와 프롬프트 개입 경로 → Table 3의 3×3이 한 스크립트로.
4. Well TPQ 루브릭을 `judge_prompts.py`에 추가.
5. stage 9 쌍둥이 결과 확인 후 E2.
