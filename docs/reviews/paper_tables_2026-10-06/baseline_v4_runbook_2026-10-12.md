# Baseline v4 실행 규약 — 2026-10-12

이 문서가 이전 v3 실행 계획보다 우선한다. [현재 Table 1·2·3](current_v4/README.md)은 생성·채점·완료 상태를 셀에 직접 표시하는 시점별 스냅샷이다. 결과 파일을 읽어 갱신하며 진행 중인 부분 점수를 최종 성능으로 표시하지 않는다.

## 고정한 대상과 분할

- Qwen2.5-7B, Gemma 3-12B, GPT-6 Luna. Ours와 Qwen3.8은 이번 실행 범위 밖이다. NLA는 별도 A6000 서버 작업이다.
- Cancer-Myth: 고정 예시와 겹치는 질문 그룹 8문항을 제외한 **577 FPQ + 147 NFP**, 기존 grouped 3-fold 공유. 고정 프롬프트는 각 문항을 한 번 생성한다. TF-IDF·Probe·GEPA만 각 fold 학습/개발 영역에서 학습·선택한다.
- CREPE: 공식 train 전체 3,437문항 유지. 공식 test에서 **500 FPQ + 500 NFP**를 seed 42로 고정 추출. GEPA 후보 선택은 공식 dev에서 **100 + 100**, seed 43. TF-IDF·Probe는 기존 전체 dev를 유지한다. 300자 길이 필터는 쓰지 않는다.
- [분할 ID manifest](current_v4/split_manifest.json), [실행 설정](current_v4/run_settings.json). 공통 bundle SHA256: `88f6f9d4ef07f2812112d1004abdbeedcd90f910171d309a5d3d93d33a16fbe1`.

## 생성과 평가

| 항목 | 적용 규칙 |
|---|---|
| Qwen GPU 생성 | GPU 0, batch 최대 8, 토큰 예산 73,728 |
| Gemma GPU 생성 | GPU 1·2, batch 최대 4, 토큰 예산 49,152 |
| Luna | 독립 CLI 요청을 병행. 공통 추출·Luna 답변·GEPA GPT 호출 전체에 동시 요청 최대 4개 |
| 생성 한도 | 로컬 greedy, 입력/출력 각각 최대 8,192. OOM 시 batch를 줄이며 출력 한도를 몰래 줄이지 않음 |
| 전제 추출 | GPT-6 Luna가 추출한 전제를 PreWoMe·Extract+Verify와 세 답변 모델이 공유. 검토·검증·답변은 평가 대상 모델이 수행 |
| Table 1 | 판정 즉시 독립 저장, 정답 라벨로 평가. Well 채점 불필요 |
| Table 2 gate | Table 1 판정 및 검토문을 질문과 함께 전달해 **실제 새 답변 생성**. 저장 Plain/Always-correct 답변 선택 방식은 사용하지 않음 |
| Table 2 최종 평가 | **Haiku**로 전부 재채점. 기존 Sonnet 점수는 원본에 보존하지만 새 결과에 혼합하지 않음 |
| Table 3 | 정답 비교로 자동 채점. 데이터셋이 달라 필요한 판정·답변을 새로 실행. 호환되는 기존 Plain QA 답변은 재사용 |
| 실패 | 길이 초과 등 문항 단위 실패는 기록하고 다음 문항 진행. 완료 건수와 실패 수를 구분하며 성공률 분모에서 임의 제외하지 않음 |

Table 3의 고정 Plain/Balanced/CoT/Direct/CoT gate는 공통 실행을 한 번 수행한다. 데이터셋별 학습·예시가 있는 방법은 Cancer-Myth와 CREPE 출처별로 평가한다. Cancer의 주 실행 버전은 **사전 고정 fold0**이다. 이전 계획의 세 fold 평균으로 보고하지 않는다.

배치 변경 전 생성된 답변은 원본 batch=1 이력을 남겨 재사용한다. Greedy라도 배치 구성에 따라 텍스트가 달라질 수 있으므로 동일 출력 보장을 주장하지 않는다. Luna의 제공자 기본 디코딩을 로컬 greedy와 같다고 표기하지 않는다.

## GEPA

탐지·직접 답변·두 단계 최적화 조건을 각각 실행한다. 학습/개발 피드백과 reflection은 GPT-6 Luna, 최종 test 채점은 Haiku로 분리한다. test Haiku 점수를 후보 선택에 사용하지 않는다. CREPE dev 200문항에서 초기 평가가 예산 대부분을 소비하지 않도록 실행 설정의 예산을 **최적화 실행당 2,000 question metric calls (= 1,000 paired metric calls)**로 잡았다. 실제 사용량은 별도 보고한다. `optimization_models.json`의 기존 잔존 예산 필드는 Worker의 예산 결정에 쓰지 않으며 `run_settings.json`의 `gepa.metric_budget`이 적용 값이다.

## 기존 결과 재사용

모델·프롬프트·디코딩·원본 코드 및 주석 규약이 맞는 Plain/Balanced/CoT 답변만 가져온다. 기존 값에서 판정기 점수를 제거한 새 원장을 만들고 과거 평가를 별도 필드에 보존한다. Qwen은 Cancer Plain 637개·Balanced 221개, Gemma는 Plain 298개·Balanced 1개를 실행 시작 시 재사용했다. Qwen/Gemma의 기존 Plain 일반 QA도 재사용했다. 자체 모델 추출을 쓴 파이프라인과 저장 답변 선택 방식 gate는 새 방식 결과로 재분류하지 않는다.

## 실행과 재개

현재 systemd user 서비스는 `paper-v4-{qwen,gemma,luna,extraction,haiku}-cli-20261012.service`이다. 생성·공통 추출·Haiku 채점이 독립적으로 진행된다. 요청 제한 시 600초 간격 재시도하며 결과 파일을 기준으로 재개한다. 시스템 서비스 PATH에는 `~/.local/bin`과 `~/.npm-global/bin`을 포함해야 한다.

생성은 `scripts/run_paper_baselines_v4.py`, 공통 추출은 `scripts/prepare_shared_luna_extractions.py`, 채점은 `scripts/judge_paper_queue.py --judge-config .../haiku.json --judgments-out .../haiku`로 실행한다. 데이터·출력 루트는 `results/paper_baselines_v4/`이다. Table 1·2 모든 방법을 Table 3의 추가 생성보다 먼저 수행한다.

상태 그림 갱신: `python3 docs/reviews/paper_tables_2026-10-06/update_v4_status.py`. 이 명령은 생성·채점을 호출하지 않는다. 집계기는 `scripts/summarize_paper_baselines.py`이며 Haiku sidecar의 답변·주석·판정기 해시가 일치하는 평가만 인정한다.
