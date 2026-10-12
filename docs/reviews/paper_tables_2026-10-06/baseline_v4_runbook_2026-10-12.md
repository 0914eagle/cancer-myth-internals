# Baseline v4 실행 규약 — 2026-10-12

## 현재 Gemma 배치: 16 (GPU 1·2·3)

세 GPU 분산 상태에서 사용자 요청으로 Gemma 배치 상한을 **8 → 16**, 배치 토큰 예산을 **98,304 → 196,608**로 높였다. 모델 BF16, greedy, 개별 입력·출력 최대 8,192, 프롬프트와 fold는 그대로다. OOM이 발생하면 기존 실행기의 배치 자동 축소를 사용한다. 기존 출력은 보존하고 미완료 요청부터 재개한다. 배치 구성 변경에 따른 수치 연산 차이로 출력이 완전히 같다고 보장하지 않으며, 각 호출에 실제 배치 크기를 남긴다.

전환 직전 같은 GPU 배정에서 최근 120개 호출은 모두 batch 8, 평균 출력 655.5토큰, 평균 배치 생성 시간 79.5초였다. 이는 batch 8 생성 구간만 환산하면 약 362문항/시간이다. 배치 16의 실제 처리량은 새 출력 로그로 확인하며, 서로 다른 문항 비교이므로 엄밀한 동일 입력 속도 실험과 구분한다.

## 최종 정정: Gemma는 GPU 1·2·3 모두 사용

사용자는 GPU 1도 계속 Gemma에 쓰도록 정정했다. 현재 배정은 **Qwen GPU 0 / Gemma GPU 1·2·3**이다. 아래의 GPU 1 유휴 기록은 이전 배정이다. `CUDA_VISIBLE_DEVICES=1,2,3`, 서비스 `paper-v4-gemma-sol-gpu123-20261012.service`, 배정 기록 `device_overrides/gpu1_gpu2_gpu3.json`을 사용한다. 모델을 세 GPU에 분산 적재하며 기존 batch 상한 8, 토큰 예산 98,304 및 과학적 실험 설정은 유지한다. 완료 결과를 재사용하고 전환 시 미완료 요청만 다시 실행한다.

## 현재 GPU 배정 변경: Gemma 2·3

사용자 요청으로 Gemma의 물리 GPU를 1·2에서 **2·3**으로 옮겼다. Qwen은 GPU 0, GPU 1은 비워둔다. Luna와 Sol은 CLI 호출이며 로컬 GPU를 사용하지 않는다. Gemma의 batch 상한 8, 토큰 예산 98,304, BF16, 출력 한도, 모델·프롬프트·분할은 유지한다. 이전에도 두 장을 사용했으므로 GPU 이동 자체를 처리량 증가로 해석하지 않는다.

`run_paper_gemma_placement.py`가 모델 적재 전에 `CUDA_VISIBLE_DEVICES=2,3`을 적용한다. 기존 답변·Probe 캐시의 과학적 설정 identity는 보존하고, 실제 배정은 `device_overrides/gpu2_gpu3.json` 및 `probe_device.json`에 기록한다. 현재 Gemma 서비스는 `paper-v4-gemma-sol-gpu23-20261012.service`이며 `launch_paper_v4_sol.py gemma`로 재개한다. Probe 특징 추출도 같은 모델/GPU lock을 공유한다.

## 현재 적용: GPT-6 Sol 공통 추출 (10월 12일 변경)

사용자 요청으로 **PreWoMe와 Extract+Verify의 추출 모델만 GPT-6 Sol, reasoning high로 변경**했다. [추출 설정](current_v4/extraction_sol.json)이 이전 Luna 추출 설정보다 우선한다. 검토·검증·답변 모델은 Qwen2.5-7B / Gemma 3-12B / GPT-6 Luna 그대로다. Well의 추출 템플릿과 고정 예시도 유지하며 정답 라벨·교정문은 추출 입력에 넣지 않는다.

- 세 답변 모델과 두 방법이 `shared/sol_extraction`의 동일 추출 캐시를 공유한다. 기존 `shared/luna_extraction`과 섞지 않는다.
- 전환 전 생성된 Qwen PreWoMe **165개**는 원본을 `archive_luna_extraction_20261012`에 보관하고 활성 집계에서 제외했다. 두 파이프라인 행은 Sol 기준으로 다시 생성한다. 다른 방법의 답변·판정·Probe와 분할 명단은 재사용한다.
- 원래 generator/probe identity를 유지하는 명시적 실행 wrapper `run_paper_baselines_sol.py`를 사용한다. 영향받는 결과에는 `extraction_model=gpt-6-sol`, `extraction_protocol=shared_premise_extraction_sol_v1`을 기록한다. 모델별 override와 공통 추출 identity를 따로 저장한다.
- 학습 문항 1개에서 Sol 호출·전제 목록 파싱을 검증한 뒤 본 추출을 시작했다. 관련 테스트 44개 통과.
- 현재 서비스는 `paper-v4-{qwen,gemma,luna,extraction,haiku}-sol-20261012.service`. 실행기는 `scripts/launch_paper_v4_sol.py ROLE`이다. 이전 서비스를 저장 후 중단하고 재개했다.
- Qwen batch 16 / Gemma batch 8, GPT 공통 동시 요청 4, Haiku 동시 4를 유지한다. Sol과 Luna는 GPT 슬롯을 공유하므로 개별 서비스당 4개씩 무제한 호출하지 않는다.

### Table 2 빈칸을 채우는 순서

| 우선순위 | 대상 | 실행 |
|---|---|---|
| 병행 채점 | 이미 생성된 모든 답변 | Haiku queue가 계속 채점. 생성 완료를 기다리지 않음 |
| 1 | Plain, Balanced, Direct gate, CoT gate, CoT, TF-IDF gate, Probe gate | 없는 답변만 생성. Table 1의 해당 fold 판정을 읽고 실제 답변을 새로 생성 |
| 2 | PreWoMe, Extract+Verify | Sol 공통 추출 → 각 대상 모델의 검토/검증 → 답변. 추출 사전 계산은 1과 병행 |
| 3 | GEPA-gate, GEPA, GEPA (both stages) | Cancer grouped 3-fold / CREPE 공식 train 및 고정 dev로 최적화 후 held-out 답변 생성 |

10월 12일 전환 시 Qwen은 우선순위 1의 7개 행을 Cancer 724개·CREPE 1,000개씩 생성 완료했다. Qwen은 바로 2로 진입하며, Gemma와 Luna는 1부터 누락을 채운다. Luna의 hidden-state Probe는 적용 대상이 아니다. Ours는 이번 baseline 생성 범위에 포함하지 않는다. Table 3 추가 생성은 Table 2 뒤에 진행한다.

아래의 이전 실행 기록 중 Luna 공통 추출과 서비스 이름은 변경 전 이력이다.

이 문서가 이전 v3 실행 계획보다 우선한다. [현재 Table 1·2·3](current_v4/README.md)은 생성·채점·완료 상태를 셀에 직접 표시하는 시점별 스냅샷이다. 결과 파일을 읽어 갱신하며 진행 중인 부분 점수를 최종 성능으로 표시하지 않는다.

## 10월 12일 정오 처리량 조정

사용자 요청에 따라 기존 실행을 현재 호출 저장 후 재개했다. **분할 명단, 프롬프트, 모델, 출력 한도, 채점 기준은 그대로** 두고 다음 실행 설정을 덮어쓴다. 원본 `run_settings.json`은 실행 식별을 위해 불변으로 보존한다. [실행 덮어쓰기 값](current_v4/transport_overrides.json), [fold 중복 검사](current_v4/fold_audit.json)를 함께 기록한다.

- Qwen batch 상한 **8 → 16**, 토큰 예산 **147,456**.
- Gemma batch 상한 **4 → 8**, 토큰 예산 **98,304**. 메모리 부족 시 동일 입력·출력 한도를 유지하며 batch만 자동 축소한다.
- Haiku 최종 채점 **1 → 동시 4 요청**. 요청 제한이 감지되면 진행 중 요청을 저장하고 600초 대기 후 재시도한다.
- GPT 공통 요청은 **동시 4개 유지**한다. Luna 답변·공통 추출·GEPA GPT 호출이 이 슬롯을 공유한다.
- **Direct·CoT 판정을 우선 저장**한 뒤 Table 2 답변을 생성한다. 답변 시 저장된 판정을 읽으며 판정 호출을 반복하지 않는다. Table 2는 질문과 판정·검토문을 받아 실제 새 답변을 만든다.
- `paper-v4-{qwen,gemma,luna,haiku}-fast-20261012.service`가 기존 `-cli` 실행을 대체한다. 공통 추출 서비스는 그대로 유지한다. 재현 실행기는 `scripts/launch_paper_v4_fast.py ROLE`이며, 기존 프로세스가 살아 있으면 강제 종료하지 않고 거부한다.

아래 표는 초기 실행 설정을 기록한 것이며, 배치와 채점 동시성은 위 변경이 우선한다.

## Probe 병행 및 fold 연결

실행 서비스: Qwen 특징 추출은 `paper-v4-qwen-probe-gpu3-20261012`, Gemma 생성·Probe 공유 실행은 `paper-v4-gemma-prefetch-20261012`, Haiku는 `paper-v4-haiku-prefetch-20261012`이다. Qwen의 `-fast` 생성 서비스와 Luna·공통 추출은 그대로 유지한다.

- Qwen 답변 생성은 GPU 0에서 계속하고, 별도 `prepare_probe_v4.py`가 GPU 3에서 같은 고정 체크포인트로 질문 특징을 추출한다. 답변 생성 호출이나 추가 판정기 호출은 없다.
- Gemma는 GPU 1·2에 이미 적재된 모델을 공유한다. `run_paper_baselines_probe_prefetch.py`가 특징 추출을 background thread로 수행하며, 생성과 forward는 하나의 GPU lock을 공유해 동시 메모리 사용을 막는다. Probe 학습의 CPU 처리는 생성과 병행한다.
- 질문 입력은 기존 Plain wrapper이며 마지막 입력 토큰의 hidden state를 쓴다. 후보 layer는 모델 깊이의 25/50/75/100%이다. 정규화 및 로지스틱 회귀는 train에만 fit하고 layer·C·threshold는 dev에서 선택한다. 기존 `fit_gate` 구현을 변경하지 않는다.
- Cancer-Myth는 3개 fold gate를 각각 저장한다. Table 1의 예측은 해당 fold test에만 작성하고, Table 2는 **같은 dataset/fold/method/question ID 경로의 판정**을 재사용한다. Table 3의 사전 선택 fold0 transfer 규칙과 혼동하지 않는다.
- CREPE는 공식 train 전체와 dev 전체로 Probe를 학습·선택하며, 고정된 500+500 test에 평가한다. GEPA의 100+100 dev 축소는 Probe에 적용하지 않는다.
- 고정 모델 특징은 질문만으로 계산하며 fold 간 재사용할 수 있다. 특징을 공유하는 것과 학습된 scaler/probe/threshold를 공유하는 것은 다르다. 후자는 각 fold에 독립적이다.

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


## 10월 12일 채점 형식 오류 및 GEPA 토크나이저 복구

Haiku가 `Rating:`을 여러 번 출력한 응답이 캐시에 남아, 자동 재시작 후에도 같은 응답을 읽고 중단되는 문제가 있었다. 형식 위반 원본은 `invalid_judge_calls`에 보관하고 **동일 프롬프트·채점 기준**으로 최대 3회 다시 요청한다. 모두 실패하면 점수를 부여하지 않고 `format_failures`에 기록해 다른 문항을 계속 처리하며 600초 후 다시 시도한다. 완료 집계에는 포함하지 않는다. 관련 회귀 테스트 12개 통과.

Qwen이 9개 비최적화 방법의 Table 2 생성을 끝낸 후 GEPA 진입 시, 로컬 토크나이저 로딩 과정에서 원격 인증 요청이 발생해 만료된 인증으로 실패했다. 고정 체크포인트와 토크나이저는 이미 로컬에 있다. `HF_HUB_OFFLINE=1`로 원격 조회를 막고 동일 토크나이저 호출이 성공하는 것을 확인했다. 실행기에도 이 환경을 추가했다. 실행 중 Gemma/Luna에는 다음 서비스 재시작부터 적용되는 runtime drop-in을 추가했다.

작업량 주의: Table 2는 Ours 및 Luna Probe를 제외해 35개 모델·방법 조합 × 1,724 = 60,340개 답변이다. Table 3은 일반 QA 17,942문항 × (Qwen 19 + Gemma 19 + Luna 17개 출처별 설정) = 986,810개 답변이다. 이 수는 GEPA 학습 중 평가 호출 및 검토·검증 등 중간 호출을 제외한 최종 답변 수다. 현재 전체 Table 3은 소규모 보존 표본이 아니라 전체 벤치마크 실행이며, 소요 시간은 Table 1·2와 분리해 보고한다.
