# A6000 GPU 0·1 기준선 실행 기록

사용자가 지정한 main 커밋은 `b81ace295dc42942d9dbed9be0916e956a7f4c6f`다.
기존 SFT 작업을 보존하기 위해 별도 worktree에서 실행한다.

- 코드: `/home/eagle0914/cancer-myth/external/paper_baselines_b81ace2`
- Python: `/data3/heejae/uv/paper_baselines_b81ace2/bin/python` (3.12.13)
- 실행/결과: `/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01`
- 가중치 캐시: `/data3/heejae/hf_cache/hub`
- GPU 0: Qwen2.5-7B-Instruct, GPU 1: Gemma 3 12B-IT

이 서버는 RTX A6000 48GB 8장이다. 사용자 요청에 따라 0·1번만 사용한다.
문서의 4090 두 장 대신 Gemma에 A6000 한 장을 배정하고 GPU당 가중치
배치 상한을 43GiB로 바꿨다. BF16, 모델 리비전, 프롬프트, 생성 한도,
seed, GEPA 예산, 데이터 분할은 원 설정을 유지한다.

## 데이터와 검증

`protocol/full_bundle.json`의 해시는
`4af42869026d2b5499b7d460d13ab5633133a7fadd0567d83cf432b2845a73c2`이며,
커밋에 있는 `baseline_split_manifest_2026-10-08.json`과 일치한다.
Cancer-Myth test는 세 fold 합계 724개(577 FPQ/147 NFP), CREPE 공식
test는 3,004개다. QA는 MedQA 1,273, PubMedQA 1,000, Medbullets 308,
MMLU 14,042, GSM8K 1,319로 총 17,942개다. PubMedQA는 공개 PQA-L
전체 풀이며 공식 500개 test 점수와 구분한다.

두 모델의 호출 없는 preflight가 완료됐다. 기존 테스트 12개와 각
데이터셋/라벨의 채점 템플릿 일치 검사 4개가 통과했다. 학습 전용 문항으로
Sonnet 5.5 제공 모델과 Well `Rating:` 형식을 확인했고, Qwen Plain·Direct
gate의 FPQ/NFP 생성·채점 4건을 `main/qwen25/smoke`에 저장했다.
이 smoke 점수를 본 평가 수치로 집계하지 않는다.

## 실행에 필요한 수정

원 코드의 bridge는 Cancer-Myth도 CREPE 전용 judge로 보내 preflight에서
실패했다. 데이터셋과 라벨에 따라 고정된 Well의 `CancerMyth_template.py`,
`CancerMythNFP_template.py`, `CREPEFPQ_template.py`, `CREPETPQ_template.py`를
선택하도록 연결을 수정했다. 원문 rubric은 수정하지 않았으며 각 템플릿의
`generate()` 출력과 렌더링 결과가 동일한지 검사했다. 수정 코드는 모델별
`identity.json`의 SHA256에 포함된다.

launcher의 `PAPER_MODELS` 옵션으로 준비된 모델만 독립 실행할 수 있다.
선택한 모델의 GPU만 잠그며 동시 집계는 `.summary.lock`으로 직렬화한다.
`run_paper_baselines_a6000_gpu01.sh`가 각 모델에서 fixed 완료 후 trained를
실행한다. 평가 backend 자체에는 자동 judge 재시도나 모델 fallback을
추가하지 않았다. 사용량 제한 후 재개는 아래의 별도 관리자가 담당한다.

## 시작/재개

아래 명령은 독립 세션을 시작한다. 이미 실행 중인 모델의 중복 시작은 GPU
lock으로 거부한다. 재개는 동일 코드/설정/분할/출력 경로를 사용한다.

```bash
cd /home/eagle0914/cancer-myth/external/paper_baselines_b81ace2
setsid -f bash scripts/run_paper_baselines_a6000_gpu01.sh qwen25 \
  </dev/null > /data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/main/logs/qwen25-supervisor.log 2>&1

# Gemma의 고정 가중치 다운로드 및 학습-only GPU smoke 완료 후 실행
setsid -f bash scripts/run_paper_baselines_a6000_gpu01.sh gemma3 \
  </dev/null > /data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/main/logs/gemma3-supervisor.log 2>&1
```

시작 시점에 Qwen 가중치는 로컬 캐시에 있었다. Gemma 가중치는 없고,
저장된 Hugging Face 인증으로 whoami와 Gemma config 다운로드가 모두
HTTP 401이었다. 따라서 Gemma 본 평가는 아직 시작하지 않았다.
접근 권한이 있는 계정으로 인증을 갱신해야 한다. 토큰을 로그/채팅에
출력하지 않는다.

```bash
HF_HOME=/data3/heejae/hf_cache \
  /data3/heejae/uv/paper_baselines_b81ace2/bin/hf auth login
```

사용자가 서버에 직접 접속할 수 없어 최신 CLI의 브라우저 device login을
별도 환경 `/data3/heejae/uv/hf_device_login`에 설치했다. 실험 환경의
Hugging Face/Transformers 버전은 변경하지 않았다. 사용자가 휴대폰
브라우저에서 코드 승인을 완료했으며, 서버 로그인의 성공과 고정 Gemma
가중치 다운로드 시작을 확인했다. 토큰은 사용자에게 전달받거나 출력하지
않았으며 최신 CLI가 인증/갱신 정보를 `HF_HOME`에 저장한다.

`scripts/bootstrap_gemma_device_login.sh`는 인증 승인 후 고정 리비전의
가중치 다운로드, 학습 전용 Plain/Direct gate 생성·Well 채점 smoke,
Gemma fixed→trained 본 평가를 순서대로 실행한다. 이 bootstrap의 로그는
`main/logs/gemma3-device-bootstrap.log`에 있다. 승인 코드가 만료되면 새
bootstrap으로 다시 발급해야 하며 코드 자체는 문서에 저장하지 않는다.

승인 후 Gemma의 고정 리비전 가중치 5개 shard(총 24,374,650,080 bytes)를
모두 다운로드했다. GPU 1 한 장에서 BF16 모델 로딩이 성공했고, 학습 전용
FPQ/NFP 각각에 Plain·Direct gate를 적용한 4건의 생성과 Well 채점이
완료됐다. 점검 중 GPU 1 VRAM 사용량은 약 24GB였다.
이후 Gemma도 `run fixed` 본 평가로 넘어갔다. GPU 0의 Qwen과 GPU 1의
Gemma가 각각 독립 세션에서 실행 중이며, 각 fixed 완료 후 trained를
이어 실행하도록 설정되어 있다.

## 상태/결과

- supervisor 상태: `main/logs/<model>-supervisor.state`
- worker 로그: `main/logs/<model>-run-fixed.log`, `...-run-trained.log`
- 생성/Well 채점: `main/<model>/table12/`
- QA 생성/채점: `main/<model>/table3/`
- 최종 집계: `main/tables/table1.csv`, `table2.csv`, `table3.csv` 및 JSON

본 평가의 모든 적격 문항이 채점되기 전까지 최종 비율/정확도는 pending으로
유지한다. 중단은 `main/<model>/STOP` 파일을 만들고 진행 중 호출을 기다린다.
재개 전에 STOP을 제거한다. Ours/Qwen3.8/NLA는 이 실행에 포함하지 않는다.

## Sonnet 사용량 제한 후 자동 재개

사용자 요청으로 `scripts/watch_paper_sonnet_quota.py`를 독립 세션에 연결했다.
기존 Qwen/Gemma 프로세스와 평가 코드, 설정, 프롬프트, 캐시 해시는 바꾸지
않았다. 각 평가 프로세스 종료 후 오류가 Sonnet 사용량 제한/429인지
확인한다. 해당 오류이면 600초 뒤부터 같은 Sonnet 5.5에 `READY`만 요청하는
짧은 확인 호출을 보낸다. 계속 제한되어 있으면 600초마다 반복한다.
두 모델이 대기 중이면 확인 호출 하나를 공유한다. 정상 실행 중에는
불필요한 주기 호출을 하지 않는다.

제공 모델과 `READY` 응답이 모두 확인되면 실패한 fixed/trained 단계의
기존 launcher를 다시 실행한다. 원래 캐시를 사용하여 저장된 생성·채점을
건너뛰고 이어간다. fixed를 재개했다면 완료 후 trained로 넘어간다. 모델
fallback이나 실패 문항의 점수 대체는 하지 않는다. 다른 오류는 자동
재개하지 않고 `failed_other`로 남긴다. `STOP`이 있으면 중단 상태를 유지한다.

- 관리 로그: `main/logs/sonnet-quota-watchdog.log`
- 상태/다음 ping 시각: `main/quota_watchdog/status.json`
- 확인·재개 이력: `main/quota_watchdog/events.jsonl`
- 재개 launcher 로그: `main/quota_watchdog/<model>-<phase>-attempt<N>.log`

공유 확인 주기, 두 모델의 실패 단계 보존, STOP 존중, 비사용량 오류의
재시도 제외, 오래된 quota 로그 제외, fixed→trained 전환 등 테스트
20개가 통과했다. 실제 Sonnet 5.5 `READY` 확인 호출도 성공했다. manager는
시작 시 기존 Qwen supervisor PID 2409042와 Gemma supervisor PID 2432772를
인계받았으며 채팅 종료와 무관하게 계속 실행된다.

## 2026-10-09 출력 길이 오류 복구

12:22 KST 점검에서 두 모델이 중단된 것을 확인했다. Sonnet 사용량 제한이
아니라 MMLU 답변이 기존 8,192 출력 토큰 한도에 도달한 오류였다.
Qwen은 QA 4,063개 다음 `mmlu:4347`, Gemma는 QA 3,820개 다음 `mmlu:1184`에서
중단됐고, 두 모델 모두 Cancer fold0 Plain 242건의 Well 채점은 완료했다.
기존 관리자는 이 오류를 `failed_other`로 기록했으므로 quota ping을 하지 않았다.

`scripts/run_paper_baselines_resilient.py`를 별도 실행 정책으로 추가했다.
기존 평가 구현, 모델/프롬프트/생성 설정, identity와 성공 캐시는 그대로
사용한다. 출력 한도에 도달한 평가 문항은 `generation_failures/`에 기록하고
나머지 문항을 계속 처리한다. 실패 문항의 답변/점수를 대체하지 않으며,
원래 적격 분모를 유지하여 해당 표는 pending/partial로 남는다. GEPA 학습
자체의 오류나 Sonnet 제한 등 다른 오류는 숨기지 않고 기존 관리자에 전달한다.

정책과 추가 runner 해시는 모델별 `execution_policy.json`에 별도 기록한다.
검증된 과거 오류는 원 로그를 보존하고 완성된 QA prefix/다음 문항을
확인하여 실패 기록을 만들었다. 예전 로그·status는
`main/logs/archive_before_length_failure_recovery_20261009T032704Z/`에 보존했다.
길이 오류 기록, 성공 결과 우선 처리, quota 오류 전파, 부분 상태 표시와
기존 quota 복구 테스트를 합쳐 25개가 통과했다.

12:27 KST부터 GPU 0(Qwen supervisor 3768893)과 GPU 1(Gemma supervisor
3768895)을 재개했고, quota manager 3769373을 다시 연결했다. manager의
후속 재개에도 같은 추가 runner가 사용된다. `PAPER_WORKER_SCRIPT`로
launcher의 실행 worker를 지정하며, 원문 평가 worker 파일은 수정하지 않았다.

## MMLU 문항 오류 처리 확대와 실행 범위

사용자 요청에 따라 `run_paper_baselines_mmlu_resilient.py`를 추가했다.
MMLU 평가 중 복구 가능한 문항별 RuntimeError/ValueError, 입력 토큰 한도,
생성 결과 parser 검증 오류, OOM은 실패 근거를 기록하고 다음 문항으로
넘어간다. OOM 이후에는 다음 문항 전에 사용하지 않는 CUDA 캐시를 해제한다.
Sonnet quota와 STOP은 기존 관리자에 전달하고, 파일/캐시/인증/치명적 GPU
오류는 문항 오류로 숨기지 않는다. 원래 길이 초과 처리도 유지한다.

새 정책/코드 해시는 모델별 `execution_policy_mmlu.json`에 별도 저장하며,
평가 코드·분할·프롬프트·생성 설정·성공 결과 cache는 그대로 유지한다.
실패 문항은 원래 분모에 남고 해당 결과는 pending/partial이다. 새로운
MMLU 오류 처리·OOM 정리 테스트 15개와 기존 길이 오류/쿼터 테스트 25개가
통과했다. 변경 적용 시 STOP으로 현재 답변을 저장한 뒤 재개했다.

이 변경 당시 큐는 Qwen/Gemma 각각 12개 기준선을 fixed→trained 순으로 처리했다.
Table 1은 탐지 기준선 5종과 both-stages 추가 행, Table 2는 답변 기준선
12종, Table 3는 같은 12종에 QA 5개 전체 분할이다. 현재 Plain을 처리
중이며 나머지 고정/학습형 조건은 후속 큐다. Ours/Qwen3.8은 제외되어 있다.

이 변경 당시 이번 Cancer/CREPE NLA 활성 추출→AV 설명→AR 복원은 아직
실행하지 않았다. `/data3/heejae/medical_nla`에 있는 July DDxPlus 실험
결과를 이번 Cancer/CREPE 결과로 간주하지 않는다. Probe 기준선의 질문
hidden state 추출은 trained 단계에 포함되지만 NLA AV/AR와는 별도 작업이다.

## 2026-10-09: FPQA 우선 실행과 기존 NLA 연결

사용자 요청에 따라 GPU 0/1 작업을 STOP으로 안전하게 끝낸 뒤,
`run_paper_baselines_fpqa_first_gpu01.sh`로 재개했다. 실행 phase는 `all`이며
모든 12개 방법의 Cancer-Myth/CREPE 작업이 일반 QA보다 먼저 진행된다.
방법별로 세 Cancer fold와 CREPE를 처리한 뒤 다음 방법으로 이동한다.
원래 평가 identity와 성공 cache는 그대로이며 별도
`execution_policy_fpqa_first.json`에 실행 순서와 코드 해시를 기록했다.
기존 MMLU 오류 건너뛰기와 10분 간격 Sonnet quota 감시/재개도 유지한다.
관련 기존 오류/쿼터/새 순서 테스트 41개가 통과했다.

NLA는 새로 학습하지 않는다. `/home/eagle0914/medical_nla` 코드와 기존
`/data3/heejae/uv/medical_nla` 환경을 재사용하며, 공식 AV/AR 체크포인트를
고정 리비전으로 연결한다. 이전 Gemma AV 가중치도 다른 캐시 경로
`/data3/heejae/hf_cache/models--kitft--nla-gemma3-12b-L32-av`에 있다.
동일 리비전의 새 캐시와 SHA256을 비교한 뒤 기존 가중치를 hardlink로
재사용한다. 검사 기록은 `gemma3/existing_av_reuse.json`에 저장한다.

| GPU | 소스 활성 | 공개 AV/AR |
| --- | --- | --- |
| 2 | 기존 기준선 Qwen2.5-7B-Instruct, hidden_states[20], d=3584 | kitft/nla-qwen2.5-7b-L20-{av,ar} |
| 3 | 기존 기준선 Gemma-3-12B-it, hidden_states[32], d=3840 | kitft/nla-gemma3-12b-L32-{av,ar} |

실험 설정: `configs/paper_baselines/nla_connection_gpu23.json`.
결과: `/data3/heejae/cancer-myth/nla_connection_gpu23_20261009/{qwen25,gemma3}`.
Cancer fold0 train과 CREPE official train에서 FPQ/NFP 각 1개를 ID 순으로
선정하고 Plain/Balanced를 적용한다. 기준선과 동일한 소스 revision,
system/user 메시지와 chat template을 사용한다. 질문 마지막 토큰과
assistant 시작 경계의 마지막 토큰 두 위치만 추출해 모델당 16개 벡터를
저장했다. 전체 토큰/레이어 저장, 답변 재입력, 평가 문항 기반 특징 선택은 없다.
**이번 활성은 답변 생성 전 prompt forward이며, 당시 답변 생성의 hidden
state를 복원한 것으로 해석해서는 안 된다.**

`nla_connection.py`는 데이터/메시지/위치 연결만 담당한다. 활성 선택은
기존 `medical_nla.src.extract_activations.select_activation`을 사용한다.
AV는 기존 `src.run_nla.main`을 그대로 호출하고, wrapper는 고정된 로컬
sidecar 파일을 읽게 해준다. 주입 토큰/이웃/배율/공식 프롬프트는 sidecar
값 그대로다. AR는 공식 `kitft/nla-inference` 커밋
`38b802a33d1d317f21b6825a9116f388c2141f86`의 `NLACritic`을 사용한다.
AR 점수는 양쪽 벡터를 공식 mse_scale로 정규화한 방향 복원 MSE와 cosine이다.
질문 텍스트만 AR에 넣는 대조 조건도 함께 저장한다. 복원 점수를 설명의
사실적 정확성이나 Table 1/2 성능으로 간주하지 않는다.

두 독립 프로세스는 GPU별 lock을 잡고 source → unload → AV → unload → AR
순서로 처리한다. 활성 추출은 기준선 환경, AV/AR는 기존 medical_nla 환경이다.
가중치 다운로드 완료를 자동으로 기다리며 로그/실패는 보존된다.
주요 파일은 `pipeline_status.json`, `download_status.json`, `provenance.json`,
`manifest.jsonl`, `av_results.jsonl`, `scored.jsonl`, `summary.json`이다.
선정/학습 분할 검증 2개와 기존 NLA 위치·주입·복원 테스트 16개가 통과했다.

연결 실험은 두 모델 모두 완료했다(Qwen 03:57:04 UTC, Gemma 03:59:57 UTC).
모델당 16개, 총 32개에서 공식 explanation 태그 파싱과 AR 복원 점수의
유한성/범위 및 `MSE = 2(1-cos)` 관계를 확인했다. 평균 방향 복원 MSE는
Qwen 0.252254, Gemma 0.009850이며, 질문 텍스트 대조 조건은 각각
0.814639, 0.026167이다. 모델 간 수치만으로 설명의 품질을 비교하지 않는다.
각 모델 및 최상위 폴더에 `connection_metrics.csv`를 저장했다.
Gemma 기존 AV의 safetensors 5개(23,532,141,328 bytes)는 두 캐시의 실제
SHA256 일치를 확인하고 기존 파일로 재사용했다. 원래 NLA 코드/환경과
이전 DDxPlus 결과는 변경하지 않았다.

## GPU 2/3: 실제 답변 생성과 희소 활성 비교 시작

사용자가 GPU가 필요한 작업부터 즉시 시작하도록 요청하여
`configs/paper_baselines/nla_generation_comparison_gpu23.json`의 별도 실행을
2026-10-09 13:20:48 KST에 시작했다. 이전 연결 실험 결과는 그대로 보존한다.
GPU 2 Qwen, GPU 3 Gemma가 각각 원래 기준선 소스 revision/BF16/SDPA/greedy,
max_input_tokens=8192, max_new_tokens=8192 설정으로 답변을 생성한다.

Cancer fold0 train 및 CREPE official train에서 각 FPQ 5개·NFP 5개, 총
20문항의 ID를 동일하게 선택해 두 모델의 Plain/Balanced를 비교한다.
각 모델 40개 답변이며 최대 280개 활성 벡터를 저장한다. 문항은 ID 순으로
선정하되 train 내 중복 group을 피한다. 위치 선택에 gold는 사용하지 않는다.
각 데이터셋 해당 train 질문에만 TF-IDF를 맞춰 질문별 상위 두 term의
마지막 subtoken을 추출한다. TF-IDF 키워드는 전제 오류를 정확히 짚은
oracle 위치가 아니므로 실제 질문/선택 단어와 함께 검토해야 한다.

소스 모델의 지정 decoder layer에 읽기 전용 forward hook을 걸어
질문 keyword 2개, 질문 마지막 토큰, assistant 경계의 마지막 토큰,
답변의 alphanumeric token rank 1/8/32만 저장한다. layer[L-1]의 출력은
내부 layer의 hidden_states[L]와 대응한다. 실제 generate() 도중 prefill과
decode를 구분해 추출하며 답변을 재입력하지 않는다. decode에서 저장한
token ID가 최종 생성 token ID와 일치하는지도 확인한다. 짧은 답변에서
관측하지 못한 위치는 unobserved로 기록하고, 8192 한도에서 끝나지 않은
답변은 completed=false / finish_reason=length로 보존한다.

`run_nla_generation_comparison_gpu23.py`는 GPU별 lock과 독립 세션을 유지하고
원본 생성/추출 → 모델 해제 → 기존 AV → 모델 해제 → 공식 AR 순서로 실행한다.
완료한 generation case는 개별 JSON과 벡터로 저장해 재개 시 재사용한다.
아직 Well 채점은 요청한 GPU 작업에 포함하지 않았으며 완료 상태에도
pending으로 표시한다. 실제 생성 token/층 정렬과 train-only 선택 테스트
2개가 통과했다.

결과는 `/data3/heejae/cancer-myth/nla_generation_comparison_gpu23_20261009/`
아래 모델별 `answers.jsonl`, `manifest.jsonl`, `generation_status.json`,
`pipeline_status.json`, `av_results.jsonl`, `scored.jsonl`에 저장되며 마지막에
질문/실제 답변/NLA 설명/복원 점수를 묶은 `comparison.csv`를 만든다.

## 20문항 제한 해제: 전체 학습 분할로 확대

사용자의 범위 수정에 따라 2026-10-09 13:30 KST부터
`configs/paper_baselines/nla_full_train_gpu23.json`으로 전환했다.
Cancer fold0 train 385개(FPQ 307/NFP 78), CREPE official train
3,437개(FPQ 904/NFP 2,533), 총 3,822개를 빠짐없이 포함한다.
Plain/Balanced를 모두 생성하므로 모델당 7,644개, 두 모델 합계 15,288개
답변이다. fold0의 train 내 같은 group 변형도 모두 포함한다. dev/test는
특징 발견 범위에 넣지 않는다. GPU 0/1의 본 기준선 평가는 별도로 계속된다.

`run_nla_full_gpu23.py`는 기존 파일럿 프로세스만 확인해 전환한다.
현재 문항 저장을 최대 60초 기다린 뒤 이전 독립 process group을 종료하고,
프로세스/GPU lock이 해제된 뒤 새 작업을 시작한다. 완료 사례와 활성 파일은
삭제하거나 변경하지 않는다. 저장되지 않은 진행 중 문항은 다시 생성한다.
새 작업은 20문항 파일럿을 prefix로 두고 기존 case JSON과 벡터 경로를
재사용한다. 실제 full train ID 집합의 완전 포함과 실제 generate()의 token/층
정렬 테스트 2개를 확인했다. 답변·manifest 인덱스는 사례별로 append하여
수만 개 벡터의 전체 파일을 반복해서 다시 쓰지 않는다.

답변/희소 활성 추출이 끝나면 NLA 처리 순서를 지정한다. Balanced에만
교정 표현이 나타난 쌍, 답변 텍스트가 달라진 쌍, 나머지 쌍 순이다.
이것은 탐색 순서를 정하는 문자열 휴리스틱이며 정확성 판정이 아니다.
7개 지정 위치만 대상으로 하며, 우선순위가 낮은 문항을 분모에서 제외하지
않는다. AV는 기존 NLA 코드, AR는 기존 공식 NLACritic을 그대로 사용한다.
대규모 AR 출력은 `nla_full_reconstruction.py`에서 사례별로 append하고
재개 시 저장한 점수를 재사용한다. Well 채점은 별도 pending 상태다.

새 결과 위치는
`/data3/heejae/cancer-myth/nla_full_train_gpu23_20261009/{qwen25,gemma3}`이다.
이전 16벡터 연결 실험과 20문항 파일럿 결과는 별도 폴더에 보존한다.

## 2026-10-12: 상태 확인과 주간 쿼터 중 생성 재개

NLA 두 모델 모두 7,644개 답변을 EOS까지 생성했다. 지정 활성은 Qwen
53,461개, Gemma 53,454개이며 짧은 답변의 관측 불가 토큰 위치는 제외해
최대 53,508개보다 적다. 10월 12일 10:26 KST 무렵 AV 설명은 Qwen
44,638개(83.5%), Gemma 3,020개(5.6%)였고 GPU 2/3에서 계속 생성 중이다.
전체 AR 단계는 아직 시작하지 않았으며, 완료한 16벡터 연결 실험과 구분한다.

본 기준선은 Sonnet 주간 사용 한도에 걸려 GPU 0/1이 비어 있었다.
로그의 리셋 안내는 Oct 15 17:00 UTC, 즉 10월 16일 02:00 KST이다.
중단 당시 Well 완료량은 Qwen Cancer Plain/Balanced 각각 724개, CREPE
Plain 3,004개·Balanced 2,935개이며 다음 답변 1개는 생성만 됐다.
Gemma는 Cancer Plain 724개, CREPE Plain 2,288개와 생성만 된 다음 1개다.
기존 QA Plain 저장량은 Qwen 4,311개, Gemma 3,926개이고 완료된 Table 3
전체 결과로 해석하지 않는다. 전체 12개 방법은 여전히 평가 대상이다.

쿼터를 기다리면서 GPU 작업을 진행하기 위해
`run_paper_fpqa_generation_only.py`를 추가하고 10:29 KST 무렵 GPU 0/1에서
`run_paper_generate_then_score_gpu01.sh`를 시작했다. Sonnet 없이 생성할 수
있는 고정 7개와 TF-IDF/Probe 2개, 총 9개 방법의 Cancer/CREPE 답변을
먼저 생성한다. GEPA 3개는 원래 Sonnet reflection이 필요하므로 후속으로
유지한다. 이 추가 큐에서는 QA를 진행하지 않고 FPQA 생성에 집중한다.
생성이 끝나면 원래 전체 12개 평가/채점 큐로 이어지고, 쿼터에 다시 막히면
기존 600초 간격 감시/자동 재개가 작동한다. 성공 답변·기존 Well 점수·모델
revision·프롬프트·생성 설정·평가 identity는 그대로다.

생성만 완료한 상태는 answers_generated, Well 채점은 pending으로 표시한다.
별도 `execution_policy_generation_only.json`에 실행 정책과 코드 해시를
보존했다. 생성 전용의 외부 reflection 제외/QA 제외 테스트와 기존 순서·쿼터
테스트 합계 23개가 통과했다. 이전 쿼터 상태/로그는 archive 폴더에 남겼다.
