# Table 2·3 서버별 실험 분배와 A6000 서버 인계

**담당 확정: 현재 4090 서버는 Qwen2.5·Luna·임베딩/SAE·공통 평가, 별도 A6000 서버는 Gemma와 Qwen3.8을 맡는다.** 서버가 달라도 [실험 설계 v2](table2_experiment_protocol_2026-10-07.md), [고정 설정](table2_run_settings_2026-10-08.json), [프롬프트 원문](table2_prompt_registry_2026-10-07.json)은 같다. 데이터셋별 독립 최적화이며 Cancer grouped 3-fold, CREPE 공식 split을 유지한다.

이 문서는 다른 서버 작업자가 그대로 읽고 실행 준비를 할 수 있는 지시서다. A6000 서버에 원격 접속하거나 작업을 전송한 상태는 아니다. 호스트 주소·로그인 정보는 Git에 넣지 않는다.

## 1. GPU 배치

GPU 번호는 **서버 내부 번호**다. 시작 시 UUID와 모델명을 확인하고 다른 작업이 사용 중인 GPU는 점유하지 않는다.

| 서버 | GPU | 담당 | 최초 작업 |
|---|---|---|---|
| 현재 4090 서버 | 0 | Qwen2.5-7B | 공통 메시지 학습 문항 smoke → 고정 baseline → Direct/CoT gate |
| 현재 4090 서버 | 1 | Qwen3-Embedding-0.6B, SAE | Luna 답변 쌍의 임베딩 → SAE feature_fit/check |
| 현재 4090 서버 | 2·3 | 예비 자원 | 기본 비워 둠; 런타임 smoke·중복 없는 후속 배치 또는 합의된 FP8 대체 배정 |
| 현재 4090 서버 | CPU/외부 호출 | TF-IDF, Luna, judge, reflection, 집계 | 공통 분할 고정·기존 답변 audit·Luna 학습 답변·공통 채점 |
| 별도 A6000 서버 | 0 | Gemma 4 12B | BF16 로드·학습 문항 smoke 후 고정 baseline |
| 별도 A6000 서버 | 1 | Qwen3.8-27B-FP8 OFF/ON | GPU/커널 호환성 확인 후 동일 모델 상주·모드별 작업 |
| 별도 A6000 서버 | 0·1, 필요 시 | 단일 큰 모델 | 한 장으로 부족할 때만 Gemma와 Qwen을 순차 실행; 두 장 예약 중 다른 작업 금지 |

현재 서버는 직접 조회에서 RTX 4090 24GB 4장이 확인됐다. A6000 2장은 사용자 제공 정보이며 그 서버의 실제 메모리·사용량·드라이버는 아직 확인하지 않았다. 배치 크기는 1에서 시작해 학습 문항으로 2, 4 순서로 검사한다. OOM이면 프롬프트나 모델 정밀도를 바꾸기 전에 batch size를 줄인다. GPU별 서로 다른 모델이 병행해도 모델별 프롬프트·채점 규약은 유지한다.

Luna/채점/반성 호출은 로컬 GPU를 쓰지 않는다. API/CLI concurrency는 계정 전체 한도를 따르며 최초 동시 호출 상한은 역할별 2개, 같은 provider 전체 4개로 둔다. 역할별 제한보다 전체 제한이 우선이다. GPU 생성이 빠르다고 judge를 무제한 실행하지 않는다.

## 2. A6000 서버에서 특히 먼저 확인할 것

**Qwen FP8는 메모리에 들어간다고 바로 실행 가능한 것은 아니다.** vLLM 공식 호환표는 Ampere에서 FP8 W8A8을 지원하지 않고, 별도 Marlin FP8 경로는 지원하는 것으로 구분한다. 실제 checkpoint의 block-wise quantization, 모델 아키텍처, 설치 버전까지 맞아야 한다. [vLLM 공식 하드웨어 호환표](https://docs.vllm.ai/en/latest/features/quantization/#supported-hardware)

1. `nvidia-smi`와 runtime의 compute capability로 실제 카드가 RTX A6000인지 확인한다. RTX 6000 Ada와 혼동하지 않는다.
2. Gemma BF16부터 로드·출력 smoke를 수행한다. 기존 스크립트의 `CUDA_VISIBLE_DEVICES=0,1` 고정 assert와 device map을 복사해서 한 장용 실행이라고 주장하지 않는다.
3. Qwen은 공식 FP8 checkpoint를 그대로 사용한 **지원 커널의 호환성 검사**부터 한다. 잘못된 dtype 변경, missing weights, CPU offload, NaN, OFF/ON 파싱을 검사한다. 모델은 한 번 올리고 OFF/ON을 순서대로 처리한다.
4. Marlin/다른 engine이 필요하면 backend·activation dtype·quantization 차이를 별도 runtime revision에 기록하고 학습 질문의 출력·점수 안정성을 확인한다. 종전 4090 커널과 수치가 같다고 가정하지 않는다.
5. 호환되는 실행이 불가능하면 Qwen 작업을 `blocked_runtime`으로 보고하고 Gemma는 계속 진행한다. **BF16/AWQ 등 다른 checkpoint로 임의 대체하지 않는다.** Qwen을 4090 GPU 2·3으로 옮기는 것은 allocation의 owner 변경과 A6000 작업 중단 확인 후 한 번만 수행한다.

## 3. 두 서버가 함께 쓸 입력

4090 서버가 공통 입력 bundle을 생성·검사하는 담당이다. A6000 서버에서 fold를 다시 랜덤 생성하지 않는다.

```text
results/table2_v2/protocol/
  split_manifest.json          # eligible/excluded/group/outer/inner/feature IDs
  question_inputs.jsonl        # 추론용 whitelist; 정답·label·reference 제외
  evaluation_annotations.jsonl # evaluator 전용; task model에 전달 금지
  smoke_ids.json               # 학습 영역에서 선정한 고정 smoke 질문
  protocol_bundle.json        # 상대 경로와 SHA-256, protocol_id, source revisions
```

이는 **준비할 파일 규격**이며 현재 생성 완료 목록이 아니다. 공통 bundle이 확정되기 전에는 두 서버 모두 test 생성·최적화를 시작하지 않는다. 그동안 GPU inventory, 설치 환경 분리, checkpoint 확인 등은 진행한다. 입력 bundle은 별도 artifact 전송으로 공유하고 양 서버에서 SHA-256을 검증한다. Git pull만으로 Git에서 제외된 데이터·체크포인트·결과가 옮겨오지는 않는다.

각 서버는 코드 commit과 working-tree 변경 파일의 해시도 기록한다. 실행 중 자동 pull로 코드를 바꾸지 않고 작업 종료 후 업데이트한다. 지금 저장소에는 과거 실행과 새 설계가 함께 있으므로 프로토콜 이름만 같은 상태를 호환으로 인정하지 않는다.

## 4. A6000 서버 작업자에게 전달할 지시

> 이 서버는 Table 2·3의 Gemma 4 12B와 Qwen3.8-27B-FP8 OFF/ON을 담당합니다. 최신 `server_handoff_2026-10-08.md`와 설계 v2를 읽고, 로컬 GPU·checkpoint·런타임을 먼저 점검하세요. 데이터셋별 독립 최적화이며 Cancer 3-fold 배정은 4090 서버에서 만든 manifest를 그대로 받습니다. Prompt registry를 모델에 맞춰 임의 수정하지 마세요.
>
> Gemma는 GPU 한 장 BF16부터, Qwen은 다른 한 장에서 FP8 커널 호환성을 먼저 확인하세요. 두 장이 필요하면 다른 큰 모델과 순차 실행하세요. A6000에서 FP8가 지원되지 않으면 상태와 오류를 남기고 Gemma 작업을 진행하세요. 다른 정밀도·모델로 바꾸거나 4090 쪽 작업을 중복 실행하지 마세요.
>
> 공통 manifest와 새 runner의 메시지/파서 검증이 끝나면 Plain/Balanced/CoT → PreWoMe/Extract+Verify → Direct/CoT/Probe gate 순서로 채우세요. GEPA는 모델별 로컬 endpoint에 연결해 별도 dataset/fold로 돌립니다. SAE는 4090에서 검증된 구현을 받되 각 모델·데이터셋·fold의 학습 구역에서 다시 특징을 만들어야 합니다. Luna의 특징을 복사해 Ours라고 하지 마세요.
>
> 최종 채점과 reflection 호출은 4090 서버의 공통 담당자에게 요청/답변 artifact를 넘겨 처리하세요. 자체 judge로 빈칸을 채우지 마세요. 게이트 판정, 검토문, 최종 답변, 원 응답, 모델 revision, 메시지 해시, 종료 사유, usage를 보존하세요. 완료·실패 수와 artifact hash를 반환하면 4090에서 수집·채점·표 반영합니다.

### 실제로 지금 실행 가능한 점검 명령

원격 작업공간에 미커밋 변경이 있으면 별도 checkout에서 최신 main을 받는다. 작업공간이 깨끗한 경우:

```bash
git pull --ff-only origin main
python3 scripts/table2_server_preflight.py --help
```

HF cache의 **해당 서버 실제 경로**를 지정한다. 아래 명령은 환경 점검 보고서만 쓰고 모델을 실행하지 않는다.

```bash
python3 scripts/table2_server_preflight.py \
  --server a6000 \
  --cache-root /YOUR/LOCAL/HF_CACHE/hub \
  --out results/table2_v2/preflight/a6000_initial.json
```

Python 3.12 이상과 GPU 추론 전용 환경을 준비한다. `scripts/setup_fpqa_prompting.sh`의 CPU/CLI 환경만으로 GPU 실행 준비가 끝나는 것은 아니다. 설치된 패키지는 실제 GPU 실행에 사용할 Python으로 다시 점검한다. 기존 `run_gemma4_diagnostics.py`, `run_qwen38_transformers.py`는 과거 프롬프트·분할·생성 설정을 포함하므로 새 baseline 전체 실행 명령으로 그대로 사용하지 않는다. **현재 제공하는 확정 실행 명령은 preflight까지**이며, 통합 runner가 준비되면 그 commit과 명령을 이 문서에 추가한다.

## 5. 실행 소유권과 결과 교환

- [서버 배정 JSON](table2_server_allocation_2026-10-08.json)을 단일 소유권 표로 사용한다. Qwen2.5와 임베딩은 4090, Gemma/Qwen3.8은 A6000이다. GPU 번호 조정은 local map만 변경하고 모델의 owner를 조용히 바꾸지 않는다.
- `job_id = protocol_id / dataset / model_setting / method / split / opt_seed / generation_replica / stage`로 만든다. 같은 job_id는 한 서버만 쓴다. 데이터셋·OFF/ON·fold·replica를 경로에서 생략하지 않는다.
- 결과 경로: `results/table2_v2/<server>/<dataset>/<model_setting>/<method>/<split>/seed42/rep0/`. 고정 baseline 전체 답변의 split은 `fixed_all`, Cancer 학습 방법은 `fold0..2`, CREPE는 `official`이다. 고정 baseline은 manifest로 나눠 OOF 집계하며 세 벌 생성하지 않는다.
- 각 job은 `run.json`, `status.json`, `answers.jsonl`, `intermediates.jsonl`, `errors.jsonl`, `artifact_manifest.json`을 남긴다. `status`에는 planned/preflight/running/complete/blocked_runtime/failed/stopped 및 완료·예상·오류 수를 둔다. 서버 간 공유 디렉터리가 없다면 local flock만으로 중복을 막았다고 주장하지 않는다. owner 파일과 전달 기록으로 조정한다.
- checkpoint·임베딩 배열·대량 원 응답은 별도 artifact로 옮긴다. Git에는 프로토콜·manifest·검증된 요약·artifact 위치/해시를 올린다. 비밀키·로그인 설정은 공유하지 않는다.
- 4090에서 수집 시 코드/protocol/입력/메시지/모델/replica 해시를 대조한다. 호환되는 judge 점수만 재사용한다. 완료 조건이 맞는 값만 원장에 넣고 PNG를 다시 만든다.

### GEPA의 서버 간 연결

Gemma/Qwen의 task model은 A6000에서 로컬 endpoint로 서비스하고, GEPA optimizer·judge·reflection의 중앙 큐는 4090에서 운영한다. 소유자는 중앙 optimizer지만 task inference owner는 A6000이다. endpoint는 인증된 사설 연결/SSH 터널을 사용하고 공개 네트워크에 무인증으로 노출하지 않는다. 초기에는 큰 모델 endpoint 동시 요청을 1개로 시작한다. fold·후보가 달라도 checkpoint는 재로드하지 않는다.

서버 간 연결이 아직 없으면 고정 baseline 생성 artifact를 먼저 교환한다. GEPA를 완료했다고 기록하거나 judge를 임의 대체하지 않는다. 원격 endpoint 주소·연결 정보는 실행 환경에만 두고 Git에 쓰지 않는다. 학습-only 요청과 outer-test 요청의 입력 채널을 분리한다.

## 6. 현재 4090 서버에서 진행한 준비와 다음 작업

- [실제 4090 환경 점검 보고서](local_4090_preflight_2026-10-08.json)를 저장했다. 이는 기록 시점의 스냅샷이며 실행 직전 사용량은 다시 확인한다. 스크립트·설정 파일 SHA-256과 모델 호출 0건을 검증했다.
- GPU 4장 및 공통 CPU/CLI Python 3.12 환경을 확인했다. 해당 Python에는 torch/transformers/sklearn/sentence-transformers가 없어 GPU 실험 환경을 별도로 준비해야 한다. 다른 환경에도 GPU 패키지가 없다는 뜻은 아니다.
- Qwen2.5, Gemma, Qwen3.8 캐시 경로를 확인했다. 메타데이터 확인은 전체 weight 검증이나 model-load 성공과 다르다. 지정한 HF cache에는 Qwen3-Embedding-0.6B가 아직 없다. 공통 split manifest도 아직 없다. 따라서 임베딩 설치·revision 고정, GPU 실행 환경 준비, 공통 manifest와 runner 통합이 새 모델 호출의 선행 작업이다.
- 이번 작업은 **서버 배정·인계 문서와 로컬 preflight 실행**이다. 새 baseline/SAE 모델 호출은 아직 시작하지 않았다.
- 다음 순서는 공통 그룹 감사·manifest 동결 → GPU 전용 환경/runner 통합 → 4090 Qwen2.5와 Luna 학습 답변 + A6000 Gemma 고정 baseline 병행 → 4090 SAE 예비 검증이다. SAE 검증이 실패해도 기존 baseline 수집은 계속한다.

Table 3은 먼저 Cancer에서 학습한 결과물을 모든 QA에 적용하고, **CREPE에서 학습한 결과물도 모든 QA에 적용**한다. 두 출처를 별도 패널로 보고한다. 이는 사용자의 후속 결정이며 설계 v2의 Cancer-only 초안보다 우선한다. 고정 Plain/Balanced/CoT는 공유하고, 학습 방법과 few-shot 팩은 출처별로 구분한다.
