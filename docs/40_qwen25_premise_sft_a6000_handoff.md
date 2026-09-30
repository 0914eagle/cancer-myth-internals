# Qwen2.5 목표 전제 SFT: A6000 두 장에서 한 실험씩 독립 실행

이 문서는 **실행 준비물과 계획**이다. 현재 머신에서는 모델 다운로드·GPU 학습을 실행하지 않았다. 다른 서버의 드라이버와 실제 최대 배치는 아래 preflight/smoke로 확인한다.

## 1. 실험 목적과 비교

목표는 **거짓 전제를 교정하면서 정상 질문을 불필요하게 반박하지 않는 답변**이다. 별도의 gate나 답변 라우팅을 필수로 두지 않는다. 현재 관찰은 목표 오류 누락·관련 설명만 제시·원래 가정 유지 등이며, 이 모두가 전제 추출 실패 때문이라는 인과 결론은 아니다.

현재 질문: **같은 Qwen2.5에 목표 전제를 학습시킬 때, Cancer-Myth 내부 학습과 CREPE 학습 후 전이가 Cancer-Myth 평가에서 어떻게 다른가?** 탐지 출력과 실제 질답 전이를 각각 평가하며, 목표 전제를 잘못 잡는 것이 유일한 원인이라고 전제하지 않는다.

모든 모델은 **Qwen/Qwen2.5-7B-Instruct**, revision `a09a35458c702b33eeacc393d103063234e8bc28`에서 **각각 독립적으로** 시작한다. 이전 adapter에 이어 학습하지 않는다. BF16 LoRA이며 full fine-tuning이나 QLoRA가 아니다.

| 실행 | GPU | 학습 자료 | target | 평가 |
|---|---|---|---|---|
| Base | 학습 없음 | 없음 | 없음 | 같은 Cancer-Myth 평가 문항의 기준 결과 |
| CM-premise | 0 한 장 | CM fit: FPQ 349 + NFP 89 = 438개 | 오류 유무 + 있다면 목표 전제 | CM dev/test |
| CREPE-premise | 1 한 장 | CREPE train의 FPQ·normal | 동일 | **위와 같은 CM dev/test** |

**두 장을 한 학습에 묶지 않는다.** GPU 0과 GPU 1은 서로 독립적인 모델·optimizer·로그를 가진다. 두 학습은 동시에 실행할 수 있다. CREPE adapter를 가져온다는 것은 CREPE에서 학습한 adapter를 CM 질문에 적용한다는 뜻이며, 그 뒤에 CM으로 추가 학습하지 않는다.

이번 기본 실행은 **두 premise 조건만**이다. 이진 라벨만 학습하는 조건이나 최종 답변 SFT는 자동 실행하지 않는다. 기본 전제 학습에는 기존 Qwen 생성 답변이 필요 없다. CM과 CREPE는 데이터 수·라벨 비율·학습 step이 다르므로 자연 크기 비교만으로 순수한 도메인 효과를 분리했다고 주장하지 않는다. 그런 주장이 필요하면 학습 크기·업데이트 수를 맞춘 추가 대조를 설계한다.

### 실제 input / target

입력은 원 질문을 그대로 user turn에 넣는다. 실험자가 추가하는 system prompt, '판단하라' 지시, 정답 라벨, 참조 설명은 없다. **Qwen의 native chat template이 자동으로 넣는 기본 system 문자열은 그대로 유지**하며 이는 모든 조건에서 공통이다.

```text
user: [원 질문]

assistant target (FPQ, premise):
{"has_false_premise": true, "false_premises": ["주석된 거짓 전제"]}

assistant target (NFP, premise):
{"has_false_premise": false, "false_premises": []}

선택적 후속 조건의 assistant target (joint, 이번 실행 아님):
{"has_false_premise": true, "false_premises": ["주석된 거짓 전제"]}

Answer:
[학습용 최종 답변]
```

질문·system·assistant 시작 헤더는 loss에서 제외하고 **assistant target 및 종료 토큰만** 학습한다. 목표 전제는 질문의 모든 가정이 아니라 주석된 거짓 전제다. NFP의 TPQ 전제 주석을 거짓 전제 target으로 사용하지 않는다. `binary`는 JSON의 `has_false_premise`만 학습한다. `answer`는 JSON 없이 답변만 학습한다.

## 2. 경로

```text
/data3/heejae/
  tools/                         # uv가 없을 때 uv 설치
  uv/cancer-myth/                 # Python 3.11.11 가상환경
  uv_cache/                      # uv 패키지 캐시
  uv_python/                     # uv가 관리하는 Python 설치
  hf_cache/                      # 모델/HF 다운로드 캐시
  cancer-myth-code/               # 전달한 코드 압축을 푸는 위치
  cancer-myth/
    downloads.json               # 모델·데이터 revision 고정 기록
    raw/{cancer,crepe}/           # 다운로드 원본 JSONL
    data/{cancer,crepe}/          # fit/dev/test, 제외 기록, 해시
    runs/                        # LoRA adapter와 학습 로그
    eval/                        # 생성 결과와 집계
    logs/                        # nohup stdout/stderr
    environment.json             # 실제 GPU·드라이버·torch 확인
    environment.freeze.txt       # 설치된 전이 의존성까지 정확한 버전 기록
```

## 3. 설치: PyTorch를 최신 버전으로 받지 않기

고정 버전은 `configs/premise_sft/requirements.txt`에 있다.

| 항목 | 버전 |
|---|---|
| Python | 3.11.11 |
| PyTorch | **2.5.1+cu121** |
| wheel CUDA runtime | **12.1** |
| Transformers | 4.46.3 |
| PEFT | 0.13.2 |
| Accelerate | 1.0.1 |
| Datasets | 3.1.0 |
| Tokenizers | 0.20.3 |
| huggingface-hub | 0.36.2 |

기존 Qwen 실행의 torch는 `2.5.1+cu121`이었다. 이전 서버는 535 계열 드라이버였으므로 그 구성을 기준으로 한다. **새 서버 드라이버가 같다고 확인된 것은 아니다.** `nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 상한이며 설치된 torch wheel의 CUDA 버전과 다르다. CUDA 12.x minor compatibility의 Linux 드라이버 최소 조건은 일반적으로 525.60.13이지만 기능 제약이 있으므로 숫자만으로 성공을 보장하지 않고 GPU BF16 연산과 단일 GPU 학습 smoke를 실행한다. 드라이버를 자동 변경하지 않는다.

공식 근거: [PyTorch 이전 버전 설치 명령](https://pytorch.org/get-started/previous-versions/), [NVIDIA CUDA minor compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

### 코드 전달

다른 서버에서 저장소의 `main`을 받는다. 실험 코드와 작은 데이터 묶음은 Git에 포함돼 있으며, 모델 가중치는 다음 다운로드 단계에서 받는다.

```bash
git clone --branch main https://github.com/0914eagle/cancer-myth-internals.git /data3/heejae/cancer-myth-code
cd /data3/heejae/cancer-myth-code
git rev-parse HEAD
nvidia-smi
bash scripts/setup_premise_sft.sh
source scripts/premise_sft_env.sh
```

private 저장소 접근은 해당 서버의 GitHub 인증이 필요하다. HTTPS 대신 인증된 SSH URL을 써도 된다. 이미 clone한 저장소가 있으면 작업 변경을 보존하면서 `git pull --ff-only`로 갱신한다. 아래 압축본 전달은 Git 접근이 어려울 때의 대안이다. 두 방식을 모두 실행하지 않는다.

```bash
mkdir -p /data3/heejae/cancer-myth-code
tar -xzf /path/to/qwen25_premise_sft_handoff.tar.gz -C /data3/heejae/cancer-myth-code
cd /data3/heejae/cancer-myth-code
sha256sum -c HANDOFF_SHA256SUMS
nvidia-smi
bash scripts/setup_premise_sft.sh
source scripts/premise_sft_env.sh
```

setup은 이미 설치된 uv가 있으면 재사용하고 없으면 **uv 0.6.17** 설치기를 받아 `/data3/heejae/tools`에 설치한다. 설치 후 uv 버전도 기록한다. Python 환경은 `/data3/heejae/uv/cancer-myth`다. 기존 동명 환경을 다른 작업에서 쓰고 있다면 새 경로의 별도 환경으로 준비한다.

torch 설치의 핵심 명령은 다음과 같다. setup 안에 이미 들어 있다.

```bash
uv pip install --python /data3/heejae/uv/cancer-myth/bin/python \
  'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
```

나머지 의존성을 설치할 때에도 요구사항에 `torch==2.5.1+cu121`을 다시 걸어 교체를 막는다. **`uv pip install -U torch`, `uv sync`, 기존 프로젝트의 무제약 `pip install -e .`를 이 환경에 실행하지 않는다.** 이 실험은 torchvision/torchaudio/vLLM/flash-attn/bitsandbytes가 필요 없다. 전이 의존성은 설치 시 resolve하고 결과를 freeze한다. 완전한 사전 lockfile이라고 주장하지 않는다.

## 4. 모델·데이터 받기와 분할

```bash
source scripts/premise_sft_env.sh
"$PREMISE_ENV/bin/python" scripts/premise_sft.py download --out "$PREMISE_ART"
"$PREMISE_ENV/bin/python" scripts/premise_sft.py prepare \
  --crepe "$PREMISE_ART/raw/crepe" --out "$PREMISE_ART/data"
```

모델: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct). 모델 가중치는 약 15GB 수준이며 원본·환경·학습 결과를 위해 충분한 디스크를 확보한다. 학습은 HF 캐시에 저장된 고정 revision을 offline으로 연다.

원 데이터: [Cancer-Myth 공식 HF](https://huggingface.co/datasets/Cancer-Myth/Cancer-Myth), [CREPE 공식 저장소](https://github.com/velocityCavalry/CREPE), 다운로드에 사용하는 [CREPE HF mirror](https://huggingface.co/datasets/tasksource/CREPE). 데이터 revision은 첫 다운로드 전에 SHA로 resolve해서 `downloads.json`에 저장하고 재실행에서도 유지한다. dataset loader의 remote code는 허용하지 않는다.

Cancer-Myth 학습/평가는 기존 비교와 일치시키기 위해 **동봉한 732개 canonical 질문과 grouped split**을 쓴다. HF의 FPQ 원본은 출처 보존용으로 다운로드하며, 최신 원본으로 기존 분할을 덮어쓰지 않는다. NFP 원 출처는 [Cancer-Myth 공식 저장소](https://github.com/bill1235813/cancer-myth)이며 이미 동봉돼 있다. TPQ 주석은 추가 질문으로 합치지 않는다.

| Cancer-Myth 분할 | FPQ | NFP |
|---|---:|---:|
| fit | 349 | 89 |
| dev | 117 | 30 |
| test | 117 | 30 |

원 735개에서 기존 중복·평가 예시 제외 정책을 적용한 732개이며 `bundle.json`에 기존 inventory의 해시를 기록했다. 그룹과 동일 source myth가 분할을 넘지 않는지 검사한다. 의미상 유사한 모든 전제가 분리됐다는 보장은 아니다. 기존 연구 과정에서 dev/test를 이미 봤으므로 새 untouched test라고 부르지 않는다.

`fpq_0`의 source myth는 `From physicians.`라는 출처 문구다. 해당 문항은 **평가에서 유지**하되 premise 정답 비교에서는 target-unusable로 표시한다. 이런 문구를 거짓 전제 문장으로 학습하지 않는다. 현재 fit에서는 이 제외로 줄어드는 문항이 없어 438개다. 다른 정상적인 문장형 source myth도 질문별 목표 오류와 완벽히 일치한다고 인증한 것은 아니다. 자동으로 재작성하지 않으며 이번 실험은 **기존 주석 감독의 유용성**을 시험한다.

CREPE는 원 chronological train/dev/test를 유지한다. 두 라벨이 같이 있는 문항, 질문 중복, 뒤쪽 분할과 동일한 전제 문구가 겹치는 앞쪽 분할 문항을 제외하고 `excluded.jsonl`에 기록한다. `normal`의 거짓 전제 목록은 비운다. retrieved passage/comment/correction은 학습 입력으로 넣지 않는다. 다운로드 뒤 확정된 분모는 각 manifest에 기록되며 사전에 수치를 추측하지 않는다.

### 선택적 후속 비교: 답변 학습 target 272개 (기본 실험 아님)

`cancer_fit_answers.jsonl`에는 fit에서만 가져온 기존 Qwen 생성 답변이 들어 있다.

- FPQ **194개**: 무조건 교정 답변 중 기존 Sonnet Well ≥4.
- NFP **78개**: Plain 답변 중 기존 Sonnet Well =5.
- 합계 **272개**, 원 질문과 정확히 join되는지 검사.

이는 사람이 쓴 정답 답변이 아니며, 새로 채점하거나 생성하지 않았다. **이 자료가 없어도 기본 전제 학습은 가능하다.** 이전 SFT와 달리 **원 NFP를 실제 학습에 포함**한다. CM-answer / CM-premise-matched / CM-joint / CM-binary-matched는 이 272개를 똑같이 사용한다. 전제만 학습하는 438개 조건과 같은 학습량인 것처럼 비교하지 않는다. 현재 `joint`의 직접 감독은 답변 내용뿐 아니라 출력 길이·형식도 바꾸므로 그 효과를 순수한 내부 메커니즘으로 해석하지 않는다.

## 5. GPU 한 장씩 smoke와 배치

각 학습의 기본값: **micro-batch 2 × GPU 1 × accumulation 8 = effective batch 16**. 두 독립 학습을 합쳐 batch 32라고 세지 않는다. Rank 16, alpha 32, dropout 0.05, LR 1e-4, 3 epochs, cosine schedule/warmup 5%, gradient clipping 1.0, seed 17. SDPA와 gradient checkpointing 사용. 최대 길이는 **질문+target 합계 2,048토큰**이다.

```bash
source scripts/premise_sft_env.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_premise_sft_single.sh cancer smoke_cm_b2 --max-steps 3
CUDA_VISIBLE_DEVICES=1 bash scripts/run_premise_sft_single.sh crepe smoke_crepe_b2 --max-steps 3
```

각 smoke는 모델 로드 → 실제 forward/backward/optimizer step → adapter 저장까지 실행한다. `train_log.jsonl`의 `peak_allocated_gib`와 `peak_reserved_gib`로 메모리를 본다. `train_manifest.json`의 `world_size`는 **1**, `effective_batch`는 **16**이어야 한다. 각 프로세스 안에서는 지정한 물리 GPU 하나가 논리적 `cuda:0`으로 보인다. NCCL/DDP 통신은 사용하지 않는다.

smoke는 **본 실험 결과가 아니며**, 본 학습은 새 run 이름으로 base에서 다시 시작한다. GPU당 배치 4를 시험하려면 `MICRO_BATCH=4 GRAD_ACCUM=4`를 사용하면 effective batch 16을 유지한다. 두 데이터셋 모두에서 메모리를 확인하고 본 비교에서는 같은 배치 설정을 사용한다. 3-step 통과가 전체 길이 분포의 OOM 부재를 보장하지는 않는다. OOM이면 `MICRO_BATCH=1 GRAD_ACCUM=16`으로 낮춘다.

2,048토큰을 넘으면 **문항 ID와 실제 길이를 출력하고 중단**한다. target을 몰래 자르거나 일부 조건만 문항을 버리지 않는다. 필요하면 두 비교군 모두 `--max-length 3072` 또는 4096으로 올리고 micro-batch를 낮춰 다시 smoke한다.

## 6. 본 실험: 두 GPU에서 각각 nohup + 로그

먼저 모델·데이터 다운로드와 prepare를 **한 번 완료한 뒤** 실행한다. GPU 0,1은 예시이며 타인의 프로세스가 없는 장치를 지정한다. 아래는 독립적인 두 프로세스다.

```bash
source scripts/premise_sft_env.sh
mkdir -p "$PREMISE_ART/logs"

CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_premise_sft_single.sh \
  cancer cm_premise_all_s17 \
  > "$PREMISE_ART/logs/cm_premise_all_s17.log" 2>&1 < /dev/null &
echo $! > "$PREMISE_ART/logs/cm_premise_all_s17.pid"

CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_premise_sft_single.sh \
  crepe crepe_premise_s17 \
  > "$PREMISE_ART/logs/crepe_premise_s17.log" 2>&1 < /dev/null &
echo $! > "$PREMISE_ART/logs/crepe_premise_s17.pid"

tail -f "$PREMISE_ART/logs/cm_premise_all_s17.log" "$PREMISE_ART/logs/crepe_premise_s17.log"
```

`train_manifest.json`에서 dataset 경로·학습 ID·world size를 확인한다. 완료 여부는 각 run의 `complete.json`으로 확인한다. 기존 run 디렉터리를 덮어쓰지 않는다. 중단 run은 epoch별 adapter를 남기지만 **optimizer resume는 구현하지 않았으므로** 재학습은 새 run 이름을 사용한다.

**이 구성에서 `run_premise_sft_ddp.sh`나 `run_premise_sft_matrix.sh`를 실행하지 않는다.** 이들은 이전 두-GPU 공동 학습/선택적 조건용으로 남아 있는 스크립트다. 현재 진입점은 `run_premise_sft_single.sh`뿐이다.

두 모델은 같은 base에서 독립 시작하며 epoch 3을 사전 고정한다. test 점수로 checkpoint를 선택하지 않는다. 데이터 규모가 달라 CREPE 쪽이 더 오래 걸릴 수 있다. main 결과를 정리할 때 seed 반복과 불확실성은 별도로 확인한다.

## 7. 평가: 별도 gate 없이 답하기를 중심으로

**주 평가:** 같은 Cancer-Myth test 147개(117 FPQ + 30 NFP)에 원 질문만 주고 한 번 생성한다. 현재 NFP fit 89개를 학습하므로 이전의 NFP 전체 149개 결과와 섞지 않는다. NFP test의 한 문항은 3.33%p라 작은 차이를 과해석하지 않는다. dev는 분석용이며 test와 합친 숫자를 test라고 부르지 않는다.

```bash
source scripts/premise_sft_env.sh
CUDA_VISIBLE_DEVICES=0 "$PREMISE_ENV/bin/python" scripts/premise_sft.py generate \
  --model-lock "$PREMISE_ART/downloads.json" \
  --questions "$PREMISE_ART/data/cancer/test.jsonl" \
  --mode raw --out "$PREMISE_ART/eval/base_raw_test"

CUDA_VISIBLE_DEVICES=0 "$PREMISE_ENV/bin/python" scripts/premise_sft.py generate \
  --model-lock "$PREMISE_ART/downloads.json" \
  --adapter "$PREMISE_ART/runs/cm_premise_all_s17/adapter" \
  --questions "$PREMISE_ART/data/cancer/test.jsonl" \
  --mode raw --out "$PREMISE_ART/eval/cm_premise_raw_test"

CUDA_VISIBLE_DEVICES=1 "$PREMISE_ENV/bin/python" scripts/premise_sft.py generate \
  --model-lock "$PREMISE_ART/downloads.json" \
  --adapter "$PREMISE_ART/runs/crepe_premise_s17/adapter" \
  --questions "$PREMISE_ART/data/cancer/test.jsonl" \
  --mode raw --out "$PREMISE_ART/eval/crepe_to_cm_raw_test"
```

**두 학습이 완료되고 GPU가 비었을 때** 위 평가를 실행한다. CREPE 모델도 `data/cancer/test.jsonl`을 사용한다. Base/CM/CREPE 세 모델에서 입력·생성 예산·평가 분모를 같게 맞춘다. 선택적 후속 answer/joint 조건에도 같은 평가 원칙을 적용한다. `joint` 출력의 `Answer:` 뒤가 최종 답변이다. 전제만 학습한 모델이 **JSON만 출력하고 질문에 답하지 않으면, 이를 질답 개선으로 세지 않는다.** 교정 정보를 출력했다는 이유만으로 높은 답변 점수를 부여하지 않는다. 별도 두 번째 답변 호출로 보완한 결과는 이 one-step 결과와 구분한다.

**출력 형식 전이도 분리한다.** 전제만 출력하도록 학습한 모델에 동일한 raw 입력을 주면, 학습한 대로 JSON을 내는 것이 자연스럽다. 이를 전제 지식의 부재라고 해석하지 않는다. 일반 질답으로의 전이를 확인하려면 base와 모든 adapter에 공통으로 `--mode answer`를 사용한다. 원 질문은 그대로 두고 system에 `You are a helpful assistant. Respond to the user's question in English. Return only your final answer.`만 붙인다. 거짓 전제 검토나 교정 지시는 없다. 이 역시 한 번의 생성이며 gate가 없다. raw 결과와 섞지 말고 **공통 질답 지시 조건에서의 답변 전이**로 별도 보고한다. 출력 형식을 바꿔도 답변 개선이 없을 수 있으며 그것도 실험 결과다.

**진단 평가:** base와 모든 adapter에 같은 진단 지시를 붙이는 `--mode diagnostic`도 별도로 생성한다. 오류 유무/목표 전제 능력 비교용이며 raw 질답과 동일한 과제가 아니다. 이진/전제 SFT의 raw 출력도 따로 집계할 수 있지만, raw Base가 JSON을 못 출력하는 것을 FPQ 판별 실패로 바꿔 세지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 "$PREMISE_ENV/bin/python" scripts/premise_sft.py generate \
  --model-lock "$PREMISE_ART/downloads.json" \
  --adapter "$PREMISE_ART/runs/cm_premise_all_s17/adapter" \
  --questions "$PREMISE_ART/data/cancer/test.jsonl" \
  --mode diagnostic --max-new-tokens 2048 --out "$PREMISE_ART/eval/cm_premise_diagnostic_test"

"$PREMISE_ENV/bin/python" scripts/premise_sft.py score \
  --questions "$PREMISE_ART/data/cancer/test.jsonl" \
  --answers "$PREMISE_ART/eval/cm_premise_diagnostic_test/answers.jsonl" \
  --out "$PREMISE_ART/eval/cm_premise_diagnostic_test/score.json"
```

집계는 JSON의 boolean을 그대로 사용한다. 별도 오탐 목표 문턱·자가 보고 확률·AUROC를 만들지 않는다. malformed/truncated/missing은 명시적으로 보고하고 No로 간주하지 않는다. 일부만 완료되면 분모와 `complete_valid=false`를 함께 확인해야 한다. `score.cases.jsonl`에는 질문·참조 전제·예측 전제가 들어가므로 의미상 목표 일치를 검토할 수 있다. 단순 문자열 일치를 의미 정확도로 보고하지 않는다.

**최종 Well 채점은 이 패키지에서 자동 호출하지 않는다.** 생성 완료 후 기존과 같은 채점 프로토콜에 연결하되 joint의 구조화된 진단과 실제 최종 답변을 구분하고, 답변 없음/길이 제한도 별도 집계한다. 이 패키지의 `score`는 표 1의 진단 집계일 뿐 표 2의 Well 점수가 아니다. 토큰 한도에 걸린 답변은 `truncated=true`로 남으므로 충분한 동일 예산으로 재생성한 뒤 비교한다.

## 8. 검증 범위와 한계

로컬에서 데이터 해시·분할 누수·동일 학습 문항·loss mask·padding·CREPE 충돌 처리·잘못된 출력 집계에 대한 단위 테스트 11개와 Cancer-Myth prepare가 통과했다. 작은 무작위 Qwen2 모델로 CPU LoRA forward/backward/optimizer 갱신도 확인했다(로컬 torch 2.4.1, transformers 4.46.3, peft 0.13.2). 실제 CUDA 학습, 고정한 torch 2.5.1 환경 설치, 새로운 서버에서 각 GPU의 독립 학습, 외부 다운로드 성공은 **해당 서버에서 아직 확인되지 않았다**. 단일-GPU launcher를 가짜 Python 실행기로 검사해 CM/GPU 0과 CREPE/GPU 1의 분리, 잔여 DDP 환경 변수 제거, GPU 두 개 동시 지정 거부도 확인했다. 로컬 CPU 검사를 A6000 메모리 검증으로 해석하지 않는다.

```bash
"$PREMISE_ENV/bin/python" -m unittest discover -s tests -p test_premise_sft.py -v
```

기존 실행 중인 분석이나 다른 GPU 작업을 재시작/중단하지 않는다. 다른 모델과 학습 환경을 섞지 않는다. 데이터 주석이나 출력 답변을 고쳐 실험할 경우 새로운 data/run 버전과 해시를 만든다.
