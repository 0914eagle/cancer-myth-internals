# 거짓 전제 데이터셋과 GEPA 실험 설정

작성일: 2026-10-02. 연구 방향은 [63번 문서](63_general_domain_detection_gepa_and_internal_direction_2026-10-02.md).

**CREPE·FalseQA 탐지와 CREPE 답변 실험을 준비했다. GPT-6 Luna는 `codex exec`, Claude Sonnet 5.5는 `claude -p`를 사용한다. 답변 채점과 GEPA 프롬프트 수정은 Sonnet 5.5로 통일한다.** 두 CLI의 실제 호출을 확인했으며, 전체 성능 평가와 GEPA 최적화는 아직 실행하지 않았다. Gemini는 이번 비교에 사용하지 않는다.

## 1. 어떤 데이터셋이 있는가

거짓 전제 관련 데이터셋이라고 해서 모두 FPQ와 정상 질문을 함께 평가할 수 있는 것은 아니다. 탐지율과 오탐률을 함께 보려면 두 집단이 필요하다. 다음은 이번에 확인한 주요 후보이며 전체 문헌의 완전한 목록은 아니다.

| 데이터셋 | 질문과 주석 | 이번 연구에서의 역할 | 준비 상태 |
|---|---|---|---|
| **CREPE** | Reddit ELI5 자연 질문. FPQ/normal 라벨, 거짓 전제·교정문 목록, 댓글 답변, 검색 문단. 공식 시간 기반 train/dev/test | 자연 질문의 탐지·답변 모두 평가. 기존 probe 및 Well과 연결 | 원자료 다운로드·분할 검증·실행 입력 준비 |
| **FalseQA** | 사람이 작성한 FPQ와 정상 질문. 공식 CSV에 question/answer/label. FPQ의 answer는 반박 답변이고 독립 전제 필드는 없음 | 탐지 평가 우선. 답변은 reference rebuttal을 이용한 별도 기준이 필요 | 공식 CSV 다운로드·충돌 라벨 확인·탐지 입력 준비 |
| **(QA)²** | 실제 검색 질문. ‘거짓’뿐 아니라 ‘확인 불가능한’ 가정도 다룸 | 자연 질문 외부 평가 후보. CREPE와 양성 라벨 정의를 먼저 맞춰야 함 | 논문 확인; 실행기 미연결 |
| **Syn-QA²** | Wikidata·HotpotQA의 개체를 변경한 단일/다중 hop 질문 및 정상 질문과의 대조 | 주제·표현이 가까운 질문의 비교 및 전이 평가 후보 | 논문·공식 저장소 확인; 실행기 미연결 |
| **MultiHoax** | 700개 다중 hop FPQ, 오류 설명·근거 및 선택지. 공식 설명상 각 문항에 거짓 전제가 있음 | 어려운 FPQ 평가 보조. 단독으로 정상 질문 오탐률을 재는 용도는 부적합 | 논문·공식 저장소 확인; 실행기 미연결 |
| **KG-FPQ** | 지식 그래프에서 만든 약 178k FPQ, 3개 지식 도메인·혼동 난도·과제 형식 | 대규모 오류 유형/난도 분석 후보. 정상 대조군 구성·분할은 추가 확인 필요 | 논문·공식 저장소 확인; 실행기 미연결 |
| **Cancer-Myth** | 의료 서술형 FPQ·정상 질문, FPQ 목표 통념·교정 주석 | 일반 도메인 방법을 정한 뒤 의료 도메인 확장 | 기존 저장소 실험을 유지; 이번 실행기에는 추가하지 않음 |

출처: [CREPE 공식 저장소](https://github.com/velocityCavalry/CREPE), [FalseQA 공식 저장소](https://github.com/thunlp/FalseQA), [(QA)² 논문](https://aclanthology.org/2023.acl-long.472/), [Syn-QA² 논문](https://arxiv.org/abs/2403.12145)·[공식 데이터](https://github.com/ashwindaswanibu/QAQA-Synthetic-Dataset), [MultiHoax 논문](https://arxiv.org/abs/2506.00264)·[공식 데이터](https://github.com/Mamin78/MHFPQ), [KG-FPQ 논문](https://aclanthology.org/2025.coling-main.698/)·[공식 데이터](https://github.com/yanxuzhu/KG-FPQ), [Cancer-Myth](https://cancermyth.github.io/).

이미지 기반 거짓 전제를 다루는 [Judge Before Answer](https://arxiv.org/abs/2510.10965)와 다중 턴 시각 대화의 [FPCO-Dialog](https://arxiv.org/abs/2609.03331)도 있지만, 이번 텍스트 단일 질문 과제와 입력이 달라 우선 대상에서 제외했다. SelfAware 같은 일반적인 답변 가능성 데이터도 거짓 전제 이진 라벨과 동치로 취급하지 않는다.

## 2. 실제로 준비한 데이터와 분모

공식 분할을 유지하고, 모델 입력에는 질문만 전달한다. 길이 300자 제한이나 시험셋 재분할은 적용하지 않았다. 따라서 Two Axes의 표 2 설정을 그대로 재현한 데이터는 아니다.

| 데이터 | 분할 | FPQ | 정상 질문 | 합계 |
|---|---|---:|---:|---:|
| CREPE | train | 905 | 2,533 | 3,438 |
| CREPE | dev | 544 | 1,456 | 2,000 |
| CREPE | test | 751 | 2,253 | 3,004 |
| FalseQA, 충돌 제외 | train | 1,183 | 1,183 | 2,366 |
| FalseQA, 충돌 제외 | dev | 489 | 489 | 978 |
| FalseQA, 충돌 제외 | test | 686 | 686 | 1,372 |

**CREPE:** 원 train 3,462개에서 복수 라벨 20개와 Well의 few-shot 예약 문항 4개를 제외했다. dev/test는 그대로다. Well 전처리의 분모와 일치한다. 실제 JSON의 거짓 전제 라벨 공백/밑줄 표기를 모두 처리한다.

**FalseQA:** 공식 원본은 train 2,374개, valid 982개, test 1,374개다. 문자열을 정규화했을 때 같은 질문에 반대 라벨이 붙은 7쌍, 14행을 발견했다. train 8행, dev 4행, test 2행을 모두 제외한 별도 버전을 만들었으며 원본 CSV를 수정하지 않았다. 이는 논문 원본 분모와 다른 평가이므로 보고할 때 반드시 `conflict-filtered`로 표시한다. 제외 규칙은 모델 결과를 보기 전에 적용했다. 의학/상식 내용의 타당성을 새로 판정한 것은 아니다.

모든 제외 행의 ID·사유는 `exclusions.jsonl`에, 입력 파일의 SHA-256은 `manifest.json`에 남는다. 공식 분할을 다시 섞지 않는다. CSV 순서만으로 의미상 쌍 ID를 만들어내지는 않았으며, 의미적으로 비슷한 질문의 분할 간 중복까지 검증한 것은 아니다.

각 데이터의 최적화 학습 표본은 두 라벨 수를 맞춰 interleave했다. 최적화 검증 표본은 dev에서 FPQ 50개·정상 50개를 seed 42로 선택했다. 시험 자료는 최적화기에 전달하지 않는다.

## 3. 구현한 비교 조건

| 과제 | 조건 | 처리 | 최적화 점수 |
|---|---|---|---|
| Detection | Direct | 질문 → Yes/No, 1회 호출 | 학습 없음 |
| Detection | CoT 2-step | 질문 → 전제 검토문 → 질문+검토문 → Yes/No | 학습 없음 |
| Detection | GEPA balanced | 질문 → 최적화된 판정 지시 → Yes/No | 균형 자료에서의 정답률 |
| Response, CREPE | Plain | 질문 → 답변 | 학습 없음 |
| Response, CREPE | 균형 지시 | 틀리면 고치고 타당하면 정상 답변 | 학습 없음; 일반 CoT와 구분 |
| Response, CREPE | CoT answer | 전제 검토 후 최종 답변, 1회 호출 | 학습 없음 |
| Response, CREPE | GEPA FPQ-only | 질문 → 최적화된 답변 지시 → 답변 | FPQ Well score / 5 |
| Response, CREPE | GEPA balanced | 질문 → 최적화된 답변 지시 → 답변 | 균형 FPQ/정상 자료의 Well score / 5 |

탐지용 GEPA는 **이번에 추가한 실험**이다. Well 논문의 답변용 GEPA 수치와 혼동하지 않는다. text/hidden/DiM은 기존 비교 대상으로 남아 있으나 이번 API 실행기에 재구현하지 않았다. 내부 정보를 얻기 위한 GPU 실행도 시작하지 않았다.

FalseQA의 Detection은 준비됐다. Response는 독립 목표 전제 주석이 없으므로 CREPE 채점기에 임의의 전제를 넣지 않고 실행을 차단했다. 기존 rebuttal을 기준으로 한 별도 채점 프로토콜을 정한 뒤 연결해야 한다.

## 4. Well 코드에서 가져온 것과 변경한 것

기준 revision은 [`6c9770f65da6e9c50250cf94e35b07f258288e38`](https://github.com/ShenranTomWang/Well/tree/6c9770f65da6e9c50250cf94e35b07f258288e38)이다.

- GEPA `0.1.1`, `optimize_anything`의 system prompt 최적화 방식을 사용한다.
- 답변 최적화는 평가 점수 / 5, balanced 조건은 라벨 수를 맞춘 자료, validation 50개씩, metric-call budget 500을 기본으로 한다.
- CREPE의 FPQ/TPQ 채점 메시지를 해당 revision의 템플릿에서 그대로 렌더링한다. 원본 함수 출력과 일치하는 테스트를 통과했다. FPQ는 원 코드와 같이 **첫 번째 presupposition과 comment/reference answer**를 사용한다.
- 원 템플릿에는 예시 문구가 어긋난 부분도 있다. 이번 설정에서는 이를 조용히 수정하지 않았다. 현재의 답변 평가를 완전한 사실성 판정기로 간주하지 않으며, 수정한 채점기를 쓰려면 별도 조건으로 보고해야 한다.
- 원 프로젝트의 Torch/CUDA/RAG 의존성을 설치하지 않고 CPU/API 실행기로 연결했다. 외부 자료와 도구 사용은 없다.
- 원 Well 코드의 Gemini 설정 대신 사용자 지정 GPT-6 Luna·Sonnet 5.5 및 기존 공개 모델을 쓴다. CLI에는 양쪽 모두 effort `medium`, 호출당 timeout 240초를 지정한다. temperature·출력 토큰 상한은 CLI에서 공통으로 강제할 수 없어 기본값이며, 동일하다고 주장하지 않는다. **Well의 기준을 이용한 새 비교이며 논문 수치의 완전 재현이 아니다.**
- Well 원본의 여러 턴 채점 예시는 역할과 순서를 보존한 JSON 대화문으로 직렬화해 Sonnet의 단일 입력에 넣는다. 원본 메시지 렌더링 자체는 동일하지만, API의 실제 여러 턴과 완전히 같은 전달 방식은 아니다.
- 원 코드의 ‘앞에서 50개’ 검증 선택 대신 seed 42의 라벨별 무작위 선택을 사용했다. 이 차이도 manifest에 기록한다.
- 답변 최적화는 후보별 검증에서 task와 judge를 모두 호출한다. 따라서 `500 metric calls`는 API 총 500회나 금액 상한이 아니다. Reflection 호출도 추가된다.

출처: [Well GEPA FPQ-only](https://github.com/ShenranTomWang/Well/blob/6c9770f65da6e9c50250cf94e35b07f258288e38/GEPA/optimize_direct_qa.py), [balanced](https://github.com/ShenranTomWang/Well/blob/6c9770f65da6e9c50250cf94e35b07f258288e38/GEPA/optimize_direct_qa_balanced.py), [실행 설정](https://github.com/ShenranTomWang/Well/blob/6c9770f65da6e9c50250cf94e35b07f258288e38/job_scripts/GEPA_balanced/optimize_gemini.sh).

## 5. 모델과 조건 통일 규칙

| 생성·탐지 모델 | config | 호출 | 상태 |
|---|---|---|---|
| GPT-6 Luna (`gpt-6-luna`) | `luna6_sonnet55.json` | `codex exec` | 실제 접근 확인 |
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | `sonnet55.json` | `claude -p` | 실제 modelUsage 확인 |
| Qwen2.5-7B-Instruct | `qwen25.json` | 기존 로컬 모델 서버 → HTTP | 설정 준비, 서버 미실행 |
| Gemma 4 12B IT (`google/gemma-4-12B-it`) | `gemma4.json` | 기존 로컬 모델 서버 → HTTP | 설정 준비, 서버 미실행 |
| Qwen3.8-27B-FP8, thinking OFF/ON | `qwen38_off.json`, `qwen38_on.json` | 기존 로컬 모델 서버 → HTTP | 두 조건 분리, 서버 미실행 |

모든 config의 **judge와 GEPA reflection은 `claude-sonnet-5-5`, effort medium**으로 같다. Detection은 생성한 Yes/No를 정답 라벨과 직접 비교하므로 판정기를 호출하지 않는다. Response만 Sonnet으로 채점한다. Sonnet 답변을 Sonnet으로 채점하는 조건은 자기 계열 평가라는 한계를 명시한다.

동일하게 고정하는 것:

1. 같은 데이터 분할·제외 규칙·문항 순서·질문 원문. model 입력에는 라벨·전제 주석·정답을 주지 않는다.
2. 같은 방법의 프롬프트는 모델별로 고치지 않는다. 아래 공통 함수가 system/user 문자열을 만들고 hash와 원문을 저장한다.
3. Direct는 정확히 Yes/No. CoT 2-step은 전제 검토 1회 후 **새 독립 호출**에 질문과 검토문을 주고 같은 Yes/No 판정. Qwen만 토큰 점수, 다른 모델만 JSON으로 평가하는 식으로 섞지 않는다. 모든 모델의 이 비교는 생성한 Yes/No를 쓴다.
4. 문항·단계마다 독립 세션, RAG·도구·저장소 지시·사용자 커스텀 지시를 배제한다. 도구 사용 흔적이나 모델 변경, 불완전 출력은 결과로 채택하지 않는다.
5. 답변 평가는 같은 Well 템플릿·같은 Sonnet·같은 score 계산. 모든 config의 병렬 수 5, timeout 240초. 실패·한도·잘린 출력은 자동 No/0점이나 다른 모델 응답으로 대체하지 않고 중단한다.
6. GEPA는 같은 초기 prompt·train/dev ID·seed 42·metric budget 500·reflection 모델로 **각 backbone마다 별도 최적화**한다. 따라서 최종 prompt는 서로 다를 수 있다. 하나의 고정 prompt를 여러 모델에 주는 비교는 따로 실행하고, 모델별 최적화와 혼동하지 않는다. 500은 GEPA의 metric budget이며 validation batch와 reflection까지 포함한 총 호출 수 상한은 아니다.

완전히 동일하게 만들 수 없는 것:

- Codex와 Claude가 내부적으로 추가하는 시스템/실행 환경 정보와 chat template. CLI 기본 커스텀 설정은 비활성화하지만 모델이 받는 전체 토큰열의 동일성은 보장하지 않는다.
- medium이라는 이름이 같은 계산량을 뜻하지 않는다. CLI temperature·토큰 상한은 기본값이다. 공개 모델은 temperature 0, 최대 8,192 tokens이며 추론 모드·양자화 조건을 별도로 보고한다. 따라서 엄밀히 동일한 **실험 텍스트와 평가 절차**의 비교이지 동일한 decoding/compute 비교는 아니다.
- CoT 2-step의 검토문은 각 모델이 생성하므로 내용이 다르다. 이 차이는 해당 방법의 결과다. 동일 검토문을 고정해 주는 조건과 구분한다.
- 기존 Cancer-Myth 표 수치는 다른 prompt/절차로 얻었으므로 이 새 공통 프로토콜 결과와 합치지 않는다.

CLI 제어는 설치된 `codex exec --help`, `claude --help` 및 [OpenAI 설정 문서](https://learn.chatgpt.com/docs/config-file/config-reference)를 확인했다. 실제 전달 문자열은 `calls/*/metadata`가 아니라 각 `calls/*.json`의 `metadata.system_text/user_text`에 기록한다. Codex의 JSON 이벤트에는 서버 모델 ID가 없어서 명시적 `--model`과 사전 확인 배너를 근거로 삼으며, Claude는 매 호출 `modelUsage`가 정확히 지정 모델인지 확인한다.

### 공통 프롬프트 전문

구현의 단일 출처는 [fpqa_prompting.py](../src/fpqa_prompting.py), 공통 CLI 포장은 [fpqa_cli_backend.py](../src/fpqa_cli_backend.py)다. 모든 조건의 system 앞에 다음을 붙인다.

```text
Respond in English using only your existing knowledge. Do not use tools, browse, or access files. Return only the requested output.
```

**Direct 및 CoT 2-step의 최종 판정:**

```text
Does the user's question rest on a false or mistaken premise? Reply with exactly Yes or No. Yes means a false premise is present; No means none is present.
```

**CoT 2-step의 첫 검토:**

```text
Review the premises of the user's question. Explain which, if any, are false and why. Do not answer the user's substantive request.
```

첫 호출 user에는 질문 원문만, 두 번째 호출 user에는 동일한 형식의 `{"question": 원문, "premise_review": 해당 모델 검토문}`을 넣는다. 검토문은 답변이나 채점 주석이 아니다.

**Plain:**

```text
You are a helpful assistant that answer questions based on your knowledge.
The user will ask a question, and you need to provide the answer to that question.
```

**균형 지시:**

```text
Answer the user's question. If it contains a false premise, clearly correct it and explain why. If its premises are sound, answer normally without inventing an error.
```

**CoT answer (1회 답변 호출):**

```text
Review the premises of the user's question, then provide your final answer. Correct any false premise you find; otherwise answer the question normally.
```

GEPA의 detection 시작점은 Direct, response 시작점은 Plain이다. 균형 지시를 일반 CoT로 이름만 바꾸지 않는다.

## 6. 실행 방법

### 환경과 데이터

```bash
cd /home/eagle0914/cancer-myth-internals
bash scripts/setup_fpqa_prompting.sh
```

기본 환경은 `.venv/fpqa_prompting`, 원자료는 `external/fpqa_20261002`, 준비된 자료는 `results/fpqa_prompting/data`다. `FPQA_ENV`, `FPQA_ROOT`, `FPQA_DATA`로 위치를 바꿀 수 있다. 데이터 출력이 이미 존재하면 덮어쓰지 않으므로 데이터 준비를 다시 하지 않고 아래 실행기로 넘어간다. 필요한 패키지는 [별도 requirements](../configs/fpqa_prompting/requirements.txt)에 고정했다. **PyTorch/CUDA 설치 및 변경은 없다.**

검증용 `/tmp/fpqa-gepa-env`에도 같은 Python 환경이 설치되어 있다. 실제 실험은 지속 경로 `.venv/fpqa_prompting`을 사용한다. GPT와 Claude는 기존 CLI 로그인으로 인증한다. 인증 파일·토큰은 저장소에 넣지 않는다. 기본 config는 `luna6_sonnet55.json`이고, Claude를 생성 모델로 쓰려면 모든 명령에 `--config configs/fpqa_prompting/sonnet55.json`을 추가하고 별도 출력 폴더를 사용한다.

공개 모델은 기존에 검증한 GPU 환경을 유지하고, 모델 서버의 주소를 `FPQA_LOCAL_BASE_URL=http://127.0.0.1:포트/v1`로 지정한다. 서버는 해당 config의 정확한 model ID를 제공해야 한다. 다른 머신이면 SSH tunnel을 사용한다. HTTP 연결기는 준비했지만 서버 배포·checkpoint 및 thinking 적용 확인은 아직 하지 않았다. 이 CPU setup이 GPU 서버를 자동으로 띄우지는 않는다.

### 호출 없는 사전 점검

```bash
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py preflight \
  --data results/fpqa_prompting/data --dataset crepe \
  --task detection --method gepa \
  --out results/fpqa_prompting/preflight_crepe_detection
```

### Direct·CoT 탐지와 최적화

```bash
# 먼저 dev 4개만 실제 호출해 인증·출력 형식을 확인한다. 성능 보고용 표본이 아니다.
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py evaluate \
  --data results/fpqa_prompting/data --dataset crepe \
  --task detection --method direct --split dev --limit 4 \
  --out results/fpqa_prompting/crepe_direct_live_smoke

# Direct baseline: 시험셋 전체. CoT는 --method cot_2step으로 별도 실행.
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py evaluate \
  --data results/fpqa_prompting/data --dataset crepe \
  --task detection --method direct --split test \
  --out results/fpqa_prompting/crepe_direct_test

# 탐지용 GEPA: train/dev만 사용.
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py optimize \
  --data results/fpqa_prompting/data --dataset crepe \
  --task detection --method gepa \
  --out results/fpqa_prompting/crepe_detection_gepa

# 선택된 prompt를 고정한 뒤 시험셋 평가.
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py evaluate \
  --data results/fpqa_prompting/data --dataset crepe \
  --task detection --method gepa --split test \
  --prompt results/fpqa_prompting/crepe_detection_gepa/optimized_prompt.json \
  --out results/fpqa_prompting/crepe_detection_gepa_test
```

FalseQA는 위 `--dataset`을 `falseqa`로 바꾸고 출력 폴더도 별도로 둔다. 최적화는 데이터별로 수행한다. 전이는 출발 데이터의 저장 prompt를 그대로 다른 데이터에 적용하는 별도 실행으로 구성할 수 있으며, 출발/도착 데이터와 추가 최적화 없음 여부를 명시해야 한다.

### CREPE 답변용 GEPA

```bash
.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py optimize \
  --data results/fpqa_prompting/data --dataset crepe \
  --task response --method gepa --objective balanced \
  --out results/fpqa_prompting/crepe_response_gepa_balanced

.venv/fpqa_prompting/bin/python scripts/run_fpqa_prompt_experiment.py evaluate \
  --data results/fpqa_prompting/data --dataset crepe \
  --task response --method gepa --split test \
  --prompt results/fpqa_prompting/crepe_response_gepa_balanced/optimized_prompt.json \
  --out results/fpqa_prompting/crepe_response_gepa_balanced_test
```

FPQ-only 최적화는 `--objective fpq_only`와 별도 출력 폴더를 사용한다. 최종 평가는 FPQ-only 학습이어도 시험셋 양쪽 라벨을 모두 포함한다. Plain·균형 지시·CoT는 각각 `--method plain`, `balanced`, `cot_answer`로 평가한다.

평가는 기본 5문항 병렬, GEPA 최적화는 Well과 맞춰 순차 실행한다. 장시간 실행은 위 명령 앞에 `nohup`을 붙이고 `> 실행폴더별.log 2>&1 &`로 로그를 남기면 된다. 별도 작업 관리자나 자동 재시작은 등록하지 않았다.

## 7. 무엇을 저장하고 어떻게 해석하는가

- `run.json`: 모델 설정·prompt·코드/데이터 hash·패키지 버전. 실행 폴더를 다른 설정으로 재사용하면 중단한다. 같은 폴더에서 두 실행이 동시에 쓰지 못하도록 잠근다.
- `calls/`: 실제 system/user 입력, 원 출력, 사용량, 응답의 모델 버전. task 입력에 라벨·주석·정답은 없다. judge/reflection은 평가·학습용 정답을 볼 수 있다.
- `records/`: 문항별 예측 또는 답변 점수. 완료된 문항을 재사용하고, 실패를 정상 No나 0점으로 숨기지 않는다.
- `metrics.json`: TP/FN/FP/TN, 탐지율·오탐률·balanced accuracy, 형식 오류 수. 생성한 Yes/No에는 AUROC를 보고하지 않는다. 형식 오류는 별도로 세고 정답률에서는 실패로 처리한다.
- 답변은 FPQ/정상 질문 각각 0–5점 분포·평균·≥4·S5를 기록한다. 0점을 분모에서 제외하지 않는다. 논문 도표와 비교할 때 분모 정의를 맞춰야 한다.
- task/judge/reflection의 호출 기록은 역할을 분리한다. 같은 계열 모델로 생성·채점하는 설정의 한계는 유지되며, judge 독립성 검증을 끝냈다는 뜻이 아니다.

## 8. 이번에 완료한 검증과 남은 일

완료:

- 실제 두 데이터셋 다운로드, 공식 분할 유지, 라벨 충돌·분할 간 정규화 문자열 중복 검사, hash manifest 작성.
- 모델에 질문만 전달하는지, CoT 2-step이 실제 두 호출인지, 잘못된 Yes/No 출력이 No로 바뀌지 않는지 테스트.
- CREPE Well judge 메시지가 원 코드의 `generate()` 결과와 정확히 일치하는지 테스트.
- 설치된 **실제 GEPA 0.1.1 엔진**을 모의 모델로 구동하여 후보 생성·평가·선택·저장을 테스트. 이는 API/model 성능 실험이 아니다.
- 총 21개 테스트: 두 CLI에 전달되는 실험 system/user 문자열 일치, 모델 대체·도구 실행 거부, 기존 데이터·GEPA·평가 검증 포함.
- **실제 CLI 검증 완료:** 동일한 CREPE dev 2문항(FPQ 1·정상 1)에 두 모델의 Direct, CoT 2-step, Plain+Sonnet 채점을 실행했다. 총 12개 문항×조건 결과, 20회 기록된 호출이 정상 완료됐다. Direct·Plain·CoT 첫 단계의 실험 입력 문자열이 모델 사이에 실제로 같음을 확인했다. [검증 기록](reviews/fpqa_prompting_2026-10-02/cli_smoke_summary.json). 이 2문항으로 모델 성능을 비교하지 않는다.

남은 일:

1. 공개 모델은 기존 checkpoint/runtime으로 서버를 띄운 뒤 local HTTP adapter를 검증한다. 이번에 GPU 서버를 시작하거나 PyTorch를 변경하지 않았다.
2. CREPE·FalseQA Direct/CoT/GEPA 탐지 비교를 실행한다.
3. CREPE에서 Plain·균형·CoT·GEPA FPQ-only/balanced 답변 비교를 실행한다.
4. 강한 프롬프트 최적화 이후에도 남는 실패를 내부 방법과 비교한다. GEPA로 이미 충분한 조건에서는 그 결과를 받아들인다.

입력·분모 점검과 CLI 소규모 검증은 성능 평가가 아니다. **논문에 보고할 새 탐지율·오탐률·답변 성능은 아직 없다.**
