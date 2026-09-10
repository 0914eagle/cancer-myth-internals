# 23. Gemma 한 모델: prompting baseline과 기본 C steering 파일럿

2026-09-10. **현재 실행 우선순위**다. 16의 코드 결손, 19의 분할 요구,
22의 방법 후보를 검토한 뒤, 교수님의 “모델 하나의 baseline부터” 지시에 맞췄다.
22의 27B·직교화 확정안과 이전 E2 격자를 이 파일럿의 실행 지시로 사용하지 않는다.
방법론은 구상 중이며 SAE·새 gate·직교화는 이번 구현에 포함하지 않는다.

## 0. 바로 실행하기 — 우선 baseline 세 행

서버 125의 기존 clone·가상환경을 사용하는 명령이다. 처음 설치하는 서버는
[EXPERIMENTS.md의 설치 절](../EXPERIMENTS.md#first-time-on-a-machine)을 먼저 따른다.
Gemma 접근 승인을 받은 HF 계정과 서버의 Codex CLI 로그인이 필요하다.
현재 기본 판정기는 **`codex exec` + `gpt-5.6-sol`**이며 OpenAI API 키는 필요하지 않다.
처음 지정한 `gpt-5`는 서버의 ChatGPT 로그인에서 지원되지 않아 중단됐으므로 변경했다.
`gpt-5.6-sol`은 GPT-5와 다른 모델이다. [공식 Codex 모델 안내](https://learn.chatgpt.com/docs/models)에
기재되어 있고 이 저장소에도 이전 실행 기록이 있지만, 서버 계정의 현재 접근은 `check-judge`로 확인한다.
`codex login status`로 확인하고 미로그인 상태면 `codex login`을 실행한다.
구버전 CLI는 `gpt-5.6-sol` 호출에 최신 버전이 필요하다는 오류를 낼 수 있다.
npm 설치 환경은 `npm install -g @openai/codex@latest` 후 `hash -r`, `codex --version`으로 확인한다.
`baselines`는 실제 GPU 생성과 Codex 판정을 실행한다. `prepare`는 CPU 준비다.
GPT-4o로 판정한 선행논문 수치와는 판정기가 다르므로 직접 재현값으로 취급하지 않는다.

```bash
cd /home/eagle0914/cancer-myth-internals
git fetch origin
git switch main
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
export DATA_ROOT=/data1/heejae
export CUDA_VISIBLE_DEVICES=0
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
export BATCH_SIZE=1
source scripts/env.sh "$DATA_ROOT"

# 연결이 끊겨도 계속 실행하도록 tmux 세션 안에서 실행한다.
bash scripts/run_gemma_pilot.sh prepare
bash scripts/run_gemma_pilot.sh check-judge
bash scripts/run_gemma_pilot.sh baselines
bash scripts/run_gemma_pilot.sh report-baselines
cat "$ART/results/pilot/gemma2_9b_v1/report_baselines_dev_codex_gpt-5.6-sol.md"
```

서버 62는 `/data1/heejae`를 `/data/heejae`로 바꾸고 사용 가능한 GPU를 지정한다.
처음에는 24GB GPU의 여유 메모리를 고려해 batch 1로 시작한다. VRAM 사용량은 실측 전이다.
wrapper 기본 batch는 4이므로 위 환경변수를 빠뜨리지 않는다. fit 단계의 활성값 추출은 문항별이다.
resume 설정을 고정하므로 batch나 모델 설정을 바꿀 때는 새 `PILOT_DIR`을 사용한다.

첫 표는 dev FPQ 117/NFP 30문항에 대한 Plain·FP Identification·전제 검토 CoT다
(현재 감사 snapshot 기준). 실제 분모는 manifest를 확인한다. 각 행의 PCR/PCS/NFP,
95% CI와 Plain 대비 rescue/harm을 먼저 읽는다. steering까지 비교하려면 이어서 실행한다.

```bash
bash scripts/run_gemma_pilot.sh fit
bash scripts/run_gemma_pilot.sh sweep
bash scripts/run_gemma_pilot.sh report
cat "$ART/results/pilot/gemma2_9b_v1/report_dev_codex_gpt-5.6-sol.md"
```

이 단계까지는 dev 탐색이다. `select`와 잠긴 test의 실행 조건은 §4에 있다.
일반 의료 QA·CREPE·MISP·내부 gate/SAE는 이번 runner에 포함되지 않는다.
[24 — MISP와의 관계](24_misp_relation.md)는 후속 평가 후보를 설명하며 실행 범위를 늘리지 않는다.

## 1. 지금 확인하는 것

`google/gemma-2-9b-it` 한 모델에서 다음 네 조건을 같은 문항·판정기로 비교한다.
9B는 첫 실행의 기본값이며, SAE가 존재한다는 것이 특정 전제 feature가 발견됐다는 뜻은 아니다.

| 행 | 실제 실행 | 해석 |
|---|---|---|
| Plain | 원 질문으로 직접 생성 | 기준 |
| FP Identification adaptation | Yes/No 전제 판별(최대 16토큰) → Yes일 때 교정 지시, No일 때 원 질문으로 답변 | Well의 두 단계 구조를 가져온 zero-shot 적용 |
| Premise-review CoT | 명시적 전제 검토(최대 128토큰) → 검토문을 포함해 최종 답변 | 우리가 구성한 두 단계 prompting 대조군; 별도 monitor 없음 |
| Unconditional paired-C steering | 원 질문 생성 시 마지막 프롬프트 토큰부터 C를 더함 | 기존 평균 차 개입의 진단; 새 방법이라는 주장 없음 |

세 답변 생성 조건의 최종 답변 예산은 동일하게 512토큰, greedy다. FP/CoT의 추가 호출은
공짜로 간주하지 않으며 검토·답변의 input/output token 수를 별도로 기록한다.
**판정기에는 최종 답변만 전달**하고 검토문은 JSONL의 `review`에 남긴다.
FP 판정은 응답 첫머리의 명확한 Yes/No를 읽는다. 뒤에 붙은 설명과 기본 Markdown 강조는
허용하고 원문을 저장한다. 첫 판정이 없거나 명시적으로 서로 충돌하는 Yes/No 응답은 중단한다.
실패를 No로 처리하지 않는다. 긴 설명은 16토큰에서 잘려도 첫 판정이 명확하면 사용할 수 있다.

Well 원 프롬프트를 그대로 재현한 행은 아니다. Gemma의 user-turn 형식에 맞춰 구현했고,
few-shot/RAG를 쓰지 않는다. 감사한 Well revision `a7ee871eadde1104f7560a2874a03cbf5221dbaa`의
`FP_identification_template.py`에는 evidence가 없을 때 조건식 때문에 user 질문이 빈 문자열이
될 수 있는 경로가 있어 그대로 복사하지 않았다. 여기서는 항상 질문을 포함한다.
출처: [Well template](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/template/FP_identification_template.py).

## 2. 분할과 방향 학습

**원본 라벨 충돌 발견:** 공개 `FPQ QID=419`와 `NFP QID=1007`은 동일한 retinoblastoma 질문이며
설명도 동일하다. 파일럿은 어느 라벨이 맞는지 임의로 선택하지 않고 두 항목을 모두 제외한다.
동일 질문이 FPQ/정상 라벨을 동시에 가진 경우를 일반적으로 검사하고 `label_conflicts`에 ID와 이유를 남긴다.
따라서 명목상 585+150행을 그대로 고유 평가 문항 수로 쓰지 않는다.

공개 데이터의 CPU 준비 검증에서는 판정 예시 `fpq_1`도 제외되어 **732문항**이 남았다.
fit=FPQ 349/NFP 89, dev=117/30, test=117/30이며 fit에서 쓸 수 있는 교정/순응 참조 쌍은 **135개**다.
[입력 snapshot·분할 감사 기록](audits/gemma_pilot_data_2026-09-10.json).
이 숫자는 해당 입력 snapshot과 분할 구현의 결과이며 서버에서는 생성된 manifest를 기준으로 확인한다.

- `prepare`: 공식 `all_data.json`과 `nfp.json` 또는 기존 E0 질문을 읽는다.
  NFP/TPQ 중복은 공식 NFP 한 행으로 합친다. TPQ만 있고 대응 NFP가 없으면 중단한다.
  판정 few-shot에 등장하는 질문은 제외하고 ID를 manifest에 기록한다.
- 알려진 source myth, 유효한 source_row, 명시한 group/pair/base ID의 연결 성분을 같은 그룹으로 둔다.
  `source_row=-1`과 `From physicians.`는 결측값이므로 서로 다른 질문을 한 그룹으로 묶지 않는다.
  의미상 같은데 원본 메타데이터에 연결되지 않은 문항은 자동으로 알아낼 수 없다. manifest의
  `group_keys`를 감사하고 필요하면 E0 입력에 수동 `group_id`를 제공한 뒤 **실험 전에** 다시 준비한다.
- 고정 seed 17의 stratified grouped 5-fold 중 fold 0=test, fold 1=dev, 나머지=fit.
  이는 **한 번의 grouped holdout 파일럿**이며 5-fold 전체 cross-fitting이나 Well 저자 test split은 아니다.
  각 분할의 FPQ/NFP 실제 수는 `manifest.json`의 `counts`가 기준이다.
- `fit`: fit FPQ의 같은 질문에서 GPT-4o 참조 점수 +1/−1 답변을 하나씩 선택한다.
  참조 저자 비율 보정도 fit 안에서만 한다. 같은 질문이라도 답변의 어투·길이 교란이 완전히 없어지지는 않는다.
- 각 참조 답변 첫 32개 **내용 토큰**을 teacher-force하고 평균 활성값의 교정−순응 차를 문항별로 계산한다.
  이를 평균·단위 정규화한 C를 저장한다. EOS/turn-end는 pooling에 넣지 않는다.
- 층 후보 기본값은 hidden-state index **14, 21, 28**이다. 최적이라는 근거가 아닌 작은 초기 grid다.
  `hidden_states[k]`와 decoder block `k−1`의 대응을 사용하며 마지막 normalization 뒤 상태는 금지한다.
- 같은 fit 질문들의 마지막 프롬프트 활성값 L2 norm 평균을 층별로 저장한다.
  `alpha_abs = alpha × 저장한 fit norm`이다. dev/test나 resume 잔여 문항에서 scale을 다시 계산하지 않는다.
- dev 강도 기본값은 **0, 0.02, 0.05, 0.1**. α=0은 Plain 경로의 일치 확인용이다.
  최종 residual 전부를 2–8배 미는 옛 E2 기본값과 혼동하지 않는다.

## 3. 현재 구현과 옛 파이프라인의 차이

| 파일 | 역할 |
|---|---|
| `scripts/run_pilot.py` | prepare / fit / generate / select / report CLI |
| `src/pilot.py` | 그룹 분할, manifest·캐시 검증, paired 지표와 source-group bootstrap |
| `src/pilot_model.py` | fit-only C와 norm 추출, 세 baseline 및 steering 생성 |
| `scripts/run_gemma_pilot.sh` | 위 단계와 공통 판정기의 서버 실행 wrapper |

방향은 `fit/directions.npz`, 학습 ID·모델 revision·tokenizer·라이브러리/구현 식별은 `fit/fit.json`에 남긴다.
모델은 fit 도중 가중치가 바뀌지 않는다. 추출 중단 시 문항별 통계 cache에서 재개한다.
생성은 기존 batch 구성을 유지하고, 중간에 끊긴 batch만 다시 계산해 완료 ID의 중복 append를 막는다.
모델·방향·강도·코드·분할·토큰 예산이 달라진 출력 파일 재사용은 거부한다.

기존 E1 전체 문항으로 학습한 `directions.npz`는 이 경로에서 사용할 수 없다.
옛 `run_steer.py`/E2 wrapper는 분할·scale provenance가 없는 탐색 경로로 남아 있으므로
held-out 파일럿에는 **새 runner를 사용**한다. 이전 Plain도 provenance가 없는 상태에서는 자동으로
가져오지 않는다. 같은 데이터·모델·생성 설정임을 별도로 확인해야 하기 때문이다.

공통 판정기 수정:

- 파싱 실패·범위 밖 점수는 최대 4회 시도 후 미완으로 남기고 nonzero exit한다. +1로 집계하지 않는다.
- 판정 출력의 `.run.json`에 backend/model/temperature, 응답·질문·예시·루브릭 hash를 저장한다.
  다른 판정기로 캐시를 이어 쓰거나, provenance 없는 옛 캐시를 그대로 재사용하지 않는다.
- 공식 FPQ/NFP만 채점한다. **TPQ 참인 주석을 NFP의 ‘가능한 환각’으로 넘기던 처리를 제거**했다.
  Well TPQ 별도 루브릭을 구현한 것은 아니다.
- 옛 미파싱 행은 `summarize_judge.py`가 분모와 분자에서 제외하고 제외 수를 보고한다.
  새 pilot의 select/report는 비교 문항 전체가 유효하게 채점될 때까지 결과표를 만들지 않는다.

## 4. 서버 실행 순서

기존 `EXPERIMENTS.md`의 bootstrap·HF 모델 접근·CUDA 환경을 준비한 뒤 실행한다.
아래는 명령 예시이며 **이 문서를 작성하며 9B 생성이나 유료 judge 호출을 실행한 것은 아니다.**
wrapper는 foreground로 실행한다. 긴 실행은 tmux 등 지속되는 세션에서 수행한다.

```bash
cd /home/eagle0914/cancer-myth-internals
source /data1/heejae/uv/cancer_myth_internals/bin/activate
export DATA_ROOT=/data1/heejae
export CUDA_VISIBLE_DEVICES=0
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
# 서버의 Codex CLI 로그인 사용. OPENAI_API_KEY는 필요하지 않다.

bash scripts/run_gemma_pilot.sh prepare
bash scripts/run_gemma_pilot.sh check-judge
bash scripts/run_gemma_pilot.sh baselines
bash scripts/run_gemma_pilot.sh fit
bash scripts/run_gemma_pilot.sh sweep
bash scripts/run_gemma_pilot.sh report
```

산출물 기본 위치: `$DATA_ROOT/cancer_myth_internals/results/pilot/gemma2_9b_v1/`.
각 명령은 실패한 단계부터 같은 설정으로 재실행할 수 있다. `PILOT_DIR`로 새 실험을 분리한다.
baseline만 먼저 표로 보고 싶으면 `bash scripts/run_gemma_pilot.sh report-baselines`를 실행한다.
개별 judge dry-run은 `scripts/run_judge.py --dry-run`을 사용한다.

dev 결과의 목적은 교정 이득·과교정·실패 유형을 확인하는 것이다. test를 열기 전, pilot용
NFP 허용폭(percentage points)을 명시하고 `select`를 실행한다. 예시의 5는 과학적으로
보장된 비열등성 한계가 아니며 최종 논문의 허용폭도 아니다.

```bash
NFP_MARGIN_PP=5 bash scripts/run_gemma_pilot.sh select
bash scripts/run_gemma_pilot.sh test
PARTITION=test bash scripts/run_gemma_pilot.sh report
```

선택 규칙은 dev point estimate에서 `ΔPCR > 0`, `ΔNFP >= -허용폭`을 만족하는 설정 중
PCR → PCS → NFP → 작은 강도 → 작은 층 순이다. 모든 후보와 점수를 `selection.json`에 저장한다.
만족하는 설정이 없으면 `selected=null`로 남겨 **test를 열지 않는다**. dev의 음성 결과를 읽고
구현·방향·데이터 문제를 점검한다. test에서는 저장한 한 방향·층·강도를 사용하며 재조정하지 않는다.

## 5. 첫 표를 읽는 법과 다음 결정

- PCR(%), PCS(−1..1), NFP(%), 실제 분모, source-group bootstrap 95% CI를 보고한다.
- JSON에는 Plain 대비 paired ΔPCR/ΔPCS/ΔNFP와 CI, +1 성공 기준 rescue/harm도 저장한다.
  그룹 수가 적은 파일럿 CI는 불안정할 수 있다. CI 포함만으로 성능 보존을 보장하지 않는다.
- FPQ 점수 −1/0/+1 분포와 원 답변을 함께 본다. **0점 전체가 ‘틀린 사실로 반박’은 아니다.**
  얼버무림·불완전 교정·틀린 교정은 답변 원문을 읽어 별도로 분류할 후속 분석이다.
- steering의 교정 이득이 있고 정상 질문이 악화되면 gate의 필요성을 검토한다.
  반박만 늘고 정확한 교정이 늘지 않으면 대조쌍과 방향 학습을 검토한다.
  기존 프롬프트보다 이점이 없으면 그 결과를 그대로 보고한다.
- 새 방법의 필요성이 구체화된 뒤 내부 probe, 일반 의료 QA, CREPE 전이를 확장한다.
  이번 코드는 최종 Table 1–3 전체나 CAST/Gated 재현을 구현한 것이 아니다.

## 6. 검증 범위

GPU 없이 실행되는 회귀 테스트에 grouped split, NFP/TPQ 중복, fit 전용 학습·scale,
재시작 설정 검증, 파싱 실패, paired 집계, test 잠금이 포함된다.
다운로드 없이 작은 무작위 Gemma2를 생성해 실제 decoder hook, 첫 토큰 개입,
α=0 Plain 일치, 짝 답변의 내용 토큰 pooling을 검사한다.
실제 9B의 VRAM·처리 시간·의학적 성능은 서버에서 첫 실행으로 확인해야 한다.

## 7. GPT-5 unsupported 오류 후 재개

`gpt-5 is not supported when using Codex with a ChatGPT account`는 판정기 접근 오류다.
생성된 Gemma 답변이나 데이터 분할의 오류가 아니다. 기존 출력과 manifest는 삭제하지 않는다.
서버에서 코드를 갱신하고 이전에 export한 모델명도 덮어쓴다. 기존 `DATA_ROOT`, `PILOT_DIR`,
`BATCH_SIZE=1`, 생성 예산과 모델 설정을 유지한다.

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
npm install -g @openai/codex@latest
hash -r
codex --version
codex login status
bash scripts/run_gemma_pilot.sh check-judge &&
  bash scripts/run_gemma_pilot.sh baselines &&
  bash scripts/run_gemma_pilot.sh report-baselines
```

`baselines`, `sweep`, `test`는 GPU 생성 전에 짧은 실제 판정기 호출을 수행한다.
검사 실패 시 즉시 중단하므로 지원되지 않는 모델로 여러 문항을 재시도하지 않는다.
이 검사도 Codex 사용량을 소모한다.
이미 완료된 생성 문항은 설정 일치 검증 후 건너뛴다(검증을 위해 Gemma 로드는 발생한다).
새 판정 결과는 `*_judge_codex_gpt-5.6-sol.jsonl`에 저장되어 실패한 GPT-5 경로와 섞이지 않는다.
`report-baselines`는 세 baseline의 판정이 모두 완료된 뒤에만 실행한다.

수정 시 로컬 CLI 0.140.0으로 실제 짧은 호출을 시도했으며 모델 응답 대신 CLI 업데이트
요구 오류를 받았다. 따라서 이 수정은 서버 호출 성공을 확인한 결과가 아니며,
업데이트 후 서버에서 `[judge check] OK`를 확인해야 한다. wrapper의 기본 모델 전달과
접근 검사 실패 시 생성·판정을 시작하지 않는 동작은 모의 호출로 검증했다.

## 8. `Invalid Yes/No identification: 'Yes ...'` 복구

서버에서 `Yes` 뒤에 설명이 붙어 strict fullmatch가 실패했다. 입력 프롬프트나 생성 설정을
바꾸지 않고 첫 판정과 설명을 분리해 읽도록 수정했다. `No treatment` 같은 설명 속 단어를
그 자체로 반대 판정으로 처리하지 않는다. 실제 오류 문자열과 정상·모호 응답 회귀 검사를 추가했다.

코드 hash가 바뀌므로 기존 폴더에서 바로 재시작하면 provenance mismatch가 발생한다.
아래 **일회성 복구**는 원본을 보존하고 새 폴더로 split·완료된 dev baseline 답변·기존 판정을 복사한다.
FP Identification 완료 행은 이전 strict 규칙에서도 유효하고 새 규칙과 판정이 같은지 확인한다.
정확히 검토한 수정 전·후 코드 hash만 허용한다. 원/새 파일 hash와 호환성 이유는
`parser_migration.json`에 남긴다. 복사한 답변은 옛 코드가 생성한 것으로 기록되며 재생성했다고
주장하지 않는다. fit·selection·test 결과는 옮기지 않는다.

기존 worker가 종료된 상태에서, 기존 가상환경과 환경변수를 유지한 채 실행한다.

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
python scripts/migrate_pilot_identification.py \
  --source /data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1 \
  --destination /data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fpfix
```

`[migrated]`가 출력되면 이어서 실행한다. 새 폴더가 이미 있으면 복구 명령은 덮어쓰지 않는다.
복구 완료 후 재접속했을 때는 아래 `PILOT_DIR`을 계속 사용한다.

```bash
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fpfix
export BATCH_SIZE=1
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
bash scripts/run_gemma_pilot.sh baselines &&
  bash scripts/run_gemma_pilot.sh report-baselines
```

완료 문항은 재사용하고 실패한 batch부터 재개한다. 부분 FP 생성의 중간 review는 예외 시
저장되지 않았으므로 그 batch의 review는 다시 생성된다. split과 test 구분은 변경하지 않는다.

## 9. 판정 완료 후 `Score provenance differs from generation run` 복구

2026-09-11 수정. 이전 parser migration이 판정 파일의 `.run.json`만 갱신하고 각 점수 행의
`response_run_hash`·`judge_run_hash`를 갱신하지 않았다. 이로 인해 기존에 복사된 판정(주로 Plain)을
report가 거부했다. SSH 단절 여부와 별개인 복구 스크립트의 누락이다.

현재 migration은 행별 hash도 일관되게 갱신한다. 이미 복구한 폴더에는 아래 명령을 사용한다.
`parser_migration.json`에 기록된 원본이 그대로 있고, 문제 행이 원본과 동일하며, 현재 답변과
판정 메타데이터가 일치할 때만 두 hash를 고친다. 점수·설명·응답 본문·판정 시각은 바꾸지 않는다.
수정 전 파일과 수정 내역은 `.before-provenance-fix-*.bak` 및 `.audit.json`으로 보존한다.
수정 후 새로 판정된 정상 행은 그대로 둔다. 원본 폴더를 삭제하지 않는다.

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
export DATA_ROOT=/data1/heejae
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fpfix
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
python scripts/migrate_pilot_identification.py --repair "$PILOT_DIR" &&
  bash scripts/run_gemma_pilot.sh report-baselines
```

이 명령은 CPU 파일 검증·복구·집계만 수행한다. Gemma 생성이나 Codex 판정을 재실행하지 않는다.
서버 원본 데이터 자체는 로컬에서 확인하지 못했으므로 검증 조건에 맞지 않으면 중단하며,
검사를 우회해 집계하지 않는다. 수정 테스트는 원래 오류, 새/옛 판정 행 혼재, 반복 복구,
실제 report 생성과 점수가 변조된 행 거부를 포함한다.
