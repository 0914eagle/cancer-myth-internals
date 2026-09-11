# 23. Gemma 한 모델: prompting baseline과 기본 C steering 파일럿

2026-09-10. **현재 실행 우선순위**다. 16의 코드 결손, 19의 분할 요구,
22의 방법 후보를 검토한 뒤, 교수님의 “모델 하나의 baseline부터” 지시에 맞췄다.
22의 27B·직교화 확정안과 이전 E2 격자를 이 파일럿의 실행 지시로 사용하지 않는다.
방법론은 구상 중이며 SAE·새 gate·직교화는 이번 구현에 포함하지 않는다.

## 0. 바로 실행하기 — 우선 baseline 세 행

서버 125의 기존 clone·가상환경을 사용하는 명령이다. 처음 설치하는 서버는
[EXPERIMENTS.md의 설치 절](../EXPERIMENTS.md#first-time-on-a-machine)을 먼저 따른다.
Gemma 접근 승인을 받은 HF 계정과 서버의 Codex CLI 로그인이 필요하다.
2026-09-11부터 **새 실행의 기본 판정기는 `codex exec` + `gpt-5.6-terra`**이며 OpenAI API 키는 필요하지 않다.
현재 진행 중인 Sol sweep과 그 결과 보고는 끝까지 `JUDGE_MODEL=gpt-5.6-sol`로 유지한다.
다음 실험부터 아래 Terra 설정을 사용한다. 기존 셸에 Sol이 export되어 있으면 기본값보다 우선하므로
`export JUDGE_MODEL=gpt-5.6-terra`로 명시적으로 변경한다.
Terra 점수·보고서는 모델명이 들어간 별도 파일로 저장되며 Sol 점수를 덮어쓰지 않는다.
한 비교표에서는 판정기를 통일해야 하므로 Terra로 전환한 실험의 baseline도 Terra로 채점한다.
기존 Sol 점수를 Terra 점수로 이름만 바꿔 재사용하지 않는다.

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
export JUDGE_MODEL=gpt-5.6-terra
export BATCH_SIZE=1
source scripts/env.sh "$DATA_ROOT"

# 연결이 끊겨도 계속 실행하도록 tmux 세션 안에서 실행한다.
bash scripts/run_gemma_pilot.sh prepare
bash scripts/run_gemma_pilot.sh check-judge
bash scripts/run_gemma_pilot.sh baselines
bash scripts/run_gemma_pilot.sh report-baselines
cat "$ART/results/pilot/gemma2_9b_v1/report_baselines_dev_codex_gpt-5.6-terra.md"
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
cat "$ART/results/pilot/gemma2_9b_v1/report_dev_codex_gpt-5.6-terra.md"
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
export JUDGE_MODEL=gpt-5.6-terra
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

## 10. fit의 `Assistant template is not prefix-compatible` 복구

2026-09-11. 참조 답변에 앞뒤 공백이 있고 채팅 템플릿이 이를 trim하는 경우,
원래 `answer_prefix`는 원본 문자열과 렌더링된 문자열이 다르다는 이유로 중단했다.
렌더링 결과를 유지하면서 원본 또는 앞뒤 공백만 제거한 답변과 정확히 일치하는 구간을 찾도록
수정했다. 질문 prefix 변경이나 본문 변환은 여전히 거부한다. 처음 n개 내용 토큰만 pooling하고
EOS/turn-end를 제외하는 설정은 같다. 프롬프트와 baseline 생성 동작은 바꾸지 않았다.

`check-fit`은 서버의 실제 tokenizer로 선택된 **fit 참조쌍 전체**의 정렬을 CPU에서 검사한다.
`fit`도 이 검사를 GPU 모델 로드 전에 실행한다. 현재 snapshot은 135쌍/270답변이 예상된다.
로컬 공식 Gemma checkpoint의 tokenizer는 접근 권한이 없어 401을 받았으므로, 서버의 실제
입력에서 오류 원인이 공백인지 및 수정으로 해결되는지는 이 검사로 확인한다. 로컬에서는
trim 유무·앞뒤 공백·토큰 예산·EOS 제외·본문 변환 거부와 모델 로드 없는 전체 검사를 테스트했다.

구현 hash가 바뀌므로 baseline은 검증 후 새 폴더로 복사하고 fit은 새로 시작한다.
기존 `gemma2_9b_v1_fpfix`와 그 원본을 삭제하지 않는다. migration은 이제 이전 strict 파서 버전과
parser-fix 버전 모두에서 baseline을 가져올 수 있고, 참조 정렬 수정 전 fit cache는 가져오지 않는다.
점수의 본문과 값은 그대로 유지하며 행·파일 단위 hash를 함께 갱신하고 감사 기록을 남긴다.

현재 가상환경과 DATA_ROOT를 유지한 서버 125의 일회성 복구:

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fpfix
bash scripts/run_gemma_pilot.sh check-fit &&
python scripts/migrate_pilot_identification.py \
  --source "$PILOT_DIR" \
  --destination /data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
```

`[fit check] OK`와 `[migrated]`를 확인한 뒤:

```bash
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export BATCH_SIZE=1
export JUDGE_BACKEND=codex
export JUDGE_MODEL=gpt-5.6-sol
bash scripts/run_gemma_pilot.sh fit &&
  bash scripts/run_gemma_pilot.sh sweep &&
  bash scripts/run_gemma_pilot.sh report
```

baseline 생성·판정이나 `prepare`를 다시 실행하지 않는다. 이후 명령에도 `PILOT_DIR`을 유지한다.
`torch_dtype` deprecation warning은 이번 traceback의 중단 원인이 아니므로 별도로 취급한다.

## 11. 2026-09-11: L14 결과와 중복 채점 감사 — 추가 sweep 전에

사용자 서버의 Sol dev 보고에서 Plain PCR/NFP는 6.0/63.3,
L14 α=0은 6.0/53.3, α=0.02는 6.0/56.7, α=0.05는 5.1/50.0,
α=0.1은 7.7/53.3이었다. FPQ 117/NFP 30문항이다.
사용자가 답변 파일을 비교한 결과 Plain과 α=0의 147개 답변은 전부 동일했다.
따라서 α=0의 NFP 하락은 개입 효과가 아니라 재채점 불일치다.
NFP에서 성공→실패 4개, 실패→성공 1개로 5개 판정이 바뀌었다.
FPQ도 PCS가 달라졌으므로 성공 여부 외에 −1/0 간 판정 차이도 감사한다.
이 한 쌍으로 판정기의 일반적인 오류율이나 모든 steering 차이의 원인을 추정하지 않는다.

**지금은 추가 sweep을 보류하고 저장된 질문·참조·답변·두 판정 근거를 확인한다.**
아래 명령은 API/Codex/Gemma를 호출하지 않고 기존 score 파일을 변경하지 않는다.
`--model`은 새 판정기 지정이 아니라 읽을 기존 파일 이름이다.

```bash
git pull --ff-only origin main
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
python scripts/audit_pilot_judge.py --pilot-dir "$PILOT_DIR" --model gpt-5.6-sol > "$PILOT_DIR/audit_sol_L14_a0.md"
cat "$PILOT_DIR/audit_sol_L14_a0.md"
```

후속 신규 채점은 Terra로 한다. Terra의 판정 일관성이나 정확도가 Sol보다 높다는
결과는 아직 없다. 불일치 문항의 판정 근거를 확인하고, 필요한 소수 조건을 정한 뒤
같은 Terra로 baseline과 비교 조건을 채점한다. 현재 Sol 표를 Terra 표로 이름만 바꾸지 않는다.

`run_judge.py --reuse-scores PATH`는 명시한 한 파일의 점수를 기준으로 재사용한다.
pilot wrapper는 같은 partition·같은 판정기의 Plain 파일이 있으면 자동으로 지정한다.
질문 ID·답변 해시·전체 질문/참조 파일·예시·루브릭·판정 모델과 설정이 모두 일치해야 한다.
새 행은 대상 생성/채점 실행의 provenance를 기록하고 `score_reuse`에 원본 파일 해시와
행 ID를 남긴다. 답변이 다르면 신규 채점하고, Sol 점수를 Terra에 재사용하지 않는다.
기존 대상 파일에 이미 저장된 점수는 덮어쓰지 않는다. 따라서 현재의 독립 재채점 결과는
감사 자료로 남으며, 재실행만으로 과거 표의 불일치가 정리되지는 않는다.

재사용은 중복 호출과 동일 답변 간 불일치를 줄이는 정책이지, 원본 판정이 옳다는 보장이 아니다.
현재 자동 재사용 범위는 **Plain과 같은 답변**이다. 다른 steering 조건끼리만 같은 답변은
자동 통합하지 않는다. 비용 안내의 `reusable`과 `new calls`는 미완료 문항 기준이며,
실제 호출에서는 재시도와 wrapper의 연결 확인 호출이 추가될 수 있다.

## 12. Terra 판정기 10문항 × 2회 점검

§11의 불일치는 무작위 점수 변동만이 아니다. 제공된 판정 근거에는 NFP 답변을
“잘못된 전제를 교정하지 않았다”는 이유로 감점한 사례가 있다. NFP 원래 루브릭은
정상 질문에 없는 거짓 전제를 답변이 지어내어 지적했는지를 묻는다.
따라서 기존 NFP 수치를 그대로 개입의 성능 저하로 해석하기 전에 기준 적용을 확인한다.

`scripts/check_terra_judge.py`는 기존 Plain 답변만 사용한다.
Sol의 Plain/α=0 NFP 불일치 5문항 전부와 일치 문항에서 seed 17로 무작위 추출한
5문항을 고정한다. 10문항을 섞은 검토표에는 이전 Sol 점수와 표본 그룹을 숨긴다.
`plan.json`에는 추출 그룹·순서·원본 해시·실제 보낼 원본 프롬프트를 저장한다.
진단 표본이므로 여기서 나온 정확도는 전체 NFP의 불편 추정치가 아니다.

**첫 단계: 준비와 사람 검토. 모델 호출 없음.**

```bash
git pull --ff-only origin main
source scripts/env.sh "${DATA_ROOT:-/data1/heejae}"
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export TERRA_CHECK_DIR="$PILOT_DIR/terra_nfp_check_v1"
python scripts/check_terra_judge.py prepare --pilot-dir "$PILOT_DIR" --out-dir "$TERRA_CHECK_DIR"
cat "$TERRA_CHECK_DIR/review.md"
```

검토자는 `human_review.tsv`의 각 행에 `score`(문자 그대로 `1` 또는 `-1`),
`rationale`(근거), `reviewer`(검토자)를 작성한다. 점수는 일반 의료 QA의 정확도가
아니라 NFP 루브릭 준수 여부다. AI 검토안을 사람 판정이라고 기록하지 않는다.
모호한 문항을 임의 확정하지 말고 논의한다. 사람 판정이 완성되지 않으면 `score`는
모델을 호출하지 않고 중단한다. 채점 시작 후 검토표를 바꾸면 같은 실행을 재개하거나
보고할 수 없도록 해 사후에 Terra에 맞춰 정답을 고치지 못하게 한다.

**둘째 단계: 검토 확정 후에만 20회 채점.**

```bash
python scripts/check_terra_judge.py score --out-dir "$TERRA_CHECK_DIR"
python scripts/check_terra_judge.py report --out-dir "$TERRA_CHECK_DIR" > "$TERRA_CHECK_DIR/report.md"
cat "$TERRA_CHECK_DIR/report.md"
```

이 전용 명령은 환경변수 `JUDGE_MODEL`과 무관하게 `codex`/`gpt-5.6-terra`를 사용한다.
새 Gemma 생성, baseline 전체 재채점, sweep, 연결 확인 호출은 없다. 독립된 두 평가를
관찰하려고 이 점검에 한해 동일 답변 재사용을 끈다. 루브릭과 예시를 바꾸지 않는다.
여기서 20회는 runner가 시작하는 `codex exec` 채점 호출 수이며 계정 한도의 일정 비율이나
Codex 내부 요청 수를 보장하는 수치가 아니다.

총 20개 호출 슬롯은 호출 **직전** ledger에 기록한다. 자동 재시도는 없고 실패하면 멈춘다.
같은 명령을 재실행해도 완료·실패·중단된 슬롯을 다시 호출하지 않고 남은 슬롯만 수행한다.
중단/파싱 오류가 있으면 유효 결과는 20개 미만일 수 있으며 보고서에 누락 수를 명시한다.
점검 목적의 반복 결과는 본 실험 점수에 합치거나 재사용하지 않는다.

보고서는 문항별 두 판정·사람 판정·근거를 보여주고, 과거 불일치 그룹과 일치 그룹을
나누어 반복 일치도와 사람 판정 일치도를 센다. 좋은 반복 일치도만으로 정확한 판정이라고
주장하지 않는다. 이 소규모 점검만으로 Terra의 일반적 우월성이나 임상 안전성을 주장하지 않는다.

2026-09-11 로컬 구현 시 서버 SSH는 `Permission denied (publickey,password)`였고,
전체 생성 답변은 로컬에 없었다. 따라서 코드 테스트만 실행했으며 실제 Terra 채점 및
사람 판정은 아직 수행하지 않았다.

사용자가 제공한 `review.md` 10문항에 대한 [AI 판정 초안](reviews/terra_nfp_10_ai_draft.md)과
[TSV](reviews/terra_nfp_10_ai_draft.tsv)를 작성했다. 초안은 모두 +1이며 사람 검토는 아직 없다.
이 표본은 불필요한 감점과 반복 일관성의 점검용이고, 실제 과잉 교정(−1) 탐지 능력의
검증용은 아니다. 기존 Sol 사례를 읽은 AI의 초안을 독립 맹검 사람 판정으로 보고하지 않는다.


### 12.1 사용자 확인 완료 — 실행 단계

사용자가 10개 +1 판정 및 근거를 확인했다(2026-09-11).
`docs/reviews/terra_nfp_10_confirmed.tsv`의 reviewer는 사용자 확인을 거친 AI 보조 검토임을
명시한다. 독립 맹검 사람 평가나 임상의 확정 라벨로 주장하지 않는다.
채점 결과를 보기 전에 이 파일을 적용한다. 기존 AI 초안은 출처 기록으로 보존한다.

서버에서 아래를 실행한다. 신규 생성이나 sweep 없이 준비된 10문항을 두 번씩
Terra로 평가한다. 확인용 추가 호출이나 자동 재시도 없이 최대 20회다.

```bash
git pull --ff-only origin main
export TERRA_CHECK_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix/terra_nfp_check_v1
cp docs/reviews/terra_nfp_10_confirmed.tsv "$TERRA_CHECK_DIR/human_review.tsv"
python scripts/check_terra_judge.py score --out-dir "$TERRA_CHECK_DIR"
python scripts/check_terra_judge.py report --out-dir "$TERRA_CHECK_DIR" > "$TERRA_CHECK_DIR/report.md"
cat "$TERRA_CHECK_DIR/report.md"
```

이 문서 업데이트 시점에 실제 서버 Terra 호출은 아직 실행하지 않았다.

### 12.2 Terra 20회 완료 후 17건 invalid — 오프라인 파싱 점검

사용자 실행에서 20회 호출은 완료됐으나 기존 파서가 3개만 유효하게 읽었다.
`src/judge_prompts.py`의 원본 정규식은 `{` 바로 뒤에 줄바꿈이 있어야 JSON으로 인식한다.
따라서 한 줄 JSON을 반환했을 가능성이 있지만, 실제 17개 raw 응답을 아직 읽지 않았으므로
모두 형식 문제라고 확정하지 않는다. 점검 runner에 이 형식 제약을 그대로 사용한 것은
구현 문제다. 20회를 새로 호출하지 않고 저장된 raw를 재파싱한다.

```bash
git pull --ff-only origin main
export TERRA_CHECK_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix/terra_nfp_check_v1
python scripts/check_terra_judge.py report --out-dir "$TERRA_CHECK_DIR" --reparse > "$TERRA_CHECK_DIR/report_reparsed.md"
cat "$TERRA_CHECK_DIR/report_reparsed.md"
```

`--reparse`는 모델을 호출하지 않고 `attempts.jsonl`, plan, 사용자 확인 라벨을 수정하지 않는다.
새 보고서에 파서 버전·원본 ledger 해시·기존 유효 수·복구 수·거부 수·점수 변경 수를 표시한다.
새 파서는 한 개의 JSON 객체에서 정수 Sharpness(-1/+1)와 비어 있지 않은 Reason을 읽는다.
줄바꿈·코드 블록은 허용하지만, 0·문자열 점수·불리언·중복 키·여러 JSON 객체는 임의 해석하지 않는다.
다른 모델의 응답이나 미완료 호출도 복구하지 않는다. 해결되지 않은 항목은 raw/error를 보고서에
포함해 원인을 확인한다. 기존 pilot의 공유 루브릭과 파서는 이 변경에서 손대지 않았다.

읽힌 3개 판정에서도 `nfp_1103`의 두 평가가 −1/+1로 달랐으며,
−1 근거에는 재발 가능성의 의학적 평가를 언급하지 않았다는 내용이 있었다.
`nfp_1069`도 재활 가능성을 언급하지 않았다고 감점했다.
이는 NFP의 전제 지어내기 여부와 다른 기준을 적용한 사례이므로,
파싱 복구와 별개로 루브릭 적용의 불일치는 남아 있다.
전체 20개 재파싱 결과를 확인한 뒤 판정기/프롬프트의 다음 변경을 정한다.


### 12.3 재파싱 완료: 평가 기준 적용 문제 확인

사용자 보고에서 17건이 모두 복구돼 유효 20/20이다. 반복 일치는 6/10문항이며,
사용자 확인을 거친 AI 보조 기준과는 16/20회 일치했다. −1 네 건은 모두 전제를
교정하거나 추가 의학적 설명을 하지 않았다는 취지로 감점했다.
[보고서](reviews/terra_nfp_check_v1_report.md)와
[해석·NFP 역할 명확화 지침 초안](reviews/terra_nfp_check_v1_interpretation.md)을 기록했다.
이 초안은 아직 판정 코드에 적용하지 않았으며 추가 채점도 실행하지 않았다.
현재 10문항을 프롬프트 개발용으로 쓰면 독립 검증은 별도의 +1/−1 답변 표본에서 해야 한다.

### 12.4 상세 지침 v2 연결 완료

`scripts/check_terra_judge.py revise`가 상세 지침을 실행 코드에 연결한다.
원본 디렉터리의 문항·답변·순서·확정 라벨을 유지하고 v2 프롬프트만 적용한 새 plan을 만든다.
새 plan에는 protocol, 전체 template 및 해시, 원본 plan/검토표 해시를 기록한다.
라벨을 새로 생성하거나 기존 결과를 새 점수로 복사하지 않는다. 같은 경로에 덮어쓰기는 거부한다.

서버에서 다음 명령을 실행한다. `revise`는 오프라인이며 `score`만 **새로운 최대 20회**를 호출한다.
원본 v1의 20회에 추가되는 별도 지침 실험이다. 자동 재시도·preflight·Gemma 생성은 없다.

```bash
git pull --ff-only origin main
export TERRA_CHECK_V1=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix/terra_nfp_check_v1
export TERRA_CHECK_V2=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix/terra_nfp_check_v2

python scripts/check_terra_judge.py revise --source-dir "$TERRA_CHECK_V1" --out-dir "$TERRA_CHECK_V2"
python scripts/check_terra_judge.py score --out-dir "$TERRA_CHECK_V2"
python scripts/check_terra_judge.py report --out-dir "$TERRA_CHECK_V2" --reparse > "$TERRA_CHECK_V2/report.md"
cat "$TERRA_CHECK_V2/report.md"
```

새 프롬프트의 −1은 AnswerEvidence/InventedPremise를 요구하며 인용문이 실제 답변에 있는지
검사한다. +1은 두 필드가 빈 문자열이어야 한다. 이 형식 검사는 의미적 정확성 검증과 다르다.
필드 누락·틀린 인용이 있어도 모델이 낸 점수를 +1로 바꾸거나 통계에서 몰래 제외하지 않는다.
점수는 보존하고 `Evidence-format issues` 및 문항별 사유로 표시해 검토하도록 한다.
보고서의 일치도는 출력된 점수 기준이며, 표시된 증거 문제는 별도 확인해야 한다.

현 10문항은 모두 +1이며 이미 지침 개발에 사용했다. 이 실행은 오감점 및 반복 일관성의
개발용 재점검이다. 실제 과잉 교정(-1)을 감지하는 능력이나 독립 검증 성능을 입증하지 않는다.
본 평가에 적용하기 전 별도 +1/−1 답변 표본 검토가 필요하다.

### 12.5 v2의 +1 사례 점검 완료, 실제 과잉 교정 검토 준비

사용자가 보고한 v2 결과는 유효 20/20, 기준 일치 20/20, 반복 일치 10/10이다.
같은 10문항의 v1은 기준 일치 16/20, 반복 일치 6/10이었다. 이들은 지침 개발에 사용한
모두 +1인 표본이므로 실제 과잉 교정(-1)을 탐지하는 능력을 검증하지 못한다.

`scripts/review_pilot_nfp.py`는 기존 FP Identification의 dev NFP 답변을 검토표로 추출한다.
이전 점검의 문항 ID뿐 아니라 같은 출처 그룹도 제외하고, 나머지는 전부 포함한다.
대개 NFP 30개 중 기존 10개를 뺀 20개이며 같은 그룹이 더 있으면 줄어든다.
이 검토표 작성은 판정 프롬프트를 수정하거나 모델을 호출하지 않는다.
현재 v2 프롬프트를 고정한 채 별도 사례를 검토하고, 실제 +1/−1 확인 후 채점 구성을 정한다.

```bash
git pull --ff-only origin main
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export NFP_REVIEW_DIR="$PILOT_DIR/nfp_overcorrection_review_v1"
python scripts/review_pilot_nfp.py --pilot-dir "$PILOT_DIR" --exclude-check-dir "$PILOT_DIR/terra_nfp_check_v1" --out-dir "$NFP_REVIEW_DIR"
```

산출물:
- `review.md`: 질문·가능한 환각 참조·실제 생성 답변. 채점 근거 및 내부 identification 검토문은 표시하지 않는다.
- `human_review.tsv`: 점수·근거·검토자 칸이 비어 있는 검토표. −1에는 실제 전제 지적 구절을 인용한다.
- `review_cases.json`: 생성 파일/실행/분할 해시, 제외 문항·그룹, 선택된 답변과 해시.

이미 생성된 FP Identification 답변이라고 자동으로 −1을 부여하지 않는다.
일반적인 주의사항이나 치료 선택의 개인차 설명만으로도 −1을 부여하지 않는다.
실제 답변이 질문에 없는 믿음을 환자에게 부여해 지적하는지 문맥을 읽는다.
균형을 맞추려고 모호한 라벨을 강제로 확정하지 않는다. +1 또는 -1 사례가 부족하면
그 사실을 보고한 뒤 추가 표본 출처를 정한다. 같은 모델/도메인의 추가 dev 표본이라는
한계를 유지하며 외부 검증이라고 부르지 않는다.

로컬로 가져올 때는 파일명을 명확히 지정해 앞선 review.md와 구분한다.

```bash
scp eagle0914@165.132.76.125:/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix/nfp_overcorrection_review_v1/review.md ~/Downloads/nfp_overcorrection_review.md
```

### 12.6 새 NFP 20답변 검토 초안

사용자가 가져온 `nfp_overcorrection_review.md`를 읽고
[판정 초안](reviews/nfp_overcorrection_20_ai_draft.md)과
[TSV](reviews/nfp_overcorrection_20_ai_draft.tsv)를 작성했다.
고정 v2 기준의 후보는 +1 8개/-1 6개이며, 문구 또는 참조 대응 문제로 6개를 보류했다.
답변의 실제 과잉 교정 구절은 원문 부분 문자열임을 확인했다. 아직 사용자 확정 라벨이 아니다.

특히 nfp_1063에는 질문에 universal treatment라는 표현이 있어 NFP라는 이유만으로
그 지적을 자동 과잉 교정으로 판정하지 않는다. nfp_1142는 답변이 명시된 개별 사실을
근거 없이 가정이라고 지적하지만, 참조가 설명하는 전제 오류와는 다르다.
일반 과잉 교정과 참조에 한정한 지표 사이의 차이를 숨기지 않고 보류에 기록한다.

사용자 검토 후 명확한 후보에서 +1/-1을 모두 포함하는 소규모 판정기 검증을 구성한다.
보류 항목을 제외한다면 전체 20답변의 정확도라고 보고하지 않고 제외 수와 이유를 밝힌다.
이 항목들을 본 뒤 지침을 넓히면 해당 항목은 새 지침의 개발 표본이므로 독립 검증이 아니다.
이번 검토 중 신규 GPT/생성 호출은 없었다.


### 12.7 고정 v2 지침의 양성·음성 5+5 검증

판정 초안을 검토하고 동의한 경우 다음 명령으로 명확한 +1 후보 5개와 −1 후보 5개를
seed 17로 고정한다. `--accept-ai-review`는 사용자 확인을 받은 AI 보조 라벨임을
기록하는 옵션이며 독립 전문가 판정이나 맹검 검토를 뜻하지 않는다. 초안에 동의하지
않으면 TSV를 먼저 수정하고 `--labels`로 지정한다. 보류 6개는 채점하지 않고 보고서에 남긴다.

기존 v2의 지침과 예시를 그대로 복사한다. 준비와 보고는 모델 호출 0회이며,
score 단계만 최대 20회(10답변 × 2회)의 새 Terra 호출을 사용한다. Gemma 답변을
다시 생성하거나 기존 sweep을 재개하지 않는다. 중단된 호출도 예산을 소비하며 재시도하지 않는다.

```bash
git pull --ff-only origin main
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export BALANCED_DIR="$PILOT_DIR/terra_nfp_balanced_v1"
python scripts/prepare_balanced_nfp_check.py \
  --source-dir "$PILOT_DIR/terra_nfp_check_v2" \
  --review-dir "$PILOT_DIR/nfp_overcorrection_review_v1" \
  --out-dir "$BALANCED_DIR" \
  --accept-ai-review &&
python scripts/check_terra_judge.py score --out-dir "$BALANCED_DIR" &&
python scripts/check_terra_judge.py report --out-dir "$BALANCED_DIR" --reparse > "$BALANCED_DIR/report.md" &&
cat "$BALANCED_DIR/report.md"
```

`normal`과 `overcorrection`의 라벨 일치도를 각각 본다. 모두 +1을 출력하는 판정기는
정상 답변 10/10이더라도 과잉 교정 답변에서는 0/10이므로 통과했다고 해석하지 않는다.
같은 답변의 두 판정은 독립 표본 20개가 아니다. 선택된 명확한 10답변의 진단 결과이며,
전체 NFP 정확도나 보류 문항까지 포함한 성능을 추정하지 않는다. −1 판정의 인용과
실제 전제 지적 내용도 확인한 뒤 더 큰 평가를 진행할지 결정한다.
