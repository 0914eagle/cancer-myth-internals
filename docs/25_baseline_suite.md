# 25. 질문 판별과 답변 생성 baseline 실행

> **2026-09-15 실행 후 갱신:** 실제 결과·변경 이유는 [26 §8–9](26_experiment_history_and_decisions.md#8-새-qwen-suite-질문-판별과-답변-평가를-분리했다)를 따른다.
> 아래 512 기본값과 `qwen25_7b_v1` 명령은 최초 suite 기록이다. 길이 점검 후 현재 전환 절차는
> [final 1024 별도 suite 준비](reviews/final_budget_check_protocol.md#user-reported-outcome-and-full-suite-transition)다.
> 새 폴더의 `generation_defaults.json`이 1024를 고정하며 원 폴더/캐시는 재명명하지 않는다.
> 사용자 마지막 로그 extract_verify 319/732에는 실행 폴더가 없어 최종 run metadata 확인이 남는다.
> 이 페이지 말미의 “아직 실행하지 않은 것”은 코드 추가 당시 기록이다. Qwen prefill과 일부 gate는 이미 실행했다.
> **2026-09-16 추가:** gate 신호 `style`/`masked`(문체 하한선), 보고서의 같은 오탐 TPR 부록, `extract --variant-file`,
> 의역·쌍둥이 통제 드라이버는 [29](29_style_confound_tests.md). 기존 `crossfit_v1`의 plan.json은 동결이라 새 신호는 새 이름으로 돌린다.

2026-09-15. 이 문서가 새 baseline 실행 순서다. 기존 Gemma pilot/quick45/overnight
파일과 점수는 그대로 보존한다. 코드 구현·오프라인 검증과 실제 모델 성능 측정은 구분한다.

## 범위와 출처

중심 모델 기본값은 `Qwen/Qwen2.5-7B-Instruct`. `SUITE_CONFIG`로 Gemma를 선택할 수 있다.
같은 모델이 선행 논문에 있다는 이유만으로 그 논문 수치를 이번 실험 결과로 사용하지 않는다.

Well 공개 GitHub는 `a7ee871eadde1104f7560a2874a03cbf5221dbaa` 기준 당시
Cancer-Myth train/dev/test 문항 목록을 제공하지 않는다. 공개 TPQ JSONL 148개에도
split 필드가 없고, generate 경로는 seed를 고정하지 않고 shuffle한다. 따라서 이번 코드는
**우리 기존 grouped manifest**를 사용한다. 새로 만든 분할을 Well의 고정 test라고 부르지 않는다.

- [Well §2.2: 답변 방법](https://arxiv.org/html/2608.06539v1#S2.SS2)
- [Well Appendix E: 0–5 답변 평가](https://arxiv.org/html/2608.06539v1#A5)
- [Two Axes Table 2: CREPE 질문 판별](https://arxiv.org/html/2607.08456v1#S5)
- [Zero-shot-CoT](https://arxiv.org/abs/2205.11916)

### 답변 생성 표

| 코드 이름 | 실행 | 성격 |
|---|---|---|
| `plain` | 질문 원문에 직접 답변 | 원 pilot과 같은 user-only 기본 대조군 |
| `zero_shot_cot` | step-by-step 추론 생성 후 별도 최종 답변 | Kojima식 동기를 적용한 의료 adaptation |
| `fp_identification` (**보고 이름: Self-gated FP Identification**) | 모델의 거짓 전제 Yes/No 판별(`direct`) 후 양성이면 교정 지시, 음성이면 Plain | **Well의 FP Identification과 다른 방법이다.** Well은 모든 질문에 교정 지시를 무조건 붙인다(Qwen2.5-7B: FPQ S5 75% / TPQ 0%). 우리 것은 Two Axes식 라우팅(게이트 → 프롬프트)에서 게이트를 모델 자신의 언어 판별로 바꾼 것이며, 판별이 거의 항상 No라 Plain에 수렴한다(28 §10.1). 코드 식별자는 진행 중인 판정 계획에 박혀 있어 배치 완료 후 `self_gated_fp`로 바꾼다 |
| `extract_verify` | 전제 JSON 추출, 각 전제 진위 판별, 결과에 따른 답변 | Well 계열 파이프라인 adaptation, RAG 없음 |
| `premise_review` | 기존 전제 검토문 생성 후 그 검토문으로 답변 | 기존 `premise_cot`의 명확한 이름 |

Well식 **무조건** FP Identification(모든 질문에 교정 지시)은 이 suite에 아직 없다. 추가하면 FPQ 상한과 NFP harm 행이 생기고 Well의 75/0과 직접 비교된다(생성 732 + 판정 732; 결정 대기, 32 §7).

원 Well의 few-shot·생성 단계·모델 설정을 그대로 재현한 행으로 표기하지 않는다.
특히 공개 FP Identification 템플릿에는 no-RAG user content 조건식 문제가 있어,
질문을 확실히 포함하는 별도 구현을 사용한다. 선행 보고값은 인용 블록으로만 둔다.
원 Well 평가 프롬프트는 `prompts/well/`에 출처와 해시를 고정해 별도로 보관한다.

생성 모델에는 `question`만 전달한다. 정답 라벨, 거짓 전제 주석, correction,
정상 질문의 Possible hallucination, Well TPQ 주석은 모델 입력으로 주지 않는다.
판별 중간 단계가 낸 신호만 답변 지시에 반영한다.

기본 final cap은 512, review cap은 1024, 추출 cap은 512다. Greedy 생성이며,
EOS/길이 도달 정보를 저장한다. 길이 도달은 정답/오답이 아니라 잘림 진단으로 보고한다.
추출 JSON이 깨졌다고 정상 질문으로 간주하거나 자동으로 Plain에 떨어뜨리지 않는다.

### 질문 판별 표

| 코드 이름 | 점수 | 학습·보정 |
|---|---|---|
| `text` | 질문 TF-IDF unigram/bigram → logistic 점수 | train 내부 group CV로 C 선택 |
| `direct` | 질문만 보고 거짓 전제 Yes/No 정규화 확률 | 고정 prompt, calibration으로 문턱 |
| `review` | 전제 검토문을 읽은 뒤 Yes/No 정규화 확률 | 검토문은 모델 생성, calibration으로 문턱 |
| `hidden` | 마지막 prompt 토큰의 block 출력 → logistic 점수 | train 내부 group CV로 층/C 선택 |
| `mean` | FPQ 평균 − 정상 질문 평균 방향으로 hidden state 투영 | train 내부 group CV로 층 선택 |

`text`는 Two Axes의 정확한 hashed-BoW 재현이 아니라 TF-IDF 대조군이다.
`mean`은 탐지 방향이며 교정 steering C가 아니다. 답변 확률 기반 output readout,
P(IK), P(True), 별도 외부 CoT monitor, RAG는 이번 첫 실행에 포함하지 않는다.

판별 성능은 AUROC, dev에서의 TP/FPQ 및 FP/NFP, TPR/FPR와 source-group bootstrap
95% CI로 보고한다. 문턱은 calibration 정상 질문에서 empirical FPR ≤5%를 만족하게 고정한다.
이는 실제 모집단 FPR ≤5% 보장이 아니다. 정상 표본이 적다는 한계를 보고서에 표시한다.

## 데이터와 분할

입력은 기존 `$PILOT_DIR/split/manifest.json`이다. 같은 원질문 중복, FPQ/NFP 충돌
격리, 과거 few-shot 제외와 group ID를 유지한다. Well 판정 예시와 겹치는 질문도
prepare 단계에서 제외하여 최종 분모는 출력된 counts를 사용한다. 735/732를 하드코딩하지 않는다.

`--tpq PATH`는 질문 원문을 대조해 정상 질문에 전제 주석을 붙일 뿐 새 질문을 추가하지 않는다.
`--exclude-ids ...`로 사용자가 명시한 few-shot 문항을 추가 제외할 수 있다.

기본 holdout:

- 기존 fit을 다시 train/calibration으로 group 분할한다.
- train 안에서 group CV로 층과 정규화 C를 정한다.
- calibration 정상 질문 점수로 문턱을 고정한다.
- 기존 dev에서 한 번 예측하고 보고한다. 이미 여러 번 본 dev이므로 탐색 결과다.
- `--evaluation test`는 사용자가 명시적으로 선택한 최종 평가 실행이다. 그 전에 설정을 고정한다.

`--scheme crossfit`은 전체 질문을 outer 5-fold로 나눈다. 매 fold의 나머지 데이터 안에서
train/calibration을 다시 나누므로 outer evaluation이 학습·설정 선택에 쓰이지 않는다.
각 질문의 OOF 예측은 한 번뿐이다. raw decision score의 크기는 fold마다 다르므로
서로 섞어 AUROC를 계산하지 않고, fold별 AUROC의 표본 수 가중 평균을 보고한다.
이것도 기존 dev 탐색을 없애거나 완전히 새로운 독립 test를 만드는 절차는 아니다.

`splits/.../fold_k/{train,calibration,evaluation}.jsonl`은 이후 GEPA/finetuning에도
사용할 수 있다. **이번 코드는 GEPA 최적화나 finetuning을 실행하지 않는다.**
비용과 선행 인용 가능성을 고려해 이 둘은 다음 단계로 미뤘다.

## 서버 실행

최종 테스트는 사용자 요청에 따라 서버에서 수행한다. 다음 두 단계는 실험 결과를
만들지 않는다. `test`는 모델을 가짜 실행기로 대체한 단위·통합 검증이고,
`test-runtime`은 무작위로 초기화한 아주 작은 Qwen/Gemma의 CPU forward 검증이다.
후자도 pretrained 가중치를 내려받거나 서버 GPU를 사용하지 않는다.

```bash
# 아래 서버 환경 설정과 git pull을 마친 뒤 실행
bash scripts/run_baselines.sh test
bash scripts/run_baselines.sh test-runtime
```

`pytest`가 없으면 활성 환경에 먼저 `uv pip install pytest`로 설치한다.
최종 변경에 대한 위 두 실행 결과는 아직 서버에서 확인하지 않았다.

```bash
cd /home/eagle0914/cancer-myth-internals
git switch main
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
export DATA_ROOT=/data1/heejae
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export SUITE_DIR=/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_v1
export SUITE_CONFIG=configs/qwen25_7b.yaml
export CUDA_VISIBLE_DEVICES=0,1
bash scripts/run_baselines.sh references
bash scripts/run_baselines.sh prepare
```

`references`는 Hugging Face의 원 FPQ 주석을 다운로드하고 revision·SHA를 고정한다.
`source_myth`와 `presupposition_correction`을 질문 텍스트로 연결해 Well 판정 입력을 만든다.
기존 manifest의 합쳐진 교정 설명을 임의 분해하지 않는다. 두 준비 단계는 모델이나 GPT를 호출하지 않는다. 모델 가중치·토크나이저 다운로드는
GPU 단계를 처음 실행할 때 필요할 수 있다. `tmux` 안에서 다음을 실행한다.

```bash
bash scripts/run_baselines.sh gpu
bash scripts/run_baselines.sh gate
```

`gpu`는 GPU0에서 5종 생성 후 출력 판별을, GPU1에서 질문 hidden state 추출을 실행한다.
동시에 채점기를 호출하지 않는다. GPU 하나만 쓰려면 `CUDA_VISIBLE_DEVICES=0`으로 설정하면
순차 실행한다. 0/1 이외 GPU를 지정하면 실행을 거부한다. 로그는 `$SUITE_DIR/logs/`이다.
GPU stages는 끝나는 시간 상한을 보장하지 않으며, SSH가 끊겨도 유지하려면 tmux를 사용한다.

작업은 단독 실행도 가능하다.

```bash
bash scripts/run_baselines.sh generate
bash scripts/run_baselines.sh detect
bash scripts/run_baselines.sh extract
bash scripts/run_baselines.sh status
```

출력은 기존 pilot 밖의 별도 폴더에 저장한다. 다시 실행하면 검증된 캐시를 재사용한다.
질문 목록·모델·토큰 한도·프롬프트를 바꿀 때는 새 `SUITE_DIR`를 사용한다.
같은 디렉터리에 서로 다른 partition 실행을 섞지 않는다. 기본은 전체 질문 생성·추출이다.
답변 생성은 전체에 할 수 있어도 전체 점수를 모두 독립 test라고 주장할 수는 없다.

### GPU 없이 텍스트 분류기부터 확인

```bash
python scripts/run_baseline_suite.py gate --out-dir "$SUITE_DIR" \
  --signals text --name text_holdout_v1
```

### 전체 질문 OOF 판별

```bash
bash scripts/run_baselines.sh crossfit
```

출력 gate 보고서는 `$SUITE_DIR/gates/holdout_v1/report.md` 또는 `crossfit_v1/report.md`다.
이 표에는 최종 답변 채점이나 선택적 steering이 들어 있지 않다.

## Well 판정: 생성과 완전히 별도

```bash
bash scripts/run_baselines.sh judge-plan
```

질문·답변·원본 평가 템플릿으로 판정 입력을 고정한다. 동일한 판정 입력은 한 번만 채점해
다른 방법에 공유한다. 실제 호출 수는 이 계획에 출력되며 단순히 문항 수×5로 지불하지 않는다.
준비와 report는 외부 모델을 호출하지 않는다.

```bash
MAX_JUDGE_CALLS=20 bash scripts/run_baselines.sh judge
bash scripts/run_baselines.sh report
```

위 judge 명령은 **최대 20개의 새 Codex Terra 호출**이다. 환경의 기존 `JUDGE_MODEL=sol` 값을
따라 몰래 모델을 바꾸지 않는다. 기본값은 `codex`, `gpt-5.6-terra`이며 계획에 고정한다.
처음 출력된 원문·이유를 확인한 뒤 같은 명령의 상한을 늘려 이어갈 수 있다.
`MAX_JUDGE_CALLS`를 생략하면 wrapper는 중단한다. 무제한 자동 채점 단계는 없다.

채점 결과는 0–5 분포, valid/expected, 0점 비율, 전체 예정 문항 기준 S5,
1–5 유효 점수 평균을 분모와 함께 보고한다. `Rating: X`가 유효하지 않거나
호출이 중단되면 보류/누락이며 성공으로 대체하지 않는다. 시작된 호출은 재개 시 자동 재시도하지 않는다.
이번 Terra 결과는 Well의 Gemini 판정 결과와 같은 원 논문 숫자라고 주장하지 않는다.
Well TPQ의 5점 미만을 모두 과잉 교정이라고 합치지 않는다. 3점에는 회피도 포함된다.

### 저장 답변 재평가

모델/프롬프트가 다른 기존 답변을 새 baseline 행처럼 몰래 합치지 않는다.
기존 JSONL과 `.run.json`을 함께 기록해 `legacy_` 이름으로 재평가할 수 있다.

```bash
python scripts/run_baseline_suite.py import-answers --out-dir "$SUITE_DIR" \
  --name legacy_gemma_plain \
  --answers "$PILOT_DIR/dev/plain.jsonl" \
  --run-metadata "$PILOT_DIR/dev/plain.jsonl.run.json"
```

이 경우는 Qwen baseline으로 쓰면 안 된다. 특정 subset/방법만 채점할 때는
`scripts/evaluate_well.py prepare --help`의 직접 인터페이스를 사용한다.

## 아직 실행하지 않은 것

이 변경은 스크립트와 오프라인 테스트다. 새 Qwen 답변, 실제 gate 성능,
새 Well/Terra 점수, GEPA/finetuning 성능, selective steering 효과를 확보한 것이 아니다.
원 논문의 unpublished split을 복원했다는 주장도 하지 않는다.
