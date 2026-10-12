# Baseline 실험 설정 — 2026-10-12 실행 기준

현재 Table 1(탐지), Table 2(최종 답변), Table 3(일반 QA 보존)에 적용하는 설정을 실행 파일과 고정 명단으로 확인해 정리했다. **과거 제안이나 초기 config 값보다 아래의 실제 실행값이 우선한다.** 결과의 진행 상태는 [현재 표](current_v4/README.md), 변경 이력은 [실행 일지](baseline_v4_runbook_2026-10-12.md)를 본다. Ours/NLA는 이 기준선 실행 범위 밖이며 별도 A6000 서버 작업이다.

## 1. 평가 범위와 분모

| 데이터 | FPQ | NFP | 총 평가 질문 | 분할 |
|---|---:|---:|---:|---|
| Cancer-Myth | 577 | 147 | 724 | 질문 그룹 기준 층화 3-fold, out-of-fold 합산 |
| CREPE | 500 | 500 | 1,000 | 공식 train/dev/test 유지, test에서 고정 표본 추출 |

Cancer-Myth 원본 583 FPQ / 149 NFP에서 고정 few-shot 예시와 연결된 그룹 8개 문항을 제외했다. 제외 ID는 `fpq_289`, `fpq_290`, `fpq_291`, `fpq_415`, `fpq_416`, `fpq_417`, `nfp_1002`, `nfp_1003`이다. 기존 583/149 표나 Luna 99/100 표와 분모가 다르다.

공통 명단: [split_manifest.json](current_v4/split_manifest.json). 세 모델·모든 방법에 같은 평가 질문 ID를 사용한다. 데이터셋별로 독립적으로 학습·최적화하며 Cancer-Myth와 CREPE 학습 자료를 합치지 않는다.

## 2. Cancer-Myth 3-fold를 만든 방식

1. 기존 `group_id`를 사용하고, 정규화한 동일 질문을 연결한다. `NFKC` 정규화, 대소문자 통일, 공백 정리를 적용한다. Cancer 전제 문자열만으로 그룹을 합치지는 않는다.
2. 고정 예시와 겹치는 그룹, 상충 라벨 그룹, 데이터셋 사이에 연결된 그룹을 제외하는 검사를 거친다. 현재 실제 제외 내역은 고정 bundle에 저장돼 있다.
3. 정렬된 질문 목록에 `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261007)`를 적용한다. 같은 그룹은 같은 test fold에 들어가고 FPQ/NFP 비율은 가능한 한 유지한다.
4. 각 outer-train 안에서 다시 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261008 + fold_index)`의 첫 분할을 사용해 train/dev를 약 80/20으로 나눈다.
5. train/dev/test 사이의 **ID와 그룹이 겹치지 않는지** 확인한 뒤 명단을 저장한다. 매 실행마다 새로 추첨하지 않는다.

| Fold | Train FPQ / NFP | Dev FPQ / NFP | Test FPQ / NFP | Test 합계 |
|---|---:|---:|---:|---:|
| fold0 | 307 / 78 | 77 / 20 | 193 / 49 | 242 |
| fold1 | 308 / 79 | 77 / 19 | 192 / 49 | 241 |
| fold2 | 308 / 78 | 77 / 20 | 192 / 49 | 241 |
| Test 합산 | — | — | **577 / 147** | **724** |

**3-fold가 필요한 방법:** TF-IDF gate, Probe gate, GEPA-gate, GEPA, GEPA (both stages). train에서 학습하고 dev에서 hyperparameter·threshold·최종 프롬프트를 고른 뒤 해당 test fold에만 적용한다. TF-IDF는 같은 분류기를 세 답변 모델이 공유하고 Probe·GEPA는 모델별로 만든다.

**fold별 재학습이 없는 방법:** Plain, Balanced instruction, CoT, Direct gate, CoT gate, PreWoMe, Extract + Verify. 고정 프롬프트로 각 질문에 한 번 답변한다. 저장 경로에 fold가 있어도 같은 질문을 세 번 생성하는 것은 아니다. 같은 입력·실행 identity의 완료 캐시는 재사용한다.

주 지표는 세 test fold의 성공 건수와 분모를 합산한다. fold별 비율의 단순 평균이 아니다. 현재는 seed 1개·replicate 0이며 반복 실행 평균이나 반복 변동 검증 완료를 뜻하지 않는다. 기존에 읽은 질문·출력의 노출 이력은 교차검증으로 없어지지 않으므로 탐색 평가라는 제한을 유지한다. 그룹 검사는 완전한 의미 중복 제거를 보장하지 않는다.

## 3. CREPE 설정

공식 train/dev/test 구분을 유지한다. 동일 그룹이 공식 split을 가로지르면 `test > dev > train` 우선순위로 낮은 쪽 문항을 제외한다. 현재 이 검사로 제외된 ID는 `crepe:2018-08635` 한 개다.

| 용도 | FPQ | NFP | 합계 | 사용 방식 |
|---|---:|---:|---:|---|
| Train | 904 | 2,533 | 3,437 | 전체 적격 train 사용 |
| TF-IDF / Probe dev | 544 | 1,456 | 2,000 | 전체 dev로 C·layer·threshold 선택 |
| GEPA dev | 100 | 100 | 200 | 공식 dev에서 고정 층화 표본 |
| 공통 test | 500 | 500 | 1,000 | 공식 test에서 고정 층화 표본 |

샘플링은 FPQ와 NFP의 ID 목록을 각각 정렬하고 Python `random.Random(seed).sample`로 추출한다. 클래스 순서는 FPQ → NFP이며 같은 RNG 인스턴스를 이어서 사용한다. test seed는 **42**, GEPA dev seed는 **43**이다. 추출한 ID를 다시 정렬해 manifest에 저장한다. 문장 길이 필터는 쓰지 않는다. CREPE에는 별도 3-fold를 적용하지 않는다.

‘Train 전체 사용’은 train 풀을 줄이지 않는다는 뜻이다. GEPA가 모든 학습 질문을 같은 횟수 평가한다는 뜻은 아니다. GEPA는 아래의 균형 쌍과 평가 예산 안에서 학습 예시를 사용한다.

## 4. 모델·디코딩·현재 배치

| 항목 | Qwen2.5 7B | Gemma 3 12B | GPT-6 Luna |
|---|---|---|---|
| 모델 | `Qwen/Qwen2.5-7B-Instruct` | `google/gemma-3-12b-it` | `gpt-6-luna` |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` | `96b6f1eccf38110c56df3a15bffe176da04bfd80` | CLI 제공 모델, 체크포인트 hash 없음 |
| 실행 위치 | 4090 GPU 0 | 4090 GPU 1·2·3에 한 모델 분산 적재 | 외부 CLI |
| 정밀도 | BF16 | BF16 | 제공자 설정 |
| 디코딩 | `do_sample=False` (greedy) | `do_sample=False` (greedy) | 제공자 기본값, reasoning `medium` |
| 입력 한도 | 8,192 tokens | 8,192 tokens | HF와 같은 한도를 강제하지 않음 |
| 새 출력 한도 | 8,192 tokens | 8,192 tokens | CLI 제공자 기본값 |
| seed | PyTorch 42 | PyTorch 42 | 샘플링 seed를 지정하지 않음 |
| 현재 batch 상한 | **16** | **32** | 질문 파이프라인 worker **4** |
| 배치 token budget | **147,456** | **393,216** | GPT 전체 공용 슬롯 **4** |

로컬 모델은 SDPA attention, `eval()`·`inference_mode()`, left padding과 attention mask를 사용한다. `temperature=0`을 넘기는 구현이 아니라 greedy(`do_sample=False`)이다. GPT CLI의 샘플링 설정을 로컬 greedy와 동일하다고 기술하지 않는다.

Gemma는 GPU 세 장에 각각 복제한 것이 아니라 한 모델의 레이어를 나누어 올린다. `device_map=auto`, GPU당 배치 전 적재 한도 `21GiB`, CPU/disk offload 금지를 사용한다. `21GiB`는 KV cache까지 포함한 실행 중 최대 사용량이 아니다.

배치는 다음 조건을 모두 만족하는 크기로 묶는다.

- 요청 수가 batch 상한 이하일 것.
- `묶음 크기 × (묶음의 최대 입력 길이 + 8192)`가 token budget 이내일 것.
- 배치 수집 대기 시간 기본값 0.03초. 남은 문항·단계 의존성 때문에 실제 크기는 상한보다 작을 수 있다.
- OOM이면 batch 상한을 절반으로 줄여 같은 입력을 다시 처리한다. 개별 출력 한도·정밀도·프롬프트는 바꾸지 않는다.
- 각 응답에 `actual_batch_size`, 입력·출력 토큰 수, `batch_elapsed_seconds`, `device_map`을 저장한다. 배치 변경에 따른 완전한 bitwise 재현성은 주장하지 않는다.

### 초기 파일과 실제 실행값의 차이

`protocol/runtime.json`은 캐시 identity 때문에 동결돼 있어 초기 Qwen 8 / Gemma 4 및 Gemma GPU 1·2가 남아 있다. 실제 서비스는 `launch_paper_v4_sol.py`의 인자로 Qwen 16 / Gemma 32를 적용하고, `run_paper_gemma_placement.py`가 GPU 1·2·3을 적용한다. 배정은 `main/gemma3/device_overrides/gpu1_gpu2_gpu3.json`에 보존한다.

Haiku 실행 인자에는 `--workers 256`이 남아 있지만 `haiku/workers_control.json`이 우선하며, 이 문서 작성 시 실제 동시성은 **128**이다. 256 시험 후 일시적 제한을 만나 축소했다. 한도 오류 시 절반으로 줄이고 600초 후 재시도한다. 따라서 설정 파일 숫자 하나만 보고 실제 배치나 동시성을 판단하지 않는다.

## 5. 공통 프롬프트와 방법 정의

프롬프트 원문·Well 템플릿 hash는 [registry](table2_prompt_registry_2026-10-07.json)에 고정돼 있다. 모든 생성 모델에 같은 의미의 system/user 메시지를 사용하며 모델별 tokenizer/chat template만 다르다. 공통 지시는 다음과 같다.

> Respond in English using only your existing knowledge. Do not use tools, browse, or access files. Return only the requested output.

| 방법 | 실제 동작 | 학습/최적화 |
|---|---|---|
| Plain | 기존 지식으로 질문에 답변 | 없음 |
| Balanced instruction | 거짓 전제는 교정, 정상 전제는 오류를 만들지 않고 답변 | 없음 |
| CoT | `Analysis` 후 `Final answer` 생성, 최종 답변을 채점 | 없음 |
| Direct gate | 질문만 보고 정확히 Yes/No 판정 → 결과를 포함해 새 답변 생성 | 없음 |
| CoT gate | `Review` 검토문과 마지막 `Verdict: Yes/No` → 검토문·판정을 포함해 새 답변 생성 | 없음 |
| TF-IDF text gate | 질문 텍스트 분류 → 판정을 포함해 새 답변 생성 | 데이터셋별/fold별 분류기 |
| Probe gate | 질문의 고정 hidden feature 분류 → 판정을 포함해 새 답변 생성 | 모델별·데이터셋별/fold별 분류기 |
| PreWoMe | 공통 전제 추출 → 질문·전제 목록에 대한 Feedback/Action → 답변 | 없음 |
| Extract + Verify | 같은 추출 목록 → 전제별 이진 검증 → 질문·목록·검증 결과로 답변 | 없음 |
| GEPA-gate | CoT 형식 detector 프롬프트만 최적화, 답변 경로 지시는 고정 | 모델별·데이터셋별/fold별 GEPA |
| GEPA | Plain을 시작점으로 직접 답변 프롬프트 최적화 | 동일 |
| GEPA (both stages) | detector와 답변 경로에 덧붙이는 supplement를 함께 최적화 | 동일 |

Direct는 현재 **생성한 Yes/No 문자열**을 파싱한다. Yes/No 토큰 logit이나 confidence를 읽는 구현이 아니다. CoT gate는 검토문과 판정을 한 호출로 내고 답변 호출이 뒤따른다.

### Table 1 → Table 2 → Table 3

- Table 1: 판정, 원문 출력, 검토문, 확률/threshold(지원 방법만), 문항 ID·fold를 저장한다. 정답 라벨은 평가에만 사용한다.
- Table 2: 해당 문항의 저장된 판정을 읽고 **실제 새 답변을 생성**한다. Yes면 교정 지시, No면 Plain 지시를 쓰되 질문과 판정 결과를 입력에 넣는다. CoT/GEPA 검토문이 있으면 함께 전달한다. 과거 Plain/Always-correct 답변을 골라 붙이는 방식이 아니다.
- 형식 오류 판정은 `Invalid`와 원문을 보존하고 현재 구현은 Plain 경로로 전달한다. 탐지 오류를 정상 판정 정확도로 해석하지 않는다.
- 학습 gate는 Table 1과 Table 2에서 같은 fold의 분류기·threshold·판정을 사용한다.
- Table 3: 질문 자체가 다르므로 판정·답변을 새로 생성한다. FPQA test의 판정값을 가져오지 않는다.

### 공유 전제 추출

추출은 **GPT-6 Sol, reasoning high, timeout 240초**로 통일했다. 동일 데이터셋·질문의 추출 목록을 세 답변 모델과 PreWoMe/Extract + Verify가 공유한다. 검토·전제 검증·최종 답변은 각 대상 모델이 맡는다. 원문 질문과 고정 예시만 추출기에 주고 test 정답·교정문을 넣지 않는다.

Well revision은 `6c9770f65da6e9c50250cf94e35b07f258288e38`, 고정 few-shot 예시는 데이터셋당 4개다. no-RAG(`passages=[]`) 조건이다. 공유 추출은 모델 간 비교를 통제하기 위한 우리 실행 설정이며, 원 논문의 전 과정 동일 모델 실행과 동일하다고 주장하지 않는다.

## 6. TF-IDF / Probe hyperparameter

| 항목 | TF-IDF | Probe |
|---|---|---|
| 입력 | 원 질문 텍스트 | Plain wrapper를 씌운 원 질문의 마지막 입력 token hidden state |
| 특징 | word 1–2gram, `min_df=2`, `sublinear_tf=True`, 최대 50,000개 | Qwen layer 7/14/21/28, Gemma layer 12/24/36/48 |
| 전처리 학습 | train에서 vocabulary·IDF 학습 | train에서 `StandardScaler` 학습 |
| 분류기 | Logistic regression | Logistic regression |
| C 후보 | 0.01 / 0.1 / 1 / 10 | 동일 |
| 기타 | `class_weight=balanced`, `solver=lbfgs`, `max_iter=2000` | 동일 |
| Threshold 후보 | 0.05~0.95, 간격 0.05 | 동일 |
| 선택 | dev balanced accuracy 최대 | dev에서 layer·C·threshold 함께 선택 |

동점이면 낮은 dev FPR → 작은 C → 낮은 layer → 큰 threshold 순서로 선택한다. test의 FPR을 보고 threshold를 맞추지 않는다. 따라서 현재 Table 1은 matched-FPR 비교가 아니다. Luna 내부 hidden state에 접근하지 않으므로 Probe는 해당 없음이다.

고정 모델의 hidden feature는 질문별로 미리 계산해 재사용할 수 있지만, scaler·분류기 학습과 layer/threshold 선택은 반드시 해당 train/dev에서만 한다. CREPE 특징 추출 총 6,437개는 train 3,437 + dev 2,000 + test 1,000이며 test를 학습에 포함한다는 뜻이 아니다.

## 7. GEPA 설정과 예산

| 항목 | 실제 적용값 |
|---|---|
| 최적화 대상 | GEPA-gate / GEPA / GEPA (both stages), 각각 독립 실행 |
| 실행 분할 | Cancer fold0/1/2 + CREPE official |
| seed | 42, 현재 1회 실행 |
| 시작점 | detector = CoT detector / answer = Plain / supplement = 빈 문자열 |
| 평가·reflection 모델 | GPT-6 Luna, reasoning medium, timeout 240초 |
| 목적 점수 | `(FPQ rating/5 + NFP rating/5) / 2` |
| 평가 예산 | 실행당 2,000 question evaluations = 최대 1,000 paired metric calls |
| Reflection minibatch | 2쌍, 즉 2 FPQ + 2 NFP |
| 프롬프트 길이 | component당 최대 512 tokens |
| 길이 tokenizer | `Qwen/Qwen3-Embedding-0.6B`, revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| GEPA evaluator 병렬성 | `parallel=False`, `max_workers=1` |
| 캐시 | `cache_evaluation=True`, 완료 후보·출력 재사용 |

train은 FPQ/NFP를 각각 seed 42로 섞고 한 쌍씩 묶는다. 작은 클래스는 순환 재사용하여 train 전체 풀을 유지한다. Cancer train 쌍 수는 fold0 307, fold1/2 308이며 CREPE는 2,533쌍이다. 이는 유일한 질문 수와 같지 않다.

후보 선택용 dev는 클래스별 그룹당 한 문항을 남기고 양쪽 수를 맞춘다. **Cancer는 실제 20+20 / 19+19 / 20+20**, CREPE는 **100+100**이다. Manifest의 Cancer dev 97/96/97개 모두가 GEPA 후보 선택에 쓰이는 것은 아니다. TF-IDF/Probe는 해당 dev 전체를 쓴다.

예산 2,000은 모델 호출 수나 금액이 아니다. 각 문항 평가 안에 판정·답변·judge 호출이 있고 reflection 호출도 따로 있다. 또 CLI GPT 슬롯 4개와 GEPA evaluator 병렬성 1은 다른 설정이다. 최적화 평가가 순차적이어서 일반 기준선의 GPU batch 상한만으로 GEPA 속도를 추정하면 안 된다.

`protocol/external.json`의 예전 `max_metric_calls=500`, `workers=5`는 현재 optimizer의 실제 예산/병렬성 기준이 아니다. 현재 실행기는 `runtime.json.gepa.metric_budget=2000`과 코드의 `EngineConfig`를 읽는다.

## 8. 채점·지표·실패 처리

| 표 | 지표 | 채점 |
|---|---|---|
| Table 1 | FPQ 탐지율(TPR) ↑, NFP 오탐률(FPR) ↓ | 저장 verdict와 정답 라벨 직접 비교 |
| Table 2 | FPQ 교정 / NFP 보존: rating ≥4 비율 | 고정 Well rubric, 최종 평가 Haiku |
| Table 3 | 정답 정확도 | 선택지·label·숫자 정규화 후 정답 비교 |

최종 judge 기본 모델은 `claude-haiku-5-5`, effort medium, timeout 240초다. 생성 완료를 기다리지 않고 저장된 답변을 독립 큐가 계속 채점한다. 현재 동시 요청 상한은 128이다. 최적화 내부 GPT 점수를 최종 Haiku 점수로 섞어 쓰지 않는다. 예외적인 직접 판정의 provenance는 별도 평가 원장에 보존하며 표에는 완료 상태만 표시한다.

입력 한도 초과·출력 EOS 없이 한도 도달은 문항 실패를 기록하고 나머지를 계속 처리한다. 실패 문항은 원래 분모에서 빼거나 성공으로 만들지 않는다. 미채점 문항이 있는 셀은 최종 성능으로 표시하지 않는다. 판정 파싱이 실패하면 별도 invalid 기록을 유지한다. Table 1의 단순 yes 비율만으로 invalid가 정상 No였다고 해석하면 안 된다.

채점 결과 재사용은 질문·답변·주석·judge 설정이 일치하는 평가 키로 검증한다. 원본 답변과 채점 원장을 분리한다. 95% 구간이나 반복 실행 분산은 현재 표에서 계산 완료했다고 주장하지 않는다.

## 9. Table 3 데이터와 적용할 시스템

| 데이터셋 | 문항 수 | 실제 source split | source revision |
|---|---:|---|---|
| MedQA | 1,273 | test | `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18` |
| PubMedQA | 1,000 | PQA-L train 풀 | `9001f2853fb87cab8d220904e0de81ac6973b318` |
| Medbullets | 308 | op4_test | `527f6fc0eebac4eeedf3dd64b9ee87c3bcf23b36` |
| MMLU | 14,042 | test | `c30699e8356da336a370243923dbaf21066bb9fe` |
| GSM8K | 1,319 | test | `740312add88f781978c0658806c59bc2815b9866` |

총 17,942문항이다. PubMedQA의 1,000개는 공식 test 성능이라고 부르지 않는다. 이 QA 질문들은 GEPA/분류기 학습이나 후보 선택에 넣지 않는다.

- Plain/Balanced/CoT/Direct gate/CoT gate: 데이터셋 학습에 의존하지 않는 공통 설정으로 한 번 평가.
- PreWoMe/Extract + Verify: Cancer와 CREPE 고정 예시 설정을 각각 적용.
- TF-IDF/Probe/GEPA 계열: 미리 지정한 Cancer **fold0** 시스템과 CREPE 시스템을 각각 고정 적용. Cancer 세 fold 평균이 아니라 두 출처 시스템의 transfer 평가다. 일반 QA 결과를 보고 더 좋은 프롬프트로 선택하지 않는다.
- 표에서 두 수치는 Cancer 시스템 / CREPE 시스템 순서다. QA 답변은 마지막 `Answer:`를 엄격히 파싱하고, 파싱 실패는 정답 처리하지 않는다.

## 10. 실행 순서·버전·재현 파일

4090에서 Qwen(GPU 0), Gemma(GPU 1·2·3), 외부 Luna 호출을 병행한다. Sol 추출과 GPT 최적화 호출은 공통 GPT 슬롯 4개를 공유한다. Haiku 채점은 별도 큐다. A6000 NLA 작업은 이 서버에서 실행하지 않는다.

각 모델은 탐지를 먼저 준비하고 Table 2의 일반 기준선 9개를 두 데이터셋 모두 생성한 뒤 GEPA-gate → GEPA → GEPA (both stages), 이후 Table 3의 남은 생성을 수행한다. Luna는 Probe가 없어 일반 기준선이 8개다. 이미 저장된 Table 3 Plain 결과는 재사용한다. Gemma Probe feature 추출은 같은 GPU lock을 공유하는 별도 스레드라 답변 생성과 GPU 연산을 동시에 수행하는 것은 아니다.

확인된 라이브러리: PyTorch `2.5.1+cu121`, Transformers `4.57.6`, scikit-learn `1.9.0`, Accelerate `1.14.0`.

- 로컬 실행 입력: `results/paper_baselines_v4/protocol/{bundle,split_manifest,runtime,external,haiku,extraction_sol}.json`
- 결과: `results/paper_baselines_v4/main/{qwen25,gemma3,luna6}/`
- Haiku 평가 및 직접 판정 원장: `results/paper_baselines_v4/haiku/`
- 동시성 제어: `results/paper_baselines_v4/haiku/workers_control.json`
- 프롬프트 원문: [registry](table2_prompt_registry_2026-10-07.json)
- 데이터 명단: [manifest](current_v4/split_manifest.json)
- 실행 검증 요약: [setup audit](current_v4/experimental_setup_audit_2026-10-12.json)
- 실행기: `scripts/launch_paper_v4_sol.py`, `scripts/run_paper_baselines_sol.py`, `scripts/run_paper_gemma_placement.py`, `scripts/judge_paper_queue.py`
- 분할 구현: `scripts/prepare_paper_baseline_bundle.py`, `scripts/prepare_paper_v4.py`

고정 bundle SHA-256:

```text
88f6f9d4ef07f2812112d1004abdbeedcd90f910171d309a5d3d93d33a16fbe1
```

## 11. Git 정리 범위

2026-10-12에 `main`에 이미 포함된 원격 `claude/document-session-o2temh`와 로컬 게시용 `codex/paper-tables-20261006`을 정리하고, 깨끗한 게시 저장소를 `main`으로 통일했다. 삭제 전 원격 브랜치 tip은 `e5fcd34`이며 해당 커밋은 main 이력에 남아 있다.

실험 작업공간의 main에는 실행 중인 코드와 미커밋 변경·미추적 파일이 다수 있으므로 강제 checkout/reset으로 덮어쓰지 않는다. 별도 `claude/paper-writing-start-a7fd1f` worktree에는 미커밋 `paper/writing_guide.md`가 있어 보존한다. 이 원고 초안의 예전 CREPE 분모·Sonnet 채점·GEPA 500 예산보다 이 문서의 확인된 현재 설정이 우선한다. 원고 파일 자체는 변경하지 않았다.
