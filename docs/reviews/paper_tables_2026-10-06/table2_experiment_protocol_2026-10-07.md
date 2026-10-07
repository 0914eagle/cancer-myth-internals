# Table 2 공통 실험 설계

2026-10-07. **실행 전 설계 v1**. 이 문서를 작성하면서 생성·채점·GEPA를 실행하지 않았다. 기존 표의 측정값을 새 프로토콜의 결과로 바꾸지 않는다. 분할 ID manifest와 실행기 통합은 아직 완료되지 않았다.

목표는 같은 질문에 대해 **필요한 교정과 정상 질문 보존을 함께 평가**하는 것이다. 같은 방법은 모든 모델에서 같은 영어 메시지와 예시, 같은 단계, 같은 파서를 사용한다. 서로 다른 방법까지 프롬프트를 같게 만들지는 않는다. 모델별 native chat template와 제공되지 않는 생성 설정은 차이로 기록한다.

### 문서와 진행 상태

| 항목 | 상태 |
|---|---|
| Cancer-Myth grouped 3-fold, 공통 모델·방법 배정 | 설계에 반영; 실제 ID 배정 전 |
| 방법별 입력·출력·프롬프트 | 아래 §3–7 및 [프롬프트 원문](table2_prompt_registry_2026-10-07.json)에 정리 |
| 예시 중복과 잠정 평가 분모 | [원자료 점검 기록](table2_source_audit_2026-10-07.json)에 보존 |
| pooled / 데이터셋별 최적화 | 미확정; pooled를 권장안으로 작성, 대안은 §12 |
| Ours의 SAE 채택 | 예비 특징 검증 후 결정; 표 이름은 Ours 유지 |
| 실행기 통합·모델 호출·새 결과 | 미완료; 기존 표의 숫자는 과거 측정값 |

문서는 분할(§2), 모델·방법·gate(§3–6), GEPA/Ours(§7), 평가·재사용(§8–9), 일반 QA와 실행 순서(§10–11) 순서로 읽으면 된다. 하이퍼파라미터와 예산의 제안값은 실행 전 manifest에 확정하고 이후 변경 이력을 남긴다.

## 1. 학습 자료 구성

이번 초안의 권장 주 설정은 **Cancer-Myth와 CREPE의 학습 자료를 모아 모델별 공통 프롬프트를 최적화**하는 것이다. Pooled/separate 선택은 미확정이다. 데이터셋별 학습을 택할 경우 §12의 대안을 사용한다.

- Cancer-Myth: 질문 그룹 기준 외부 3-fold. fold마다 두 블록에서 학습하고 나머지 한 블록에서 평가한다.
- CREPE: 공개 train/dev/test를 유지한다. Cancer의 fold와 관계없이 test는 최적화·특징 발견에서 제외한다.
- pooled 실행 f: Cancer의 해당 outer-train 중 **inner-dev를 제외한 opt_train** + CREPE train에서 학습하고, Cancer inner-dev + CREPE dev에서 후보를 선택한다. Cancer outer-test와 CREPE test에서 각각 평가한다. Inner-dev의 예시는 반성·특징 학습에 사용하지 않는다.
- 따라서 모델·학습 방법마다 3개의 동결된 결과물이 생긴다. Cancer에서는 문항별 해당 OOF 결과 하나를 합친다. CREPE에서는 같은 test에 대한 세 결과물의 성능을 각각 계산하고 평균한다. 이를 CREPE의 3-fold OOF라고 부르지 않는다.
- 같은 모델에서는 Table 2의 Cancer/CREPE와 Table 3의 의료/일반 QA에 같은 결과물을 적용한다. 평가 데이터셋 이름으로 프롬프트를 선택하지 않는다.
- TF-IDF도 pooled 텍스트로 학습한 공통 탐지기를 사용한다. Probe는 각 모델의 pooled 질문 표현으로 학습한다. GEPA와 고정 기준선에만 서로 다른 자료 범위를 주지 않는다.

## 2. 무작위 3-fold: 질문 행이 아니라 그룹을 섞는다

### 2.1 현재 자료에서 확인한 사항

`configs/premise_sft/cancer_questions.jsonl`에는 732문항(FPQ 583 / NFP 149), 기존 group_id 516개가 있다. FPQ group 367개, NFP group 149개다. 정규화한 동일 premise_text가 여러 기존 group_id로 나뉜 경우는 없었다. **동의어 통념·의역의 의미상 중복까지 검증했다는 뜻은 아니다.**

Well 고정 예시의 질문을 원문으로 대조하면 `fpq_291`, `fpq_417`, `nfp_1002`, `nfp_1003`이 전체 자료와 겹친다. 이들의 기존 group_id에 속하는 문항은 총 8개다. 예시 그룹을 공통 평가에서 제외하면 잠정 적격 집단은 **FPQ 577 / NFP 147 = 724개**다. 기존 199문항에서 한 예시만 제외한 것과 전체 자료의 제외 범위가 다르다.

자세한 원문 대조와 파일 해시는 `table2_source_audit_2026-10-07.json`에 저장한다. few-shot에 남는 내용은 모든 fold의 고정 지원 자료로 공개하고, 그 그룹을 outer-train/dev/test에서 모두 제외한다. 이 사례를 통념 미노출 평가에 섞지 않는다.

### 2.2 분할 규칙

1. 평가·생성 예시, 원본 질문, 의역, 정상 쌍둥이, 동일 통념/출처 연결을 감사한다. 출처 URL이 같다는 이유만으로 관계없는 주장까지 한 그룹으로 묶지는 않는다. 연결 관계의 전이적 폐쇄를 group_id로 만든다.
2. FPQ/NFP 라벨 충돌, 정확한 중복, 예시 중복을 처리한 적격 ID와 제외 사유를 고정한다. 최종 분모가 감소하면 실제 수를 보고한다.
3. `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261007)`로 외부 분할한다. 레코드는 dataset/id 순서로 정렬한 뒤 입력한다. FPQ/NFP 비율은 가능한 한 맞추되 그룹 보존이 우선이다. 각 fold에 정확히 같은 개수를 강제하지 않는다.
4. 외부 학습 집단은 다시 그룹 단위 약 80/20으로 opt_train/opt_dev에 나눈다. 내부 5-fold 분할의 첫 블록을 dev로 선택하는 규칙과 seed `20261008 + outer_fold`를 미리 고정한다. 성능을 보고 더 좋은 seed를 고르지 않는다.
5. Ours feature_fit/check는 opt_train 내부에서만 약 75/25로 분할한다. 질문의 모든 모델 출력·반복·답변 쌍·방향 반전은 같은 구역에 둔다.
6. 모든 모델·방법은 **하나의 manifest**를 공유한다. `dataset, id, group_id, outer_fold, inner_role_by_fold, feature_role_by_fold, exclusion_reason, exposure_history`와 데이터/코드/분할 해시를 저장한다.
7. CREPE는 공식 test 우선으로 중복을 제거하고 train/dev와의 의미상 그룹 중복을 감사한다. test에 연결된 train/dev 그룹을 학습에서 제외한다. 결과를 보고 test를 새로 섞지 않는다. Cancer와 CREPE 간 중복도 검사한다.

단순 계산상 fold당 정상 질문은 약 49개다. 실제 수는 그룹 감사와 분할 결과로 결정한다. 과거 읽은 문항은 OOF여도 연구자의 미노출 문항이 되지 않는다. 기존 노출 이력은 계속 명시한다.

## 3. 모델과 메시지를 맞추는 규칙

| 표시 | 설정 파일의 모델 식별자 | 고정할 차이 |
|---|---|---|
| Qwen2.5 | Qwen/Qwen2.5-7B-Instruct | checkpoint/tokenizer/runtime revision |
| GPT-6 Luna | gpt-6-luna | served model ID, CLI 버전, reasoning effort=medium |
| Gemma 4 | google/gemma-4-12B-it | checkpoint/tokenizer/runtime revision |
| Qwen3.8 OFF | Qwen/Qwen3.8-27B-FP8 | enable_thinking=false |
| Qwen3.8 ON | Qwen/Qwen3.8-27B-FP8 | enable_thinking=true |

위 이름은 현재 저장소 설정의 식별자다. 실제 서버에 해당 체크포인트가 로드됐는지 preflight에서 확인한다. 다른 버전으로 자동 대체하지 않는다. Qwen OFF/ON은 별도 설정으로 보고한다.

- 사용자 입력은 원 질문 그대로다. 추론에 test 라벨, reference, 정답 전제, 채점 이유, 파일 경로를 주지 않는다.
- 저장·해시의 단위는 **최종 렌더링한 system/user 메시지**다. 영어, closed-book, 도구/검색 없음이라는 공통 실행 문구를 모든 task 단계에 동일하게 적용한다. Well template 본문은 유지하되 이 공통 transport wrapper를 사용한다는 점을 재현 설명에 밝힌다.
- 공통 문구는 `table2_prompt_registry_2026-10-07.json`의 `transport_common`이다. CLI만 추가 문구를 받거나 local_http만 빠지는 일이 없게 한다. 기존 exact_well_messages 경로와 자동으로 호환된다고 보지 않는다.
- local backend는 temperature=0, do_sample=false, max_new_tokens=8192를 기본으로 제안한다. 불필요한 top_p/top_k를 함께 조정하지 않는다. 제공되지 않는 CLI temperature/seed/token cap은 `null / provider_default`로 기록한다. 동일 decoding 또는 완전 재현을 주장하지 않는다.
- 단계별 토큰/시간/중간 출력/종료 사유를 기록한다. reasoning 모델의 내부 사고 토큰과 출력 설명은 구분한다. 자원 제한 때문에 모델별 유효 한도가 다르면 표시한다.
- 모델당 동일 backend·checkpoint·생성 설정을 모든 방법에 유지한다. 요청마다 새 대화를 사용한다. 모델 내 방법 순서는 seed로 섞어 시점 효과를 줄이고 날짜·served ID를 기록한다.
- preflight는 학습 구역의 사전 선택한 문항만 사용한다. 최종 test로 형식·길이·프롬프트를 튜닝하지 않는다.

## 4. Table 2의 13개 행 정의

| 행 | 질문에서 최종 답변까지 | 최적화 대상 | 기본 task 호출/질문 |
|---|---|---|---:|
| Plain | 질문 → 답변 | 없음 | 1 |
| Balanced instruction | 교정/정상 답변을 함께 지시 → 답변 | 없음 | 1 |
| CoT | 일반적인 단계별 검토 → final answer | 없음 | 1 |
| PreWoMe | 전제 목록 → Feedback/Action → 답변 | 없음; Well 예시 고정 | 3 |
| Extract + Verify | 전제 목록 → 전제별 true/false → 답변 | 없음; Well 예시 고정 | k+2 |
| TF-IDF text gate | 텍스트 분류 → No: Plain / Yes: 교정 경로 | 분류기·threshold | 생성 1 + 로컬 분류 |
| Direct gate | Yes/No → No: Plain / Yes: 교정 경로 | 없음 | 2 |
| CoT gate | 검토문+Yes/No를 한 번에 생성 → 분기 답변 | 없음 | 2 |
| Probe gate | 질문 hidden states 분류 → 분기 답변 | layer·분류기·threshold | hidden forward + 생성 1 |
| GEPA-gate | 최적화한 검토문+판정 → 고정 분기 답변 | detector prompt | 2 |
| GEPA | 최적화한 직접 답변 prompt → 답변 | answer prompt | 1 |
| GEPA (both stages) | 최적화한 detector → 최적화한 분기 답변 | detector + 공통 answer supplement | 2 |
| Ours | 특징 피드백으로 최적화한 직접 답변 prompt → 답변 | answer prompt | 1 |

호출 수에는 judge·학습·오류 재시도가 포함되지 않는다. k는 실제 추출된 전제 수다. PreWoMe/Extract+Verify의 추출을 공유하면 공동 실행 비용이 줄지만 각 방법의 독립 실행 비용은 위대로 보고한다.

### 4.1 CoT와 CoT gate를 구분

`CoT`는 **전제 오류를 찾으라는 추가 지시 없이** 일반적으로 검토하고 답하는 조건이다. 현재 `src/fpqa_prompting.py:COT_ANSWER`는 전제 검토를 명시하므로 이 새 CoT의 구현으로 재사용하지 않는다. 과거 행에 CoT가 쓰여 있어도 메시지가 다르면 재생성한다.

`CoT gate`는 원 질문의 전제를 검토하고 마지막에 Yes/No를 출력하는 **한 호출**, 이어지는 답변 **한 호출**이다. 기존 `review → 별도 yes/no 판정 → 답변`의 3단계와 구분한다. `premise_review_cot`라는 과거 내부 key는 새 protocol_id와 함께 기록하거나 새 key를 사용한다. 이름만 바꿔 과거 결과를 옮기지 않는다.

### 4.2 PreWoMe와 Extract + Verify의 구현 출처

Well revision `6c9770f65da6e9c50250cf94e35b07f258288e38`의 `prompting/run_prewome.py`, `prompting/run_presupposition_pipeline.py` 템플릿·few-shot·파서를 사용한다. registry에 해당 클래스 원문과 SHA-256을 보관한다. 이는 **Well 구현을 사용한 다른 모델 비교**이지 원 PreWoMe의 모델·데이터·예산까지 동일한 재현은 아니다.

- 같은 모델·질문·반복에서 추출 메시지가 byte-identical이면 추출 출력 하나를 공유한다. 모델 간 추출 출력은 공유하지 않는다.
- PreWoMe: 질문과 전체 추출 목록으로 Feedback/Action을 생성한다. 원 코드의 단일 답 전제 추가도 보존하고 명시한다.
- Extract+Verify: 전제를 하나씩 **원 질문 없이** true/false로 판정한다. False 목록으로 원 코드가 만든 지침과 질문을 답변기에 준다. RAG, scope 규칙, 확인 불가 선택지는 추가하지 않는다.
- 원 파서의 invalid→False, 소문자화 등을 로그에 노출한다. 이러한 parser fallback은 결과 해석의 일부이며 조용히 수정하지 않는다. parser 수정판은 별도 ablation이다.
- 데이터셋별 Well 예시는 원 구현대로 고정하고 모든 모델에서 같다. 이 파이프라인은 few-shot, Plain/Balanced/CoT는 zero-shot이므로 전체 시스템 비교이며 단계 구조만의 인과 효과라고 주장하지 않는다. 필요하면 같은 예시를 준 직접 답변 대조를 별도로 수행한다.

## 5. Gate 인터페이스와 답변 프롬프트

Direct detector는 registry의 지시와 질문을 받고 `Yes` 또는 `No`만 출력한다. CoT/GEPA detector는 `Review: ...` 다음 줄에 `Verdict: Yes` 또는 `Verdict: No`를 출력한다. 마지막 verdict 한 개만 파싱하고 임의 본문 키워드로 결정하지 않는다.

**주 Table 2의 Direct는 모든 모델에서 생성한 Yes/No를 사용한다.** 공개 모델에서는 같은 입력의 label likelihood를 보조 분석으로 저장할 수 있다. Luna가 제공하지 않는 logits를 LLM이 적은 confidence로 대체하지 않는다. 토큰 확률을 thresholding하는 Direct는 별도 버전이며 주 행에 섞지 않는다. 이 설계를 Two Axes의 정확한 재현이라고 부르지 않는다.

Gate 공통 경로:

- No → 동일 모델의 Plain 지시 + 원 질문. 검토문은 보내지 않는다. 캐시 조건이 같으면 Plain 답변을 그대로 사용한다.
- Yes → registry의 `correct_route` 지시 + 원 질문. CoT/GEPA-gate만 `Detector review`에 실제 생성된 검토문을 추가한다. 검토는 모델 생성 정보라는 표기를 고정한다. Direct/TF-IDF/Probe는 근거를 새로 만들어 넣지 않는다.
- 따라서 CoT gate와 Direct gate 차이는 탐지와 검토문 전달을 모두 포함한다. 탐지 자체 효과는 **동일한 두 저장 답변으로 label-only routing하는 보조 분석**에서 비교한다.
- Always correct는 모든 질문을 Yes 경로로 보낸 분석용 대조다. 주 행은 추가하지 않되 해당 답변 풀은 gate 분석용으로 확보한다. FPQ 성능의 상한이라고 부르지 않는다.
- 오류로 verdict가 파싱되지 않으면 No로 숨기지 않는다. end-to-end 시스템은 Plain fallback을 사용하고 fallback 횟수를 보고한다. Table 1에는 invalid를 별도 집계한다. 인프라 실패는 generation 실패와 구분하여 재개한다.

GEPA-gate는 CoT gate의 detector 지시에서 시작해 **최종 답변의 공통 점수**로 detector를 최적화한다. 따라서 Table 1 값은 답변 성능을 목표로 최적화된 detector의 부가 측정이다. detection balanced accuracy를 직접 최적화한 GEPA와 혼용하지 않는다.

GEPA (both stages)는 동일 detector와, No/Yes 양쪽 답변에 붙이는 **공통 answer supplement** 두 문자열을 최적화한다. 최초 supplement는 빈 문자열이다. route 규칙·verdict 형식·원 질문 입력은 고정한다. supplement가 바뀌면 No 경로도 기존 Plain과 같지 않으므로 캐시를 공유하지 않는다. GEPA-gate와 같은 시작 상태와 총 평가 예산으로 비교한다.

## 6. TF-IDF와 Probe 학습

두 방법의 y=1은 FPQ, y=0은 정상 질문이다. 분류 학습 목표를 GEPA의 답변 점수 최적화와 구분한다.

- TF-IDF 초기 고정 설정: 질문 원문, word unigram/bigram, lowercase, min_df=2, sublinear_tf=True, max_features=50,000, L2 정규화. LogisticRegression, solver=lbfgs, C∈{0.01,0.1,1,10}, intercept 포함, max_iter=2000. tokenizer·solver·버전을 저장한다.
- pooled 학습에서는 네 집단(dataset × FPQ/normal)의 총 sample weight를 같게 한다. 단순 전체 평균으로 CREPE 정상 질문이 지배하지 않게 한다.
- Probe: 답변 생성 전 질문의 마지막 non-padding 토큰 hidden state. 동일한 질문 wrapper를 사용하고 reasoning/generated token을 포함하지 않는다. 후보 layer는 깊이의 25/50/75/100% 위치를 사전 지정한다. scaler는 opt_train에서만 fit. 동일 C grid와 가중치의 선형 로지스틱 회귀를 사용한다.
- C/layer와 threshold는 opt_dev에서 선택한다. 주 threshold grid는 0.05부터 0.95까지 0.05 간격. dataset별 balanced accuracy의 macro 평균 최대화 → 동률 시 normal FPR 낮음 → 작은 C → 얕은 layer → 큰 threshold 순으로 선택한다. 정해진 기준을 test에서 바꾸지 않는다.
- 최종 fit은 opt_train만 사용한다. 선택 후 dev를 합쳐 재학습해 calibration을 바꾸지 않는다.
- TF-IDF 결과는 모델 공통으로 한 번 학습한다. Probe는 모델별이며 Luna의 native hidden states는 제공되지 않으므로 n/a다. 외부 임베딩 분류기를 Luna hidden probe라고 부르지 않는다.
- dev-derived target-FPR threshold나 answer-utility threshold는 추가 분석으로 이름을 나눠 보고한다. 전체 OOF 정답으로 threshold를 다시 정하지 않는다.

## 7. GEPA/Ours 최적화 규약

### 공통 목적함수

점수는 고정 Well template의 0–5 점수를 5로 나눈다. pooled 목적함수는 다음 네 집단 평균의 평균이다.

`J = 1/4 × (Cancer FPQ 평균 + Cancer NFP 평균 + CREPE FPQ 평균 + CREPE TPQ 평균)`

원 데이터의 질문 수를 같게 버릴 필요는 없다. 반성용 미니배치는 네 집단에서 1개씩, 총 4개를 기본으로 순환 샘플링한다. 각 집단 내 질문을 중복 없이 순회한 뒤 다시 섞는다. 그룹이 큰 통념이 과대표집되는 정도를 별도 보고한다. test는 원래의 적격 문항 전체를 사용한다.

- GEPA와 Ours의 시작 프롬프트는 동일 Plain. 모델별, fold별로 최적화한다. 모델에 따라 seed prompt를 더 친절하게 고치지 않는다.
- 동일 train/dev ID, 반성 모델, judge, GEPA revision, 수치 목적함수, 초기 seed, 후보 제안 상한을 고정한다. `seed=42`를 최초 최적화 실행에 사용한다.
- 후보 선택용 dev는 opt_dev의 각 집단에서 최대 16개를 group-aware로 미리 선택한다. 한 집단에 16개 미만이면 네 집단 모두 그 최소 수를 쓴다. 선택 seed와 ID를 저장하고 조건·모델 간 공유한다. 나머지 dev를 보고 선택을 뒤집지 않는다. 작은 후보 선택 집단의 과적합 한계를 명시한다.
- pilot은 최대 100, 본 비교는 최대 500 **문항 단위 metric evaluation**/fold/condition을 초기 예산으로 제안한다. train/val 및 반복 평가가 소비하는 수를 모두 센다. 500은 생성 호출 500회가 아니다. 연결 확인에서 seed 평가·선택 예산만으로 소진되는지 확인하고, 본 실험 시작 전 동일 예산을 재등록한다. pilot 점수로 특정 방법에만 예산을 더 주지 않는다.
- 서로 다른 파이프라인은 호출 수가 다르므로 동등 metric budget과 실제 task/judge/reflection 호출·토큰·시간을 함께 보고한다. SAE/텍스트 특징 구축 비용도 포함한다.
- 최종 prompt 상한은 512 reflection-model tokenizer tokens/최적화 문자열로 맞춘다. fixed wrapper·예시는 별도이다. 길이 초과 시 사전 정의된 1회 재제안 뒤 후보를 거부하고 비용을 센다. 두 단계 조건은 최대 두 문자열이므로 총 길이·비용을 별도 보고한다.
- 최종 후보는 dev J 최대, 동률이면 normal 두 집단 평균, 이후 짧은 prompt, 이후 먼저 평가한 후보 순으로 정한다. test 성능으로 후보나 seed를 선택하지 않는다.
- 학습 주석과 평가 설명은 학습 feedback에 사용할 수 있으나 test 주석은 전달하지 않는다. reference 문장이 prompt에 복사되는지 검사하고 audit를 남긴다. 보고할 때 train 기반 지식 주입과 일반적인 지시 개선을 구분한다.

### Ours의 변경점

GEPA의 기존 텍스트 feedback에 검증된 특징 설명을 추가한다. SAE 활성값은 점수에 더하지 않는다. 최종 추론에는 선택된 answer prompt와 질문만 필요하다.

- feature_fit/check는 각 fold의 opt_train 안에서만 구성한다. Cancer outer-test뿐 아니라 CREPE test·opt_dev를 사용하지 않는다.
- 답변 풀은 같은 모델의 Plain/Balanced/CoT와 고정한 동일 프롬프트 반복, 학습 전용 공통 pilot GEPA 출력으로 구성한다. 모델·조건 출처, 길이·문체 혼입을 점검한다.
- 선형 행동 구분 검사는 상한 증명이 아니다. FPQ 교정과 NFP 오반박을 따로 검사하고 질문 그룹을 나눠 조건 내·조건 간 평가한다.
- SAE 초기 후보는 원 논문 설정 M=32, K=4. 기존 상세안의 M=16/K=2는 별도 개발 비교로 남긴다. 임베딩 모델·입력 형식은 feature pilot 시작 전에 revision과 함께 동결한다. 답변-only와 question+answer 입력을 섞지 않는다.
- 의미 있는 특징이 확보되지 않으면 SAE 본 실험을 확정하지 않는다. 표에는 계속 Ours로 표시하고 실제 구현을 본문에서 명시한다.
- SAE의 추가 가치는 같은 답변 풀·특징 검증·feedback 길이/비용을 사용하는 **LLM 텍스트 특징 대조**와 비교한다. 이는 ablation이며 Table 2에 기존 방법 행을 늘리지 않는다. 추가 예시만 주는 대조도 feature pilot에서 확인한다.
- 이는 WIMHF의 선호 분석을 GEPA feedback으로 확장하는 설계다. 원 논문이 사실 교정 특징의 유용성을 입증했다고 주장하지 않는다.

## 8. 최종 채점과 통계

- 공통 judge는 현재 설정 파일의 `claude-sonnet-5-5`, effort=medium을 사용하되 실제 served ID/이용 가능성을 preflight에서 확인한다. 모델 대체 시 새 evaluation revision으로 전체 호환성을 다시 검사한다.
- Well pinned Cancer FPQ/NFP 및 CREPE template와 reference를 사용한다. 데이터셋별 rubric은 다르지만 **같은 문항의 모든 방법·모델**에는 같은 rubric/messages renderer/judge 설정을 사용한다. judge에는 방법명·모델명을 주지 않는다.
- judge에 전달하는 것은 최종 답변이다. gate 검토문·PreWoMe Feedback·CoT의 별도 Analysis 필드는 제외한다. CoT `Final answer:`가 없거나 모호하면 형식 실패로 기록하고 원 출력 전체를 답변으로 채점한다. 더 좋은 답을 얻기 위한 형식 재생성은 하지 않는다.
- 주 값은 FPQ Well≥4, NFP/TPQ Well≥4 각각의 비율. 보조 값은 평균 Well, 5점 비율, 교정/불필요한 반박 행동, invalid/실패율, 비용이다. 낮은 Well 점수를 모두 오반박으로 해석하지 않는다.
- Cancer: 각 eligible ID의 해당 outer-fold 예측을 한 번만 모아 분자/분모를 합산한다. fold별 퍼센트 단순 평균은 주 값이 아니다.
- CREPE: 3개의 동결 결과물 각각 test 전체를 평가해 비율을 낸 후 평균. 고정 기준선의 동일 답변을 세 번 복사해 독립 표본 수를 늘리지 않는다. Table 3도 같은 규칙이다.
- 질문 그룹 단위 paired bootstrap 2,000회로 차이와 95% 구간을 구한다. 같은 질문의 모든 조건·반복·모델 결과는 함께 resample한다. 이는 동결된 학습 결과물에 조건부인 구간이며 최적화 불확실성을 모두 포함하지 않는다.
- 최초 표는 optimization seed 1개/fold와 generation replicate 1개다. 3-fold를 3개 최적화 seed로 부르지 않는다. 핵심 Plain/Balanced/GEPA/Ours는 최종 비교에서 generation replicate를 추가해 총 3회 보고한다. 반복은 test prompt 선택용으로 사용하지 않는다. optimization seed 반복은 별도 축이다.
- 독립 판독 표본은 결과를 보기 전에 group/label 층화로 정한다(예: 데이터셋당 FPQ 최대 30, normal 최대 30). 같은 선택 질문의 핵심 조건을 전부 가린 채 판독한다. 불일치·임상 판단 모호함을 보고한다. 실패 사례만 고른 감사와 구분한다.
- 동등/보존 주장은 비유의성만으로 하지 않는다. 비열등성 margin을 예컨대 3%p로 사전 등록하되 현재 정상 질문 수로 구간이 넓으면 판단 불가로 보고한다.

## 9. 답변·채점 재사용과 오류 처리

| 저장 자료 | 재사용 조건 |
|---|---|
| 고정 방법 답변 | 동일 question bytes, checkpoint/served ID, rendered messages, few-shot, wrapper, decoding/effort, parser, 중간 입력 |
| 최종 점수 | 위 조건 + 동일 추출된 최종 답변, judge ID/설정, rubric/reference revision |
| gate 선택 결과 | 같은 판정·threshold·route prompt·review 전달 규칙·답변 풀 |
| TF-IDF/Probe | 새 manifest와 train/dev, 학습/threshold까지 정확히 같을 때만 |
| GEPA/Ours prompt | 새 fold의 train/dev/feature manifest와 정확히 같을 때만; 과거 100/100 prompt는 불가 |

고정 방법은 적격 전체 질문에서 한 번 생성한 답변을 fold별로 나눠 집계할 수 있다. 방법명을 기준으로 재사용하지 않고 message hash를 기준으로 한다. 표의 기존 숫자를 그대로 새 표에 복사하지 않는다.

cache key는 `(protocol_id, model_revision, settings_hash, rendered_messages_hash, parser_revision, replicate_id)`를 포함한다. replicate_id가 다른 반복은 cache를 공유하지 않는다. 답변 문자열뿐 아니라 실제 답변기에 전달되는 설명/검토문도 같아야 재사용할 수 있다. 판정 라벨이 같다는 것만으로 충분하지 않다.

전송 오류는 같은 요청으로 제한된 재시도 후 중단·재개한다. 오류 문항을 제외해 분모를 유리하게 줄이지 않는다. 내용이 정상적으로 생성됐으나 파싱이 실패한 경우의 처리는 §4–5대로 고정한다. 해결되지 않은 인프라 결측이 있으면 최종 완료 표로 표시하지 않고 분모·결측 수와 성공률 범위를 보고한다.

## 10. Table 3와의 연결

의료 QA에는 Cancer prompt, 일반 QA에는 CREPE prompt를 임의로 고르는 대신, pooled 주 설정에서는 같은 3개 fold 결과물을 모든 QA에 적용한다. 모델별 다른 prompt를 가지는 것은 허용하며 학습된 절차를 비교한다.

MedQA/PubMedQA/Medbullets/MMLU/GSM8K의 평가 문항은 GEPA/feature/threshold 선택에서 제외한다. 데이터 revision, few-shot, 공통 답안 출력 형식과 scoring을 별도 QA manifest로 고정한다. QA wrapper가 필요한 경우 같은 문항에서 모든 방법에 동일하게 붙이고 원 교정 지시는 유지한다. 객관식 정답·숫자 EM은 원 task scoring을 쓰며 Well 점수로 대체하지 않는다. QA 결과로 Table 2 prompt를 다시 선택하지 않는다.

고정 파이프라인 PreWoMe/Extract+Verify에는 원 구현의 데이터셋별 few-shot 팩이라는 별도 선택이 있다. Table 3 주 설정은 **CREPE 예시 팩을 모든 의료·일반 QA에 똑같이 사용**하고 support source를 캡션에 적는다. 이는 CREPE 설정에서의 보존 평가다. Cancer 예시 팩의 결과를 보려면 같은 QA 전체에 별도 적용해 source별로 보고한다. 의료 QA 점수가 높은 팩을 사후 선택하지 않는다. Pooled GEPA가 모든 데이터셋에 같은 prompt를 쓰는 것과 이 고정 few-shot 기준선의 설정을 혼동하지 않는다.

## 11. 실행 순서와 완료 조건

1. **분할·메시지 동결:** 제외/그룹 감사 → 공통 outer/inner/feature manifest → template·예시·parser·model revision 기록. 현 상태는 source audit와 prompt registry 초안까지다.
2. **작은 adapter 검증:** 같은 학습 질문에서 모든 backend의 렌더링 메시지, 반환 모델, 도구 미사용, 파서, 캐시/반복 분리를 확인한다. 비용·길이를 확인하고 최종 budget을 동결한다.
3. **고정 기준선 채우기:** 호환성 audit 후 Plain/Balanced/CoT, 공통 추출을 쓰는 PreWoMe/Extract+Verify. 적격 Cancer + CREPE test를 목표로 누락분만 생성하고 공통 judge와 맞지 않는 점수만 재채점한다.
4. **gate 채우기:** 공유 TF-IDF, 공개 모델 Probe, Direct/CoT. 원 gate 판정과 최종 답변을 함께 저장한다. 저장 두 답변 routing은 별도 분석으로 계산한다.
5. **특징 예비 실험 병행:** 학습 구역의 답변으로 SAE/텍스트 특징을 비교한다. test 결과를 보고 특징을 바꾸지 않는다. stage 실행 상태와 다음 수정의 평가 노출 이력을 기록한다.
6. **GEPA 최적화:** GEPA와 Ours를 같은 fold/예산으로 먼저 비교한다. GEPA-gate/both-stages도 registry의 독립 조건으로 실행한다. 한 모델의 test를 보고 다른 모델의 규약을 바꾸지 않는다.
7. **최종 비교·QA:** 동결 후보로 held-out 답변/반복/독립 판독, Table 3. 테스트 수정이 생기면 protocol v2로 구분하고 v1 결과를 보존한다.

처음부터 5설정×모든 행×전체 test를 무조건 실행하지 않는다. **설계는 다섯 모델에 공통**으로 적용하고, 실행은 Luna 및 공개 모델 한 개의 학습-only preflight부터 시작한다. 표에 모델 열이 있다는 사실은 완료를 뜻하지 않는다.

비용 산식: 모델별 고정 생성 `N × (Plain+Balanced+CoT+PreWoMe+Extract 호출수+gate 단계)` + 최적화 `3 folds × conditions × optimization seeds × metric budget` + test 생성·judge + feature 구축. TF-IDF는 공유, 같은 추출/Plain은 합법적인 cache 범위에서만 공유한다. 예를 들어 GEPA/Ours/GEPA-gate/both-stages 네 조건, 3-fold, seed 1개, 각 500은 **모델당 6,000 metric evaluations**이며 추가 feature/최종 평가 비용은 별도다.

## 12. 데이터셋별 최적화를 선택할 경우

Cancer 3-fold와 CREPE 공식 split에서 각각 별도 학습하고, TF-IDF/Probe/GEPA/Ours 모두 그 자료 범위를 따른다. Table 2 각 열은 해당 데이터셋에서 학습된 시스템이다. Table 3은 학습 출처를 행/패널로 나눠 각 prompt의 모든 QA 점수를 보이거나, 사전 고정한 하나의 source만 공통 적용한다. 의료/일반이라는 이름으로 source를 자동 선택하면 domain-aware routing 방법이라고 명시해야 한다. pooled 결과와 같은 행에 섞지 않는다.

## 관련 구현과 참고

- `src/fpqa_prompting.py`: 현재 Plain/Balanced/CoT 메시지와 evaluator. 새 CoT/weighted pooled 목적함수는 그대로 구현돼 있지 않다.
- `src/fpqa_cli_backend.py`: COMMON wrapper, exact Well transport, cache와 served-model 검사.
- `src/fpqa_well_pipelines.py`: Well 클래스 추출·공유 추출·단계별 호출.
- `scripts/prepare_fpqa_prompt_data.py`: CREPE source pin·중복·예시 제외.
- `gepa_sae_detailed_protocol_2026-10-06.md`: 특징 비교의 상세 초안. 본 문서와 충돌하는 이전 test100/feature 크기/예산은 실행 전에 새 manifest 기준으로 갱신한다.

작성 환경의 기본 `python3`는 3.8이며 Well 원문에는 그 버전으로 파싱되지 않는 f-string 구문이 있다. 템플릿 스냅샷은 소스를 텍스트로 보존했다. 실행 preflight에서는 호환 Python 환경을 명시하고 원문을 임의 수정해 실행하지 않는다. 현재 `RecordedCaller`의 cache에는 독립 replicate 구분이 없고, 기존 evaluator는 pooled macro objective와 새 2-call CoT gate를 그대로 구현하지 않는다. 이 세 부분을 통합하기 전 표 채우기 실행을 시작하지 않는다.

이 문서는 실험 규약이며 기존 실행기를 이미 통합·검증했다는 보고가 아니다.
