# 21. Well-Actually Table 1 사실 확인·RAG 재현 점검

2026-09-10. GitHub 코드를 내려받아 정적으로 확인하고, README가 연결한 공개 데이터 파일을 실제 다운로드해 필드·개수·ID를 검사했다. 모델 가중치 다운로드, embedding 검색, 답변 생성, judge API 호출은 수행하지 않았다. 따라서 **재실험에 필요한 공개 부품을 확인한 것이며, 논문 수치를 재현한 결과가 아니다.**

**정정:** 사용자가 가져오려던 표는 Well 본문 Table 1의 사실 확인 정확도다. 이전 버전이 Table 8의 최종 응답 S5 평가로 바꾼 것은 잘못된 범위 변경이다. 아래는 Table 1 기준이며, 공개 파일 감사의 버전·개수·누락 사실은 유지한다.

## 1. 결론: 무엇을 가져올 수 있는가

**RAG 코드와 정상 질문의 근거 문서는 가져올 수 있다. 당시의 전체 RAG 실행 결과를 그대로 가져올 수 있는 상태는 아니다.** 검색과 답변·평가를 우리 분할에서 다시 실행하는 adaptation은 구성할 수 있다. 원 논문과 같은 숫자를 재현하려면 당시 FPQ 문서 snapshot, split manifest, 모델·판정기 설정을 추가로 확보·대응해야 한다.

| 항목 | 확인 결과 | 사용할 때의 의미 |
|---|---|---|
| RAG 검색 코드 | 공개됨. Qwen3-Embedding-0.6B, top-k 검색 | 새로 검색을 수행하는 코드를 활용 가능 |
| 정상 질문과 수동 전제 주석 | HF JSONL에 148개, 고유 ID 148개 | NFP 150개 중 정상 few-shot 2개를 뺀 집합임을 ID로 확인 |
| 정상 질문의 수집 문서 | 148개 모두 `passages` 필드, 147개는 비어 있지 않음 | 이미 수집된 자료를 재스크래핑 없이 재사용 가능 |
| FPQ 질문·정답 전제·출처 | 원 Cancer-Myth HF 데이터에 공개 | 원 데이터에는 `passages`가 없어 근거 문서 준비가 필요 |
| 당시 Top-4 검색 결과 | 공개 데이터에서 `top_passages` 없음 | 검색을 다시 실행하고 결과를 별도로 저장해야 함 |
| 원 train/dev/test ID 목록 | 점검한 GitHub·HF 파일에 명시적 manifest 없음 | 148개 정상 질문 목록이 곧 원 test 100개 목록은 아님 |
| 기존 모델 최종 응답·채점 결과 | 점검한 Git 트리에 결과 JSONL 없음 | 논문 집계값 인용은 가능, 개별 응답 재채점은 현재 불가 |
| 방법별 실행·판정 코드 | Direct QA, FP Identification, 추출·검증 등과 Gemini 판정 공개 | 모델·환경·입력 파일을 준비해 재실행 필요 |

GitHub의 `response/`와 `job_cache/`는 실행 결과 모음이 아니라 클래스·캐시 처리 코드다. 이 디렉터리가 있다는 이유로 응답과 API 결과가 공개되었다고 판단하지 않았다.

## 2. 확인한 버전과 데이터

- [Well GitHub 고정 revision](https://github.com/ShenranTomWang/Well/tree/a7ee871eadde1104f7560a2874a03cbf5221dbaa): `a7ee871eadde1104f7560a2874a03cbf5221dbaa`.
- [TPQ 데이터 고정 revision](https://huggingface.co/datasets/shenranw/CancerMyth-TPQ/tree/f44ef11fc86805e2e09b5ac66536a37b8e1b410e): `f44ef11fc86805e2e09b5ac66536a37b8e1b410e`.
- JSONL 크기 4,949,027 bytes, SHA256 `d7160733b1468443ffdce9796e85b65440458f2414523890e1ef510f0f9c55d0`.
- 원 NFP 150개 ID에서 `1002`, `1003`을 제외한 집합과 공개 148개 ID 집합이 일치한다. 두 ID는 공개 few-shot 파일의 정상 예시다. 148개의 질문에 수동 전제 문장은 총 178개이며, 질문 수와 전제 문장 수를 구분한다.
- 공개 JSONL에는 `question`, `cancer`, `id`, `presuppositions`, `doctor_suggestion`, `passages`, `few_shot_data`, `eval_few_shot_data` 등이 있다. `split`, 당시 Top-4, 최종 모델 답변 필드는 없다. `doctor_suggestion`은 전제 진위 라벨을 새로 만드는 필드로 쓰지 않는다.
- 로컬 원자료는 Git에서 제외되는 `external/well-rag-audit-20260910/`에 저장했다. Git에는 [검사 manifest](audits/well_rag_inventory_2026-09-10.json)를 남긴다. 기존 정상 데이터의 명목 크기는 여전히 150개이며 148개를 새 독립 집합으로 더하지 않는다.

출처: [HF 데이터 설명과 파일](https://huggingface.co/datasets/shenranw/CancerMyth-TPQ/blob/f44ef11fc86805e2e09b5ac66536a37b8e1b410e/README.md), [NFP 원 파일](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMythNFP/nfp.json), [few-shot 파일](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMythNFP/few_shot_data.json).

## 3. RAG는 어떤 검색인가

이 코드의 Top-4는 매번 인터넷 전체를 검색하는 기능이 아니다. **각 질문 row에 이미 들어 있는 `passages` 목록 안에서 관련 문단을 고르는 검색**이다. RAG=all도 전 세계 문서가 아니라 그 row의 문서 목록 전체를 뜻한다. Gemini의 web 조건은 별도다.

[`utils/RAG_utils.py`](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/utils/RAG_utils.py)는 기본 모델 `Qwen/Qwen3-Embedding-0.6B`로 질의와 문단을 embedding하고 유사도로 상위 k개를 선택한다. 제어 문자 등으로 손상된 문단을 제거하고 긴 문단을 기본 1,024 **공백 구분 단어** 단위로 나눈다. 함수 설명의 ‘token’ 표현과 실제 `split()` 구현을 구분한다. `top_passages`에 선택 텍스트를 저장하지만 복수 전제에 대해 같은 row를 반복 검색하면 마지막 값으로 덮일 수 있으므로, 재현용으로는 단계·질의별 문서 목록을 별도로 저장해야 한다.

**방법마다 검색 질의와 문서를 주는 단계가 다르다.** Direct QA는 원 질문으로 검색해 답변 프롬프트에 문서를 넣는다. FP Identification은 원 질문으로 검색한 문서를 판별 단계에 주고, 최종 응답 단계에는 그 판별 결과를 전달한다. 전제 추출·사실 확인은 모델이 추출한 전제별로 검색하는 경로가 있다. 따라서 원 방법을 재현하면서 모든 방법에 동일한 질문 기반 Top-4만 강제하면 adaptation이다. 이 설명은 최종 응답 pipeline의 감사 기록이다. 현재 계승할 Table 1은 아래의 gold 전제별 사실 확인 경로를 사용한다.

출처: [Direct QA pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_direct_qa_pipeline.py), [FP Identification pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_fp_identification_pipeline.py), [전제 추출·검증 pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_presupposition_pipeline.py).

## 4. 그대로 실행하면 원 숫자가 나오지 않는 이유

**자료 snapshot.** FPQ 전처리는 Wikipedia와 각 질문의 `source` URL을 다시 수집한다. 같은 암종에서는 문서를 캐시하므로 질문별 source를 독립적으로 모두 수집한다고 가정할 수 없다. TPQ 기본 전처리는 Wikipedia에서 수집하며, 공개 TPQ JSONL의 저장된 문서를 읽는 절차와 다르다. 원 코드의 수집 시점·캐시 순서와 지금 웹의 내용이 달라질 수 있다. [FPQ 전처리](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMyth/prepare_dataset.py) · [TPQ 전처리](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMythNFP/prepare_dataset.py)

**분할.** 두 `generate` 경로는 `random.shuffle`을 쓰지만 그 실행 경로에 고정 seed가 없다. `amend --seed 42`는 기존 파일을 수정하는 경로이며 당시의 미공개 파일을 복원하는 기능이 아니다. TPQ `generate` 기본값은 dev 30·나머지 test로 저장하므로, 논문의 train/dev/test=30/18/100과도 바로 대응하지 않는다. 새 manifest를 만들 수는 있지만 원 split 재현이라고 부르지 않는다.

**사실 확인·집계.** Table 1은 참·거짓 출력을 정답 라벨과 비교한다. 공개 `TransformersCheckOperator`의 fact_check 경로는 전제별로 검색해 `max_new_tokens=36`으로 판정한다. `run_check_gold_eval.py`의 `expected_result=0`은 FP 전제, `=1`은 TP 전제를 뜻한다. 이 스크립트 출력은 오답률이므로 Table 1 정확도는 `100 × (1 − incorrect_rate)`다. 최종 응답 생성 helper의 2,048토큰과 response-level Gemini S5 judge는 Table 8 계열이며 현재 Table 1의 채점 절차로 쓰지 않는다.

**실행 환경.** `pyproject.toml`은 Python≥3.12와 torch·transformers·sentence-transformers 등을 지정하고 uv lock도 있다. 배치 셸은 저자 서버의 환경·모델·데이터 경로를 사용한다. 경로와 GPU 환경을 맞추고 한 모델·한 조건으로 확인한 뒤 확대해야 한다. 이번 감사에서는 환경 설치나 GPU 실행을 하지 않았다.

## 5. Table 1 재사용 경로

1. 원 평가의 거짓 전제 100개·참 전제 116개에 해당하는 질문 ID와 전제 index 목록이 필요하다. 공개 TPQ 148질문·178전제 전체를 원 test로 간주하지 않는다. 원 목록이 없으면 우리 목록을 고정하고 baseline까지 함께 재실행한다.
2. 원 질문 ID·전제 index·전제 텍스트·FP/TP 라벨·few-shot 여부·그룹·문서 snapshot을 manifest에 기록한다. 질문이 같은 전제들은 같은 fold에 둔다. Table 3 진단은 주석 전제를 입력으로 제공하지만 Table 1 질문 탐지와 Table 2 실제 답변에는 gold 전제를 주지 않는다.
3. `--check_gold --pipeline fact_check`를 사용한다. 전제 문장이 retrieval query이며 None/Top-4/All은 각각 `no_passages` / `use_RAG --k 4` / `use_passages`다. 전제별 검색 결과를 저장한다. 원 template class, few-shot, reasoning, 모델 revision을 기록한다.
4. `doctor_suggestion`은 진위 라벨이 아니다. 공개 CancerMythNFP 전용 템플릿은 이 주석이 참이면 의사가 환자에게 고려하라고 제안한 내용이라는 문맥을 추가한다. 원 템플릿 적용 여부를 확인하고, 출력/내부 비교에는 같은 문맥을 준다. 이 필드를 임의로 FP/TP 정답으로 바꾸지 않는다.
5. 출력은 `factcheck_results`의 전제별 0/1이다. 생성 파싱 실패를 정답으로 처리하지 않는다. 원 코드의 실패 처리와 변경점을 기록하고, 재실행 주 정확도의 분모는 전체 평가 전제로 고정해 무효 판정은 오답으로 포함하고 무효율도 보고한다. 유효 출력에만 한정한 정확도는 보조로 분리한다.
6. 내부 readout 추가는 같은 전제·근거 입력의 별도 진단으로 명명한다. fit에서 probe를 학습하고 dev에서 층·문턱을 고정한다. 현재 질문 A gate와 C steering을 그대로 넣는 실험은 아니다. 실제 교정 효과는 Table 2로 검증한다.

### 확인한 실행 명령 — 아직 실행하지 않은 예시

다음은 Well checkout·GPU·의존성과 입력 JSONL이 준비된 뒤의 명령이다. `our_frozen_fp_test.jsonl`은 우리 목록이며 원 test 재현을 의미하지 않는다. Top-4 예시이며 다른 근거 조건은 source subcommand를 바꾼다.

```bash
python -m prompting.run_fact_check transformers \
  --file our_frozen_fp_test.jsonl \
  --out_file out/our_split/Qwen/fp_top4_checked.jsonl \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --check_gold --pipeline fact_check \
  use_RAG --k 4 --RAG_model Qwen/Qwen3-Embedding-0.6B

python run_check_gold_eval.py evaluate \
  --file out/our_split/Qwen/fp_top4_checked.jsonl \
  --expected_result 0
```

TP 전제 파일은 `--expected_result 1`로 집계한다. 원 TPQ 전용 template class 적용 여부까지 실제 config에 명시해야 하므로 이 예시 하나만으로 원 표 재현 완료라고 부르지 않는다.

출처: [gold 전제 실행기](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_fact_check.py), [전제별 검색·36토큰 판정](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/pipeline_operator/check_operator/transformers_check_operator.py), [오답률 집계](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/run_check_gold_eval.py), [TPQ 전용 템플릿](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/template/CancerMythNFP_template.py).

## 6. 논문용 Table 3 — Well Table 1의 사실 확인 평가

**현재 위치와 확정 범위.** 가져올 선행표는 Well-Actually Table 1로 확정한다. 이 표는 교수님 발표의 기존 연구 근거로 사용할 수 있다. 아래의 우리 Table 3 및 추가 probe 행은 사실 확인 진단을 독립 연구 질문으로 채택할 경우의 후보이며 본 실험으로 확정한 것은 아니다. 기존 결과를 인용하는 것만으로 우리 실험 결과표가 되지 않는다. 현재 질문 gate의 검증이 목적이면 Table 1의 질문 단위 탐지 비교에 RAG 기반 전제 추출·검증을 추가하는 방안도 가능하며, gold 전제를 받는 원 Table 1의 숫자와 직접 섞지 않는다.

**계승하는 표는 Well-Actually의 본문 Table 1, “Fact-checking accuracy on CancerMyth”다.** 이전 문서의 Table 8 S5 표로 대체한 구성은 철회한다. 사람이 주석한 전제 문장을 입력으로 주고 참·거짓 판정 정확도를 잰다. 최종 환자 답변 생성이나 PCR·PCS·S5를 평가하는 표가 아니다.

**Table 3(a): Well-Actually Table 1의 원 보고값 — 참고 자료.** Model → RAG → FPQ Accuracy → TPQ Accuracy의 원 구조를 유지했다. 수치는 정확도 % (정답 수/전제 수)다. FPQ Accuracy는 거짓 전제를 거짓으로 판정한 비율, TPQ Accuracy는 참 전제를 참으로 판정한 비율이다. 원 표의 분모 100·116은 전제 문장 수이며 116개의 정상 질문을 뜻하지 않는다. [원문 §3.2·Table 1](https://arxiv.org/html/2608.06539v1#S3.SS2)

| Model | RAG | FPQ Accuracy | TPQ Accuracy |
|---|---|---|---|
| Llama3-Med42-8B | None | 95 (95/100) | 31 (36/116) |
| Llama3-Med42-8B | Top-4 | 96 (96/100) | 32.8 (38/116) |
| Llama3-Med42-8B | All | 98 (98/100) | 20.7 (24/116) |
| Llama-3-8B-Instruct | None | 97 (97/100) | 22.4 (26/116) |
| Llama-3-8B-Instruct | Top-4 | 94 (94/100) | 42.2 (49/116) |
| Llama-3-8B-Instruct | All | 100 (100/100) | 16.4 (19/116) |
| Olmo-3-7B-Instruct | None | 100 (100/100) | 17.2 (20/116) |
| Olmo-3-7B-Instruct | Top-4 | 93 (93/100) | 8.6 (10/116) |
| Olmo-3-7B-Instruct | All | 100 (100/100) | 11.2 (13/116) |
| Qwen2.5-7B-Instruct | None | 96 (96/100) | 15.5 (18/116) |
| Qwen2.5-7B-Instruct | Top-4 | 97 (97/100) | 12.9 (15/116) |
| Qwen2.5-7B-Instruct | All | 100 (100/100) | 6.9 (8/116) |
| Gemini-3-flash | None | 93 (93/100) | 32.8 (38/116) |
| Gemini-3-flash | None + reasoning | 98 (98/100) | 21.6 (25/116) |
| Gemini-3-flash | Top-4 | 95 (95/100) | 26.7 (31/116) |
| Gemini-3-flash | Top-4 + reasoning | 96 (96/100) | 17.2 (20/116) |
| Gemini-3-flash | All | 98 (98/100) | 25 (29/116) |
| Gemini-3-flash | All + reasoning | 96 (96/100) | 12.9 (15/116) |
| Gemini-3-flash | Web | 96 (96/100) | 31 (36/116) |
| Gemini-3-flash | Web + reasoning | 98 (98/100) | 20.7 (24/116) |
| Gemma-4-E4B-it | None | 98 (98/100) | 23.3 (27/116) |
| Gemma-4-E4B-it | Top-4 | 96 (96/100) | 16.4 (19/116) |
| Gemma-4-E4B-it | All | 61 (61/100) | 37.1 (43/116) |
| MiniCheck | Top-4 | 99 (99/100) | 10.3 (13/116)† |
| MiniCheck | All | 95 (95/100) | 11.2 (14/116)† |

† MiniCheck 두 TPQ 셀은 원문·사용자 첨부 표를 그대로 옮겼다. 다만 13/116≈11.2%, 14/116≈12.1%로 원문의 앞 백분율과 일치하지 않는다. 원 결과 파일 확인 전에는 어느 값을 정정값으로 채택할지 단정하지 않으며, 이 두 셀로 차이·순위를 계산하지 않는다.

**Table 3(b): 같은 전제·근거를 사용하는 재실행 및 내부 readout 진단 — 미측정 초안.** 독립 진단을 채택할 경우 Qwen에서 아래 조건을 비교할 수 있다. (a)의 기존 모델·RAG 조건은 보고값으로 소개하고, 직접 우위 비교는 같은 우리 분할에서 baseline까지 재실행한 (b) 안에서만 한다. 원 split·문서와 일치하지 않는 새 결과를 (a)에 그대로 붙이지 않는다.

| Model / 판별 방식 | RAG | FPQ Accuracy (%) ↑ | TPQ Accuracy (%) ↑ |
|---|---|---|---|
| Qwen2.5-7B / 원 사실 확인 프롬프트 | None | — | — |
| Qwen2.5-7B / 원 사실 확인 프롬프트 | Top-4 | — | — |
| Qwen2.5-7B / 원 사실 확인 프롬프트 | All | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | None | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | Top-4 | — | — |
| Qwen2.5-7B / 전제 검토 CoT 후 판정 | All | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | None | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | Top-4 | — | — |
| Qwen2.5-7B / 전제 입력 내부 probe (진단) | All | — | — |

**추가 행의 범위.** 내부 행은 주석 전제와 해당 근거를 입력받는 판별기의 진단이다. 원 질문만 읽도록 학습한 현재 A gate를 그대로 옮기거나 C steering의 최종 응답 점수를 붙이는 행이 아니다. 동일한 전제·근거·few-shot 문맥에서 출력 판정과 내부 readout을 비교한다. probe는 전제 단위 fit 자료에서 별도로 학습하고 층·문턱은 dev에서 고정한다. 같은 질문의 여러 전제와 같은 myth는 같은 fold에 묶는다. 추가 CoT 행은 전제를 검토한 뒤 참·거짓을 판정하는 비교이며 별도 모니터가 환자 답변을 채점하는 절차가 아니다. 원 모델이 지원하는 reasoning 설정과 추가 CoT 프롬프트도 구분한다.

**캡션 초안.** *Fact-checking accuracy on annotated medical presuppositions under external evidence conditions.* (a) Published results from Well-Actually Table 1. (b) New evaluations on shared held-out presuppositions and frozen evidence. Accuracy is computed separately for false and true presuppositions. New entries report estimates, 95% confidence intervals, and correct/total counts. The premise-input probe is a diagnostic readout, distinct from the question-level deployment gate. Published and new results are not pooled.

**세 표의 차이.** 우리 Table 1은 원 질문에서 잘못된 전제를 탐지하는 능력, Table 2는 최종 교정·정상 질문·의료 QA 성능, Table 3는 전제를 직접 제공해 추출 단계를 제거한 사실 확인 능력을 평가한다. Table 3에서만 주석 전제를 입력으로 제공하며 정답 진위 라벨은 입력하지 않는다. 이는 Task 1의 원인 진단 확장이다. 같은 개입량의 선택 비교는 부록 A1, 동일 gate 뒤의 prompt/C 효과는 Table 2에 남긴다.

**해석 범위.** RAG 후에도 참 전제를 부정한다면 그 근거 조건에서 사실 확인 문제가 남았다는 결과다. 문서의 관련성·충분성을 확인하지 않은 채 지식 부족을 배제하지 않는다. 전제 입력 probe가 더 좋아도 실제 환자 질문을 읽는 A gate의 우위나 steering의 교정 효과를 입증한 것은 아니다.

## 7. 아직 필요한 원 저자 산출물

정확한 원 Table 1 재현에는 평가 질문 ID와 전제 index 목록, 당시 FPQ 문서 snapshot, 전제별 Top-4 결과, factcheck_results 원 파일, 실행 config·template class·few-shot·모델 revision이 필요하다. MiniCheck의 분자/분모와 백분율 불일치도 원 결과로 확인해야 한다. 당시 최종 환자 답변과 S5 judge 출력은 Table 8 재현의 자료이며 Table 1의 필수 채점 산출물과 구분한다. 이번 작업에서는 저자 연락·embedding·모델 생성·API 실행을 하지 않았다.
