# 21. Well-Actually RAG 재현 점검과 새 Table 3

2026-09-10. GitHub 코드를 내려받아 정적으로 확인하고, README가 연결한 공개 데이터 파일을 실제 다운로드해 필드·개수·ID를 검사했다. 모델 가중치 다운로드, embedding 검색, 답변 생성, judge API 호출은 수행하지 않았다. 따라서 **재실험에 필요한 공개 부품을 확인한 것이며, 논문 수치를 재현한 결과가 아니다.**

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

**방법마다 검색 질의와 문서를 주는 단계가 다르다.** Direct QA는 원 질문으로 검색해 답변 프롬프트에 문서를 넣는다. FP Identification은 원 질문으로 검색한 문서를 판별 단계에 주고, 최종 응답 단계에는 그 판별 결과를 전달한다. 전제 추출·사실 확인은 모델이 추출한 전제별로 검색하는 경로가 있다. 따라서 원 방법을 재현하면서 모든 방법에 동일한 질문 기반 Top-4만 강제하면 adaptation이다. 반대로 우리 prompt/C 통제 비교에서는 의도적으로 동일 질문 기반 문서와 선택 목록을 공유한다.

출처: [Direct QA pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_direct_qa_pipeline.py), [FP Identification pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_fp_identification_pipeline.py), [전제 추출·검증 pipeline](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/prompting/run_presupposition_pipeline.py).

## 4. 그대로 실행하면 원 숫자가 나오지 않는 이유

**자료 snapshot.** FPQ 전처리는 Wikipedia와 각 질문의 `source` URL을 다시 수집한다. 같은 암종에서는 문서를 캐시하므로 질문별 source를 독립적으로 모두 수집한다고 가정할 수 없다. TPQ 기본 전처리는 Wikipedia에서 수집하며, 공개 TPQ JSONL의 저장된 문서를 읽는 절차와 다르다. 원 코드의 수집 시점·캐시 순서와 지금 웹의 내용이 달라질 수 있다. [FPQ 전처리](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMyth/prepare_dataset.py) · [TPQ 전처리](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/data_gen/CancerMythNFP/prepare_dataset.py)

**분할.** 두 `generate` 경로는 `random.shuffle`을 쓰지만 그 실행 경로에 고정 seed가 없다. `amend --seed 42`는 기존 파일을 수정하는 경로이며 당시의 미공개 파일을 복원하는 기능이 아니다. TPQ `generate` 기본값은 dev 30·나머지 test로 저장하므로, 논문의 train/dev/test=30/18/100과도 바로 대응하지 않는다. 새 manifest를 만들 수는 있지만 원 split 재현이라고 부르지 않는다.

**생성·판정.** 공개 transformers helper의 기본 생성은 greedy, 최대 2,048개 새 토큰이다. 현재 우리 주 실행안의 512와 다르다. response-level 판정기 기본 모델은 `gemini-3-flash-preview`이며, 모델 이름·버전·thinking 설정·채점 템플릿을 맞춰야 한다. API 모델 이름만 같다고 당시 출력까지 동일함이 보장되지 않는다. [생성 helper](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/utils/transformers_utils.py) · [판정기 기본값](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/constant/response_level_score.py)

**실행 환경.** `pyproject.toml`은 Python≥3.12와 torch·transformers·sentence-transformers 등을 지정하고 uv lock도 있다. 배치 셸은 저자 서버의 환경·모델·데이터 경로를 사용한다. 경로와 GPU 환경을 맞추고 한 모델·한 조건으로 확인한 뒤 확대해야 한다. 이번 감사에서는 환경 설치나 GPU 실행을 하지 않았다.

## 5. 이번 계획에서 선택하는 재사용 방식

첫 목표는 **공개 코드·문서와 고정한 우리 split으로 baseline과 우리 방법을 함께 재실행하는 것**이다. 원 논문의 당시 결과를 똑같이 복원하는 exact replication은 별도 목표로 남긴다.

1. 공개 TPQ 148개를 기존 NFP ID에 연결하고 few-shot 2개를 명시한다. Table 3의 주석 기반 평가에서 쓰는 정확한 문항과 분모를 고정한다. 이를 Table 2의 150개와 같은 분모라고 쓰지 않는다.
2. FPQ는 공개 질문·출처로 문서 snapshot을 만들고, TPQ 저장 문서는 그대로 사용한 경로와 새로 수집한 경로를 구분한다. 양성·음성의 문서 준비 방식 차이가 shortcut이 될 수 있음을 진단한다. gold 전제·정답 답변을 전체 QA의 검색 질의에 넣지 않는다.
3. `RAG=0/4/all` 자료·질의·정규화·문단 분할·모델 revision을 저장한다. 자료가 비어 있으면 정상 질문으로 처리하지 말고 빈 근거 조건으로 기록한다. 관련 자료를 못 찾았다는 것과 전제가 거짓이라는 것은 다르다.
4. 우선 Qwen2.5-7B에서 Direct QA, FP Identification, 전제 추출·검증을 재실행한다. 근거가 판별/검증/응답 중 어디에 들어가는지 원 절차와 변경점을 명시한다.
5. 우리 두 Hidden 행은 같은 원 질문 gate·문턱·목록을 공유한다. 근거를 제공할 때는 선택 여부와 관계없이 답변 생성 입력에 문서를 넣고, 비선택 질문은 둘 다 동일한 RAG Direct QA 응답을 사용한다. 선택 질문에서는 FP prompt 또는 C를 적용한다. **RAG로 gate 자체가 개선되는지 보는 실험은 아니다.**
6. 우리 gate/C는 Table 2에서 잠근 값을 우선 유지해 근거 추가 효과를 본다. RAG용 재학습·강도 재조정은 별도 adaptation이다. 두 Hidden 행에만 유리한 test 조정은 하지 않는다.
7. 같은 응답들을 Well의 0–5 판정기로 평가한다. S5, 전체 분포와 무효 S0 비율을 기록한다. 재실행 수치의 CI와 문항별 차이는 원 집계값에서 만들어 내지 않는다.

**집계:** S5는 유효한 0–5 판정을 받은 응답 중 score=5인 비율이며 S0도 분모에 포함한다. 파싱 실패는 S0와 구분해 처리한다. Figure 1의 평균 점수는 원문처럼 S0 제외 평균이므로 S5와 같지 않다. [공개 점수 집계 코드](https://github.com/ShenranTomWang/Well/blob/a7ee871eadde1104f7560a2874a03cbf5221dbaa/run_response_level_score.py)

### 실행 명령의 출발점 — 아직 실행하지 않은 예시

아래는 Well checkout과 Python/GPU 환경, 검증한 입력 JSONL이 준비된 뒤 사용하는 단일 조건 예시다. `our_frozen_test.jsonl`은 우리가 별도로 만든 manifest의 자료이며 원 논문 test라고 가정하지 않는다.

```bash
python -m prompting.run_direct_qa_pipeline \
  --backend transformers \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset_path our_frozen_test.jsonl \
  --output_dir out/our_split/Qwen/Direct_QA \
  --RAG 4 --thinking false --batching true --batch_size 4
```

FPQ와 TPQ를 각각 해당 dataset 이름으로 채점한다. 예를 들어 TPQ 파일은 `--dataset CancerMythNFP`, FPQ는 `--dataset CancerMyth`다. 아래 호출은 judge API를 사용하므로 이번 문서 감사에서는 실행하지 않았다.

```bash
python run_response_level_score.py response_level_score_submit \
  --file out/our_split/Qwen/Direct_QA/RAG=4.jsonl \
  --dataset CancerMythNFP \
  --evaluator_model_name gemini-3-flash-preview \
  --thinking_level minimal --disable_batching
```

## 6. 새 Table 3 — 원 수치와 재실행 수치를 분리한다

Table 1은 판별 신호 10개, Table 2는 PCR·PCS·NFP와 의료 QA, **Table 3은 외부 근거 조건의 Well 평가**다. 이전 Table 3(b)의 같은 gate prompt/C 비교는 Table 2에 합친다. 같은 C·같은 K의 무작위/텍스트/CoT/내부 선택 비교는 부록 Table A1에 남긴다.

**Table 3(a): 원 논문 보고값 — Qwen2.5-7B, 참고 전용.** 각 셀은 FPQ S5 / TPQ S5 (%). 아래는 Table 8의 주요 행 발췌이며 새로운 실험 결과가 아니다. `—`는 원문 해당 조건의 보고값이 없음을 뜻한다. [Well-Actually Table 8](https://arxiv.org/html/2608.06539v1)

| 방법 | 근거 없음 | Top-4 | 전체 근거 |
|---|---|---|---|
| Direct QA | 1 / 100 | 3 / 100 | 0 / 78 |
| FP Identification | 75 / 0 | 53 / 34 | 70 / 6 |
| 전제 추출 + 사실 확인 — LLM | 28 / 56 | 30 / 48 | 37 / 47 |
| 전제 추출 + 사실 확인 — MiniCheck | — | 5 / 99 | 1 / 98 |
| PreWoMe | 14 / 94 | 13 / 95 | 5 / 89 |
| Self-Dual-Critique | 6 / 85 | 7 / 85 | 1 / 61 |

**Table 3(b): 동일한 우리 분할·근거 조건으로 재실행 — 미측정.** 이 패널 안에서만 직접 비교한다. 원 표의 숫자와 우리 cross-fitting 숫자를 섞어 순위·차이·CI를 만들지 않는다. C의 주 실험과 Well식 생성 길이가 다르면 조건별로 다시 실행하며 이름을 명시한다.

| 방법 | 근거 없음 | Top-4 | 전체 근거 |
|---|---|---|---|
| Direct QA | 미측정 | 미측정 | 미측정 |
| FP Identification | 미측정 | 미측정 | 미측정 |
| 전제 추출 + 사실 확인 — LLM | 미측정 | 미측정 | 미측정 |
| 균형 전제 검토 CoT | 미측정 | 미측정 | 미측정 |
| Hidden gate + FP prompt | 미측정 | 미측정 | 미측정 |
| Hidden gate + C steering | 미측정 | 미측정 | 미측정 |

**캡션 초안.** *Premise correction with external evidence under the Well-Actually evaluation.* (a) Published reference percentages from Well-Actually, Table 8. (b) New evaluations on a shared, frozen medical split and documented evidence snapshots. Entries report FPQ/TPQ S5 percentages; new results include confidence intervals and sample sizes. S5 is not Cancer-Myth PCR or NFP. Published and new evaluations are not pooled. For the two hidden-gated methods, gate decisions are shared and evidence is supplied to answer generation; evidence-conditioned detection is outside this comparison.

Table 3는 RAG가 있는 기존 방법으로 충분한지, 같은 근거·같은 대상에서 steering의 효과가 남는지, 근거와 개입이 보완적인지를 본다. RAG 후에도 실패했다는 사실만으로 지식 부족을 배제하지 않는다. 검색 근거의 실제 관련성과 충분성을 따로 확인한다. 이 실험을 넣었다는 사실만으로 새 알고리즘 기여가 생기는 것도 아니다.

## 7. 아직 필요한 원 저자 산출물

정확한 원 표 재현을 위해 필요한 것은 FPQ/TPQ의 당시 train/dev/test와 few-shot ID 목록, 당시 FPQ 문서 snapshot, 단계별 Top-4 결과, 모델별 최종 응답·judge 출력, 실행 config와 모델 revision이다. 공개 자료만으로 우리 재실험은 진행할 수 있으므로 이 자료가 없다고 모든 작업을 멈출 필요는 없다. 이번 작업에서 저자에게 메시지를 보내지는 않았다.
