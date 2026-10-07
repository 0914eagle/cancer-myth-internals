# Qwen2.5 / Gemma 3 기준선 Table 1·2·3 실행

사용자 결정: Qwen3.8 OFF/ON 제외, Gemma 4 12B → **Gemma 3 12B-IT**, Ours 제외. 데이터셋별 독립 학습과 Cancer grouped 3-fold / CREPE 공식 split은 유지한다. 과거 Gemma 4 수치를 Gemma 3으로 옮기지 않는다.

## 서버 배정

| 서버/GPU | 작업 |
|---|---|
| 여기 4090 GPU 0 | Qwen2.5-7B-Instruct 기준선 |
| 여기 4090 GPU 1·2 | Gemma-3-12B-IT 기준선; BF16 모델 분산 |
| 여기 4090 GPU 3 | 예비 |
| 별도 A6000 2장 | 공개 NLA의 대상 활성 추출·AV 설명·AR 복원. Ours/GEPA 연결은 후속 |

4090 네 장을 실제 확인했다. Gemma 3 가중치가 약 24.4GB이므로 24GB 한 장에 억지로 올리거나 자동 양자화하지 않는다. 두 서버의 VRAM을 합산하지 않는다. A6000에서는 처음에 target→AV→AR를 순차 실행하고 메모리를 실측한 후 동시 로드 여부를 정한다. 본 변경에서 원격 NLA는 실행하지 않았다.

## 기준선과 표

- Table 1: TF-IDF, Direct, CoT, Probe, GEPA-gate. both-stages detector는 추가 분석 행으로 따로 출력한다.
- Table 2: Plain, Balanced, CoT, PreWoMe, Extract+Verify, TF-IDF gate, Direct gate, CoT gate, Probe gate, GEPA-gate, GEPA, GEPA both-stages.
- Table 3: 같은 12개 방법. Cancer에서 학습한 3개 시스템의 평균 / CREPE에서 학습한 1개 시스템을 각각 모든 QA에 적용한다. 고정 zero-shot은 공유한다.

한 worker가 탐지·답변을 함께 저장하고, 집계기가 세 CSV를 만든다. 검증되지 않은 과거 값으로 빈칸을 메우지 않는다. 생성 실패/미채점은 분모에서 제거하지 않고 pending/partial로 남긴다. Table 1의 binary decision에는 AUROC를 만들어 넣지 않고 TF-IDF/Probe 연속 점수에만 계산한다.

## 데이터 동결

- `prepare_paper_baseline_bundle.py`: 기존 Cancer 그룹+동일 질문 연결, CREPE 동일 전제 연결, 공식 test 우선 제외, few-shot 그룹 제외 후 분할한다. 추가 검토한 그룹은 `--group-overrides` JSON의 ID 목록으로 넣는다.
- 기존 그룹+정확한 문자열 연결로 만든 최초 Cancer 집단은 **577 FPQ / 147 NFP**다. 의미상 동의어 전부를 검증한 독립 평가라고 주장하지 않는다. 연구자 노출 이력도 유지한다.
- 원래 평가 주석은 bundle에 저장되지만 task backend에는 Pipeline이 렌더링한 question-only 메시지밖에 전달하지 않는다. 정답/주석은 judge/학습 feedback에서만 사용한다.
- QA는 `prepare_paper_qa.py`로 source revision, split, ID를 고정한다. 기본은 전체 split이다. `--limit-per-dataset`을 쓰면 답변과 무관한 해시 표본이며 표에 부분집합임을 밝혀야 한다.

| QA | 고정한 공개 출처 | 분할 / 이번 준비 개수 | 평가 |
|---|---|---|---|
| MedQA | GBaker/MedQA-USMLE-4-options | test / 1,273 | 4지선다 accuracy |
| PubMedQA | qiaojin/PubMedQA, pqa_labeled | HF train이라는 이름의 공개 PQA-L 1,000개 풀 | context 포함 yes/no/maybe accuracy; **공식 500개 test 성능이라고 부르지 않음** |
| Medbullets | mkieffer/Medbullets | op4_test / 308 | A–D accuracy; 빈 E 항목 제거 |
| MMLU | cais/mmlu, all | test / 14,042 | 생성 답의 letter accuracy, 질문 단위 micro 평균; loglikelihood/few-shot 원 점수와 구분 |
| GSM8K | openai/gsm8k, main | test / 1,319 | 최종 숫자 EM; comma/Decimal 표준화 |

원형 QA 입력/context는 보존하고 정답 해설은 입력에 넣지 않는다. 모든 방법에 같은 `Answer:` 출력 형식을 요청한다. FPQ/NFP Well 점수와 QA 지표를 혼합하지 않는다. PubMedQA의 평가 풀 범위가 원 논문 공식 test와 다름을 캡션에도 표시해야 한다.

## 실행 순서

저장소 루트에서 실행한다. 기존 GPU Python은 `/data1/heejae/uv/cancer_myth_internals/bin/python`이다. Well 원문에 Python 3.12 문법이 있어 템플릿 렌더링만 `.venv/fpqa_prompting/bin/python` subprocess를 사용한다. 원문을 3.11 문법으로 고쳐 쓰지 않는다.

```bash
# 1. QA 공개 revision/문항 준비. 이미 있으면 같은 내용만 허용.
/data1/heejae/uv/cancer_myth_internals/bin/python scripts/prepare_paper_qa.py \
  --out results/paper_baselines_v3/qa

# 2. QA 포함 공통 bundle. 기존 smoke용 bundle은 덮어쓰지 않음.
/data1/heejae/uv/cancer_myth_internals/bin/python scripts/prepare_paper_baseline_bundle.py \
  --qa-jsonl results/paper_baselines_v3/qa/questions.jsonl \
  --out results/paper_baselines_v3/protocol/full_bundle.json

# 3. 공유 GPU 환경을 바꾸지 않고 GEPA만 별도 설치.
uv pip install --python /data1/heejae/uv/cancer_myth_internals/bin/python \
  --target .venv/paper_baseline_extras --no-deps gepa==0.1.1

# 4. 프롬프트 길이를 재는 tokenizer만 pin/download. 임베딩 모델 가중치 불필요.
/data1/heejae/uv/cancer_myth_internals/bin/python scripts/prepare_paper_runtime.py \
  --out results/paper_baselines_v3/protocol/runtime.json

# 5. 호출 없는 점검 → 학습-only 생성 smoke → 고정 기준선 → 학습형 기준선.
bash scripts/run_paper_baselines_4090.sh preflight all
bash scripts/run_paper_baselines_4090.sh smoke fixed --methods plain,direct_gate --generate-only
bash scripts/run_paper_baselines_4090.sh run fixed
bash scripts/run_paper_baselines_4090.sh run trained
```

전체를 순서대로 돌리려면 `bash scripts/run_paper_baselines_4090.sh run all`을 사용할 수 있다. 첫 실행은 위 단계별 순서를 권한다. launcher가 두 모델을 병행하고 GPU lock을 잡으며, 한쪽 실패를 exit status로 보고한다. 다른 독립 worker는 계속 완료할 수 있다. 다른 연구 프로세스까지 lock을 지키는 것은 아니므로 GPU 사용량은 실행 전 확인한다.

`PAPER_CONFIG`, `PAPER_BUNDLE`, `PAPER_OUT`, `PAPER_PYTHON` 환경변수로 경로를 바꿀 수 있다. 설정/코드/분할/주석/judge가 바뀌면 새 out을 사용한다. `--generate-only`는 답변·판정만 저장하고 Well 채점은 건너뛴다. 같은 명령에서 이 옵션을 빼면 저장 답변만 채점하여 이어간다. 다만 GEPA 최적화 자체는 학습 judge를 필요로 하므로 generate-only로 judge-free GEPA가 되지는 않는다.

중단은 해당 모델의 출력 폴더에 `STOP` 파일을 만든다. 진행 중 호출은 마치고 다음 호출 전에 중단한다. 재개 전 STOP을 제거하고 동일 명령을 실행한다. 전송 오류를 정상 답변 실패 점수로 바꾸지 않는다. 현재는 자동 재시도 없이 실패를 기록하고 같은 요청으로 재개한다.

## 결과와 상태

- 기본 결과: `results/paper_baselines_v3/main/{qwen25,gemma3}/`
- stage cache: `calls/` (정확한 렌더링 메시지·model revision·설정·replica 해시)
- 고정된 분할: `protocol/full_bundle.json`
- 학습기: `shared/gates/`의 TF-IDF, 모델별 `gates/`의 Probe, `gepa/.../selected.json`
- 생성/채점: `table12/`, `table3/`; 학습 smoke는 별도 `smoke/`
- 표: `main/tables/table1.csv`, `table2.csv`, `table3.csv` 및 JSON. 역사적 PNG는 자동 덮어쓰지 않는다.
- Table 2는 전체 적격 분모가 채점됐을 때만 최종 비율을 출력한다. Table 3 역시 모든 문항/시스템이 끝난 뒤 평균을 출력한다.

학습 분할, parser, 캐시, 지표 집계에 대한 단위 테스트와 실제 GEPA 라이브러리의 mock optimizer 연결 테스트를 수행했다. Qwen/Gemma 실제 GPU smoke는 FPQ·NFP 각 1개에서 Plain·Direct gate 생성 경로를 확인했다. 이는 점수 비교나 전체 파이프라인 품질 검증이 아니다. 전체 GEPA 최적화, 공통 judge 실호출, TF-IDF/Probe 학습, 전체 QA 생성은 아직 완료하지 않았다.

## 원문 근거

- [NLA 공식 대상/레이어](https://github.com/kitft/natural_language_autoencoders#released-checkpoints)
- [MedQA 데이터](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [PubMedQA 데이터](https://huggingface.co/datasets/qiaojin/PubMedQA)
- [Medbullets op4 정의](https://huggingface.co/datasets/mkieffer/Medbullets)
- [MMLU](https://huggingface.co/datasets/cais/mmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k)
