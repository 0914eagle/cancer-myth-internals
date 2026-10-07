# 논문용 Table 1 · Table 2 — 2026-10-06

**2026-10-07: 저장된 실험 결과 100칸을 반영했다.** Table 1은 16칸, Table 2는 28칸, S1/S2/S3는 각각 6/46/4칸이다. 새 생성·채점·학습 호출은 하지 않았다. `—`는 결과가 없거나 현재 모델·방법에 그대로 넣을 수 없는 조합이다.

각 수치의 위첨자 **H/L/R/T는 평가 집단**이다. 기존 공개 모델의 전체 문항 결과와 Luna의 시험 부분집합·다른 채점자를 구분한다. 이 표는 **현재 확보한 결과와 남은 실험을 보여주는 논문 형식의 현황표**이며, 모든 모델을 통일된 조건으로 다시 평가한 최종 비교표는 아니다. [분모·출처·미반영 사유](result_sources.md), [기계 판독 원장](measured_results.json)에 기록했다.

| 위첨자 | 사용한 평가 자료 | 해석 |
|---|---|---|
| H | Cancer-Myth 전체 583 FPQ / 149 NFP 대상, 일부 Qwen 행은 유효 점수 분모가 작음 | 기존 자동 Well 채점과 기본 gate 판정. 모델별 지시·실행 조건 차이 있음 |
| L | GPT-6 Luna Cancer-Myth 공통 99 FPQ / 100 NFP | few-shot 중복 `fpq_291`을 Plain·GEPA·PreWoMe·Extract+Verify 모두에서 제외. 이미 분석에 노출됨 |
| R | GPT-6 Luna CREPE 751 FPQ / 2,253 TPQ | 저장된 전체 test 평가, dev 최적화 점수가 아님 |
| T | Qwen LoRA 비교의 234 FPQ / 자연 NFP 149 | 합성 쌍둥이는 학습 자료이며 NFP 평가에 섞지 않음. Plain도 같은 평가 ID로 재집계 |

Luna Cancer-Myth의 Plain **54.5/91.0**, GEPA **78.8/72.0**은 공통 199문항 기준이다. 기존 100/100 표의 **55.0/91.0, 79.0/72.0**과 차이는 재실행 때문이 아니라 한 문항 제외 때문이다.

## 바로 볼 표

- [Table 1 PDF](table1.pdf) · [LaTeX](table1.tex) · [CSV](table1.csv): Detection.
- [Table 2 PDF](table2.pdf) · [LaTeX](table2.tex) · [CSV](table2.csv): Response + 의료/일반 QA. **a/b/c로 나누지 않은 단일 표**다.
- [Table S1 PDF](table_s1.pdf) · [LaTeX](table_s1.tex): SFT·DPO 및 NLA/AO 추가 효과.
- [Table S2 PDF](table_s2.pdf): CoT·균형 지시·무조건 교정·Gate 답변 기준선.
- [Table S3 PDF](table_s3.pdf): 개인 상황 보호·Hidden probe·TF-IDF 탐지 기준선.

![Table 2](table2.png)

![Table 1](table1.png)


상세 방법 계획: [GEPA SAE 데이터·특징·피드백·대조 실험 프로토콜](../gepa_sae_detailed_protocol_2026-10-06.md).

## 모델과 표 배치

본표 모델은 `Qwen/Qwen2.5-7B-Instruct`, `gpt-6-luna`, `google/gemma-4-12B-it`, `Qwen/Qwen3.8-27B-FP8` thinking OFF/ON이다. Qwen의 OFF/ON은 동일 가중치의 두 추론 조건이지 독립적인 두 backbone이 아니다. 정확한 실행 버전·추론 설정은 실험 manifest에 남긴다. 예전 GPT-5.6 결과를 GPT-6 칸에 옮기지 않는다. Claude는 추가하지 않는다.

Gemma-3-12B는 Gemma 4의 대체 모델이 아니라 NLA용 추가 대상이다. Table S1의 Gemma-2-9B는 AO 호환 후보다. 추가 트랙은 체크포인트·가중치·메모리와 예산을 확인한 후 실행한다. 사전학습 설명기가 새 미세조정 모델에서도 그대로 유효하다고 가정하지 않는다.

모델 5개 × 지표 9개를 가로로 놓으면 45개 수치 열이 생긴다. 논문용 확장판은 **행을 모델·방법으로, 열을 데이터셋·지표로** 배치해 한 페이지 폭에 맞췄다. 모델명은 반복 인쇄하지 않고 세로로 병합했다. 이전 가로 모델 두 개 시안 대신 이 버전을 사용한다.

## 데이터셋과 역할

| 데이터셋 | 역할 | 지표와 계획 |
|---|---|---|
| Cancer-Myth | 의료 거짓 전제 교정 / 정상 질문 보존 | Detection TPR/FPR; Response Well≥4 각각 보고 |
| CREPE | 일반 도메인 거짓 전제 교정 / 정상 질문 보존 | 위와 동일; TPQ는 정상 질문 칸 |
| MedQA | 의료 지식 QA 보존 | USMLE 4-option 정답 정확도 |
| PubMedQA | 초록을 이용한 의료 QA 보존 | yes/no/maybe 정확도. 입력에서 결론·정답 누출 방지 |
| Medbullets | 추가 의료 QA 보존 | 고정한 버전·옵션 수·시험 ID의 정확도 |
| **MMLU (추가)** | 여러 분야의 지식 QA 보존 | 공식 test, 과목별 정확도의 macro 평균. 의료 과목도 포함되는 다분야 평가이며 순수 비의료 부분집합은 아님 |
| **GSM8K (추가)** | 일반 수학 문장제 해결 능력 보존 | 공식 main/test, 최종 숫자 정답의 정규화 exact match |

MMLU는 [Measuring Massive Multitask Language Understanding 공식 저장소](https://github.com/hendrycks/test), GSM8K는 [Training Verifiers to Solve Math Word Problems 공식 데이터](https://github.com/openai/grade-school-math)를 사용한다. 이 둘은 FPQ/NFP 데이터셋이 아니라 **교정 방법을 적용했을 때 일반 능력이 손상되는지 확인하는 대조 평가**다. 의료와 일반 QA 다섯 열을 서로 평균하지 않는다.

새 두 데이터의 초기 비교 계획은 zero-shot이며 모든 방법에 같은 데이터·출력 형식·예산을 적용한다. 데이터 revision과 시험 ID 목록, MMLU 과목 집계 규칙, GSM8K 정답 추출기는 실행 전에 동결한다. 전체 시험셋이 부담되어 일부를 쓰면 사전 선정 ID·표본 수를 함께 보고하고 공식 전체 점수로 부르지 않는다. 무응답·파싱 실패도 평가 분모에 포함한다. 일반 QA를 만났다고 교정 방법을 끄지 않는다.

## 방법 추가 및 의미

- 본문: Plain, 전제 검토 CoT, PreWoMe식, Extract+Verify, GEPA, **GEPA + text feedback**, **GEPA + SAE feedback**.
- `text feedback`: 동일 학습 답변 쌍에서 LLM이 텍스트만 보고 특징을 요약해 제공한다. 기존 GEPA도 자연어 피드백을 쓰므로 이 이름은 'GEPA에 처음 텍스트를 준다'는 뜻이 아니다.
- `SAE feedback`: WIMHF식 답변 쌍 특징을 설명·검증한 후 반성 입력에 제공하는 제안 조건. 탐지와 답변 각각 따로 최적화한다.
- Table S1: 기존 **SFT/DPO (FPQ + 합성 twins)** 결과 두 행을 별도로 추가했다. 이는 계획 중인 같은 chosen의 SFT, 자연 FPQ+NFP 기본 DPO, 특징 선별 DPO와 다르므로 그 칸을 대신 채우지 않는다. NLA/AO 설명 추가도 계획 상태다. 학습 데이터 수·구성·예산을 맞추며 해당 요소의 추가 효과를 본다.
- Table S2/S3: 기존 CoT·균형 지시·무조건 교정·Gate·학습 분류기 기준선을 삭제하지 않고 별도 표로 보존했다.
- 본표에 포함되지 않은 NLA/AO 조합은 미지원 확정이 아니라 현재 계획 범위 밖이다. 모델에 따라 방법을 바꾼 결과를 같은 조건으로 묶지 않는다.

## 공통 평가 규칙

저장 수치를 분자·분모에서 계산했고, 한 번 실행한 결과를 반복 평균으로 표시하지 않았다. 위첨자가 다른 결과를 동일 문항·채점 조건의 모델 순위로 해석하지 않는다. H의 결측 점수는 실패로 바꾸지 않고 유효 분모와 누락 수를 원장에 남겼다. 특히 기존에 반복 노출한 Cancer-Myth 199문항은 새 독립 시험셋이 아니다. 특징 발견·GEPA·DPO 학습에서 최종 평가 문항 및 연결된 질문/통념 그룹을 제외한다. 주석 오류와 채점 불일치를 점검하고, 최종 조건은 반복 생성으로 비교한다. 공식 데이터셋을 쓴다고 사전학습 오염이 없다는 뜻은 아니다.

## 수정·재생성

`table_specs.json`에 모델·방법·열을 편집한 뒤 실행한다.

```bash
python3 docs/reviews/paper_tables_2026-10-06/render_tables.py
```

Python 3, reportlab, DejaVu Serif 폰트, `pdftoppm`(Poppler)이 필요하다. 결과는 PDF·PNG·CSV·LaTeX이다. 생성기는 `measured_results.json`과 `sources/`의 고정 집계를 읽는다. 출처 SHA-256, 원 집계의 분자·분모, 중복 셀, 모델·방법·지표 일치를 검사한 뒤 PDF·PNG·CSV·LaTeX를 갱신한다. **결과를 CSV에 직접 입력하지 않는다.** CSV는 재생성되는 출력물이며 수치는 원장에 보존한다.

`collect_results.py`는 로컬 원 실험 파일이 있을 때만 출처를 다시 수집하는 도구다. 일반적인 표 재생성은 `render_tables.py`만으로 가능하며, 대용량 원 답변이나 비공개 서버가 필요하지 않다.

LaTeX 조각은 `booktabs`, `multirow`, `graphicx`가 필요하며 `table*`로 두 단 폭에 들어간다. PDF/PNG는 ReportLab으로 렌더링한 미리보기다. 이 환경에는 TeX 엔진이 없어 LaTeX 컴파일은 하지 않았으며, 괄호·열 수를 정적으로 점검했다.
