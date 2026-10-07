# 논문 비교표 — Detection / Response

2026-10-07 갱신. **본표·보조표를 나누지 않고 Table 1과 Table 2에 기존 방법을 모두 모았다.** 저장 결과 100칸은 그대로 유지했다(Table 1: 20칸, Table 2: 80칸). 새 생성·채점·학습은 하지 않았다.

- [Table 1 PDF](table1.pdf) · [PNG](table1.png) · [LaTeX](table1.tex) · [CSV](table1.csv): 거짓 전제 탐지.
- [Table 2 PDF](table2.pdf) · [PNG](table2.png) · [LaTeX](table2.tex) · [CSV](table2.csv): 답변의 FPQ 교정·NFP 보존과 의료/일반 QA.
- [분모·출처·미반영 사유](result_sources.md) · [수치 원장](measured_results.json).

**제안 방법의 표시명은 `Ours`다. SAE 사용을 확정하지 않는다.** `GEPA + text features` 행은 삭제했다. 과거 S1/S2/S3의 측정값을 옮겼으며, 표를 나눈 이전 버전은 Git 이력에 남아 있다. 모든 모델·방법을 볼 수 있는 통합 현황표라 Table 2는 세로로 길다. 최종 논문 지면 배치는 확정된 실험 구성으로 조정하며 수치를 임의로 줄이지 않는다.

![Table 1](table1.png)

![Table 2](table2.png)

## GEPA와 삭제한 text features의 차이

기존 GEPA도 실행한 질문·답변·점수와 자연어 채점 이유를 이용해 프롬프트를 수정한다. 이전에 제안한 `GEPA + text features`는 여러 학습 답변을 별도로 비교해 공통 특징을 요약하고, 그 설명을 기존 GEPA 피드백에 추가하는 **우리가 설계한 대조 조건**이었다. 독립적인 기존 논문 방법이나 공식 GEPA 변형 이름이 아니다. 최적화 알고리즘과 점수는 같고 추가 입력만 달라지는 조건이므로 주 결과표의 기존 방법에서 제외했다.

Ours의 구체 구현은 미정이다. [GEPA·SAE 상세 계획](../gepa_sae_detailed_protocol_2026-10-06.md)은 가능한 후보와 대조 실험의 기록이며 최종 방법을 확정한 문서가 아니다. 기존 GEPA 수치는 과거 실행 결과이고, Ours 비교에서 조건·예산을 변경하면 GEPA도 같은 조건으로 다시 평가한다.

## 기존 비교 방법과 배치

| 방법군 | Table 1: Detection | Table 2: Response | 역할 |
|---|---|---|---|
| 직접 판정 | Direct gate | Direct-gated response | 명시적인 LLM 판단으로 답변 경로 선택 |
| 전제 검토 | Review CoT gate (2-step) | Review CoT (2-step), CoT-gated response | 검토 후 판단하거나 답변 |
| 내부 분류기 | **Probe gate (hidden states)** | **Probe-gated response** | 내부 표현을 학습한 분류기로 선택적 교정 |
| 텍스트 분류기 | TF-IDF text (모델 독립 공통 행) | Text-gated response | 내부 접근 없이 질문 텍스트로 판정 |
| 구조화 파이프라인 | 별도 성능 대입 없음 | PreWoMe-style, Extract + Verify | 전제 추출·검토·답변 |
| 프롬프트 최적화 | GEPA | GEPA | 최적화한 지시의 성능 |
| 학습 | — | 기존 SFT/DPO (FPQ + twins) | 합성 정상 질문을 함께 학습한 과거 기준선 |
| 단순 지시 대조 | Direct + scope protection | Plain, CoT, Balanced instruction, Always correct | 구조화 방법·gate의 효과를 읽기 위한 대조 |
| 제안 방법 | Ours | Ours | 구체 방법과 측정 결과는 아직 미정 |

[Two Axes of LLM Abstention: Answer Correctness and Question Answerability](https://arxiv.org/html/2607.08456v1)의 §6은 내부 probe로 일반 답변과 전제 검토 지시를 선택하는 비교를 제공한다. **Probe gate를 넣을 직접적인 선행 연구 근거**다. 다만 현재 저장된 우리 response gate 값은 Plain/무조건 교정의 저장 답변을 선택한 값이다. 그 논문의 지시·학습 분할·threshold·평가까지 그대로 재현한 수치라고 부르지 않는다. 앞으로 선행 방법을 재현할 때에는 gate별로 같은 답변 경로를 쓰고 분할·threshold 선택을 통제한다.

Probe는 해당 모델의 내부 상태가 필요하다. 공개 모델 Qwen/Gemma에 배치했고 GPT-Luna에는 native probe 행을 두지 않았다. 다른 모델의 probe로 Luna를 라우팅한다면 별도의 transfer 조건이다. TF-IDF 판정은 모델 독립적이지만 그 판정으로 고르는 답변의 품질은 모델별로 다르므로 Response에서는 모델별 행이 필요하다.

## Review CoT gate가 하는 일

1. 원 질문에서 사실적 전제를 검토하는 글을 생성한다. 잘못된 전제와 교정 정보를 찾고 타당한 전제에서는 오류를 만들지 않도록 지시한다.
2. **질문 + 검토문**으로 오류 있음/없음을 판정한다. Qwen2.5는 Yes/No 토큰 점수 비교, Gemma/Qwen3.8은 생성 JSON이라 판정 출력 절차는 다르다.

Table 1은 이 판정 결과다. Table 2의 `Review CoT (2-step)`는 2단계에서 판정 대신 실제 답변을 생성한다. `CoT-gated response`는 gate 판정으로 저장된 Plain/무조건 교정 답변을 선택한다. 2-step gate가 최종 답변 생성까지 두 번 호출했다는 뜻은 아니다.

## 데이터·모델·측정 범위

| 위첨자 | 평가 자료 | 주의점 |
|---|---|---|
| H | Cancer-Myth 전체 583 FPQ / 149 NFP 대상 | 일부 Qwen 행은 유효 점수 분모가 작음. 모델별 지시·실행 조건 차이 있음 |
| L | GPT-6 Luna Cancer-Myth 99 FPQ / 100 NFP | few-shot 중복 fpq_291을 네 방법 모두에서 제외. 이미 분석에 노출된 문항 |
| R | GPT-6 Luna CREPE 751 FPQ / 2,253 TPQ | 저장된 전체 test 평가. dev 최적화 점수가 아님 |
| T | Qwen 학습 비교의 234 FPQ / 자연 NFP 149 | FPQ+합성 쌍둥이로 학습한 기존 SFT/DPO. 같은 ID의 Plain을 별도 행에 표시 |

- 수치 단위는 %다. 탐지는 TPR↑/FPR↓, 답변은 FPQ/NFP/TPQ 각각 Well≥4↑다. 답변 보존과 gate 오탐은 같은 지표가 아니다.
- 모든 값은 과거 한 번 실행한 결과이며 반복 평균이 아니다. 모델 간 평가 문항·judge·지시 차이가 있으므로 통일 조건의 backbone 순위로 읽지 않는다. 유효 점수 분모와 누락 수를 원장에 보존했다.
- Luna Cancer-Myth Plain 54.5/91.0, GEPA 78.8/72.0은 공통 199문항 기준이다. 예전 100/100 결과 55.0/91.0, 79.0/72.0과 차이는 한 문항 제외 때문이다.
- 이전 첨부 표의 공개 모델 4조건 및 TF-IDF **76칸은 모두 유지**했다. GPT-5.6-Luna 값을 GPT-6 칸에 넣지 않았고 Claude는 제외 요청을 유지한다. 이전 값은 [45번 표](../../45_complete_performance_tables_2026-10-01.md)에 있다.
- `Extract+Verify (scope)`는 예전 Qwen의 범위 보존 지시·문맥 제공 변형이다. Well 공식 Extract+Verify와 합치지 않는다.
- 기존 SFT/DPO는 자연 NFP가 아닌 합성 쌍둥이로 학습했으며, 쌍둥이 질문 품질 문제에 대한 기존 감사 단서는 유지한다. 새 feature-curated DPO나 확정된 Ours의 성능이 아니다.
- `—`는 결과가 없거나 현재 모델·방법에 옮길 수 없는 조합이다. 새 독립 평가나 비교의 완료를 뜻하지 않는다.

모델: Qwen2.5-7B, GPT-6 Luna, Gemma 4 12B, Qwen3.8-27B-FP8 OFF/ON. OFF/ON은 같은 가중치의 추론 조건이다. NLA/AO를 위해 검토한 Gemma-3/2 및 미확정 조합은 현재 주 결과표의 고정 방법으로 싣지 않고 상세 계획에 남긴다.

의료 QA는 MedQA(MQ)·PubMedQA(PQ)·Medbullets(MB), 일반 QA는 [MMLU](https://github.com/hendrycks/test)와 [GSM8K](https://github.com/openai/grade-school-math)다. MMLU는 의료 과목도 포함하는 다분야 평가이고 GSM8K는 수학 문장제다. 정확도와 최종 숫자 exact match를 각각 쓰며 서로 평균하지 않는다. 현재 이 조건들의 완료 결과는 없어 공란이다. 실행 전 데이터 revision·문항 ID·출력 형식·채점 규칙을 동결하고, 교정 방법을 적용한 채 일반 능력 보존을 평가한다.

## 수정·재생성

```bash
python3 docs/reviews/paper_tables_2026-10-06/render_tables.py
```

`table_specs.json`은 표의 구조·방법명이고 `measured_results.json`은 값과 출처 원장이다. `sources/`의 고정 집계를 SHA-256과 분자/분모로 검증한 뒤 PDF·PNG·CSV·LaTeX를 생성한다. **CSV는 출력물이므로 직접 수정하지 않는다.** 원 데이터가 있는 환경에서 출처를 다시 모으려면 `collect_results.py`를 사용한다. 새 모델 호출은 하지 않는다.

ReportLab, DejaVu Serif, pdftoppm이 필요하다. PDF/PNG는 전체 비교를 보여주는 미리보기다. LaTeX는 booktabs·multirow·graphicx를 쓰는 table* 조각이며, 긴 통합표의 최종 페이지 배치는 논문 편집 시 결정한다. 여기서는 TeX 엔진이 없어 컴파일하지 않았고 구조와 수치 일치를 검사했다.
