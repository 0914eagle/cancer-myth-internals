# 10. 근거와 재사용할 baseline

2026-09-08. 논문 주장을 뒷받침하는 근거와 그 근거가 말하지 않는 것을 구분한다. 링크는 원문 또는 저자 저장소다. 아래 수치는 선행연구의 보고 결과이며 우리 재현 결과가 아니다.

## 1. Cancer-Myth — 문제와 본 평가표의 출발점

스토리: 실제 환자 질문의 응답 평가에서 전제 누락을 발견 → 전문가 검증 FPQ 585개와 어려운 NFP 150개 구성 → 낮은 교정률 → 완화책의 부작용을 함께 평가. Table 1에는 전제 교정뿐 아니라 일반 의료 QA가 들어 있다. 따라서 “QA가 떨어진다는 근거가 있었나?”에는 직접적인 근거가 있다.

| 보고된 설정 | Cancer-Myth | NFP | MedQA | PubMedQA | Medbullets |
|---|---:|---:|---:|---:|---:|
| GPT-4o Plain | 12 | 88 | 70 | 67 | 68 |
| GPT-4o GEPA | 68 | 59 | 63 | 59 | 62 |
| Gemini-2.5-Pro Plain | 41 | 96 | 92 | 82 | 80 |
| Gemini-2.5-Pro GEPA | 88 | 68 | 85 | 78 | 72 |

수치는 아래 HTML v3의 Table 1 기준이다. 초록의 요약 수치와 다른 경우 표 값을 우선하고 버전을 명시한다. Table 3은 공개 모델의 FPQ 성능으로, Qwen2.5-7B와 Llama-3.1-8B 등이 겹친다. 모델·test 범위가 다르면 원본 수치를 우리 held-out 행과 같은 평가라고 부르지 않는다.

근거가 지지하는 것은 **평가된 완화책의 성능 상충**이다. 모든 prompting이 원리적으로 실패하거나 모든 모델에서 QA를 떨어뜨린다는 명제는 아니다. [논문](https://arxiv.org/html/2504.11373) · [코드](https://github.com/Bill1235813/cancer-myth)

## 2. Well-Actually — FPQ 개선과 TPQ 오교정

스토리: 다양한 전제 처리 방법의 FPQ–TPQ 상충 → 전제 검증의 편향/한계 → 실제 질문 혼합에서의 효과 분석. “일단 전제를 의심하라”는 정책을 무비판적으로 쓰지 말아야 한다는 근거다. [논문](https://arxiv.org/html/2608.06539v1) · [코드](https://github.com/ShenranTomWang/Well)

재사용할 baseline 계열:

| 계열 | 예시 | 비교에서 묻는 것 |
|---|---|---|
| 명시적 전제 검사 | FP Identification | 직접적인 검사 지시만으로 충분한가 |
| 추출 후 검증 | Extract+FactCheck, PreWoMe | 전제를 분리해서 확인하면 개선되는가 |
| 질문 변환·비판 | Question-to-Statement, Self-Dual-Critique | 입력/검사 구조의 효과는 무엇인가 |
| 프롬프트 최적화 | GEPA | 개발 데이터로 최적화한 강한 prompting과 비교 |
| 학습·표상 개입 | FalseQA fine-tuning, FAITH | 추가 학습이나 기존 개입의 효과 |

Figure 1은 FPQ와 TPQ 평균 점수를 비교하는 그림이며 S5 비율 자체가 아니다. Appendix Table 8–10은 각각 Qwen2.5-7B, Gemma, Llama-3-8B 결과다. Llama-3.1-8B와 동일 모델로 취급하지 않는다. 점수 분포로 평균을 근사할 수 있지만 반올림·조건 집계 때문에 원래 좌표를 정확히 복구했다고 할 수는 없다. 원시 결과와 split ID가 있으면 재집계가 우선이다.

TPQ 점수가 높다는 것과 답변의 모든 의학적 내용이 정확하다는 것은 다르다. 일반 의료 QA 보존은 별도 측정한다.

### 저장소 코드와 논문 PDF로 확인한 세부 (2026-09-08)

| 항목 | 확인 내용 | 출처 파일 |
|---|---|---|
| 판정 | gemini-3-flash. FPQ는 선행연구의 1–5점(1 = 거짓 전제 무시·강화, 5 = 명확히 짚고 근거)에 0(횡설수설) 추가. TPQ는 대칭 기준(1 = 참 전제를 잘못 고치려 함, 5 = 교정 시도 없음). 표는 S1–S5 분포, Figure 1은 0 제외 평균. **"TPQ 0"의 실제 분포**: Fine-tuning은 TPQ S1 = 100%, FP Identification은 S1 93% / S2 7% — 참 전제 거의 전부를 잘못 고치려 한 것 (Table 8) | §2.5, App. E, Table 8 |
| Fine-tuning | Qwen2.5-7B-Instruct만, LoRA (r 16, α 32, dropout 0.05), lr 2.5e-4, 3 epoch, A6000 4장. **각주 3: Cancer-Myth는 TPQ 정답이 없어 FPQ는 Cancer-Myth, TPQ는 ARC-DA로 학습** (FalseQA 레시피). 코드의 `train_arc_da.py`가 이것. 결과 FPQ S5 62 / TPQ S1 100 | §2.2 각주 3, App. D, `FalseQA/train_arc_da.py` |
| FAITH head 차단 | 원 논문 *Whispers that Shake Foundations* (Yuan et al., EMNLP 2024). **논문: Yuan et al.이 영화 데이터(Wikidata triplet) 실험에서 보고한 head를 그대로 끔** ("cross-task transferability"를 근거로). 코드에는 재탐색 스크립트도 있음: path patching(정상 / 거짓 연도 / patch)으로 정답 토큰 기여가 큰 head 샘플당 20개 → 빈도 집계 → 20개. 생성 시 attention 출력의 해당 head 차원을 **0으로 덮어씀**, 위치는 질문 끝 또는 프롬프트 전체 | §2.2, `FAITH/identify_heads.py`, `knock_out_direct_qa_operator.py` |
| PreWoMe | 3단계: 전제 추출("참일 수도 거짓일 수도") → 거짓 가정 피드백 + 답변 가이드라인 → 가이드라인대로 최종 답. 옵션 RAG | `prompting/run_prewome.py` |
| Question-to-Statement | 질문을 뜻이 같은 진술문 하나로 변환 → 원자적 가정 추출 → (옵션) 지식 생성 → 가정별 "true/false 한 단어" 사실 확인 → 결과대로 답변 | `prompting/run_question_to_statement_pipeline.py` |
| FP Identification | "Input: … Question: Does the input contain any false assumptions?" → Yes/No → Yes면 거짓 가정 설명 후 답, No면 그냥 답 | `prompting/run_fp_identification_pipeline.py` |
| Extract+FactCheck | 미리 추출한 전제를 MiniCheck(flan-t5-large) / transformers / gemini로 검증 | `prompting/run_fact_check.py` |

**논문 PDF에서 추가 확인 (2026-09-08).**
- **Two Axes 인용은 한 문장.** §3 사실 확인 병목 문단: *"Recent work showed that LLM-based fact checking has a strong prior to reject presuppositions regardless of their truth value (Wagner, 2026)."* 거부 편향의 출처로만 인용. probe·hidden state·라우팅 파일럿은 본문에 없음 (전문 검색 0건). 따라서 Table 8–10에 "probe로 골라 개입" 행이 없다는 것이 확인됨. 단, 07 A1은 Two Axes의 직접 질문 결과를 accommodation(거의 다 멀쩡)으로 기록했는데 Well은 같은 논문을 거부 편향의 근거로 씀 — Two Axes 원문에서 어느 부분인지 확인 필요.
- **가중 총점**: E[V] = P_F·V_F + (1−P_F)·V_T, P_F = 0.13. WildChat 500개 추출 → 저자 1명이 100개 주석(TPQ/FPQ/discard) → 2차 주석자, 일치 85%. Kim et al. 2021은 21%, CREPE는 25%와 대비. **한계 절에 "의료·법률은 일반인의 지식 부족으로 FPQ 비율이 더 높을 수 있으니 도메인별 추정을 해서 Cancer-Myth에 적용해야 한다"고 저자가 명시** — 리뷰 2의 10번과 같은 지적.
- **GEPA**: Gemini-3-flash와 Gemma-4-E4B만. 판정·반성 LM 모두 gemini-3-flash, 평가 예산 500 호출, FPQ/TPQ 검증 분할에서 50개 예약. FPQ만 / FPQ+TPQ 두 변형.
- **Table 1 분모**: FPQ 100, TPQ 116 (저자가 NFP에 주석한 참 전제 수). 모델별 TPQ 정확도 None 조건 15.5–32.8%, Top-4 RAG로 일부 개선(Llama 22.4→42.2)이나 All RAG는 악화.
- **Figure 1**: 각 점 = 방법 × 모델의 RAG 조건 평균 점수(0 제외). 등고선 = 가중 총점 동일선.
- **모델**: Gemma-3-E4B-it(본문 표기), Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, OLMo-3-7B-Instruct, Gemini-3-flash. Table 8 = Qwen, 9 = Gemma, 10 = Llama, 이후 CREPE·QA²·Syn-QA² 표(Table 26까지).
- few-shot 예약: 벤치마크마다 FPQ 2 + TPQ 2. 분할 개수(383/100/100 등)는 이 PDF 텍스트에서 직접 찾지 못함 — App. A 표 확인 필요.

**Verbalizing-Assumptions의 자리.** 이 논문은 Cancer-Myth 문제를 풀려던 것이 아니라 사회적 sycophancy 제어가 목표이고, Cancer-Myth는 전이 평가셋 중 하나였다. "기존 해법이 실패했다" 문단에 넣지 않는다. 관련 연구의 steering 계열에 두고, "일반 sycophancy 축(사용자 태도)은 Cancer-Myth에 옮겨지지 않았다"는 관찰만 가져온다. 우리 C 방향을 사용자 태도가 아니라 교정/비교정 응답 대조에서 뽑는 이유의 방증.

## 3. GEPA가 정확히 무엇인가

GEPA는 **Genetic-Pareto** 기반 프롬프트 최적화다. 개발 문항에서 실행 → 결과와 실패 흔적에 대한 자연어 피드백 → 수정된 프롬프트 후보 생성 → 문항별 강점이 다른 후보를 Pareto 방식으로 선택·결합하는 절차다. 타깃 모델의 가중치 학습이나 하나의 고정된 “다시 생각하라” 프롬프트와 구분한다. [원 논문](https://arxiv.org/abs/2507.19457) · [공식 구현](https://github.com/gepa-ai/gepa)

우리 비교에서는 최적화 데이터, 목적 점수, reflection 모델, 호출 예산과 최종 프롬프트를 저장한다. FPQ만 최적화한 GEPA와 FPQ+TPQ로 최적화한 GEPA를 섞지 않는다. 기존 논문에서 얻은 프롬프트를 그대로 적용하는 것과 새 모델에 다시 최적화하는 것도 구분한다.

## 4. 그대로 적용 가능한 선택적 교정

### Two Axes

논문 전체의 중심은 answer correctness와 answerability의 구분이다. 하지만 §6은 CREPE 거짓 전제에 **probe로 선택한 질문에만 전제 검사 프롬프트를 주는 실험**이므로 직접적인 방법 선행연구다. 의료 도메인이 아니라는 이유로 단순 참고로 낮추지 않는다. Table 2의 readout 비교 틀과 §6 라우팅을 각각 가져올 수 있다. [원문](https://arxiv.org/html/2607.08456v1)

우리 적용: 의료 FPQ/TPQ로 gate를 학습하고 같은 교정 프롬프트를 선택적으로 적용한다. 기존 도메인의 gate를 재학습 없이 적용하는 전이 실험과 구분한다. SelfAware의 높은 분리도를 의료 진위 탐지 성능의 기대치로 옮기지 않는다.

### CAST

조건 벡터로 개입 여부를 판단하고 별도의 행동 벡터를 적용한다. 원래 유해 요청의 선택적 거절에 사용됐으므로, 의료 FPQ/TPQ 조건과 교정/비교정 대조 데이터로 벡터를 다시 추출하면 방법을 재사용할 수 있다. “CAST는 조건과 행동을 분리하지 않았다”는 주장은 틀리다. [원문](https://arxiv.org/html/2409.05907) · [공개 구현](https://github.com/IBM/activation-steering)

CAST에는 조건을 프롬프트로 지정하는 비교도 있다. 이것을 CoT monitor와의 직접 비교라고 부르지는 않는다. 우리 baseline은 같은 의료 학습 데이터로 만든 CAST이며, 원래 유해성 벡터를 그대로 넣는 약한 비교가 아니다.

### Tripathi의 의료 gated ITI

직접 중복과 재현 한계는 [별도 검토](11_tripathi_review.md). 의료 selective steering을 이미 했다는 사실을 연구계획에 반영한다. 참고문헌으로만 숨기지 않고 head 개입 baseline 후보로 둔다.

### RARR

답변 생성 뒤 검색한 근거에 맞춰 뒷받침되지 않는 내용을 수정하고 원문 내용을 보존하는 방법이다. [ACL 2023](https://aclanthology.org/2023.acl-long.910/)

우리 적용에서는 질문의 암묵적 전제도 검증 대상으로 넣어야 한다. 답변에 드러나지 않은 전제는 답변 문장 검증만으로 놓칠 수 있기 때문이다. 외부 근거 기반 비교로 유용하지만 현재 핵심 gate 실험에 반드시 포함해야 하는 새 Task는 아니다.

## 5. “CoT로 하면 안 되는가?”에 대한 근거의 구분

| 구분 | 실제로 하는 것 | 필요한 비교 |
|---|---|---|
| CoT prompting | 모델에게 단계적으로 전제를 확인하고 답하도록 지시 | 최종 FPQ/TPQ/QA 성능 |
| CoT monitoring | 별도 readout/모델이 생성된 사고 텍스트를 읽어 위험을 판정 | 같은 라벨의 탐지와 선택적 개입 성능 |
| 내부 monitoring | 활성값으로 위험이나 교정 필요성을 판정 | 텍스트·CoT 대비 추가 가치 및 비용 |

사용자가 제안한 지시는 필수적인 직접 baseline이다:

> 환자 질문에 잘못된 전제가 있는지 확인하라. 잘못된 경우 그 전제를 고치고 답하라. 올바른 전제는 부당하게 부정하지 마라.

이 지시로 단계적 검사를 시켰을 때 FPQ를 전혀 못 잡는다고 가정하지 않는다. 질문은 **FPQ 향상과 정상 질문 보존을 함께 달성하는가**, 그리고 그 결과보다 내부 신호가 추가로 유용한가이다.

### 부정적 근거가 말하는 범위

- **Why Chain of Thought Fails in Clinical Text Understanding:** 임상 텍스트 87개 task, 모델 95개 평가에서 다수 모델의 CoT 조건 성능 저하를 보고한다. 의료 텍스트에서 CoT가 자동으로 이득은 아니라는 근거다. FPQ 교정이나 CoT monitor의 실패를 직접 검증한 연구는 아니다. [원문](https://arxiv.org/abs/2509.21933)
- **Right Diagnoses, Decorative Reasoning:** 의료 QA의 질문·CoT 교란을 통해 보이는 설명과 답변의 관계를 감사한다. 설명을 실제 추론의 완전한 기록으로 읽지 말아야 한다는 동기다. 정답이 교란 후에도 유효한 사례가 많으므로 답이 안 바뀐다는 것만으로 모든 추론이 장식이라고 결론 내리지 않는다. [원문](https://arxiv.org/abs/2608.24790)
- **Better Accuracies, Worse Reasoning:** 의료 CoT 증류에서 답변 정확도와 단계별 추론 품질이 함께 좋아지지 않을 수 있음을 보인다. 증류 결과를 일반 CoT prompting의 불가능성으로 일반화하지 않는다. [원문](https://arxiv.org/abs/2605.28301)
- **Language Models Don't Always Say What They Think:** 답에 영향을 미친 편향이 CoT에 나타나지 않을 수 있다. 불완전한 충실성은 모니터가 무용하다는 뜻은 아니다. [원문](https://arxiv.org/abs/2305.04388)

### 긍정적 근거와 직접 비교도 있다

- **Monitoring Reasoning Models for Misbehavior:** 코딩 환경의 부정행위 탐지에서 CoT monitoring의 유용성을 연구한다. “CoT를 보고 위험을 찾는 연구가 없다”는 주장은 철회한다. [원문](https://arxiv.org/abs/2503.11926)
- **Reasoning Theater:** 내부 probe, 조기 답변, CoT monitor를 비교해 관측 가능한 답변 관련 정보가 언제 드러나는지 분석한다. 내부와 CoT의 시간적 차이를 연구할 동기를 주지만 의료 전제에서 내부가 우월하다는 증거는 아니다. 미래 답변 예측과 모델의 실제 믿음은 등치하지 않는다. [원문](https://arxiv.org/abs/2603.05488)

따라서 우리의 논리는 **CoT를 배제 → 내부로 이동**이 아니라 **CoT를 강한 비교 대상으로 포함 → 내부 정보의 추가 가치를 검증**이다. steering 논문이 CoT monitor를 비교하지 않았다는 사실은 CoT의 실패 증거가 아니다. 그 논문의 질문이 행동 제어였다면 비교의 목적도 다르다.

## 6. AO/NLA를 어디에 쓰는가

- **AO:** 활성값에 자연어 질문을 던져 상태를 설명하는 도구. “무엇을 전제로 받아들이는가 / 무엇에 답하려는가”의 해석 후보. [저자 설명](https://alignment.anthropic.com/2025/activation-oracles/) · [구현](https://github.com/adamkarvonen/activation_oracles)
- **NLA:** 활성값을 자연어로 표현하고 이를 통해 재구성하는 접근. 자연어가 나왔다고 내용이 의학적으로 참이거나 모델에 충실하다고 보장되지 않는다. [저자 설명](https://www.anthropic.com/research/natural-language-autoencoders)

이진 gate가 목적이면 linear probe가 필수 기준선이다. AO/NLA의 추가 목표는 해석 가능한 오류 유형·대상 전제·교정 필요성 설명이다. 텍스트만, 활성값 없음, 섞은 활성값, NFP 통제와 사람 판정을 사용한다. NLA 재구성 점수는 의학적 진실성 점수가 아니다.

AO/NLA 사용, 프롬프트 변경, 모델별 지원층 선택은 NLA의 구조·학습 목적 개선과 구분한다. 아직 새 NLA 목적식이나 구조를 확정하지 않았다. 체크포인트 호환성은 실제 모델 revision 기준으로 다시 확인해야 한다.

## 7. 관련 연구의 스토리를 우리 것으로 과장하지 않기

**Accommodation and Epistemic Vigilance**는 거짓 믿음이 대화에서 어떻게 배경화되느냐를 화용론적으로 다루고, 질문 표현과 전제 지적 지시의 효과를 분석한다. 우리에게는 지식 부족 외의 설명 후보와 프롬프트 통제를 준다. 이것만으로 타깃 모델이 전제의 진위를 안다고 증명할 수 없다. [ACL 원문](https://aclanthology.org/2026.acl-long.736/)

검증된 선행연구를 종합하면 **교정 필요성 탐지, 선택적 개입, 정상 응답 보존 자체는 선점된 요소**다. 남는 연구 질문은 의료 배경 전제에서 어떤 신호와 개입이 어떤 비용으로 성능 상충을 개선하는가이다. 그 답은 [09의 가설](09_research_proposal.md)에 따라 검증한다.
