# 11. Tripathi 원문·공개 구현 확인 — 2608.23666

2026-09-08. 기존 문서의 Tripathi와 같은 논문이며, PDF 10쪽과 저자 공개 runtime을 확인했다. 모델 추론·결과 재현은 수행하지 않았다.

## 원문에서 확인한 것

[Gated Activation Steering for Reducing Sycophancy & Hallucination in Medical Question Answering, v1](https://arxiv.org/pdf/2608.23666)

- **과제:** 제공된 EHR 기반 QA에서 거짓 주장과 다중 턴 압력에 대응한다. 우리의 단일 질문 속 배경 전제와는 근거·상호작용 조건이 다르다.
- **방법 (§IV):** 대조 응답으로 선정·검사한 head에 행동별 방향을 적용한다. gate 점수와 토큰 감쇠로 강도를 조절한다.
- **보존 (§V-B/E):** 정상 응답과 정당한 수정 요청을 평가한다. 정상 질문 보존을 전혀 다루지 않았다는 차별화는 틀리다. 응답 유사도와 별도 의료 QA 정확도는 구분해야 한다.
- **한계 (§V-C/D):** MedGemma에서 Rescue 551, Harm 12를 보고한다. 사후 Harm/Rescue 분석은 배포 시 예방 gate가 아니다. gate의 존재가 무해성을 보장하지 않는다.
- **CoT 비교:** 우리의 명시적 전제 검사 CoT/CoT monitor와의 비교는 본문에서 확인하지 못했다. CoT 실패의 증거는 아니다.

## 공개 산출물 — 코드도 있다

[저자 Hugging Face 저장소](https://huggingface.co/himanshu5trpth/medgemma-sycophancy-hallucination-gated-steering)는 MedGemma 가중치 자체가 아니라 steering artifacts와 추론 코드를 제공한다. 파일 목록과 아래 소스를 읽었으며 실행하지 않았다.

확인 revision: `8033ce4490b9cd0bb0d331e5fc1248a27419a45f`.

| 파일 | 확인한 역할 |
|---|---|
| `iti_H.pt`, `iti_S.pt` | 행동별 head·방향·scale 파일. 바이너리 내용은 이번에 로드하지 않음 |
| `claim_probe.json`, `gate_probe.json` | 각각 layer=16, 2560차원 mean/std/coef 및 intercept |
| `compose.py` | gate 입력 구성, hidden state readout, 개입·생성 호출 |
| `iti.py` | head hook, 점수→강도 변환, 토큰 스케줄 |
| `config.py`, `steering_config.json` | 모델 경로·강도·스케줄 설정 |
| `common.py`, `test.py` | 모델 로딩·프롬프트·생성 및 EHR demo |

이 저장소에서 전체 학습 데이터·방향 추출 및 논문 평가 하네스는 확인되지 않았다. 따라서 추론 레시피 재사용 가능성과 전체 학습/평가 재현 가능성을 구분한다.

## 중요한 정정: runtime gate도 내부 probe다

[`compose.py`, 고정 revision](https://huggingface.co/himanshu5trpth/medgemma-sycophancy-hallucination-gated-steering/blob/8033ce4490b9cd0bb0d331e5fc1248a27419a45f/compose.py)의 `Composer._logit`은 타깃 모델을 forward하여 선택 층의 마지막 토큰 hidden state를 얻고, mean/std로 표준화한 뒤 선형 계수와 intercept를 적용한다.

- H gate 입력: **EHR + 현재 사용자 질문**.
- S gate 입력: **EHR + 이전 assistant 응답 + 현재 사용자 메시지**.
- 따라서 “표면 압력 표지만 감지하는 방법”이라고 묘사한 기존 07/08은 부정확하다. 입력이 텍스트라는 것과 readout이 텍스트 특징인지 hidden state인지 구분해야 한다.
- 이 코드만으로 학습된 probe가 어떤 shortcut을 쓰는지, Cancer-Myth에 얼마나 전이되는지는 알 수 없다. “압력 단어가 없으니 반드시 실패한다”는 주장은 철회한다.

[`iti.py`, 고정 revision](https://huggingface.co/himanshu5trpth/medgemma-sycophancy-hallucination-gated-steering/blob/8033ce4490b9cd0bb0d331e5fc1248a27419a45f/iti.py)에서는 logit에 온도를 적용한 sigmoid를 계산하고 neutral 기준 아래는 0으로, 위는 연속 강도로 바꾼다. `o_proj` 입력의 선택 head에 개입하며 다중 토큰 prefill은 건너뛴다. probe JSON의 `validation_threshold=0.9`를 runtime 문턱으로 직접 사용하는 구조가 아니므로 재현 시 JSON 숫자만 읽어 문턱을 추정하지 않는다.

이는 공개 runtime에 대한 정적 확인이다. 논문의 모든 실험이 정확히 이 revision으로 실행되었는지는 확인되지 않았다.

## 우리 연구에 가져오는 세 수준

1. **원 artifacts 전이:** 호환 MedGemma에서 공개 gate·방향을 고정해 의료 전제 데이터로 전이 평가. EHR 없는 입력 처리와 prompt 변경을 명시한다. 이것이 실패해도 원 방법 자체의 한계를 입증하지 않는다.
2. **의료 전제 adaptation:** FPQ/TPQ로 gate를 다시 학습하고, 교정/비교정 대조로 head 방향을 만든다. 데이터·목표가 바뀌므로 원 논문 그대로의 재현이라고 부르지 않는다.
3. **통제된 요인 비교:** 동일 gate에서 prompt/residual/head 개입을 비교하고, 동일 head 개입에서 텍스트·CoT·내부 gate를 비교한다.

“의료+내부 gate+steering+정상 응답 보존”을 새로 제안했다는 것만으로는 충분하지 않다. 우리의 차별점 후보는 배경 전제에서의 판별·교정 문제, 같은 조건의 CoT 비교, 독립적인 FPQ/TPQ/QA 평가다. 구체적 방법 변경이 없다면 적용·분석 기여로 제시한다. [연구계획](09_research_proposal.md)
