# 14. 교수님 보고 2 — 후속 연구와 연구 방향 (2026-09-08 발송)

첫 보고(Cancer-Myth 소개, 후속 연구를 확인하겠다는 약속)에 이어 2026-09-08에 보낸 글. 아래 본문이 발송본(작성자 초안에 여섯 군데 수정 반영). Task 1–3과 표는 이메일에서 빼고 목요일 미팅에서 [13](13_task_spec.md)으로 설명한다. anchor 표를 Well-Actually로 바꾸는 대안은 이메일에 넣지 않았고 [12 §11](12_discussion_decisions.md)에 기록. 근거는 [09](09_research_proposal.md), [10](10_evidence_and_baselines.md), [13](13_task_spec.md). 목요일 미팅 전에 보내고, 피드백은 [12](12_discussion_decisions.md)의 미결정 항목에 반영한다.

**이 논문이 될 수 있는가에 대한 답 (보고서에는 마지막 문단으로 압축).**
- method 논문은 아니다. 골라서 개입하는 틀은 CAST, Two Axes, Tripathi에 있다.
- empirical·analysis 논문으로는 된다. 조건은 Table 1에 "의사 검증 의료 벤치마크에서 교정률을 올리면서 정상 질문 손실을 허용폭 안에 둔 행"이 나오는 것. Cancer-Myth도 Well-Actually도 그 행을 못 냈다.
- 좋은 경우: 내부 > 텍스트·CoT 그리고 steering > 프롬프트. "내부에서 읽고 내부에서 고쳐야 한다". 본 트랙 급.
- 약한 경우: 텍스트 분류기로 골라도 같다. "선택적 프롬프팅으로 시소가 끊긴다". Findings·워크숍.
- 안 되는 경우: 골라도 NFP가 무너진다. 음성 결과, 워크숍.

---

안녕하세요 교수님.

저번에 말씀드린 Cancer-Myth의 후속 논문과 의료 domain에서 비슷한 sycophancy 논문, 일반 domain에서 잘못된 전제가 깔린 질문을 다룬 논문들을 찾아봤습니다.

**1. Cancer-Myth의 문제를 직접 다룬 후속 연구**: *Don't 'Well, Actually' Me Unless You Know What You're Talking About: Weak Presupposition Verification Degrades General QA Performance*

이 논문은 Cancer-Myth 논문이 발견한, 잘못된 전제를 잘 고치게 만들수록 오히려 옳은 전제를 망치게 되는 문제를 다뤘습니다. Cancer-Myth 585개를 False Presupposition Question (FPQ)으로, Cancer-Myth의 NFP 150개에 참 전제를 주석해 True Presupposition Question (TPQ)으로 만들어 총 8가지 방법을 적용했습니다. "먼저 잘못된 가정이 있는지 판별하라"는 prompt, 질문에서 전제를 뽑아 모델에게 직접 사실 확인, 질문을 평서문으로 바꿔 T/F 판별, self-critique, prompt 최적화(GEPA), finetuning, 특정 attention head 차단입니다. 결과는 전부 FPQ 성능을 올리면 trade-off로 TPQ 성능이 내려가는 현상을 보였습니다.

특히 WildChat이라는 실제 사용자 대화 기록을 표본 추출했을 때 잘못된 전제가 깔린 질문이 약 13%였고, "잘못된 전제 질문 점수 × 0.13 + 정상 질문 점수 × 0.87"로 계산하면 아무 방법도 적용하지 않은 바닐라 모델이 가장 높았습니다.

또한 질문에서 전제 문장을 따로 꺼내 모델에게 "이 문장이 참인가 거짓인가"를 물으면, 거짓 전제는 96–98% 거짓이라 답하지만 참 전제도 58–93%를 거짓이라 답했습니다.

**2. 관련 연구**

*Two Axes of LLM Abstention: Answer Correctness and Question Answerability*. 일반 domain에서 모델에게 직접 물으면 잘못된 전제 질문을 잘 못 판정하지만(AUROC 0.64–0.67) 내부 hidden state에 classifier를 걸면 더 잘 판정하고(0.69–0.78), 그 점수로 고른 질문에만 교정 prompt를 주면 정상 질문을 잘못 지적하는 것이 57%에서 14%로 줄었습니다.

*Gated Activation Steering for Reducing Sycophancy & Hallucination in Medical Question Answering*. 의료 domain에서 사용자가 멀티 턴에 걸쳐 자신의 의견을 주장하는 상황에서 내부 classifier로 steering 여부를 결정하여, 정상적인 상황에서의 output은 바꾸지 않으면서 sycophancy를 줄였습니다.

모델이 전제를 내부에서 어떻게 처리하는지 분석한 연구들도 있었는데(*Language Models Encode the Contextual Truth of Propositions* 등), 문장의 참/거짓이 내부에 선형으로 읽힌다는 것까지만 보였고 개입은 없었습니다.

결국 출력 수준에서 교정을 강화하는 방법은 전부 정상 질문의 성능이 낮아졌고, 개입할 질문을 내부 신호로 골라서만 개입하면 그 trade-off가 줄어든다는 것이 일반 domain과 의료 기록 상황에서 각각 확인됐지만, Cancer-Myth처럼 환자 질문 안에 깔린 전제에 대해서는 시도된 적이 없습니다.

그래서 위 두 논문을 포함해 최근 여러 조건부 steering 연구가 쓰는 방식, 즉 모델 내부 hidden state에 classifier를 걸어 "이 질문에 잘못된 전제가 있는가"를 점수로 매기고 threshold를 넘는 질문에만 개입하는 방식을 Cancer-Myth에 적용해서, FPQ 성능을 올리면서 TPQ와 일반 QA 성능은 지키는 것이 되는지 보려고 합니다. 개입은 교정 prompt와 steering 둘 다 해보고, 결과는 Cancer-Myth Table 1의 열을 그대로 써서 기존 방법들과 같은 표에 놓을 생각입니다.

다만 이 방식 자체는 이미 여러 논문에서 쓰고 있어서, 이걸 Cancer-Myth에 적용해 아직 아무도 못 낸 "FPQ를 올리면서 TPQ를 지키는" 결과를 내는 것으로 논문이 되는지, 아니면 방법론 자체가 새로워야 하는지 여쭤보고 싶습니다.

감사합니다.
