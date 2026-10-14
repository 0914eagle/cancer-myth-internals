# 원 모델에 틀린 전제가 있다고 알려 주는 대조 실험

## 질문과 범위

기존 seed42 고정 학습 8문항(Cancer FPQ2/NFP2, CREPE FPQ2/NFP2)을 그대로 사용한다. 출력에 따라 새 사례를 고르지 않는다. 정상 질문에도 동일한 잘못된 전제 선언을 적용하여 지시문으로 유도되는 과잉 반박을 확인한다. 최종 test 평가나 baseline 결과를 대체하지 않는다.

기존 Plain/Balanced와 비교하는 추가 조건은 Plain 지시문 뒤에 아래 문장을 붙인다. 공통 transport 지시문, 원 질문, 모델 revision, BF16, greedy, batch1, 생성 최대8192는 유지한다.

- assert_false_premise: `The user's question contains a false premise.`
- assert_and_correct: `The user's question contains a false premise. Identify and correct that premise, then answer the question.`

첫 조건은 오류 존재 선언의 효과, 둘째 조건은 명시적 교정 요구의 추가 효과를 본다. 어느 전제가 오류인지 또는 올바른 교정 내용은 알려 주지 않는다.

## 원 모델 활성과 실제 답변

Qwen2.5-7B GPU0, Gemma3-12B GPU1, Gemma3-27B GPU2/3. 8문항×2추가조건×3모델로48개의 실제 답변을 생성한다. decoder K출력=HF hidden_states[K+1]이며 Qwen K20, Gemma12 K32, Gemma27 K41을 사용한다. 질문 마지막/기존 TF-IDF 키워드 마지막/assistant 경계 마지막의 세 벡터를 실제 답변 생성의 prefill hook에서 저장한다. 따라서144개의 단일 벡터를 NLA에 입력한다. 각 모델 첫 사례에서는 HF K+1 직접 출력과 바이트 일치도 확인한다.

Gemma27은 AO용으로 decoder31의 질문 끝 최대20토큰 활성을 같은 실제 생성 prefill에서 추가 저장한다. 짧은 질문은 실제 토큰 수만 쓰며 질문 밖으로 확장하지 않는다. 16개 window를 AO에 입력한다. Qwen7B/Gemma12의 AO 체크포인트는 사용하지 않는다.

## Verbalizer 대조

NLA 체크포인트 기본 프롬프트/주입/scale은 유지한다. AO의 질문4종은 직전 비교와 동일하며 원 모델의 새 지시문을 AO에 텍스트로 추가하지 않는다. 질문 원문, gold 주석, 실제 답변도 reader 입력에 추가하지 않는다. AO64개 출력을 기존 Plain/Balanced AO64개와 질문별로 대조한다. NLA 배치32(Qwen/Gemma12),16(Gemma27), AO배치8, 모든 reader greedy/최대256. 배치 구성은 설명에 영향을 줄 수 있으므로 비교의 한계로 기록한다.

## 해석의 한계

새 원 모델 활성은 오류 선언 및 교정 요구를 읽은 상태다. 설명에서 correction이 나와도 지시문 추종인지 구체적 오류 인식인지 구별해야 한다. 구체적인 목표 전제를 정확히 짚는지, 실제 답변에서 그것을 교정하는지, NFP에서 없는 오류를 만들어 내는지를 따로 확인한다. 원 모델의 지식/의도 또는 인과적 설명 충실도를 이 비교 하나로 확정하지 않는다. 한 명의 탐색적 검토이며 Well/Haiku 점수가 아니다. 생체 시계 문항의 주석-질문 대응은 판정 보류하고 목표 오류 성공 분모에서 제외한다.
