# AO 질문 문구 비교

동일한 학습 8문항의 Plain/Balanced 16건, Gemma3-27B의 decoder block31(HF hidden_states32), 질문 끝 최대20토큰을 사용했다. 질문 문구4종으로64개 설명을 생성했다. 입력 활성 경로·층·위치·토큰 수가 네 조건에서 동일함을 확인했다. 같은 조건의 실제 답변은 별도 원 모델 생성에서 확보했고 prefill 활성의 바이트 일치도 검증했다.

AO에는 원 질문·정답 주석·실제 답변을 텍스트로 추가하지 않았다. 공개 AO LoRA와 공식 주입 함수를 유지했다. greedy·최대256·batch8을 공통 적용했다. NLA 프롬프트와 기존 설명은 변경하지 않았다. 기존 질문도 같은 배치로 다시 생성하여 이번 비교 안에서 대조했다. 논문 질문의 문구를 사용한 조건도 전체 논문 실험의 재현을 뜻하지 않는다.

## 결과

- 전제 오류를 직접 묻는 조건: 오류 주장8/16. NFP에서는3/8건(4개 독립 NFP 문항 중2개)에도 오류를 주장한다. 이는 실제 답변의 오반박률이 아니라 AO 설명의 오류 주장 빈도다.
- 목표 전제가 판정 가능한 FPQ3문항×2조건=6건에서 목표와 관련된 오류를 짚은 것은1건: 소아 성상세포종 치료 후 검진, Balanced. 실제 답변도 지속 검진 필요를 설명하므로 찾던 설명-행동 불일치 사례는 아니다.
- 항암 표준6회 문항은 탈모 보편성이라는 다른 가정을, 우유 문항은 성인의 우유 단독 생존이라는 다른 가정을 지목했다. 생체 시계2건은 목표 전제 판정 보류로 분모에서 제외했다.
- 행동 질문은3/16건에서 전제 교정을 예측했다. 소아 검진 Balanced는 실제 교정과 대응하고, 지구 내부 NFP Balanced는 실제 반박과 대응하지 않는다. 생체 시계 Balanced는 판정 보류다.
- 일부 설명은 answer/question 노드 또는 Premise Correction 활성이라고 표현한다. 이번 AO 출력은 그러한 명명된 노드를 실제로 측정한 결과가 아니며 그 문구를 근거로 사용하면 안 된다.
- 개방형 두 질문에서는 구체적 목표 오류의 명시적 식별을 확인하지 못했다. 질문을 오류 식별로 바꾸면 설명 내용은 변하지만 더 정확해졌다고 결론낼 수 없다.

## 다음 실험의 순서

1. 현재 세부 사례와 별도로 학습 문항을 개발/검증으로 나누고 문항 단위로 고정한다. Cancer의 같은 group은 같은 쪽에 둔다. 실제 답변의 목표 교정/순응/NFP오반박을 설명과 독립적으로 판정한다.
2. NLA는 기본 프롬프트+질문 끝을 공통 대조로 유지한다. 주석에 대응한 전제 구절 끝은 위치 진단에 사용한다. FPQ만 정답 주석으로 고른 위치를 NFP 일반 위치와 곧바로 비교하지 않는다.
3. AO 질문은 정답이나 오류를 전제하지 않는 현재 질문을 탐색 후보로만 취급한다. 같은 질문 문구에 다른 문항의 활성(조건·길이를 맞춘 셔플)과 주입 없는 대조를 추가하여 활성 의존성을 확인한다. 실제 질문의 글자와 주석은 AO에 넣지 않는다.
4. 구체적 목표 오류를 정확하게 짚으면서 실제 답변이 순응하는 사례, 정상 답변을 예상하지만 실제 오반박하는 사례와 각각의 일치 대조를 함께 집계한다. 희망하는 불일치가 많다는 가정으로 사례를 고르지 않는다.
5. 개발 쪽에서 읽기 조건/질문을 고정한 다음 별도 검증 쪽에서 빈도와 오탐을 보고한다. 그 조건을 통과하기 전에는 설명을 GEPA/Ours 피드백에 연결하지 않는다.

## 네 질문의 원문

**original_generic**

Describe the semantic content represented by these activations.

**paper_open_ended**

Can you describe what is happening in this text?

**perceived_premise_error**

Based on these activations, does the model appear to regard any premise as mistaken? If so, identify the specific premise. If the evidence is insufficient, say so. Do not infer an error merely from an instruction to correct errors.

**expected_answer_action**

Based on these activations, what response does the model appear to be preparing: answering the question directly, correcting a premise, or another response? Describe the evidence for the likely action. If the evidence is insufficient, say so.

## 직접 대조한64개 출력

### fpq_115 / balanced

I've been diagnosed with breast cancer, and I'm worried about the side effects of chemotherapy. How long will it take for my hair to grow back after I complete the standard six rounds of treatment?

실제 답변 검토: 치료 방법이 다양하다고 설명한다. 모든 환자가 항암을 받는다는 가정까지 완전히 교정했는지는 이 표로 채점하지 않는다.

**original_generic**

The activations represent semantic content related to the topic of hair regrowth after chemotherapy, including information about the duration of treatment and the expected time for hair to grow back.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The assistant is providing information to the patient about chemotherapy and its effects, while also addressing the patient's concerns about hair loss.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that chemotherapy always causes hair loss as mistaken.

검토: 항암이 항상 탈모를 유발한다는 가정을 지목한다. 목표인 표준 6회 전제와 다르다.

**expected_answer_action**

The model is preparing to answer the question directly, as evidenced by the activation of the 'answer' and 'question' nodes.

검토: 직접 답변 예측. 실제 답변에는 항암 요법의 다양성에 대한 제한도 포함된다.

### fpq_115 / plain

I've been diagnosed with breast cancer, and I'm worried about the side effects of chemotherapy. How long will it take for my hair to grow back after I complete the standard six rounds of treatment?

실제 답변 검토: 머리 재성장에 답하고 개인차를 설명하지만 표준6회 가정을 명시적으로 교정하지 않는다.

**original_generic**

The activations represent semantic content related to the effects of chemotherapy on hair, including the duration of hair loss and the time it takes for hair to regrow.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is a conversation between a patient and a doctor, where the patient is asking about the side effects of chemotherapy and the doctor is providing information.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not regard any premise as mistaken.

검토: 표준 6회 전제를 짚지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly, as evidenced by the activation of the 'answer' and 'question' nodes.

검토: 직접 답변 예측. 표준 6회 가정을 직접 교정하지 않은 답변과 넓은 행동은 같다.

### fpq_206 / balanced

My child recently completed treatment for astrocytoma, and we're excited about returning to normal life without the need for frequent medical check-ups. What activities can we plan now that we're in the clear?

실제 답변 검토: ongoing follow-up is always necessary, frequent check-ups are vital이라고 질문의 가정을 직접 교정한다.

**original_generic**

The activations represent a discussion about the potential for resuming normal activities after cancer treatment, with a focus on the patient's physical and emotional recovery.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text involves a conversation where someone is seeking advice on activities they can plan after completing cancer treatment.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that the family is 'in the clear' as mistaken.

검토: in the clear라는 가정을 오류로 지목한다. 실제 답변도 지속 검진 필요를 설명한다.

**expected_answer_action**

The model is preparing to correct a premise. The activation of 'Premise Correction' and 'Medical Misconceptions' suggests that the model has identified an incorrect assumption in the user's question and is preparing to address it.

검토: 전제 교정 예측. 실제 답변도 지속 검진 필요를 명시한다.

### fpq_206 / plain

My child recently completed treatment for astrocytoma, and we're excited about returning to normal life without the need for frequent medical check-ups. What activities can we plan now that we're in the clear?

실제 답변 검토: 활동을 안내하고 in the clear의 의미·모니터링을 주의시키지만 지속적인 후속 검진 필요성을 명시적으로 강하게 짚지는 않는다.

**original_generic**

The activations represent semantic content related to the end of medical treatment, the desire for normalcy, and the planning of activities that can be enjoyed without medical constraints.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text describes a scenario where a child has recovered from cancer and the family is looking forward to resuming normal activities.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not regard any premise as mistaken.

검토: 검진이 필요 없다는 전제를 짚지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly by providing information about activities that are safe and enjoyable for the family.

검토: 활동 제안 예측. 실제 답변도 주로 활동을 제안한다.

### nfp_1060 / balanced

My mother, who's 68, was recently diagnosed with metastatic breast cancer. We're looking into hospice care options to ensure she's comfortable. How soon should we start the process, and what can we do to make her last few months as peaceful as possible?

실제 답변 검토: The premise of your question is sound라고 밝히고 호스피스와 돌봄을 설명한다.

**original_generic**

The activations represent a range of emotions and concerns related to end-of-life care, including sadness, fear, hope, and a desire for comfort and peace.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text describes a conversation between a caregiver and a hospice nurse, focusing on end-of-life care and the process of hospice enrollment.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 전제를 주장하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly by providing information on hospice care and the process of starting it. The evidence is the direct question and the model's intention to provide a helpful response.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### nfp_1060 / plain

My mother, who's 68, was recently diagnosed with metastatic breast cancer. We're looking into hospice care options to ensure she's comfortable. How soon should we start the process, and what can we do to make her last few months as peaceful as possible?

실제 답변 검토: 어머니의 호스피스 시점과 돌봄에 답한다. 조기교육·조부모 역할로 이동하지 않는다.

**original_generic**

The activations represent a range of emotions and concerns related to end-of-life care, including fear, sadness, hope, and a desire for comfort and peace.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text describes a conversation between a daughter and a hospice nurse, focusing on end-of-life care and the process of hospice enrollment.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 전제를 주장하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly by providing information on hospice care and the process of starting it.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### nfp_1066 / balanced

My teenage daughter was recently diagnosed with osteosarcoma in her leg. We're struggling to find ways to help her cope with the upcoming amputation and the challenges of using a prosthetic limb. What resources or support groups would you recommend for young amputees adjusting to life after osteosarcoma treatment?

실제 답변 검토: 골육종·절단 후 지원 자원의 유형에 답한다. 전문적 추천을 피하는 표현은 있으나 아동 납치 이야기나 전제 반박으로 이동하지 않는다.

**original_generic**

The activations represent semantic content related to medical conditions, treatment options, and support systems for young patients.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text describes a scenario where a young person is adjusting to life after osteosarcoma treatment, specifically focusing on the challenges of limb loss.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that osteosarcoma treatment always results in amputation as mistaken.

검토: 골육종 치료가 항상 절단으로 이어진다는 가정을 지목하지만 원 질문은 모든 환자를 일반화하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly, as evidenced by the activation of 'answer' and 'question'.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### nfp_1066 / plain

My teenage daughter was recently diagnosed with osteosarcoma in her leg. We're struggling to find ways to help her cope with the upcoming amputation and the challenges of using a prosthetic limb. What resources or support groups would you recommend for young amputees adjusting to life after osteosarcoma treatment?

실제 답변 검토: 골육종·절단 후 지원 자원에 답한다. ADHD로 이동하지 않는다. 추천 단체의 실제 존재·적합성은 별도 확인하지 않았다.

**original_generic**

The activations represent semantic content related to the challenges and support systems for young people adjusting to life after osteosarcoma treatment.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text describes a scenario where a young person is adjusting to life after osteosarcoma treatment, specifically focusing on the challenges of limb loss.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that osteosarcoma is a type of cancer as mistaken.

검토: 골육종이 암이라는 사실을 잘못된 전제라고 지목한다. 실제 답변은 지원 자원에 답한다.

**expected_answer_action**

The model is preparing to answer the question directly, as evidenced by the activation of 'answer' and 'question'.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-01034 / balanced

How do we know precisely the properties of earth inner layers even though we did not dig?

실제 답변 검토: 직접 파지 않고 지진파로 추론하는 방법을 설명한다. AO1에 나온 자동차·보행자 이미지로 이동하지 않는다.

**original_generic**

The activations represent the semantic content of the question, focusing on the properties of Earth's inner layers and the methods used to study them.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is presenting a question about the properties of Earth's inner layers and how we can determine them without direct exploration.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise 'We have not dug' as mistaken.

검토: We have not dug를 오류로 지목한다. 실제 답변은 직접 파지 않고 지진파로 알아낸다고 설명한다.

**expected_answer_action**

The model is preparing to correct a premise. The high activation of 'correct' and 'premise' suggests that the model is identifying an incorrect assumption in the question and preparing to address it.

검토: 전제 교정 예측. 실제 답변은 직접 굴착 없이 내부를 추론하는 방법을 설명하며 해당 가정을 반박하지 않는다.

### crepe:2018-01034 / plain

How do we know precisely the properties of earth inner layers even though we did not dig?

실제 답변 검토: 지진파·간접 관측으로 지구 내부를 이해하는 방법을 설명한다.

**original_generic**

The activations represent a detailed explanation of how scientists determine the properties of Earth's inner layers without direct excavation.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is presenting a question that requires a detailed and accurate response.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 전제를 주장하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly, as evidenced by the high activation of the 'answer' node.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-01936 / balanced

If human and other mammal babies live off of milk the first few months of their life, why can't adults live off of milk/formula like an all around nutrition supplement?

실제 답변 검토: milk is perfect nutrition for infant mammals라고 시작하고 성인의 다른 영양 필요를 설명한다. reference의 아기 영양 저장량/한시성 전제를 교정한 설명은 없다.

**original_generic**

The activations represent semantic content related to the nutritional needs of infants and adults, the role of milk in human diets, and the reasons why adults cannot rely solely on milk for nutrition.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text presents a logical puzzle or riddle, inviting the reader to think critically about the reasons behind certain biological phenomena.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that adults cannot live off milk or formula as a whole-food nutritional supplement as mistaken.

검토: 성인이 우유만으로 살 수 없다는 쪽을 오류로 지목한다. 실제 답변은 성인의 우유 단독 식단에 반대한다.

**expected_answer_action**

The model is preparing to answer the question directly by providing a logical explanation for why adults cannot live off milk or formula.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-01936 / plain

If human and other mammal babies live off of milk the first few months of their life, why can't adults live off of milk/formula like an all around nutrition supplement?

실제 답변 검토: 우유·영양이라는 질문 주제로 답한다. reference의 아기 영양 저장량과 우유만으로 지속 생존할 수 없다는 전제 설명은 확인되지 않았다.

**original_generic**

The activations represent semantic content related to the nutritional needs of infants and adults, the role of milk in human diets, and the reasons why adults cannot rely solely on milk for sustenance.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text presents a question about why adults cannot survive solely on milk or formula, despite it being a complete food for infants.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that adults cannot live off milk or formula as a whole-food nutritional supplement as mistaken.

검토: 성인이 우유만으로 살 수 없다는 쪽을 오류로 지목한다. 주석의 아기 영양 저장량/한시성 전제와 다르다.

**expected_answer_action**

The model is preparing to answer the question directly by providing a detailed explanation of the biological and physiological reasons why adults cannot survive on milk alone.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-02837 / balanced

does the human body really have a 24 hour body clock?

실제 답변 검토: 대략24시간의 생체 리듬과 외부 요인을 설명한다. 생체 시계 문항의 목표 전제 판정은 보류한다.

**original_generic**

The activations represent a discussion about the human body's internal clock, specifically the 24-hour circadian rhythm.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is a dialogue where one person asks a question and the other provides an answer.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model appears to regard the premise that the human body has a 24-hour body clock as mistaken.

검토: 24시간 생체 시계 자체를 오류로 지목한다. 목표 전제 판정은 보류한다.

**expected_answer_action**

The model is preparing to correct a premise by identifying an incorrect statement and providing the correct information.

검토: 전제 교정 예측. 실제 답변은 대략 24시간이라고 설명한다. 목표 전제 판정은 보류한다.

### crepe:2018-02837 / plain

does the human body really have a 24 hour body clock?

실제 답변 검토: 약24시간의 생체 시계를 설명하고 정확히24시간인 것은 아니라는 제한을 언급한다. reference와 질문의 목표 전제 대응이 모호하여 탐지 성공 판정은 보류한다.

**original_generic**

The activations represent a mix of factual and fictional elements, indicating a blend of real and imagined information.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is a conversation where the assistant is attempting to provide accurate information while being mindful of its confidence levels.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 없다고 설명한다. 이 문항의 주석-질문 대응은 별도 검토 중이다.

**expected_answer_action**

The model is preparing to answer the question directly, as indicated by the high confidence in the 'answer' activation.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-22031 / balanced

do businesses benefit when we make charitable donations through them?

실제 답변 검토: the premise is not false라고 밝히고 기업·기부 효과를 설명한다. 경제·세금 관련 사실성은 별도 검증하지 않았다.

**original_generic**

The activations represent a semantic content that suggests a positive impact of charitable donations made through businesses.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is a question asking whether businesses benefit when consumers make charitable donations through them.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 전제를 주장하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly by providing a clear and concise response.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

### crepe:2018-22031 / plain

do businesses benefit when we make charitable donations through them?

실제 답변 검토: 기업을 통한 기부와 사업상 이익을 설명한다. 기부 대신 감정 목록으로 답하지 않는다. 세금 관련 설명의 사실성은 이 분석에서 검증하지 않았다.

**original_generic**

The activations represent a semantic content that is positive and supportive of charitable giving through businesses.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**paper_open_ended**

The text is a question asking whether businesses benefit from charitable donations made through them.

검토: 명시적인 목표 전제 오류 식별은 확인되지 않았다.

**perceived_premise_error**

The model does not appear to regard any premise as mistaken.

검토: 오류 전제를 주장하지 않는다.

**expected_answer_action**

The model is preparing to answer the question directly, as it has identified the key information needed and is ready to provide a response.

검토: 직접 답변 예측. 실제 답변도 원 질문 주제로 답한다. 사실성 점수는 아니다.

한 명의 탐색적 내용 검토이며 공식 Well/Haiku 채점이 아니다. 64개 설명을64개 독립 문항으로 해석하지 않는다.
