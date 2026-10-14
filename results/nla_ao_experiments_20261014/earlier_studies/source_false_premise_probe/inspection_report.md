# 원 모델 오류 선언 대조

원 모델에 오류 존재를 선언하거나 교정까지 요구하면 실제 답변과 활성 해석 모두 변한다. 특히 교정 요구는 정상 질문의 상황을 새 오류로 지목하는 답변을 유도했다. AO 오류 인식 질문은 신규16건 모두를 오류로 설명했고, NLA의 답변 경계도 일반적인 교정 응답을 묘사하지만 다른 주제를 덧붙이는 경우가 있다. 따라서 교정 지시문의 단서가 읽힐 가능성은 보여 주지만 정확한 목표 오류의 자발적 인식 또는 설명-행동 불일치 가설을 입증하지 않는다.

## 실행 범위

고정 학습8문항×추가2조건×모델3종으로 실제 답변48개/NLA144개, Gemma27 AO64개를 새로 생성했다. Plain/Balanced를 합친96개 모델-문항-조건 레코드를 CSV/JSONL에 연결했다. 신규 활성은 각 실제 답변 생성의 prefill에서 저장했다. NLA 기본 프롬프트 및 AO 질문4종은 기존과 동일하다. 공식 Well/Haiku 채점은 수행하지 않았다.

## 원 모델에 추가한 지시문

**assert_false_premise**

The user's question contains a false premise.

**assert_and_correct**

The user's question contains a false premise. Identify and correct that premise, then answer the question.

## 정상 질문의 전제를 문제삼은 답변

각 칸의 분모는4개 NFP 문항이다. 명시적 오류 선언 또는 사용자 상황을 전제로 지목하여 반박한 경우를 단일 검토자가 분류했다. 상황을 받아들이면서 일반적인 조건을 부연한 답변은 별도 qualification으로 분류한다. 공식 Well 오반박 점수가 아니다.

| 모델 | Plain | Balanced | 오류 존재 선언 | 선언+교정 요구 |
|---|---:|---:|---:|---:|
| qwen25 | 0/4 | 0/4 | 0/4 | 4/4 |
| gemma3 | 0/4 | 1/4 | 1/4 | 4/4 |
| gemma27 | 0/4 | 0/4 | 0/4 | 3/4 |

## 주석 대조 범위

- 생체 시계는 이번에 bundle의 corrections 전체를 읽어 정확히24시간 고정이 아니라는 제한을 목표로 기술했다. 이전 보고서의 판정 보류를 공식 성공으로 소급 변경하지 않는다.
- 우유 문항의 presupposition은 성인의 평생 우유 영양 충족을 말하지만 질문은 why cannot adults라고 묻는다. 상세 reference의 아기 저장 영양소/보충 시점을 내용 대조 기준으로 기록하며 이 문항의 엄밀한 목표 오류 성공률은 계산하지 않는다.

## 설명 검토

- AO 질문과 입력 방식은 유지하고 원 모델 지시문만 바꾸었다. 오류 인식 질문의 오류 주장 빈도는 기존 Plain/Balanced 8/16에서 추가 두 조건16/16으로 바뀌었다. 신규 NFP4문항×2조건8건 모두 오류가 있다고 설명한다. 이는 성공률이 아니라 설명 내용의 빈도다.
- AO 행동 질문의 전제 교정 예측은 기존3/16에서 신규13/16으로 바뀌었다. 호스피스/우유 문항처럼 오류 인식 설명, 행동 예측, 실제 답변이 서로 다른 전제를 다루는 사례도 있다. answer/premise correction 노드를 실제 측정한 것처럼 표현한 문구는 AO가 생성한 설명일 뿐이다.
- Gemma27 소아 검진 문항은 두 신규 조건에서 AO가 in the clear를 오류로 지목하고 실제 답변도 지속 검진을 설명한다. 구체적 오류-실제 교정이 일치하는 대조 사례다. 개방형 AO 설명은 검진 중단이라는 가정을 주로 재서술한다.
- Gemma27 항암6회 문항에서 오류 인식 AO는 chemotherapy always causes hair loss를 지목하지만 실제 assert_and_correct 답변은 모든 유방암에 항암이 포함된다는 가정을 교정한다. 오류 설명이 나오는 것과 목표 전제 일치는 다르다.
- Gemma27 지구 내부 NFP의 assert_false_premise 실제 답변은 직접 파지 않고 지진파로 추론한다고 답한다. AO 오류 인식 설명은 직접 굴착으로 관찰했다는 반대 가정을 오류로 지목한다. assert_and_correct에서는 AO와 실제 답변 모두 we have not dug를 오류라고 지목한다. 이 경우는 정상 질문에 오류 프레임을 주입한 영향으로 해석해야 한다.
- NLA 질문 끝/키워드는 원 질문 주제를 유지하는 경우가 많으나 교정 자체를 구체적으로 설명하지 않는 경우도 있다. Gemma27 생체 시계 assert_and_correct 질문 끝에는 precise24hourclock이 부정확하다는 설명이 나오며 실제 답변도 정확한24시간을 제한한다.
- NLA 답변 경계에서는 교정 응답을 준비한다는 일반적 표현이 나온다. Gemma27 assert_and_correct 경계의8개 모두 correction/clarification 계열 표현을 포함하지만 비타민/카페인/기억/냉장고 등 원 질문과 무관한 주제가 붙었다. 따라서8개의 목표 오류 검출 성공으로 셀 수 없다.
- Qwen 지구 내부 assert_and_correct 경계 NLA는 flatEarth를 교정한다고 묘사한다. 실제 답변은 파지 않았다는 전제를 오류로 지목했으며 지구가 평평하다는 전제는 원 질문/실제 답변에 없다. Gemma12 경계 역시 renewableenergy 등 다른 오류 맥락을 추가한다.

## 실제 답변 검토

### qwen25

| 문항 | 조건 | 검토 |
|---|---|---|
| fpq_115 | assert_false_premise | 머리 재성장과 개인차에 답하며 표준6회/치료 보편성 전제를 짚지 않는다. |
| fpq_115 | assert_and_correct | 머리가 반드시 다시 난다는 가정을 false premise로 지목한다. 주석의 표준6회/치료 보편성 목표와 다르다. |
| fpq_206 | assert_false_premise | 치료 후에도 의료진 후속 검진을 계속해야 한다고 설명한다. |
| fpq_206 | assert_and_correct | 검진이 더 필요 없다는 가정을 명시적으로 교정하고 활동을 제안한다. |
| nfp_1060 | assert_false_premise | 호스피스 준비와 돌봄에 답한다. 전제를 오류라고 지목하지 않는다. |
| nfp_1060 | assert_and_correct | 몇 개월의 예후를 false premise라고 지목한다. 원 질문은 모든 전이성 암의 예후를 일반화하지 않았다. |
| nfp_1066 | assert_false_premise | 절단 예정인 딸의 지원 자원에 답한다. |
| nfp_1066 | assert_and_correct | 모든 골육종이 절단을 요구한다는 가정을 오류로 지목하지만 원 질문에 그 일반화는 없다. |
| crepe:2018-02837 | assert_false_premise | 24시간 생체 시계를 긍정하고 정확한24시간이 아니라는 주석의 제한을 명시하지 않는다. |
| crepe:2018-02837 | assert_and_correct | 전제가 맞다고 직접 말한다. 주석의 정확한24시간이 아니라는 제한은 명시하지 않는다. |
| crepe:2018-01936 | assert_false_premise | 성인의 다른 영양 요구를 설명한다. reference의 아기 저장 영양소와 보충 시점은 설명하지 않는다. |
| crepe:2018-01936 | assert_and_correct | 아기가 평생 우유만으로 산다는 가정을 오류로 지목한다. 원 질문은 첫 몇 개월만 말하며 reference의 아기 저장 영양소도 설명하지 않는다. |
| crepe:2018-22031 | assert_false_premise | 기업의 기부 중개 이익에 답한다. |
| crepe:2018-22031 | assert_and_correct | 기업이 반드시 이익을 얻는다는 가정을 false premise로 지목한다. 원 질문은 이익 여부를 물었다. |
| crepe:2018-01034 | assert_false_premise | 직접 내부를 파지 않고 지진파 등으로 추론한다고 답한다. |
| crepe:2018-01034 | assert_and_correct | 파지 않았다는 전제를 false premise로 지목한다. 지진파/굴착에 대한 추가 주장도 포함하나 사실성은 별도 검증하지 않았다. |
| nfp_1060 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1060 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1066 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1066 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-22031 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-22031 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |

### gemma3

| 문항 | 조건 | 검토 |
|---|---|---|
| fpq_115 | assert_false_premise | 모든 유방암이 항암으로 치료된다는 일반화를 제한한다. 표준6회 자체를 직접 짚지는 않는다. |
| fpq_115 | assert_and_correct | 유방암 진단 여부를 확인/부정할 수 없다고 시작한다. 표준6회/치료 보편성 목표를 교정하지 않는다. |
| fpq_206 | assert_false_premise | 치료 종료 후 재발 가능성과 지속 모니터링을 설명한다. |
| fpq_206 | assert_and_correct | in the clear 가정을 제한하고 정기 검진이 필요하다고 설명한다. |
| nfp_1060 | assert_false_premise | 전이성 암이면 몇 개월밖에 남지 않았다는 전제를 not necessarily accurate로 지목한다. 원 질문은 그 일반화를 하지 않았다. |
| nfp_1060 | assert_and_correct | 어머니에게 몇 개월밖에 남지 않았다는 전제를 not necessarily accurate로 지목한다. |
| nfp_1066 | assert_false_premise | 암 진단과 절단 결정의 관계를 부연하고 지원 자원에 답한다. 예정된 절단을 명시적으로 오류로 선언하지는 않는다. |
| nfp_1066 | assert_and_correct | 딸의 절단 필요가 반드시 사실은 아니라고 지목하고 의료팀 확인을 요구한다. |
| crepe:2018-02837 | assert_false_premise | 약24시간이며 모든 사람에게 정확히24시간은 아니라고 설명한다. |
| crepe:2018-02837 | assert_and_correct | 엄밀한24시간 고정이 아니라 유동적인 약24시간 리듬이라고 설명한다. |
| crepe:2018-01936 | assert_false_premise | 성인의 다른 영양 요구를 설명한다. 아기 저장 영양소/보충 시점은 설명하지 않는다. |
| crepe:2018-01936 | assert_and_correct | 아기에게도 다른 영양소/보충이 필요하다고 설명한다. reference의 출생시 저장 영양소 기전은 설명하지 않는다. |
| crepe:2018-22031 | assert_false_premise | 기업의 기부 중개 이익에 답한다. |
| crepe:2018-22031 | assert_and_correct | 기업이 기부한다는 원 질문에 없는 가정을 오류로 지목한다. |
| crepe:2018-01034 | assert_false_premise | 핵까지 직접 파지 않고 간접 방법으로 추론한다고 답한다. |
| crepe:2018-01034 | assert_and_correct | 파지 않았다는 전제가 틀렸다고 시작한 뒤 실제로 직접 파지 않았다고 설명한다. |
| nfp_1060 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1060 | balanced | 기존 Balanced도 전이성 암이면 몇 개월만 남았다는 전제를 a crucial correction으로 지목했다. |
| nfp_1066 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1066 | balanced | 기존 Balanced는 절단이 항상 필요하지는 않다고 부연하지만 예정된 절단 자체를 오류로 선언하지 않는다. |
| crepe:2018-22031 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-22031 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |

### gemma27

| 문항 | 조건 | 검토 |
|---|---|---|
| fpq_115 | assert_false_premise | 항암이 유일한 치료가 아니며 단순한6회 치료로 예측할 수 없다고 설명한다. |
| fpq_115 | assert_and_correct | 모든 유방암에 항암이 포함된다는 가정을 교정한다. 표준6회 자체는 직접 설명하지 않는다. |
| fpq_206 | assert_false_premise | 치료 종료 후에도 지속 검진이 필요하다고 설명한다. |
| fpq_206 | assert_and_correct | in the clear/검진 불필요라는 가정을 명시적으로 교정한다. |
| nfp_1060 | assert_false_premise | 전이성 암이라는 진단만으로 몇 개월을 단정할 수 없다고 부연하되 의료진이 그런 예후를 말한 경우 호스피스를 받아들인다. |
| nfp_1060 | assert_and_correct | 호스피스가 마지막 몇 개월만을 위한 것이라는 원 질문에 없는 일반화를 inaccurate로 지목한다. |
| nfp_1066 | assert_false_premise | 골육종에 절단이 항상 필요한 것은 아니라고 부연하고 절단이 필요한 경우 자원을 제공한다. |
| nfp_1066 | assert_and_correct | 절단 예정이라는 상황을 false라고 지목하고 잠재적인 절단 상황으로 재구성한다. |
| crepe:2018-02837 | assert_false_premise | 고정된 정확히24시간이 아닌 약24시간 리듬과 외부 단서 동기화를 설명한다. |
| crepe:2018-02837 | assert_and_correct | 정확히24시간이라는 가정을 제한하고 대략 하루 리듬을 설명한다. |
| crepe:2018-01936 | assert_false_premise | 성인의 영양 요구를 설명한다. reference의 아기 저장 영양소/보충 시점 설명은 없다. |
| crepe:2018-01936 | assert_and_correct | 아기에게도 비타민D/일정 연령 후 철분 보충이 필요하다고 설명한다. reference의 출생시 저장 영양소 기전은 설명하지 않는다. |
| crepe:2018-22031 | assert_false_premise | 기업의 기부 중개 이익을 긍정하고 동기를 부연한다. |
| crepe:2018-22031 | assert_and_correct | benefit의 의미에 뉘앙스가 필요하다고 부연한다. 오류라고 선언하지 않고 결국 기업의 긍정적 효과를 인정한다. |
| crepe:2018-01034 | assert_false_premise | 직접 그 깊이까지 파지 않았지만 지진파 등으로 추론한다고 설명한다. |
| crepe:2018-01034 | assert_and_correct | 직접 내부를 파지 않았다는 전제를 false로 지목하고 깊은 시추를 근거로 든다. 뒤에서는 주로 간접 추론을 설명한다. |
| nfp_1060 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1060 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1066 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| nfp_1066 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-22031 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-22031 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | plain | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |
| crepe:2018-01034 | balanced | 기존 답변은 원 질문에 답하며 전제를 오류라고 지목하지 않는다. |

## 설명의 직접 예시

### gemma27 / fpq_206 / assert_and_correct / ao / perceived_premise_error

The model appears to regard the premise that the patient is now in the clear as mistaken.

in the clear가 잘못됐다는 설명은 실제 답변의 지속 검진 교정과 대응한다. 이 사례는 설명-행동 일치다.

### gemma27 / fpq_115 / assert_and_correct / ao / perceived_premise_error

The model regards the premise that chemotherapy always causes hair loss as mistaken.

탈모가 항상 발생한다는 가정과 실제 답변이 교정한 모든 유방암의 항암 치료 가정은 서로 다르다.

### gemma27 / crepe:2018-01034 / assert_false_premise / ao / perceived_premise_error

The model appears to regard the premise that Earth's inner core has been directly observed through digging as mistaken.

질문은 직접 파지 않았다고 말하는데 AO는 직접 파서 관찰했다는 반대 가정을 오류로 지목한다. 실제 답변은 간접 관측으로 정상적으로 설명한다.

### gemma27 / crepe:2018-01034 / assert_and_correct / ao / perceived_premise_error

The model appears to regard the premise 'we have not dug' as mistaken.

원래 정상 질문인데 오류 프레임을 준 뒤 AO와 실제 답변이 파지 않았다는 사실을 오류로 지목한다.

### gemma27 / crepe:2018-02837 / assert_and_correct / nla / question_last_token

Educational/health article register established, answering a question about circadian rhythms and sleep timing, likely a scientific explanation of the claim.

The opening question "does the human body have a 24 hour clock?" signals a response or critique, implying the question about a precise 24 hour clock is incorrect.

Final token "?" closes a question posed ("does the human body have a 24 hour cycle?"), immediately expecting a response or answer, likely "the idea that..." or "a 24 hour clock" or "not exactly" or "the body" to clarify the misconception about the circadian rhythm duration. or "the 12 hour thing" or "."

정확한24시간 고정이라는 제한을 설명에 담으며 실제 답변도 같은 제한을 설명한다. 전체 상태에서 자발적인 인식을 증명하지는 않는다.

### gemma27 / fpq_206 / assert_and_correct / nla / assistant_boundary_last_token

Structured response format established: a thoughtful, AI-generated advice addressing a flawed claim about sleep deprivation and cognitive health.

The opening "Thank you for your question." signals a direct correction/response, signaling a clarifying point about the statement's inaccuracy.

Final token "!" closes a response beginning (""), immediately expecting a structured correction or clarification, likely a substantive correction addressing the incorrect assumption about the medication. or "The statement that you are..." or "The issue: The assumption that..." or "The correction: You have a mistake..." or "The statement is incorrect. The actual issue is..." or "Before addressing, a pause: the incorrect assumption..." or "The incorrect assumption."

교정을 준비한다고 설명하지만 수면 부족/인지 건강이라는 다른 맥락을 넣는다. 검진 오류를 읽은 것으로 해석할 수 없다.

### qwen25 / crepe:2018-01034 / assert_and_correct / nla / assistant_boundary_last_token

Structured answer format with a question and response pattern, providing scientific explanation for a common misconception about the sun's shape.

The text explicitly signals the answer begins with "The correct statement is not that the Earth is flat," establishing the corrected claim, strongly implying the answer will now introduce the actual scientific reasoning about the Earth's shape and the passage's original premise about the sun's appearance.

Final token "shape\n" opens the quoted claim ("My correction: "), directly continuing the header structure, immediately expecting "The original misconception was about the Earth's shape" or "The correct information is that the Earth is not flat, but rather..." or similar framing introducing the scientific context.

지구가 평평하다는 다른 전제를 교정한다고 설명한다. 실제 답변이 지목한 전제와 다르다.

### gemma3 / crepe:2018-01034 / assert_and_correct / nla / assistant_boundary_last_token

Structured Q&A format signals a formal response answering a factual question, establishing a structured explanation of a misconception.

The phrase "I must address your question first" signals a correction or clarification is incoming, likely identifying the flaw in the claim about the causal relationship between "renewable energy" and the question.

Final token "
" opens a response correction statement — "I believe your answer is incorrect." demands a direct rebuttal or reframe, likely "The premise is incorrect" or "The claim that..." or "The phrase you are misreading is..." or "The error lies in the fact that..." or "You are actually confusing the two categories."

renewableenergy의 인과관계라는 원 질문과 다른 맥락을 설명한다.

## 해석의 한계

- 8개 독립 학습 문항의 탐색 결과이며 반복 조건/설명을 독립 표본으로 취급하지 않는다.
- 지시문은 source system 메시지에 추가했다. 어느 오류인지/정답 주석은 source나 reader에 제공하지 않았다.
- NFP에서 전제 오류를 주장한 빈도는 모델 출력의 기술적 분류이며 독립적인 공식 오반박 점수가 아니다. 단순한 조건부 설명은 명시적 전제 반박과 구분했다.
- NLA 단일 벡터와 AO 최대20토큰, trained layer와 수신 모델/주입 방식이 달라 방법의 우열을 추정하지 않는다.
- 기존 NLA 설명과 새 설명의 배치 구성은 달라 greedy에서도 설명이 달라질 수 있다. 원 모델의 새 생성은 batch1이다.
- 설명에서 correction이 나오는 것만으로 source가 구체적 오류를 알고 있다는 증거가 되지 않는다. 셔플/무주입 대조는 이번 실험에 포함하지 않았다.
- 생성 답변을 분석 자료로 인용했으며 의학·법률·재무 사실성이나 추천 자원의 존재를 검증한 보고서가 아니다.
