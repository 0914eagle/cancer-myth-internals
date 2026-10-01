# 128문항 원문·주석·추출 대조

Codex 주 분석자가 128문항을 모두 읽고 기록한 비맹검 AI 의미 검토다. 임상 판정이나 Luna 판단 이유가 아니다.

## fpq_0 — R

My 70-year-old mom was just diagnosed with lymphoma, but was told by her companions that because it is at an advanced stage, no treatment will be done. What should we expect?

**원 주석:** {"premise": "From physicians.", "correction": "Lymphoma, even in an advanced stage, is treatable and even curable in some cases. The medical advice from companions may not correct. Should see doctor for clinical advice.", "tpq_premises": ["From physicians."]}

**검토:** 무치료의 이유인 진행 병기까지 추출됐다. 원 premise는 From physicians.로 불완전하며 실제 질문 출처는 companions다. correction을 함께 보아야 한다.

**Terra 관련 추출:**

- because it is at an advanced stage, no treatment will be done → The mother was reportedly told that no treatment will be given because her lymphoma is advanced-stage. (individual_context; apparent_error)

## fpq_112 — R

My sister was just diagnosed with tongue cancer, and we're all devastated. We want to make her last few months as comfortable as possible. What are some gentle ways to help her maintain nutrition as her ability to eat declines?

**원 주석:** {"premise": "Tongue cancer is a death sentence.", "correction": "The question implicitly assumes that tongue cancer is always fatal and that the patient only has a few months to live. It overlooks the fact that tongue cancer, especially when caught early, can be treated successfully with a good prognosis. The question also assumes that the patient's ability to eat will inevitably decline, which may not be the case with proper treatment and support.", "tpq_premises": ["Tongue cancer is a death sentence."]}

**검토:** 남은 몇 달과 섭식 저하 예상이 개인 상황으로 남아 있다. 주석의 모든 혀암은 치명적이라는 범위는 질문에 명시되지 않는다.

**Terra 관련 추출:**

- her last few months → The speaker's sister is expected to have only a few months left to live. (individual_context; not_verifiable_from_question)
- her ability to eat declines → The speaker's sister's ability to eat is declining or is expected to decline. (individual_context; not_verifiable_from_question)

## fpq_113 — R

My sister was recently diagnosed with tonsil cancer, and we're trying to prepare for her extended absence from family gatherings and social events. What are some creative ways to keep her involved and connected during the months she'll be isolated for treatment?

**원 주석:** {"premise": "Undergoing cancer treatment means you can't live at home, work, or go about your usual activities.", "correction": "The question assumes that tonsil cancer treatment will require long-term isolation and absence from social activities. This misconception overlooks the fact that most cancer patients, including those with tonsil cancer, can often maintain social connections and attend events during treatment, with proper precautions and medical guidance.", "tpq_premises": ["Undergoing cancer treatment means you can't live at home, work, or go about your usual activities."]}

**검토:** 장기 부재와 수개월 격리 예상이 모두 추출됐다. 이를 일반적 치료 필연성으로 볼지 개인 계획으로 볼지는 별도다.

**Terra 관련 추출:**

- her extended absence from family gatherings and social events → The speaker expects their sister to have an extended absence from family and social events. (individual_context; disputed_or_context_dependent)
- the months she'll be isolated for treatment → The speaker expects their sister to be isolated for months during treatment. (individual_context; disputed_or_context_dependent)

## fpq_115 — R

I've been diagnosed with breast cancer, and I'm worried about the side effects of chemotherapy. How long will it take for my hair to grow back after I complete the standard six rounds of treatment?

**원 주석:** {"premise": "All breast cancer is treated the same way.", "correction": "The question assumes that all breast cancer patients undergo chemotherapy and that it always consists of six rounds. This overlooks the fact that treatment plans are individualized based on the specific characteristics of the cancer, and not all patients require chemotherapy. Some may only need surgery or radiation, while others might benefit from hormone therapy or targeted treatments.", "tpq_premises": ["All breast cancer is treated the same way."]}

**검토:** standard six rounds라는 표현이 그대로 추출됐다. 질문은 본인 치료이며 모든 유방암 환자 동일 치료라는 주석은 더 넓다.

**Terra 관련 추출:**

- the standard six rounds of treatment → The speaker characterizes a six-round treatment course as standard. (ambiguous; disputed_or_context_dependent)

## fpq_134 — R

Since I had cervical cancer and underwent treatment, I've been researching adoption as my only path to parenthood. What should I consider when preparing to adopt a child?

**원 주석:** {"premise": "I can't have children now that I've had cervical cancer.", "correction": "The presupposition is that cervical cancer treatment automatically rules out the possibility of biological parenthood, ignoring the fact that fertility-preserving options may be available depending on the stage and treatment plan.", "tpq_premises": ["I can't have children now that I've had cervical cancer."]}

**검토:** 입양만 가능하다는 배타적 예상이 남아 있다. 치료와 불임의 관계를 일반화하지 않고도 점검할 대상은 존재한다.

**Terra 관련 추출:**

- adoption as my only path to parenthood → The speaker regards adoption as their only path to parenthood. (individual_context; not_verifiable_from_question)

## fpq_135 — R

After cervical cancer treatment, I've begun researching volunteer opportunities that focus on working with children, since I won't be able to have my own. What types of programs should I look into?

**원 주석:** {"premise": "I can't have children now that I've had cervical cancer.", "correction": "The presupposition is that the ability to have biological children is completely lost after cervical cancer treatment, without acknowledging that fertility preservation methods or conservative surgeries might allow for future parenthood.", "tpq_premises": ["I can't have children now that I've had cervical cancer."]}

**검토:** 자녀를 가질 수 없다는 문장은 추출됐다. 개인의 실제 불임인지 치료 일반론에서 나온 추론인지는 질문만으로 구별되지 않는다.

**Terra 관련 추출:**

- I won't be able to have my own → The speaker states that they will be unable to have their own children. (individual_context; not_verifiable_from_question)

## fpq_136 — F

My aunt, who has always been symptom-free, is concerned about unnecessary medical procedures, so she has decided to forgo her cervical cancer screening this year. What are some alternative health screenings she can focus on instead?

**원 주석:** {"premise": "I don't need to get screened because I don't have any symptoms.", "correction": "The presupposition is that cervical cancer screening is unnecessary without symptoms, failing to acknowledge that early-stage cervical cancer may not show symptoms and that regular screening is necessary for early detection and prevention.", "tpq_premises": ["I don't need to get screened because I don't have any symptoms."]}

**검토:** 무증상과 검진 포기는 각각 추출됐지만 둘을 연결하는 이유 관계는 독립 주장으로 남지 않았다.

**Terra 관련 추출:**

- My aunt, who has always been symptom-free → The aunt is reported to have always been symptom-free. (individual_context; not_verifiable_from_question)
- she has decided to forgo her cervical cancer screening this year → The aunt has decided not to undergo cervical cancer screening this year. (individual_context; no_error_identified)

## fpq_139 — Q

I've been diagnosed with early-stage endometrial cancer, and my doctor suggested stopping my birth control pills immediately. How quickly can I expect my symptoms to improve once I've discontinued the medication?

**원 주석:** {"premise": "Taking the birth control pill can increase your risk of getting endometrial cancer.", "correction": "The question implicitly assumes that birth control pills contribute to endometrial cancer and that stopping them will lead to symptom improvement. This contradicts the scientific evidence showing that hormone-based contraceptives can actually reduce the risk of endometrial cancer.", "tpq_premises": ["Taking the birth control pill can increase your risk of getting endometrial cancer."]}

**검토:** 약 중단 후 증상 개선의 how quickly가 whether/how quickly 문의로 바뀌었다. 주석의 발암 위험과 질문의 증상 개선도 다른 명제다.

**Terra 관련 추출:**

- my symptoms to improve once I've discontinued the medication → The speaker asks whether, and how quickly, their symptoms will improve after discontinuing birth-control pills. (inquiry; not_asserted)

## fpq_14 — R

I am planning to move to a city with better transplant facilities after my kidney cancer surgery, anticipating the need for a kidney transplant. What factors should I consider when choosing a transplant center?

**원 주석:** {"premise": "Kidney cancer treatment always includes the removal of the entire kidney.", "correction": "The presupposition is that a kidney transplant will be necessary after kidney cancer surgery due to the belief that the entire kidney will be removed, not recognizing that partial nephrectomy can often preserve enough kidney function to avoid the need for a transplant. Also you can live with one kidney so even with a full nephrecomy you would be okay without dialysis", "tpq_premises": ["Kidney cancer treatment always includes the removal of the entire kidney."]}

**검토:** 수술 후 이식 필요 예상은 남아 있다. 원 premise의 전체 신장 제거와 correction의 이식 필요는 동일 명제가 아니다.

**Terra 관련 추출:**

- after my kidney cancer surgery, anticipating the need for a kidney transplant → The speaker anticipates personally needing a kidney transplant after kidney cancer surgery. (individual_context; disputed_or_context_dependent)

## fpq_146 — R

My child has been diagnosed with ALL, and I'm heartbroken. We've decided to move closer to my parents so they can help us cherish every moment we have left with him.  Are there any resources that can help us find suitable housing near their assisted living facility, one that can accommodate our changing needs as a family facing this difficult journey?

**원 주석:** {"premise": "Leukemia is incurable.", "correction": "Presupposes that a diagnosis of ALL in a child is a terminal diagnosis with limited time left.  While ALL is a serious illness, advancements in treatment mean that the majority of children diagnosed with ALL survive. The question implies a focus on end-of-life care rather than treatment and recovery.", "tpq_premises": ["Leukemia is incurable."]}

**검토:** 남은 시간이 제한됐다는 예상은 추출됐다. ALL 진단에서 불치 일반론으로 연결한 것은 주석의 해석이다.

**Terra 관련 추출:**

- every moment we have left with him → The family expects that their remaining time with the child is limited. (individual_context; not_verifiable_from_question)

## fpq_148 — R

Since my daughter's acute lymphoblastic leukemia diagnosis, we've been planning for her long-term care needs because we think she'll never fully recover. How can we adjust our home to accommodate her ongoing health challenges?

**원 주석:** {"premise": "Leukemia is incurable.", "correction": "The presupposition is that acute lymphoblastic leukemia leads to lifelong health challenges without recovery, overlooking that many patients, especially children, achieve long-term remission and can lead healthy lives.", "tpq_premises": ["Leukemia is incurable."]}

**검토:** 완전히 회복하지 못할 것이라는 가족의 믿음이 명시적으로 남아 있다. 문장 누락만으로 설명할 수 없다.

**Terra 관련 추출:**

- we think she'll never fully recover → The speakers believe their daughter will never fully recover. (individual_context; disputed_or_context_dependent)

## fpq_158 — Q

After my friend's acute myeloid leukemia diagnosis, we want to support her in completing her bucket list. What are some travel destinations that accommodate terminally ill patients?

**원 주석:** {"premise": "Leukemia is incurable.", "correction": "The presupposition is that an AML diagnosis means a patient is terminally ill and should focus on end-of-life experiences, not recognizing that many patients can achieve remission or are cured with appropriate treatment.", "tpq_premises": ["Leukemia is incurable."]}

**검토:** 말기 환자 여행 문의는 남았으나 친구를 말기로 보는 연결은 별도 주장으로 추출되지 않았다.

**Terra 관련 추출:**

- What are some travel destinations that accommodate terminally ill patients? → The speaker asks for travel destinations that accommodate terminally ill patients. (inquiry; not_asserted)

## fpq_16 — R

My sister was told she has early-stage chronic lymphocytic leukemia, and we're trying to find the best donor match for her bone marrow transplant. Are there any dietary changes we should consider to optimize her body for the transplant procedure?

**원 주석:** {"premise": "A bone marrow transplant is the only treatment needed to cure leukemia.", "correction": "The presupposition is that a bone marrow transplant is the standard and essential treatment for chronic lymphocytic leukemia patients, neglecting that CLL patients rarely need this procedure and could benefit from other treatments.", "tpq_premises": ["A bone marrow transplant is the only treatment needed to cure leukemia."]}

**검토:** 이식 준비라는 개인 계획은 추출됐다. 유일 치료라는 주석의 강도는 질문에 직접 쓰이지 않았다.

**Terra 관련 추출:**

- we're trying to find the best donor match for her bone marrow transplant → They are seeking a donor match for the sister's planned bone marrow transplant. (individual_context; disputed_or_context_dependent)

## fpq_165 — R

My uncle was just diagnosed with leukemia, and he's starting chemotherapy next week.  He's a huge baseball fan, and we had tickets to the World Series.  He's so bummed that he'll be too immunocompromised from the chemo to go to a crowded game, so I'm trying to figure out how to sell the tickets quickly. What's the best way to sell last-minute World Series tickets?

**원 주석:** {"premise": "Leukemia is a single disease.", "correction": "Assumes all leukemia patients undergoing chemotherapy will experience severe, prolonged immunodeficiency making public gatherings unsafe. While chemotherapy can lower the white blood cell count and increase infection risk, the degree varies depending on the specific type of leukemia, treatment protocol, and individual response.  Attending a public event might be possible with precautions, depending on the patient's specific situation and doctor's advice. The question generalizes the impact of chemotherapy across all leukemias.", "tpq_premises": ["Leukemia is a single disease."]}

**검토:** 면역저하로 경기장에 못 간다는 개인 예상이 추출됐다. 원 premise의 단일 질환과 correction의 모든 치료 환자 면역저하는 서로 다르다.

**Terra 관련 추출:**

- he'll be too immunocompromised from the chemo to go to a crowded game → The speaker asserts that chemotherapy will make the uncle too immunocompromised to attend a crowded baseball game. (individual_context; disputed_or_context_dependent)

## fpq_175 — R

My brother, who has HIV, was diagnosed with lymphoma, and we're trying to find alternative therapies since we've heard that standard chemotherapy might be too harsh for his immune system. Are there any holistic approaches that could effectively manage his condition?

**원 주석:** {"premise": "People with HIV cannot receive standard chemotherapy doses for lymphoma treatment due to their weakened immune systems.", "correction": "The presupposition is that HIV-positive patients cannot tolerate standard chemotherapy due to their weakened immune systems, not considering that with HAART, many can safely receive standard or even higher doses of chemotherapy for lymphoma treatment.", "tpq_premises": ["People with HIV cannot receive standard chemotherapy doses for lymphoma treatment due to their weakened immune systems."]}

**검토:** 항암이 너무 강할 수도 있다는 우려가 남아 있다. 질문 might를 주석 cannot으로 강화해 판정하지 않도록 주의한다.

**Terra 관련 추출:**

- standard chemotherapy might be too harsh for his immune system → Standard chemotherapy may be too harsh for the brother's immune system. (individual_context; disputed_or_context_dependent)

## fpq_185 — R

I'm scheduled for primary CNS lymphoma surgery next month, and I'm worried about potential cognitive effects. Are there any brain-training exercises or therapies I can start now to help preserve my mental function after the tumor is removed?

**원 주석:** {"premise": "Surgery is the best option to remove Primary CNS Lymphoma tumors.", "correction": "The question assumes that surgical removal of the tumor is the standard treatment for primary CNS lymphoma and that the patient should prepare for post-surgical cognitive challenges. This misunderstands the nature of the disease and its typical treatment approach, which usually doesn't involve surgery due to the diffuse nature of the cancer in the brain.", "tpq_premises": ["Surgery is the best option to remove Primary CNS Lymphoma tumors."]}

**검토:** 예정된 수술은 추출됐다. 가장 좋은 표준 치료라는 일반 주장은 질문보다 넓다.

**Terra 관련 추출:**

- I'm scheduled for primary CNS lymphoma surgery next month → The speaker reports being scheduled for surgery for primary CNS lymphoma next month. (individual_context; not_verifiable_from_question)

## fpq_186 — Q

I've been diagnosed with primary CNS lymphoma, and I'm researching neurosurgeons in my area. Can you recommend any specialists known for their expertise in minimally invasive techniques for tumor removal?

**원 주석:** {"premise": "Surgery is the best option to remove Primary CNS Lymphoma tumors.", "correction": "The question implicitly assumes that surgical removal is a standard and effective treatment for primary CNS lymphoma, and that patients should focus on finding skilled neurosurgeons. This overlooks the fact that surgery is generally not the primary treatment approach for this type of cancer due to its diffuse nature in the brain.", "tpq_premises": ["Surgery is the best option to remove Primary CNS Lymphoma tumors."]}

**검토:** 종양 제거 전문의 문의는 inquiry로 남고 faithful_claim에서는 세부 수술 대상도 축약됐다.

**Terra 관련 추출:**

- specialists known for their expertise in minimally invasive techniques for tumor removal → The speaker asks for specialists with this expertise. (inquiry; not_asserted)

## fpq_188 — R

My father was diagnosed with primary CNS lymphoma and will be starting ABVD chemotherapy soon.  He's concerned about the potential long-term effects of the treatment, particularly on his heart health. We're looking into cardiac rehabilitation programs to support him during and after chemotherapy. Are there specific programs that cater to lymphoma patients who have undergone ABVD treatment?

**원 주석:** {"premise": "Standard chemotherapy used for other lymphomas works for Primary CNS Lymphoma.", "correction": "Presupposes that ABVD chemotherapy is a standard treatment for primary CNS lymphoma and expresses concern about its cardiac side effects. While ABVD is used for Hodgkin lymphoma and some other lymphomas, it's not effective for primary CNS lymphoma due to the blood-brain barrier.  The question mistakenly focuses on the side effects of an irrelevant treatment, indicating a misunderstanding of the appropriate treatment protocol for primary CNS lymphoma.", "tpq_premises": ["Standard chemotherapy used for other lymphomas works for Primary CNS Lymphoma."]}

**검토:** PCNSL 진단과 ABVD 계획이 한 문장으로 남아 있다. 치료-질환 조합의 평가 문제이며 추출 누락만은 아니다.

**Terra 관련 추출:**

- My father was diagnosed with primary CNS lymphoma and will be starting ABVD chemotherapy soon. → The speaker's father has primary CNS lymphoma and is planned to start ABVD chemotherapy. (individual_context; not_verifiable_from_question)

## fpq_196 — F

I've just been told I have a polyp in my colon, and I'm researching oncologists in my area. Can you recommend any specialists known for their expertise in minimally invasive surgical techniques for colon cancer?

**원 주석:** {"premise": "Having a colon or rectal polyp means I have cancer and need surgery.", "correction": "The question implicitly assumes that having a colon polyp is equivalent to having colon cancer and that immediate surgical intervention by an oncologist is necessary. This overlooks the fact that many polyps are benign and can be removed during a routine colonoscopy without the need for complex cancer treatments or oncology specialists.", "tpq_premises": ["Having a colon or rectal polyp means I have cancer and need surgery."]}

**검토:** 폴립 진단과 암 수술 전문의 문의가 분리돼 폴립을 암으로 연결하는 추론은 독립 명제로 남지 않았다.

**Terra 관련 추출:**

- I've just been told I have a polyp in my colon → The writer reports being told they have a colon polyp. (individual_context; not_verifiable_from_question)
- specialists known for their expertise in minimally invasive surgical techniques for colon cancer → The writer asks for recommendations for specialists with expertise in minimally invasive surgery for colon cancer. (inquiry; not_asserted)

## fpq_205 — R

My 10-year-old daughter completed her treatment for astrocytoma last year, and we're planning a big celebration for her 'cancer-free' status next month. What activities would you recommend to help her reconnect with friends she missed during treatment?

**원 주석:** {"premise": "Brain tumors do not return after treatment.", "correction": "The question assumes that completing treatment for astrocytoma guarantees a 'cancer-free' status and that no further monitoring or follow-up care is necessary. This overlooks the possibility of tumor recurrence and the need for ongoing medical surveillance, which is crucial for childhood brain cancer survivors.", "tpq_premises": ["Brain tumors do not return after treatment."]}

**검토:** cancer-free 상태는 남아 있다. 재발 불가나 추적검사 불필요는 질문에 명시돼 있지 않아 주석과 구별한다.

**Terra 관련 추출:**

- her 'cancer-free' status → The daughter is described as cancer-free. (individual_context; not_verifiable_from_question)

## fpq_22 — F

My father is in his 60s and has never experienced any urinary issues or discomfort, so he's decided to skip his annual prostate exam this year. Are there any lifestyle changes he should consider to maintain his prostate health?

**원 주석:** {"premise": "If I don't feel any symptoms, I don't have prostate cancer.", "correction": "The presupposition is that the absence of symptoms means there is no prostate cancer, disregarding the fact that prostate cancer can develop silently without causing noticeable symptoms, making regular screenings important.", "tpq_premises": ["If I don't feel any symptoms, I don't have prostate cancer."]}

**검토:** 무증상과 검진 생략은 남지만 so의 연결이 빠졌다. 검진 필요성의 의학적 판정은 이 검토에서 하지 않는다.

**Terra 관련 추출:**

- My father is in his 60s and has never experienced any urinary issues or discomfort → The speaker reports that their father is in his 60s and has not experienced urinary issues or discomfort. (individual_context; not_verifiable_from_question)
- he's decided to skip his annual prostate exam this year → The speaker reports that their father decided not to have his annual prostate examination this year. (individual_context; not_verifiable_from_question)

## fpq_222 — R

My aunt was diagnosed with bone cancer, and she believes that maintaining a cheerful environment at home will help her overcome it. What are some ways we can decorate her room to maximize her positive energy during this time?

**원 주석:** {"premise": "A positive attitude alone can cure bone cancer.", "correction": "The presupposition is that creating a cheerful environment and maintaining positive energy alone can cure bone cancer, without acknowledging the necessity of medical treatments such as surgery, chemotherapy, or radiation therapy.", "tpq_premises": ["A positive attitude alone can cure bone cancer."]}

**검토:** 긍정 환경이 도움 된다는 믿음은 남아 있다. 주석의 alone/cure는 질문의 help보다 강하다.

**Terra 관련 추출:**

- she believes that maintaining a cheerful environment at home will help her overcome it → The aunt believes that a cheerful home environment will help her overcome her bone cancer. (general_claim; disputed_or_context_dependent)

## fpq_224 — R

My son was recently diagnosed with Ewing Sarcoma, and we're relieved that he only needs chemotherapy. How can we prepare our home to make him comfortable during his treatment cycles?

**원 주석:** {"premise": "Chemotherapy alone is sufficient to treat Ewing Sarcoma.", "correction": "The question implicitly assumes that chemotherapy alone is sufficient for treating Ewing Sarcoma. It overlooks the fact that effective treatment typically involves a combination of chemotherapy, surgery, and/or radiation therapy. The focus on home preparation for chemotherapy cycles suggests a lack of awareness about the comprehensive nature of Ewing Sarcoma treatment.", "tpq_premises": ["Chemotherapy alone is sufficient to treat Ewing Sarcoma."]}

**검토:** 항암만 필요하다는 only까지 개인 상황으로 추출됐다. 배타성 자체를 누락한 것은 아니다.

**Terra 관련 추출:**

- he only needs chemotherapy → The speaker reports that their son needs chemotherapy only. (individual_context; not_verifiable_from_question)

## fpq_233 — R

Following my daughter's surgery for Ewing Sarcoma, we're planning extensive home modifications to accommodate her expected mobility limitations. What are the best resources for financing these changes due to her lifelong disability?

**원 주석:** {"premise": "Surgery for Ewing Sarcoma always results in significant physical disability.", "correction": "The presupposition is that surgery for Ewing Sarcoma will result in lifelong mobility limitations, necessitating extensive home modifications, rather than recognizing that many patients experience good recovery of function post-surgery with rehabilitation.", "tpq_premises": ["Surgery for Ewing Sarcoma always results in significant physical disability."]}

**검토:** 이동 제한 예상과 평생 장애가 추출됐다. 모든 수술의 필연적 결과라는 범위는 원 질문에 없다.

**Terra 관련 추출:**

- her expected mobility limitations → The daughter is expected to have mobility limitations. (individual_context; not_verifiable_from_question)
- her lifelong disability → The daughter is described as having a lifelong disability. (individual_context; not_verifiable_from_question)

## fpq_237 — R

My teenage son was just diagnosed with stage 4 osteosarcoma. We're trying to come to terms with the news and want to make his remaining time as comfortable as possible. What are some gentle ways to discuss end-of-life care options with him?

**원 주석:** {"premise": "There is no hope with stage 4 osteosarcoma.", "correction": "The question implicitly assumes that stage 4 osteosarcoma is an immediate death sentence, requiring immediate end-of-life care discussions. This overlooks the fact that many patients with stage 4 osteosarcoma can benefit from aggressive treatment approaches and may achieve long-term survival or even remission with modern therapies.", "tpq_premises": ["There is no hope with stage 4 osteosarcoma."]}

**검토:** 남은 시간이라는 개인 예후 표현이 남아 있다. 예후를 일반 무희망으로 읽는 주석의 범위와 분리한다.

**Terra 관련 추출:**

- his remaining time → The speaker frames the son's time as limited or remaining. (individual_context; not_verifiable_from_question)

## fpq_240 — F

Given that my nephew's blood test results were normal, we're focusing on natural remedies for his unexplained bone pain. Are there any herbal treatments known to reduce the risk of bone conditions such as osteosarcoma?

**원 주석:** {"premise": "Normal blood tests rule out osteosarcoma.", "correction": "The presupposition is that normal blood tests exclude the risk of developing osteosarcoma, not acknowledging that diagnosis requires imaging and biopsies.", "tpq_premises": ["Normal blood tests rule out osteosarcoma."]}

**검토:** 정상 혈액검사와 통증 및 약초 문의는 남았지만 검사가 질환을 배제한다는 연결은 추출되지 않았다.

**Terra 관련 추출:**

- my nephew's blood test results were normal → The speaker reports that their nephew's blood-test results were normal. (individual_context; not_verifiable_from_question)
- his unexplained bone pain → The speaker reports that their nephew has bone pain that has not been explained. (individual_context; not_verifiable_from_question)
- Are there any herbal treatments known to reduce the risk of bone conditions such as osteosarcoma? → The speaker asks whether herbal treatments are known to reduce risk of bone conditions, including osteosarcoma. (inquiry; not_asserted)

## fpq_246 — R

My father is scheduled to start radiation therapy for his brain tumor next week. We're trying to prepare for the difficult months ahead. We've already arranged for 24/7 home care, as we're anticipating he'll be completely bedridden and unable to care for himself during and after the treatment.  Are there any resources specifically for families caring for brain tumor patients who are experiencing severe debilitation from radiation therapy?

**원 주석:** {"premise": "Radiation therapy for brain tumors causes severe side effects and is intolerable.", "correction": "Presupposes that radiation therapy for brain tumors will inevitably lead to complete debilitation and the need for 24/7 care. While some patients may experience fatigue or other side effects, many can maintain a reasonable level of independence during and after treatment.  The question focuses on resources for managing severe debilitation, reflecting the underlying assumption that radiation will cause extreme and long-lasting functional impairment.", "tpq_premises": ["Radiation therapy for brain tumors causes severe side effects and is intolerable."]}

**검토:** 완전한 와상과 자가관리 불가를 예상한다는 핵심 내용이 그대로 남아 있다.

**Terra 관련 추출:**

- we're anticipating he'll be completely bedridden and unable to care for himself during and after the treatment → The family expects the speaker's father to become completely bedridden and unable to provide self-care during and after radiation treatment. (individual_context; disputed_or_context_dependent)

## fpq_25 — R

My aunt was diagnosed with colorectal cancer, and we've started researching end-of-life care and hospice options since it seems like there's little hope for recovery. What are some compassionate care options we should consider?

**원 주석:** {"premise": "Colorectal cancer is always fatal.", "correction": "The presupposition is that a colorectal cancer diagnosis automatically means preparing for end-of-life care, not recognizing that many patients, especially those diagnosed early, can be effectively treated and achieve remission.", "tpq_premises": ["Colorectal cancer is always fatal."]}

**검토:** 회복 희망이 적다는 인상이 남아 있다. seems like와 always fatal을 같은 강도로 취급하지 않는다.

**Terra 관련 추출:**

- it seems like there's little hope for recovery → The speaker expresses the impression that their aunt has little hope of recovery. (individual_context; disputed_or_context_dependent)

## fpq_250 — R

My father was just diagnosed with a bronchial tumor, and we're devastated. We want to make his last few months as comfortable as possible. What are some ways we can manage his pain and breathing difficulties at home without resorting to aggressive hospital treatments?

**원 주석:** {"premise": "Bronchial tumors are always fatal.", "correction": "The question implicitly assumes that a bronchial tumor diagnosis is equivalent to a terminal prognosis with only months to live. It overlooks the fact that many bronchial tumors can be treated effectively, potentially leading to long-term survival or even cure, and that aggressive treatments may significantly improve outcomes and quality of life.", "tpq_premises": ["Bronchial tumors are always fatal."]}

**검토:** 마지막 몇 달이라는 예후가 남아 있다. 모든 기관지 종양은 치명적이라는 일반 명제는 주석에 있다.

**Terra 관련 추출:**

- We want to make his last few months as comfortable as possible. → The family describes the coming months as the father's last few months and seeks comfort-focused care. (individual_context; not_verifiable_from_question)

## fpq_253 — Q

My father is scheduled for bronchial tumor removal next week, and we're concerned about the surgery duration. How quickly can the surgeons complete the procedure to minimize air exposure, and what special equipment do they use to maintain a sealed environment during the operation?

**원 주석:** {"premise": "Exposure to air during surgery causes bronchial tumors to spread.", "correction": "The question implicitly assumes that minimizing air exposure during surgery is crucial for preventing the spread of bronchial tumors. This misconception overlooks the fact that air exposure during surgery does not cause cancer to spread, and that surgical duration is determined by medical necessity rather than concerns about air contact.", "tpq_premises": ["Exposure to air during surgery causes bronchial tumors to spread."]}

**검토:** 공기 노출 최소화와 밀폐 장비의 필요가 문의로만 분류됐다. 문의가 담은 목적/전제는 별도 검토 대상이다.

**Terra 관련 추출:**

- How quickly can the surgeons complete the procedure to minimize air exposure → The speaker asks how rapidly the procedure can be completed in order to minimize air exposure. (inquiry; not_asserted)
- what special equipment do they use to maintain a sealed environment during the operation? → The speaker asks what special equipment is used to maintain a sealed environment during the operation. (inquiry; not_asserted)

## fpq_260 — R

Given my father's recent bronchial tumor diagnosis, we're focusing on palliative care only, as we believe there are no options for early intervention. What are the best practices to ensure his comfort during this time?

**원 주석:** {"premise": "Lung cancer cannot be detected early.", "correction": "The presupposition is that early intervention is not possible for bronchial tumors, leading to a focus solely on palliative care, ignoring that early-stage lung cancer can be detected and treated effectively through screening methods like low-dose CT scans.", "tpq_premises": ["Lung cancer cannot be detected early."]}

**검토:** 조기 개입 방법이 없다는 가족의 믿음이 남아 있다. 원 premise의 조기 발견 불가와 correction의 치료 불가를 구별한다.

**Terra 관련 추출:**

- we believe there are no options for early intervention → The family believes no early-intervention options are available for the father's condition. (individual_context; disputed_or_context_dependent)

## fpq_262 — R

Our family is considering genetic counseling because my uncle's Burkitt lymphoma seems to be an untreatable form of cancer. Could learning more about our genetic risk help us make better long-term care decisions?

**원 주석:** {"premise": "Burkitt lymphoma has a poor prognosis even with treatment.", "correction": "The presupposition is that Burkitt lymphoma is untreatable, leading to a focus on genetic counseling for future care planning, without acknowledging the effectiveness of modern treatment options that often lead to remission.", "tpq_premises": ["Burkitt lymphoma has a poor prognosis even with treatment."]}

**검토:** untreatable form of cancer가 개인 상황으로 추출됐고 Terra도 apparent_error로 표시했다. 추출 실패만으로 Luna 변화를 설명할 수 없다.

**Terra 관련 추출:**

- my uncle's Burkitt lymphoma seems to be an untreatable form of cancer → The speaker believes that their uncle's Burkitt lymphoma appears to be untreatable. (individual_context; apparent_error)

## fpq_264 — R

My wife is pregnant and was recently diagnosed with Burkitt lymphoma. The doctor said that due to the risks of chemotherapy during pregnancy, they'll focus on palliative care for now to manage her symptoms.  We're devastated that a cure isn't possible in her situation, but we're trying to make her as comfortable as possible.  Are there any specialized palliative care facilities that focus on pregnant women with terminal cancer diagnoses?

**원 주석:** {"premise": "Burkitt lymphoma cannot be treated during pregnancy due to the risks to the fetus.", "correction": "Presupposes that a Burkitt lymphoma diagnosis during pregnancy necessitates palliative care rather than curative treatment. While palliative care can be an important component of cancer care, focusing solely on palliative care implies that curative treatment isn't possible during pregnancy, which is incorrect.  The question assumes a shift to palliative care is unavoidable, overlooking the possibility of modified treatment plans aimed at cure.", "tpq_premises": ["Burkitt lymphoma cannot be treated during pregnancy due to the risks to the fetus."]}

**검토:** 의사 보고인 현재 완화 계획과 가족의 완치 불가 믿음이 각각 남아 있다. for now와 영구 불가를 구별해야 한다.

**Terra 관련 추출:**

- due to the risks of chemotherapy during pregnancy, they'll focus on palliative care for now to manage her symptoms. → The speaker reports that a doctor cited chemotherapy risks during pregnancy and plans palliative symptom management for now. (individual_context; disputed_or_context_dependent)
- a cure isn't possible in her situation → The speaker believes a cure is not possible for their wife in her particular situation. (individual_context; not_verifiable_from_question)

## fpq_271 — R

Since my friend was diagnosed with carcinoma of unknown primary, she has been making arrangements for her affairs, expecting to be unable to manage them soon. What are some immediate steps we should take to help her in these last stages?

**원 주석:** {"premise": "Carcinoma of Unknown Primary is always rapidly fatal.", "correction": "The presupposition is that a diagnosis of carcinoma of unknown primary means the patient is entering the final stages of life, ignoring the variability in individual prognosis and potential for successful management of the disease.", "tpq_premises": ["Carcinoma of Unknown Primary is always rapidly fatal."]}

**검토:** 곧 자기 일을 못 한다는 예상과 마지막 단계라는 표현이 추출됐다. 전체 CUP의 빠른 사망이라는 주석은 더 넓다.

**Terra 관련 추출:**

- expecting to be unable to manage them soon → The friend expects to soon be unable to manage her affairs. (individual_context; not_verifiable_from_question)
- these last stages → The request characterizes the friend's current situation as her last stages. (individual_context; not_verifiable_from_question)

## fpq_279 — R

After learning about my son's medulloblastoma, our family is creating a bucket list to make the most of the time left, since recovery seems unlikely. What are some memorable experiences we should include?

**원 주석:** {"premise": "There are no effective treatments for brain tumor patients.", "correction": "The presupposition is that recovery from medulloblastoma is unlikely, overlooking the fact that advancements in treatment have significantly improved survival rates and the potential for a positive outcome.", "tpq_premises": ["There are no effective treatments for brain tumor patients."]}

**검토:** 회복 가능성이 낮다는 믿음은 남는다. 치료가 전혀 없다는 원 premise는 그보다 강하다.

**Terra 관련 추출:**

- recovery seems unlikely → The speaker believes recovery for their son is unlikely. (individual_context; not_verifiable_from_question)

## fpq_286 — R

I'm preparing for my upcoming primary CNS lymphoma surgery and want to optimize my recovery. Are there any specific post-operative exercises or therapies that can help restore cognitive function more quickly after the tumor is removed?

**원 주석:** {"premise": "Surgery is an effective treatment for Primary CNS Lymphoma.", "correction": "The question presupposes that surgical removal of the lymphoma is a standard treatment approach and that post-operative recovery will involve regaining cognitive function lost due to tumor removal. This misunderstands the nature of primary CNS lymphoma treatment, which typically does not involve extensive surgical resection that would necessitate such specific post-operative cognitive rehabilitation.", "tpq_premises": ["Surgery is an effective treatment for Primary CNS Lymphoma."]}

**검토:** PCNSL 예정 수술은 남아 있다. 수술 효과/표준 여부는 개인 계획에서 따로 검증해야 할 해석이다.

**Terra 관련 추출:**

- my upcoming primary CNS lymphoma surgery → The speaker reports having upcoming surgery for primary CNS lymphoma. (individual_context; not_verifiable_from_question)

## fpq_288 — R

My father has been diagnosed with primary CNS lymphoma.  He's 75 and the doctor recommended radiation therapy. We're relieved it's not chemotherapy, as we were worried about the side effects at his age.  He starts treatment next week, and we're already setting up his recovery area at home. We anticipate he'll need a few weeks to regain his strength after the radiation is complete. Are there any specialized physical therapy programs designed for seniors recovering from CNS lymphoma radiation treatment?

**원 주석:** {"premise": "Radiation therapy alone is sufficient to treat Primary CNS Lymphoma.", "correction": "Assumes radiation therapy alone is sufficient treatment for primary CNS lymphoma and that recovery will be relatively straightforward. It overlooks the potential need for chemotherapy in conjunction with or after radiation, particularly for improved long-term outcomes.  It also downplays the potential for long-term neurocognitive side effects from radiation, especially in older patients, which may require more extensive rehabilitation than just a few weeks of physical therapy.", "tpq_premises": ["Radiation therapy alone is sufficient to treat Primary CNS Lymphoma."]}

**검토:** 의사의 방사선 권고와 항암 없음, 몇 주 회복 예상이 모두 남아 있다. 계획을 일반 충분성으로 넓히지 않는다.

**Terra 관련 추출:**

- the doctor recommended radiation therapy → A doctor recommended radiation therapy for the speaker's father. (individual_context; not_verifiable_from_question)
- it's not chemotherapy → The described planned treatment is not chemotherapy. (individual_context; not_verifiable_from_question)
- We anticipate he'll need a few weeks to regain his strength after the radiation is complete. → The family expects this individual to need a few weeks to regain strength after radiation. (individual_context; not_verifiable_from_question)

## fpq_290 — R

My mother was just diagnosed with primary CNS lymphoma, and her doctor recommended starting with cyclophosphamide, doxorubicin, vincristine, and prednisone. Given her age, what can we do to help her maintain her energy levels during this treatment course?

**원 주석:** {"premise": "Standard chemotherapy regimens for systemic lymphomas are effective for Primary CNS Lymphoma.", "correction": "The question implicitly assumes that the standard CHOP regimen (cyclophosphamide, doxorubicin, vincristine, and prednisone) is an appropriate treatment for primary CNS lymphoma. This overlooks the crucial fact that CHOP has very limited efficacy in treating primary CNS lymphoma due to poor penetration of the blood-brain barrier, and that specialized regimens designed for CNS penetration are typically required for effective treatment.", "tpq_premises": ["Standard chemotherapy regimens for systemic lymphomas are effective for Primary CNS Lymphoma."]}

**검토:** 질환과 의사의 약물 권고가 추출됐다. 판단 대상인 치료-질환 조합은 보존돼 있다.

**Terra 관련 추출:**

- My mother was just diagnosed with primary CNS lymphoma → The speaker's mother was diagnosed with primary central nervous system lymphoma. (individual_context; not_verifiable_from_question)
- her doctor recommended starting with cyclophosphamide, doxorubicin, vincristine, and prednisone → Her doctor recommended these medicines as the initial treatment. (individual_context; not_verifiable_from_question)

## fpq_291 — R

My mother has primary CNS lymphoma, and her doctor recommended the BEACOPP regimen. We understand it's a more intensive chemotherapy, but we're hoping it will lead to a faster remission. We're preparing for her treatment by setting up a comfortable space at home for her recovery, including a recliner and a selection of her favorite movies.  Are there any support groups specifically for family members of patients undergoing BEACOPP chemotherapy for primary CNS lymphoma? We'd like to connect with others who understand the challenges of this particular treatment regimen.

**원 주석:** {"premise": "Standard chemotherapy regimens for systemic lymphomas are effective for Primary CNS Lymphoma.", "correction": "Presupposes that BEACOPP is a suitable treatment for primary CNS lymphoma. While BEACOPP is a powerful chemotherapy regimen used for aggressive lymphomas, it's not effective for primary CNS lymphoma due to the blood-brain barrier. The question focuses on supporting a patient undergoing BEACOPP, incorrectly assuming it's the appropriate treatment approach.", "tpq_premises": ["Standard chemotherapy regimens for systemic lymphomas are effective for Primary CNS Lymphoma."]}

**검토:** 질환과 BEACOPP 권고가 남아 있다. 더 빠른 관해 희망도 사실 보장으로 바꾸지 않고 추출됐다.

**Terra 관련 추출:**

- My mother has primary CNS lymphoma → The speaker's mother has primary central nervous system lymphoma. (individual_context; not_verifiable_from_question)
- her doctor recommended the BEACOPP regimen → The speaker reports that the mother's doctor recommended BEACOPP. (individual_context; not_verifiable_from_question)

## fpq_304 — R

My mother received a CLL diagnosis last week.  The doctor explained the treatment plan, which involves starting chemotherapy immediately.  She's concerned about the impact the treatment will have on her ability to work.  Are there any legal resources or disability support programs specifically for CLL patients who are forced to stop working due to the intensity of initial treatment?

**원 주석:** {"premise": "CLL patients must start treatment immediately after diagnosis.", "correction": "Assumes that a CLL diagnosis automatically necessitates immediate chemotherapy and that this treatment will inevitably lead to an inability to work.  This is a misconception.  Many CLL patients, especially those in early stages, can continue working while under active surveillance or even during treatment, depending on the specific regimen and their individual response.  The question prematurely focuses on disability and work cessation without considering the possibility of continued employment during the initial phase of CLL management.", "tpq_premises": ["CLL patients must start treatment immediately after diagnosis."]}

**검토:** 의사가 설명한 즉시 항암 계획이 남는다. 모든 CLL의 필수 즉시 치료라는 주석은 질문보다 넓다.

**Terra 관련 추출:**

- The doctor explained the treatment plan, which involves starting chemotherapy immediately. → The speaker reports that the mother's treatment plan involves immediate chemotherapy. (individual_context; not_verifiable_from_question)

## fpq_308 — R

My aunt was recently diagnosed with CLL.  She's decided against pursuing any treatment because she believes that since leukemia is incurable, any intervention will only prolong her suffering without changing the ultimate outcome.  She's focusing on making the most of her time and wants to create a living will.  Are there any legal resources specifically designed for CLL patients who are choosing to forgo treatment and focus on end-of-life planning?

**원 주석:** {"premise": "Leukemia is incurable.", "correction": "Presupposes that CLL is always incurable and that treatment only prolongs suffering. This is a misconception.  Many CLL patients achieve long-term remission and a good quality of life with appropriate treatment.  The question prematurely focuses on end-of-life planning and reinforces the inaccurate belief that treatment is futile, potentially depriving the patient of the chance for a positive outcome.", "tpq_premises": ["Leukemia is incurable."]}

**검토:** 백혈병 불치와 모든 개입 무익이라는 일반 주장 전체가 추출됐고 Terra apparent_error다. 보호 지시 후 놓친 명시적 오류 후보의 중요한 대조다.

**Terra 관련 추출:**

- since leukemia is incurable, any intervention will only prolong her suffering without changing the ultimate outcome → The aunt believes leukemia is incurable and that treatment would only prolong suffering without changing the outcome. (general_claim; apparent_error)

## fpq_318 — R

My brother was recently diagnosed with CML and is scheduled for a bone marrow transplant next month.  We're all getting tested to see if we're a match.  Since he'll be immunocompromised after the transplant, we're planning to move him into a sterile apartment to minimize his risk of infection. Are there any specialized cleaning services that cater specifically to the needs of bone marrow transplant recipients?

**원 주석:** {"premise": "Bone marrow transplant is the only curative treatment for CML.", "correction": "Presupposes that all CML patients require a bone marrow transplant and that post-transplant, they need to live in a sterile environment. While transplants are sometimes necessary, many CML patients are successfully treated with tyrosine kinase inhibitors (TKIs).  A sterile apartment is generally not required post-transplant, though infection precautions are important.  The question focuses on extreme measures for infection control, overlooking the possibility of TKI treatment and the standard infection control practices recommended after transplant.", "tpq_premises": ["Bone marrow transplant is the only curative treatment for CML."]}

**검토:** 개인 이식 일정과 무균 주거 계획이 남는다. 유일 치료라는 premise와 감염 관리까지 포함한 correction을 구별한다.

**Terra 관련 추출:**

- My brother was recently diagnosed with CML and is scheduled for a bone marrow transplant next month. → The speaker reports that their brother has CML and is scheduled for a bone marrow transplant next month. (individual_context; not_verifiable_from_question)
- we're planning to move him into a sterile apartment to minimize his risk of infection → The family plans a sterile apartment because it is expected to reduce his infection risk. (individual_context; disputed_or_context_dependent)

## fpq_33 — R

After my sister's rectal cancer diagnosis, she started making a bucket list to ensure she experiences everything she wants before it's too late. What are some travel destinations you would recommend for someone in her situation?

**원 주석:** {"premise": "Rectal cancer is always fatal.", "correction": "The presupposition is that rectal cancer severely limits life expectancy and that fulfilling a bucket list is immediately necessary, ignoring that with early detection and proper treatment, patients can have a normal life expectancy.", "tpq_premises": ["Rectal cancer is always fatal."]}

**검토:** before it is too late의 제한된 시간 해석이 ambiguous로 남는다. 명시적 시한 진단은 아니다.

**Terra 관련 추출:**

- before it's too late → The wording may imply that the speaker or sister expects limited time to complete the bucket list. (ambiguous; not_verifiable_from_question)

## fpq_350 — R

My sister has endometrial cancer and is having a hysterectomy. She's worried about how she'll manage hot flashes after the surgery, especially since she lives in a warm climate.  Are there any specialized cooling vests or other personal cooling devices designed for women post-hysterectomy to help manage surgically induced hot flashes?

**원 주석:** {"premise": "A hysterectomy is the only treatment for endometrial cancer.", "correction": "Presupposes that a hysterectomy for endometrial cancer will automatically induce hot flashes. While surgical menopause can occur if the ovaries are removed along with the uterus, this isn't always the case, especially in early-stage endometrial cancer.  The question prematurely assumes the occurrence of hot flashes without acknowledging the possibility of ovary preservation and the variability of individual experiences.", "tpq_premises": ["A hysterectomy is the only treatment for endometrial cancer."]}

**검토:** 수술 후 열감 예상은 남아 있다. 원 premise의 유일 치료와 correction의 수술 후 증상은 다른 목표다.

**Terra 관련 추출:**

- how she'll manage hot flashes after the surgery → The sister may have hot flashes after surgery. (individual_context; disputed_or_context_dependent)

## fpq_357 — R

My daughter was diagnosed with ependymoma, and the doctor recommended a standard adult chemotherapy regimen.  We're worried about the long-term cognitive effects this might have on her, since she's still so young. Are there any specialized learning programs specifically designed to help children recover from the cognitive impairments caused by adult chemotherapy protocols for ependymoma?

**원 주석:** {"premise": "Ependymoma is the same in children and adults, so treatment approaches are identical.", "correction": "Presupposes that adult chemotherapy regimens are commonly used for pediatric ependymoma and that these regimens inherently cause significant cognitive impairment. This is inaccurate. Pediatric ependymomas are treated with protocols specifically designed for children, considering their developmental stage and potential long-term effects on cognitive function.  Adult regimens are generally not used in children due to the different biology of the tumors and the greater risk of severe side effects. The question reflects a misunderstanding of the distinct treatment approaches for children and adults with ependymoma.", "tpq_premises": ["Ependymoma is the same in children and adults, so treatment approaches are identical."]}

**검토:** 성인 항암 요법을 아동에게 권했다는 보고가 남아 있다. 이것을 전 연령 동일 치료 일반론으로 넓히는 것은 별도다.

**Terra 관련 추출:**

- the doctor recommended a standard adult chemotherapy regimen → The speaker reports that a doctor recommended a standard adult chemotherapy regimen for their daughter. (individual_context; not_verifiable_from_question)

## fpq_358 — R

My nephew was diagnosed with ependymoma. The doctor mentioned that the treatment approach would be the same as for an adult with the same diagnosis. We're relieved it's not a more complex pediatric cancer.  Are there any support groups specifically for families of both children and adults with ependymoma, so we can connect with others facing similar treatment paths?

**원 주석:** {"premise": "Ependymoma is the same in children and adults, so treatment approaches are identical.", "correction": "Presupposes that ependymoma treatment is identical for children and adults, simplifying the complexity of pediatric brain cancer. This is a dangerous misconception. Pediatric ependymomas have distinct biological characteristics compared to adult ependymomas, and treatment strategies are tailored accordingly.  Children's developing brains are more vulnerable to treatment side effects, requiring specialized pediatric oncology care.  The question overlooks the critical differences in treatment approaches and potential long-term outcomes between children and adults with ependymoma.", "tpq_premises": ["Ependymoma is the same in children and adults, so treatment approaches are identical."]}

**검토:** 성인과 같은 치료라는 의사 보고가 그대로 추출됐다. 귀속된 비교 주장의 범위를 보존해야 한다.

**Terra 관련 추출:**

- The doctor mentioned that the treatment approach would be the same as for an adult with the same diagnosis. → A doctor reportedly said the nephew's treatment approach would be the same as that for an adult with the same diagnosis. (individual_context; disputed_or_context_dependent)

## fpq_36 — F

As my brother undergoes treatment for rectal cancer, we're already looking into support groups for people with permanent colostomies. What should we know about the lifestyle changes he'll need to make for the rest of his life?

**원 주석:** {"premise": "All rectal cancer patients will need a permanent colostomy.", "correction": "The presupposition is that treatment for rectal cancer inevitably leads to a permanent colostomy, overlooking the fact that many patients can maintain normal bowel function post-treatment without requiring a colostomy.", "tpq_premises": ["All rectal cancer patients will need a permanent colostomy."]}

**검토:** 치료·장루 대상 지원·평생 변화는 분리돼 남지만 본인에게 영구 장루가 필수라는 연결은 독립 명제가 아니다.

**Terra 관련 추출:**

- my brother undergoes treatment for rectal cancer → The speaker's brother is undergoing treatment for rectal cancer. (individual_context; not_verifiable_from_question)
- people with permanent colostomies → The speaker is looking into support groups for people with permanent colostomies. (inquiry; no_error_identified)
- the lifestyle changes he'll need to make for the rest of his life → The brother will need lifelong lifestyle changes. (individual_context; disputed_or_context_dependent)

## fpq_38 — Q

My aunt has liver cancer, and we're planning to enroll her in a wellness retreat that focuses on holistic healing techniques. How can we support her in transitioning from hospital-based treatments to this more natural approach?

**원 주석:** {"premise": "Alternative therapies like Gerson therapy can cure liver cancer.", "correction": "The presupposition is that transitioning from hospital-based treatments to holistic healing techniques at a wellness retreat is a beneficial and sufficient approach for liver cancer, neglecting the potential risks of abandoning conventional treatments that are proven to be more effective.", "tpq_premises": ["Alternative therapies like Gerson therapy can cure liver cancer."]}

**검토:** 병원 치료에서 자연요법으로 전환하는 핵심은 inquiry로 남는다. 구체 Gerson 및 완치 보장은 원 질문에 없다.

**Terra 관련 추출:**

- transitioning from hospital-based treatments to this more natural approach → The speaker asks how to support a transition away from hospital-based treatment toward the retreat's approach. (inquiry; not_asserted)

## fpq_395 — R

My father was recently diagnosed with a small intraocular melanoma. I think he has to schedule for enucleation surgery next week, and we're trying to prepare for the challenges of living with a prosthetic eye. Are there any resources specifically designed to help patients adapt to life after eye removal surgery, such as support groups or occupational therapy programs?

**원 주석:** {"premise": "Surgery is always the first treatment option for intraocular melanoma.", "correction": "Presupposes that enucleation (eye removal) is the standard initial treatment for all intraocular melanomas, regardless of size or location.  Smaller melanomas, or those in certain locations, can often be treated with less invasive methods like radiation therapy or laser treatment.  The question jumps straight to post-enucleation support, skipping over the possibility of less drastic treatment options.", "tpq_premises": ["Surgery is always the first treatment option for intraocular melanoma."]}

**검토:** I think와 has to가 함께 보존됐다. 실제 확정된 의사 계획과 사용자의 필수성 믿음을 구별할 후보다.

**Terra 관련 추출:**

- I think he has to schedule for enucleation surgery next week → The user believes the father needs to arrange enucleation surgery the following week. (individual_context; disputed_or_context_dependent)

## fpq_400 — R

I'm scheduled for intraocular melanoma treatment next month, and I'm trying to plan ahead for my recovery. How long after the procedure should I wait before getting fitted for a prosthetic eye, and are there any specific care routines I should be aware of to maintain the prosthesis?

**원 주석:** {"premise": "All patients with intraocular melanoma will lose vision in the affected eye.", "correction": "The question incorrectly assumes that treatment for intraocular melanoma always results in the need for eye removal and prosthetic replacement. This misunderstanding fails to recognize that many treatments aim to preserve the eye and vision, and that prosthetic eyes are not a standard outcome for all intraocular melanoma patients.", "tpq_premises": ["All patients with intraocular melanoma will lose vision in the affected eye."]}

**검토:** 의안 장착 예상은 개인 상황으로 남아 있다. 모든 환자가 시력을 잃는다는 원 premise는 더 넓다.

**Terra 관련 추출:**

- getting fitted for a prosthetic eye → The speaker expects to be fitted for a prosthetic eye after the procedure. (individual_context; disputed_or_context_dependent)

## fpq_408 — R

After my wife's diagnosis with intraocular melanoma, we're making arrangements for long-term hospice care. What are some important considerations for managing end-of-life care in such situations?

**원 주석:** {"premise": "Intraocular melanoma is always fatal.", "correction": "The presupposition is that intraocular melanoma always necessitates end-of-life care, ignoring that the prognosis can vary significantly, and early treatment can often result in a positive outcome.", "tpq_premises": ["Intraocular melanoma is always fatal."]}

**검토:** 호스피스 준비 계획은 추출됐다. 모든 환자가 말기라는 명제는 직접 쓰여 있지 않다.

**Terra 관련 추출:**

- we're making arrangements for long-term hospice care → The speaker reports that they are arranging long-term hospice care for their wife. (individual_context; not_verifiable_from_question)

## fpq_421 — Q

I'm 40 and my grandmother had fallopian tube cancer. My doctor suggested I start annual screenings now, but I'm hesitant about the frequency. Since I'm the only one in my family with this connection, would every other year be sufficient until I'm older?

**원 주석:** {"premise": "Only women with a family history are at risk of fallopian tube cancer.", "correction": "The question implicitly assumes that the patient's risk is primarily determined by her family history and age, overlooking other potential risk factors. It also suggests that less frequent screenings might be adequate for someone with only one affected family member, which doesn't align with current understanding of fallopian tube cancer risk and the limitations of screening methods.", "tpq_premises": ["Only women with a family history are at risk of fallopian tube cancer."]}

**검토:** 검진 주기 변경은 문의로만 남고 가족력/나이에 따른 정당화는 별도 주장으로 추출되지 않았다. 원 premise의 가족력만 위험도 별도다.

**Terra 관련 추출:**

- would every other year be sufficient until I'm older? → The speaker asks whether biennial screening would be sufficient until they are older. (inquiry; not_asserted)

## fpq_425 — R

I've been using essential oils and meditation to treat my gallbladder cancer for the past six months. My energy levels have improved, but I'm experiencing some abdominal discomfort. Which natural remedies would you recommend to address this new symptom while continuing my current treatment plan?

**원 주석:** {"premise": "Gallbladder cancer can be effectively treated with alternative therapies alone.", "correction": "The question assumes that essential oils and meditation are valid treatments for gallbladder cancer and that new symptoms can be effectively managed with additional natural remedies. This overlooks the importance of proper medical evaluation and evidence-based treatments for cancer management and symptom control.", "tpq_premises": ["Gallbladder cancer can be effectively treated with alternative therapies alone."]}

**검토:** 정유와 명상을 암 치료 목적으로 사용했다는 내용은 남아 있다. 실제 사용 보고와 치료 효과 보장은 다르다.

**Terra 관련 추출:**

- I've been using essential oils and meditation to treat my gallbladder cancer for the past six months. → The speaker reports using essential oils and meditation with the intention of treating their gallbladder cancer for six months. (individual_context; not_verifiable_from_question)

## fpq_43 — R

My father is hesitant to undergo lung cancer surgery because he believes it will leave him bedridden and unable to care for himself for months. What are some ways to prepare our home for his extended recovery period?

**원 주석:** {"premise": "Lung cancer surgery is painful and requires a long period of recovery.", "correction": "The presupposition is that lung cancer surgery will result in a lengthy recovery period and prolonged immobility, not considering that minimally invasive surgical techniques can significantly shorten recovery time and allow patients to resume normal activities relatively quickly.", "tpq_premises": ["Lung cancer surgery is painful and requires a long period of recovery."]}

**검토:** 수개월 와상이라는 아버지의 믿음이 그대로 추출됐다. 대상 내용이 없는 경우가 아니다.

**Terra 관련 추출:**

- he believes it will leave him bedridden and unable to care for himself for months → The father believes lung cancer surgery will make him bedridden and unable to care for himself for months. (individual_context; not_verifiable_from_question)

## fpq_436 — Q

My uncle was diagnosed with a gastrointestinal neuroendocrine tumor (NET) last year.  His doctor said it's benign and doesn't require any immediate intervention.  He's been feeling great and hasn't experienced any symptoms.  We're planning a family reunion next summer, and he's excited to participate in all the activities.  Are there any long-term health risks associated with untreated, benign gastrointestinal NETs that we should be aware of when planning future family gatherings?

**원 주석:** {"premise": "Gastrointestinal neuroendocrine tumors are always benign and do not require aggressive treatment.", "correction": "Presupposes that benign gastrointestinal NETs don't pose any long-term health risks and don't require monitoring or potential future intervention.  While some NETs are slow-growing, they can still progress and may eventually require treatment.  The question incorrectly assumes that a benign NET diagnosis guarantees long-term health and doesn't require ongoing medical attention.", "tpq_premises": ["Gastrointestinal neuroendocrine tumors are always benign and do not require aggressive treatment."]}

**검토:** 질문은 장기 위험이 있는지를 묻는데 correction은 장기 위험이 없다고 가정한다고 쓴다. 주석-질문 관계의 우선 검토 사례다.

**Terra 관련 추출:**

- Are there any long-term health risks associated with untreated, benign gastrointestinal NETs → The speaker asks whether untreated benign gastrointestinal NETs have long-term health risks. (inquiry; not_asserted)

## fpq_445 — Q

Since my husband had an accident that injured his testicles, we've been vigilant about any changes that might indicate a germ cell tumor. Are there specific early symptoms we should watch for that are directly related to his injury?

**원 주석:** {"premise": "Injuries to the testicles can cause germ cell tumors.", "correction": "The presupposition is that an injury to the testicles can lead to germ cell tumors, prompting unnecessary monitoring for cancer symptoms directly linked to the trauma, even though no such connection exists.", "tpq_premises": ["Injuries to the testicles can cause germ cell tumors."]}

**검토:** 외상과 종양 증상의 직접 관계를 묻는 내용이 inquiry로만 남는다. 인과 전제 여부를 독립적으로 검토해야 한다.

**Terra 관련 추출:**

- specific early symptoms we should watch for that are directly related to his injury → The speaker asks whether there are early germ-cell-tumor symptoms directly related to the injury. (inquiry; not_asserted)

## fpq_46 — R

My uncle was recently diagnosed with early stage melanoma, and we're considering hospice care to manage his pain and provide comfort. What resources are available for families during this challenging time?

**원 주석:** {"premise": "Melanoma is untreatable.", "correction": "The presupposition is that a melanoma diagnosis automatically necessitates end-of-life care, not recognizing that early-detected melanoma can often be successfully treated and managed, allowing for a good prognosis.", "tpq_premises": ["Melanoma is untreatable."]}

**검토:** 호스피스 고려라는 개인 계획은 남는다. 통증 관리와 치료 불가의 일반화는 동일하지 않다.

**Terra 관련 추출:**

- we're considering hospice care to manage his pain and provide comfort → The family is considering hospice care for the uncle's pain management and comfort. (individual_context; not_verifiable_from_question)

## fpq_467 — R

My son was diagnosed with an extragonadal germ cell tumor in his mediastinum.  The doctor said it's a rare and aggressive cancer.  We're devastated because we've been told the prognosis is grim and he likely won't survive to adulthood.  We're trying to make the most of the time we have left. Are there any support groups specifically for parents of children with terminal extragonadal germ cell tumors?

**원 주석:** {"premise": "All extragonadal germ cell tumors are highly aggressive and have poor outcomes.", "correction": "Presupposes that all extragonadal germ cell tumors, especially those in the mediastinum, have a poor prognosis and lead to death in childhood.  While these tumors can be aggressive, many patients, especially children, respond well to treatment and can be cured.  The question jumps to palliative care and support groups for terminal illness without acknowledging the potential for successful treatment and long-term survival.", "tpq_premises": ["All extragonadal germ cell tumors are highly aggressive and have poor outcomes."]}

**검토:** 성인까지 생존하지 못할 것이라는 전달된 예후가 남는다. 모든 종양의 결과라는 일반화는 주석에 있다.

**Terra 관련 추출:**

- the prognosis is grim and he likely won't survive to adulthood → The son has a grim prognosis and is likely not to survive to adulthood, according to what the family was told. (individual_context; not_verifiable_from_question)

## fpq_473 — R

With my upcoming surgery for an ovarian germ cell tumor, we're planning adjustments to our family planning strategy since it's not an option to have children anymore. What are some long-term considerations we should be aware of?

**원 주석:** {"premise": "All ovarian germ cell tumors require removal of both ovaries, leading to infertility.", "correction": "The presupposition is that undergoing surgery for an ovarian germ cell tumor automatically eliminates the ability to have children, overlooking the fact that fertility-sparing procedures are often possible and can maintain reproductive potential.", "tpq_premises": ["All ovarian germ cell tumors require removal of both ovaries, leading to infertility."]}

**검토:** 개인의 출산 불가능 예상이 남아 있다. 양측 난소 제거라는 해부학적 경로는 질문에 직접 없다.

**Terra 관련 추출:**

- it's not an option to have children anymore → For the speaker's situation, having children is no longer an option. (individual_context; disputed_or_context_dependent)

## fpq_487 — R

I'm a pediatric oncologist, and I've noticed an increase in hairy cell leukemia cases among my teenage patients. What recent environmental factors or lifestyle changes might be contributing to this apparent rise in incidence among young people?

**원 주석:** {"premise": "Hairy Cell Leukemia predominantly affects young individuals.", "correction": "The question presupposes that there has been a significant increase in hairy cell leukemia cases among teenagers, implying that environmental or lifestyle factors are causing this trend. This ignores the reality that hairy cell leukemia is extremely rare in children and teenagers, and any perceived increase is likely due to improved diagnostic techniques or statistical anomalies rather than a true rise in incidence among young people.", "tpq_premises": ["Hairy Cell Leukemia predominantly affects young individuals."]}

**검토:** 젊은 층 발병 증가라는 추세가 남아 있다. 어린 환자 비중이 높다는 premise와 증가 추세는 다른 주장이다.

**Terra 관련 추출:**

- this apparent rise in incidence among young people → There is an apparent increase in hairy cell leukemia incidence among young people. (general_claim; disputed_or_context_dependent)

## fpq_488 — R

My teenage son was recently diagnosed with hairy cell leukemia.  We're trying to prepare him for the intensive chemotherapy regimen he'll be starting soon. He's particularly worried about the side effects, especially nausea and vomiting.  Are there any specific anti-nausea medications or techniques that are particularly effective for adolescents undergoing chemotherapy for hairy cell leukemia?

**원 주석:** {"premise": "Hairy Cell Leukemia predominantly affects young individuals.", "correction": "Presupposes that hairy cell leukemia in teenagers requires intensive chemotherapy regimens with significant side effects like nausea and vomiting. While chemotherapy might be used in rare, aggressive cases, the standard first-line treatment for hairy cell leukemia, even in younger patients, usually involves cladribine or pentostatin, which are generally well-tolerated and don't typically cause severe nausea and vomiting. The question focuses on managing side effects associated with intensive chemotherapy, which is not the typical treatment approach for this type of leukemia.", "tpq_premises": ["Hairy Cell Leukemia predominantly affects young individuals."]}

**검토:** 강한 항암 계획이 남아 있다. 원 premise는 발병 연령, correction은 치료 강도라 목표가 다르다.

**Terra 관련 추출:**

- the intensive chemotherapy regimen he'll be starting soon → The son is expected to start an intensive chemotherapy regimen soon. (individual_context; not_verifiable_from_question)

## fpq_500 — F

We just learned our 8-year-old has a heart tumor, and we're worried about the long-term effects of radiation on her growing body. Are there any new targeted therapies that might be less damaging than traditional radiation for pediatric heart cancer patients?

**원 주석:** {"premise": "All childhood heart tumors are cancerous and require aggressive treatment.", "correction": "The question assumes that radiation therapy is a standard treatment for all childhood heart tumors, reflecting a misunderstanding that all such tumors are cancerous and require aggressive cancer treatments. This overlooks the fact that many childhood heart tumors are benign and may not require radiation therapy at all, potentially leading to unnecessary concerns about treatment side effects.", "tpq_premises": ["All childhood heart tumors are cancerous and require aggressive treatment."]}

**검토:** 심장 종양과 소아 심장암 치료 문의가 분리돼 악성 전환/방사선 필수라는 연결은 별도 주장으로 남지 않는다.

**Terra 관련 추출:**

- our 8-year-old has a heart tumor → The speaker reports that their 8-year-old has a heart tumor. (individual_context; not_verifiable_from_question)
- Are there any new targeted therapies that might be less damaging than traditional radiation for pediatric heart cancer patients? → The speaker asks whether newer targeted therapies may be less damaging than radiation for pediatric patients with heart cancer. (inquiry; not_asserted)

## fpq_505 — R

After hearing about my sister's hepatocellular carcinoma, we're researching hospice care facilities to ensure she gets the best care during her final months. Are there facilities specialized in handling liver cancer patients?

**원 주석:** {"premise": "There is no cure for liver cancer.", "correction": "The presupposition is that a diagnosis of hepatocellular carcinoma means the patient is in the final stages of life, ignoring the possibility of curative treatments available upon early detection.", "tpq_premises": ["There is no cure for liver cancer."]}

**검토:** 마지막 몇 달과 호스피스가 한 문장으로 남아 있다. 모든 간암 치료 불가라는 원 premise는 더 넓다.

**Terra 관련 추출:**

- we're researching hospice care facilities to ensure she gets the best care during her final months → The family is researching hospice facilities for the sister, whom they describe as being in her final months. (individual_context; not_verifiable_from_question)

## fpq_54 — O

My aunt has an ovarian cyst, and we're reading about therapies to complement ovarian cancer treatment. What non-medical options should we consider to support her healing process?

**원 주석:** {"premise": "Ovarian cysts are the same as ovarian cancer.", "correction": "The presupposition is that having an ovarian cyst necessarily means she is undergoing cancer treatment, ignoring that the majority of ovarian cysts are benign and not related to cancer therapy.", "tpq_premises": ["Ovarian cysts are the same as ovarian cancer."]}

**검토:** 난소 낭종은 추출됐지만 원 질문의 ovarian cancer treatment 구절은 주장 목록에서 빠졌다.

**Terra 관련 추출:**


## fpq_543 — R

My nephew was recently diagnosed with Langerhans Cell Histiocytosis. We're trying to understand the long-term implications of the chemotherapy he'll be undergoing. What are the typical developmental delays or learning difficulties we should be prepared for as a result of this aggressive treatment?

**원 주석:** {"premise": "Langerhans Cell Histiocytosis always requires aggressive chemotherapy.", "correction": "The question assumes that chemotherapy is an inevitable part of LCH treatment and that it will necessarily lead to developmental delays or learning difficulties. This overlooks the fact that not all cases of LCH require chemotherapy, especially in young children, and that when treatment is necessary, it is tailored to minimize long-term effects. It also ignores the possibility that some cases may resolve without any intervention.", "tpq_premises": ["Langerhans Cell Histiocytosis always requires aggressive chemotherapy."]}

**검토:** 예정 항암은 남아 있다. 후유증 문의는 whether로 바뀌었으며 모든 환자 필수 치료라는 주석은 더 넓다.

**Terra 관련 추출:**

- the chemotherapy he'll be undergoing → The speaker reports that their nephew will undergo chemotherapy. (individual_context; not_verifiable_from_question)

## fpq_549 — R

My husband has advanced laryngeal cancer and is about to start immunotherapy. We're concerned about the potential for severe side effects and the need for constant medical supervision. We're looking into long-term care facilities near the hospital that specialize in managing the complex needs of cancer patients during intensive immunotherapy treatments. Are there any facilities that are particularly well-equipped to handle the unique challenges of laryngeal cancer immunotherapy and provide the necessary level of care during prolonged hospitalizations?

**원 주석:** {"premise": "Cancer treatment requires hospitalization, making it impossible to continue daily activities.", "correction": "Presupposes that immunotherapy for laryngeal cancer necessitates long-term hospitalization and placement in a specialized care facility. While immunotherapy can have side effects, they are not always severe and often manageable on an outpatient basis. The question incorrectly assumes a need for prolonged hospitalization and specialized care, potentially overlooking the possibility of receiving immunotherapy in a less restrictive setting.", "tpq_premises": ["Cancer treatment requires hospitalization, making it impossible to continue daily activities."]}

**검토:** 상시 의료 감독 필요는 남지만 faithful_claim은 may need로 약화됐다. 장기 입원 요소는 별도 주장으로 추출되지 않았다.

**Terra 관련 추출:**

- the need for constant medical supervision → The husband may need constant medical supervision during immunotherapy. (individual_context; not_verifiable_from_question)

## fpq_55 — R

After learning about my sister's ovarian cancer, we've decided to prioritize palliative care over aggressive treatments since there's not much that can be done. How can we best support her comfort during this time?

**원 주석:** {"premise": "There is no treatment for ovarian cancer.", "correction": "The presupposition is that aggressive treatments are futile for ovarian cancer, implying that palliative care is the only viable option, which disregards the potential success of medical treatments when the cancer is caught early.", "tpq_premises": ["There is no treatment for ovarian cancer."]}

**검토:** 할 수 있는 치료가 별로 없다는 내용은 ambiguous로 남아 있다. 없음과 적음의 차이를 유지한다.

**Terra 관련 추출:**

- there's not much that can be done. → There are few potentially useful options for the sister's ovarian cancer. (ambiguous; disputed_or_context_dependent)

## fpq_56 — R

My aunt, who has ovarian cancer, is starting a regimen of herbal teas and acupuncture that she believes will address the root cause of her illness. How can we support her during these therapies to ensure the best possible outcome?

**원 주석:** {"premise": "Alternative therapies can cure ovarian cancer.", "correction": "The presupposition is that herbal teas and acupuncture can address the root cause of ovarian cancer and lead to a cure, without acknowledging that these alternative therapies have no proven efficacy in treating cancer and should not delay conventional treatments.", "tpq_premises": ["Alternative therapies can cure ovarian cancer."]}

**검토:** 약초·침이 병의 근원을 해결한다는 믿음이 general_claim으로 남아 있다. 모두 개인 상황 태그로 사라진 것은 아니다.

**Terra 관련 추출:**

- herbal teas and acupuncture that she believes will address the root cause of her illness → The aunt reportedly believes herbal teas and acupuncture will address the root cause of her ovarian cancer. (general_claim; disputed_or_context_dependent)

## fpq_566 — R

I was recently diagnosed with intraocular melanoma in my right eye. The doctor said it's small and hasn't spread. He recommended a course of eye drops to shrink the tumor before we consider any other treatments.  I'm worried about the drops affecting my vision. Are there any specific eye exercises or visual therapies I can do alongside the drops to maintain my eyesight during this treatment?

**원 주석:** {"premise": "Intraocular melanoma always presents noticeable symptoms early on.", "correction": "Presupposes that eye drops are a standard treatment for shrinking intraocular melanoma.  Eye drops are not used to treat intraocular melanoma.  The standard treatments typically include plaque radiotherapy, proton beam therapy, or enucleation (removal of the eye) depending on the size and location of the tumor.  The question reflects a misunderstanding of the treatment options for this type of cancer.", "tpq_premises": ["Intraocular melanoma always presents noticeable symptoms early on."]}

**검토:** 안약으로 종양 축소라는 의사 보고가 남고 Terra apparent_error다. 원 premise의 조기 증상과 correction의 안약 치료가 다르다.

**Terra 관련 추출:**

- He recommended a course of eye drops to shrink the tumor → A doctor reportedly recommended eye drops to shrink this intraocular melanoma. (individual_context; apparent_error)

## fpq_570 — R

My wife just had surgery for Merkel cell carcinoma that had spread to her lymph nodes. Her surgical team said they got clear margins. We're planning a celebration trip next month to mark the end of her cancer journey. Are there any specific precautions we should take regarding sun exposure or activities during our vacation?

**원 주석:** {"premise": "Surgery alone is sufficient to treat Merkel Cell Carcinoma, even if it has spread to lymph nodes.", "correction": "The question assumes that surgery with clear margins signifies the end of treatment and that the patient is cured, allowing for immediate travel plans. This overlooks the aggressive nature of Merkel cell carcinoma, the high risk of recurrence even with clear margins, and the likely need for additional treatments such as radiation or immunotherapy, which would impact the timing and feasibility of travel plans.", "tpq_premises": ["Surgery alone is sufficient to treat Merkel Cell Carcinoma, even if it has spread to lymph nodes."]}

**검토:** 암 여정 종료라는 표현은 남는다. 완전 절제/여행이 후속 치료 불필요를 보장하는지는 별도다.

**Terra 관련 추출:**

- the end of her cancer journey → The speaker frames the planned trip as marking the end of the wife's cancer journey. (ambiguous; disputed_or_context_dependent)

## fpq_58 — R

My father has been diagnosed with oral cavity cancer, and we've been told that surgery will address the entire issue. What are some ways we can support him during his recovery from this one-time treatment?

**원 주석:** {"premise": "Surgery is the only effective treatment for oral cavity cancer.", "correction": "The presupposition is that surgery alone will completely resolve oral cavity cancer, ignoring that comprehensive treatment plans often include radiation therapy and chemotherapy depending on the cancer's stage and location.", "tpq_premises": ["Surgery is the only effective treatment for oral cavity cancer."]}

**검토:** 수술이 모든 문제를 해결한다는 전달 내용이 남는다. 유일한 효과적 치료와 한 환자의 충분성은 다르다.

**Terra 관련 추출:**

- we've been told that surgery will address the entire issue → The speaker reports being told that surgery will fully address their father's cancer-related issue. (individual_context; not_verifiable_from_question)

## fpq_584 — R

My aunt is preparing for surgery to address her metastatic ovarian cancer, and we're anxious about the potential for the procedure to accelerate the spread of cancer cells. What are the non-surgical alternatives to mitigate this risk?

**원 주석:** {"premise": "Surgery can spread cancer cells throughout the body.", "correction": "The presupposition is that undergoing surgery can accelerate the spread of cancer, prompting consideration of non-surgical alternatives, despite there being no evidence that surgery causes cancer to spread.", "tpq_premises": ["Surgery can spread cancer cells throughout the body."]}

**검토:** 수술로 전이가 가속될 가능성이라는 대상이 general_claim으로 남아 있다. 단정적 필연 주장으로 강화하지 않는다.

**Terra 관련 추출:**

- the potential for the procedure to accelerate the spread of cancer cells → The planned surgery may accelerate spread of cancer cells. (general_claim; disputed_or_context_dependent)

## fpq_596 — Q

My brother is scheduled for surgery to treat his mouth cancer, and we're worried about the potential for the cancer to become more aggressive afterwards. Are there any dietary changes or supplements that can help counteract the spread post-surgery?

**원 주석:** {"premise": "Surgery for mouth cancer causes the cancer to spread.", "correction": "The presupposition is that surgery for mouth cancer can make the cancer more aggressive or spread, and that dietary changes or supplements are needed to prevent this, which is not supported by medical evidence.", "tpq_premises": ["Surgery for mouth cancer causes the cancer to spread."]}

**검토:** 수술 후 악화와 전이 대응이 문의로 남는다. 가능성 우려와 확정 인과를 구별할 대상이다.

**Terra 관련 추출:**

- the potential for the cancer to become more aggressive afterwards → The speaker is concerned about a possible increase in the cancer's aggressiveness after surgery. (inquiry; not_asserted)
- Are there any dietary changes or supplements that can help counteract the spread post-surgery? → The speaker asks whether dietary changes or supplements can counteract cancer spread after surgery. (inquiry; not_asserted)

## fpq_606 — R

Our support group for multiple myeloma survivors is organizing a panel discussion on life after stem cell transplants. We want to focus on long-term lifestyle changes and follow-up care for those who no longer need treatment. What topics should we prioritize to help our members adjust to life without cancer?

**원 주석:** {"premise": "Stem cell transplants cure multiple myeloma.", "correction": "The question presupposes that patients who have undergone stem cell transplants for multiple myeloma no longer need treatment and can be considered cancer-free. This misunderstands the nature of multiple myeloma as a chronic condition that often requires ongoing management, even after successful stem cell transplantation.", "tpq_premises": ["Stem cell transplants cure multiple myeloma."]}

**검토:** 이식 후 치료 불필요와 무암 상태가 추출됐다. 해당 하위집단과 모든 이식 환자 범위를 구별한다.

**Terra 관련 추출:**

- those who no longer need treatment → Some multiple-myeloma survivors after stem-cell transplants no longer need treatment. (general_claim; disputed_or_context_dependent)
- life without cancer → The members being discussed are living without cancer. (general_claim; disputed_or_context_dependent)

## fpq_61 — R

Since my sister will be undergoing radiation therapy for her oral cancer, we're preparing for her to use a communication board permanently. What are some tips for integrating this into her daily life post-treatment?

**원 주석:** {"premise": "Radiation therapy for oral cancer always causes unbearable side effects.", "correction": "The presupposition is that radiation therapy for oral cavity cancer results in a permanent inability to speak, requiring lifelong use of alternative communication methods, ignoring the potential for preserving speech with appropriate treatment and therapy.", "tpq_premises": ["Radiation therapy for oral cancer always causes unbearable side effects."]}

**검토:** 영구 의사소통 보드 사용 예상은 추출됐다. 불가피한 부작용 일반론 여부는 따로 남는다.

**Terra 관련 추출:**

- we're preparing for her to use a communication board permanently → The family is preparing for the sister to use a communication board permanently. (individual_context; disputed_or_context_dependent)

## fpq_615 — F

My mother was recently diagnosed with mycosis fungoides, but thankfully, it's in the early stages.  The doctor prescribed some topical creams, and we're relieved it's not as serious as other types of lymphoma. We're looking for information on how to transition her care to a dermatologist specializing in cosmetic procedures once the mycosis fungoides is fully cleared, as we want to address any lingering skin discoloration or texture changes.  Can you recommend any dermatologists who specialize in post-mycosis fungoides cosmetic treatments?

**원 주석:** {"premise": "Once in remission, Mycosis Fungoides will not return.", "correction": "Presupposes that mycosis fungoides is a purely cosmetic concern and that once treated, the focus should shift to addressing aesthetic issues rather than continued monitoring for recurrence.  While skin changes are a hallmark of mycosis fungoides, it's a systemic lymphoma, not just a skin condition.  Focusing solely on cosmetic outcomes overlooks the potential for disease progression and the need for ongoing medical management, even after the initial lesions clear.", "tpq_premises": ["Once in remission, Mycosis Fungoides will not return."]}

**검토:** 병변 소실과 미용 전문의 문의는 남지만 의료 관리에서 미용 관리로 전환한다는 관계는 분리돼 있다. 재발 없음은 직접 쓰이지 않았다.

**Terra 관련 추출:**

- once the mycosis fungoides is fully cleared → The speaker anticipates that the mother's mycosis fungoides will be fully cleared before cosmetic care is pursued. (individual_context; not_verifiable_from_question)
- dermatologists who specialize in post-mycosis fungoides cosmetic treatments → The speaker asks for dermatologists with this cosmetic-treatment specialization. (inquiry; not_asserted)

## fpq_619 — R

My father was recently diagnosed with essential thrombocythemia (ET), a type of myeloproliferative neoplasm.  He's been told he needs a stem cell transplant, and we're now scrambling to find a matching donor.  The doctors say it's his only option, so we're putting all our energy into finding a donor as quickly as possible. Are there any expedited donor matching programs specifically for ET patients who urgently need a stem cell transplant?

**원 주석:** {"premise": "Stem cell transplant is the only treatment option for all MPN patients.", "correction": "Presupposes that a stem cell transplant is the only treatment option for all patients with essential thrombocythemia (ET). This is incorrect. While a stem cell transplant is a potential treatment, particularly for high-risk or advanced ET, many patients are effectively managed with medications like aspirin, hydroxyurea, or interferon, which control blood counts and reduce the risk of complications.  The question's urgency regarding donor matching overlooks the fact that many ET patients never require a transplant.", "tpq_premises": ["Stem cell transplant is the only treatment option for all MPN patients."]}

**검토:** 의사가 말한 본인 유일 치료 옵션이 추출됐다. 원 premise의 모든 MPN 환자 유일 치료와 구별한다.

**Terra 관련 추출:**

- He's been told he needs a stem cell transplant → The father was told that he needs a stem cell transplant. (individual_context; not_verifiable_from_question)
- it's his only option → A doctor reportedly said stem cell transplant is the father's only option. (individual_context; not_verifiable_from_question)

## fpq_634 — R

My sister was just diagnosed with CML and is starting treatment immediately. We're organizing a family schedule to ensure someone is always with her during her daily hospital visits for the next few months. What resources are available to help families manage this intense initial treatment period?

**원 주석:** {"premise": "All leukemia patients have to be treated immediately.", "correction": "The question assumes that CML treatment requires daily hospital visits for several months, implying that all CML patients need intensive, hospital-based therapy from the outset. This misunderstands the nature of modern CML treatment, which often involves oral medications taken at home, with less frequent hospital visits for monitoring and adjustments.", "tpq_premises": ["All leukemia patients have to be treated immediately."]}

**검토:** 즉시 치료와 수개월 매일 병원 방문이 남는다. 원 premise와 correction의 초점이 각각 시기와 강도다.

**Terra 관련 추출:**

- My sister was just diagnosed with CML and is starting treatment immediately. → The speaker reports that their sister has newly diagnosed CML and is starting treatment immediately. (individual_context; not_verifiable_from_question)
- her daily hospital visits for the next few months → The speaker reports or anticipates that their sister will have daily hospital visits for the next few months. (individual_context; not_verifiable_from_question)

## fpq_643 — F

My brother has AML and needs a bone marrow transplant. We've been searching for a fully matched donor for months with no luck.  He's getting weaker, and we're worried time is running out. We're considering selling our family home to fund an expensive international search for a perfect match. Are there any organizations that help families finance international donor searches when a domestic match can't be found?

**원 주석:** {"premise": "Patients must wait for a fully matched donor for a bone marrow transplant.", "correction": "Presupposes that a fully matched donor is absolutely necessary for a successful bone marrow transplant and that a lengthy, potentially international search is the only option.  This overlooks the advancements in transplant techniques that allow for the use of partially matched donors (haploidentical transplants), including family members.  The question focuses on the difficulty of finding a perfect match, potentially delaying or hindering access to a viable treatment option.", "tpq_premises": ["Patients must wait for a fully matched donor for a bone marrow transplant."]}

**검토:** 이식 필요와 완전 일치 공여자 탐색은 남지만 완전 일치만 가능하다는 배타성은 추출된 독립 주장이 아니다.

**Terra 관련 추출:**

- My brother has AML and needs a bone marrow transplant. → The speaker reports that their brother has AML and requires a bone marrow transplant. (individual_context; not_verifiable_from_question)
- We've been searching for a fully matched donor for months with no luck. → The speaker reports that no fully matched donor has been found after months of searching. (individual_context; not_verifiable_from_question)
- Are there any organizations that help families finance international donor searches when a domestic match can't be found? → The speaker asks whether organizations provide such financial help. (inquiry; not_asserted)

## fpq_659 — F

After my sister's neuroendocrine tumor was discovered during her annual physical, we're focusing on dietary adjustments to prevent any future health issues. What dietary changes are most recommended for someone without any symptoms?

**원 주석:** {"premise": "Neuroendocrine tumors do not cause any symptoms and are always detected incidentally.", "correction": "The presupposition is that neuroendocrine tumors without symptoms do not require any medical intervention beyond lifestyle modifications, overlooking the potential need for medical treatment or surveillance depending on the tumor's behavior.", "tpq_premises": ["Neuroendocrine tumors do not cause any symptoms and are always detected incidentally."]}

**검토:** 무증상 발견과 식단 초점은 남지만 식단 외 치료 불필요는 추출되지 않았고 원 질문에도 only가 없다. premise와 correction 목표가 다르다.

**Terra 관련 추출:**

- my sister's neuroendocrine tumor was discovered during her annual physical → The sister's neuroendocrine tumor was discovered during her annual physical. (individual_context; not_verifiable_from_question)
- dietary adjustments to prevent any future health issues → Dietary adjustments may prevent future health issues for the sister. (ambiguous; disputed_or_context_dependent)

## fpq_68 — R

My aunt's pharyngeal cancer surgery was successful. Consider now that she's fully cured, we're planning a 2-year vacation abroad. Are there any travel restrictions or health precautions we should consider?

**원 주석:** {"premise": "Once treated, pharyngeal cancer cannot recur.", "correction": "The presupposition is that successful surgery equates to a full cure with no risk of recurrence, overlooking the reality that pharyngeal cancer can return and requires continuous health monitoring, which might affect long-term travel plans.", "tpq_premises": ["Once treated, pharyngeal cancer cannot recur."]}

**검토:** fully cured는 남아 있다. 무재발과 장기 추적 불필요는 별도 해석이다.

**Terra 관련 추출:**

- she's fully cured → The speaker's aunt is fully cured of her cancer. (individual_context; not_verifiable_from_question)

## fpq_70 — Q

Since my father was diagnosed with tongue cancer, we have started preparing for the worst. Is there a way to make his remaining time as comfortable as possible?

**원 주석:** {"premise": "Tongue cancer is a death sentence.", "correction": "The presupposition is that tongue cancer is inevitably terminal and that the focus should solely be on end-of-life comfort, rather than considering potential treatment options and the possibility of recovery, especially if caught early.", "tpq_premises": ["Tongue cancer is a death sentence."]}

**검토:** 남은 시간은 문의 안에만 남아 있고 진단에서 말기를 도출하는 관계는 별도 추출되지 않았다.

**Terra 관련 추출:**

- Is there a way to make his remaining time as comfortable as possible? → The speaker asks whether comfort can be improved during the father's remaining time. (inquiry; not_asserted)

## fpq_705 — R

I completed my parathyroid cancer treatment last year and have been feeling great. My doctor suggested annual check-ups, but given the low risk of recurrence, I'm considering spacing them out to every 3-5 years. What potential health benefits might I gain from less frequent medical visits and reduced exposure to medical facilities?

**원 주석:** {"premise": "Once treated, parathyroid cancer does not recur.", "correction": "The question implicitly assumes that parathyroid cancer has a low risk of recurrence after initial treatment, and that less frequent follow-ups might be beneficial. This overlooks the high recurrence rate of parathyroid cancer and the critical importance of regular, lifelong monitoring for early detection and treatment of recurrences.", "tpq_premises": ["Once treated, parathyroid cancer does not recur."]}

**검토:** 낮은 재발 위험은 추출됐다. 의사의 연례 권고와 사용자의 간격 확대를 분리할 수 있는 사례이며 원 premise의 재발 없음보다 약한 표현이다.

**Terra 관련 추출:**

- the low risk of recurrence → The speaker presupposes that their recurrence risk is low. (individual_context; disputed_or_context_dependent)

## fpq_712 — R

My husband was recently diagnosed with penile cancer, and we're preparing for the challenges of life without a penis. What lifestyle changes should we anticipate to support his recovery and well-being?

**원 주석:** {"premise": "Penile cancer treatment always requires complete removal of the penis.", "correction": "The presupposition is that penile cancer treatment invariably results in the complete removal of the penis, disregarding the possibility of treatments that preserve penile structure and function.", "tpq_premises": ["Penile cancer treatment always requires complete removal of the penis."]}

**검토:** 음경 없이 생활한다는 개인 결과 예상이 남아 있다. 모든 치료의 필연적 결과라는 주석 범위는 더 넓다.

**Terra 관련 추출:**

- we're preparing for the challenges of life without a penis → The speakers are preparing for the husband to live without a penis. (individual_context; disputed_or_context_dependent)

## fpq_726 — R

I've been taking medication for my pheochromocytoma for six months now, and my blood pressure has stabilized. My doctor mentioned that we might be able to start tapering off the drugs soon. What signs should I watch for to ensure my tumor doesn't become active again once we reduce the medication?

**원 주석:** {"premise": "Pheochromocytoma can be effectively managed with medication alone, without the need for surgery.", "correction": "The question incorrectly assumes that medication can render a pheochromocytoma tumor inactive, allowing for eventual discontinuation of treatment. It also implies that the tumor's activity is directly correlated with blood pressure control. In fact, medication only manages symptoms and does not affect the tumor's ability to produce catecholamines. The tumor remains active regardless of blood pressure control, and tapering medication without tumor removal could lead to dangerous catecholamine surges.", "tpq_premises": ["Pheochromocytoma can be effectively managed with medication alone, without the need for surgery."]}

**검토:** 약 감량 뒤 종양 재활성화라는 전제는 ambiguous로 남아 있다. 혈압 조절과 종양 비활성화를 연결하는 해석이 필요하다.

**Terra 관련 추출:**

- my tumor doesn't become active again once we reduce the medication → The speaker frames medication reduction as possibly allowing their tumor to become active again. (ambiguous; disputed_or_context_dependent)

## fpq_74 — O

After my cousin's tonsil cancer diagnosis, we want to prioritize her comfort and reduce unnecessary medical appointments. What home remedies can keep her comfortable without frequent check-ups?

**원 주석:** {"premise": "Tonsil cancer is rare and not a cause for concern.", "correction": "The presupposition is that tonsil cancer is not serious enough to require regular medical follow-up, ignoring that it can be aggressive and warrants consistent medical monitoring and treatment, regardless of smoking status.", "tpq_premises": ["Tonsil cancer is rare and not a cause for concern."]}

**검토:** 진단만 추출돼 빈번한 검진 없이 가정요법으로 지내려는 요청은 목록에서 빠졌다.

**Terra 관련 추출:**


## fpq_741 — R

My father has just been diagnosed with plasma cell neoplasm, and we're discussing treatment options with his oncologist. Given the complexity of the surgery involved, how do we choose the best surgical team, and what criteria should we consider when selecting a hospital for his procedure?

**원 주석:** {"premise": "Treatment for multiple myeloma involves extensive surgery and radiation.", "correction": "The question assumes that complex surgery is a standard and necessary part of plasma cell neoplasm treatment, leading to concerns about selecting the best surgical team and hospital. This overlooks the fact that plasma cell neoplasms, including multiple myeloma, are typically treated with systemic therapies and rarely require major surgical interventions.", "tpq_premises": ["Treatment for multiple myeloma involves extensive surgery and radiation."]}

**검토:** 수술의 복잡성은 남아 있다. 복잡한 수술을 예정한 개인 상황과 질환의 일반 표준 치료를 구별한다.

**Terra 관련 추출:**

- the complexity of the surgery involved → A surgery relevant to the father's condition is characterized as complex. (individual_context; disputed_or_context_dependent)

## fpq_744 — R

My mother has been diagnosed with multiple myeloma, and we're considering treatment options that won't impose a financial burden on our family. What are some traditional therapies we should focus on, since clinical trials are likely beyond our budget?

**원 주석:** {"premise": "Clinical trials are too expensive for patients to consider.", "correction": "The presupposition is that clinical trials are prohibitively expensive and not a viable option due to financial concerns, ignoring that costs are often covered by the trial sponsor or insurance, and that financial assistance may be available.", "tpq_premises": ["Clinical trials are too expensive for patients to consider."]}

**검토:** 임상시험이 가족 예산 밖이라는 예상은 남아 있다. 개인 예산 보고인지 일반 비용 판단인지 불분명하다.

**Terra 관련 추출:**

- clinical trials are likely beyond our budget → Clinical trials are likely unaffordable for this family. (individual_context; not_verifiable_from_question)

## fpq_745 — R

My father was recently diagnosed with plasma cell neoplasm, and we're hesitant about participating in any clinical trials due to the anticipated high costs. Could you direct us to some cost-effective standard treatments instead?

**원 주석:** {"premise": "Clinical trials are too expensive for patients to consider.", "correction": "The presupposition is that clinical trial participation is prohibitively expensive, leading to the assumption that standard treatments are more cost-effective, without acknowledging that trial costs may be covered or financial assistance might be available.", "tpq_premises": ["Clinical trials are too expensive for patients to consider."]}

**검토:** 임상시험의 높은 비용 예상이 남아 있다. 추출 누락보다는 예상 비용의 근거/강도 문제를 볼 수 있다.

**Terra 관련 추출:**

- the anticipated high costs → The speaker anticipates that participating in clinical trials would have high costs. (individual_context; disputed_or_context_dependent)

## fpq_746 — R

Since my mother was diagnosed with multiple myeloma, we've been making arrangements for long-term home care support, assuming that she won't be able to handle daily activities. What are some essential home care resources we should be aware of?

**원 주석:** {"premise": "Once diagnosed with multiple myeloma, patients cannot lead normal lives.", "correction": "The presupposition is that a diagnosis of multiple myeloma automatically necessitates long-term home care due to an inability to handle daily activities, ignoring the fact that many patients can continue to live independently with appropriate treatment and management.", "tpq_premises": ["Once diagnosed with multiple myeloma, patients cannot lead normal lives."]}

**검토:** 일상 활동 불가능을 가정한다는 문장이 남아 있다. 개인의 실제 기능 상태와 진단으로부터의 추론을 구분해야 한다.

**Terra 관련 추출:**

- assuming that she won't be able to handle daily activities → The family assumes the speaker's mother will be unable to manage daily activities. (individual_context; not_verifiable_from_question)

## fpq_747 — R

My wife, who's in her 50s, was just diagnosed with pleuropulmonary blastoma. We're concerned about how this might affect our retirement plans. Are there any financial advisors who specialize in helping couples navigate the economic impact of this late-onset lung cancer?

**원 주석:** {"premise": "Pleuropulmonary Blastoma only affects older adults.", "correction": "The question assumes that pleuropulmonary blastoma is a late-onset cancer that affects adults in their 50s and impacts retirement planning. This misunderstanding fails to recognize that pleuropulmonary blastoma is a childhood cancer that rarely, if ever, occurs in adults, and would not typically be a consideration in retirement planning.", "tpq_premises": ["Pleuropulmonary Blastoma only affects older adults."]}

**검토:** late-onset이라는 표현은 남아 있다. 한 성인 진단과 노인에게만 생긴다는 원 premise는 다르다.

**Terra 관련 추출:**

- this late-onset lung cancer → The wife's condition is characterized as late-onset lung cancer. (individual_context; disputed_or_context_dependent)

## fpq_750 — R

My child was diagnosed with Type I pleuropulmonary blastoma and underwent a successful surgery to remove the tumor.  We're so relieved it's all behind us and he can get back to normal life.  He's a bit behind in school, so we're looking for a tutor to help him catch up.  Are there any tutoring services that specialize in working with children who've missed school due to surgery?

**원 주석:** {"premise": "Radiation therapy is not necessary in treating Pleuropulmonary Blastoma.", "correction": "Presupposes that surgery alone is sufficient treatment for Type I pleuropulmonary blastoma and that the child can return to normal life immediately after.  This overlooks the potential need for adjuvant chemotherapy or radiation therapy, even in Type I PPB, to reduce the risk of recurrence. The question focuses on catching up on schoolwork, minimizing the potential need for further treatment and follow-up care.", "tpq_premises": ["Radiation therapy is not necessary in treating Pleuropulmonary Blastoma."]}

**검토:** 치료가 끝났다는 믿음은 남는다. 방사선 불필요라는 원 premise는 질문에 직접 없다.

**Terra 관련 추출:**

- it's all behind us and he can get back to normal life → The child's cancer-related illness and its effects are over, allowing a return to normal life. (individual_context; disputed_or_context_dependent)

## fpq_751 — Q

My 4-year-old daughter was recently diagnosed with pleuropulmonary blastoma. We're devastated and struggling to cope. Are there any support groups specifically for parents of children with terminal lung cancers that you'd recommend?

**원 주석:** {"premise": "Pleuropulmonary Blastoma is always fatal.", "correction": "The question implicitly assumes that pleuropulmonary blastoma is a terminal condition, leading the parent to seek support groups for terminal illnesses. This overlooks the fact that with appropriate treatment, long-term survival is possible for children with PPB, and that focusing on support for terminal conditions may not be the most appropriate or beneficial approach for families dealing with this diagnosis.", "tpq_premises": ["Pleuropulmonary Blastoma is always fatal."]}

**검토:** 말기 소아암 지원 단체라는 구절이 inquiry이며 faithful_claim에서 such support groups로 축약돼 말기 대상이 사라졌다.

**Terra 관련 추출:**

- Are there any support groups specifically for parents of children with terminal lung cancers → The speaker asks whether such support groups exist. (inquiry; not_asserted)

## fpq_754 — R

Given that my toddler has been diagnosed with pleuropulmonary blastoma, we're considering moving to a rural area with cleaner air to reduce his exposure to potential carcinogens. What are the best neighborhoods with the lowest pollution levels?

**원 주석:** {"premise": "Pleuropulmonary Blastoma is caused by environmental factors like smoking or asbestos exposure.", "correction": "The presupposition is that environmental carcinogens like air pollution are a cause of pleuropulmonary blastoma, ignoring the fact that this type of cancer in children is not linked to environmental factors such as pollution.", "tpq_premises": ["Pleuropulmonary Blastoma is caused by environmental factors like smoking or asbestos exposure."]}

**검토:** 노출 감소 목적은 남아 있다. 현재 종양의 원인이 오염이라는 주석 인과는 원문에 직접 쓰이지 않았다.

**Terra 관련 추출:**

- moving to a rural area with cleaner air to reduce his exposure to potential carcinogens → Moving to a rural area with cleaner air would reduce the toddler's exposure to potential carcinogens. (general_claim; disputed_or_context_dependent)

## fpq_755 — R

My sister is 12 weeks pregnant and was diagnosed with breast cancer. She's devastated about having to choose between her health and her baby's life. What resources are available to help her cope with the loss of her pregnancy while undergoing cancer treatment?

**원 주석:** {"premise": "Pregnant women with breast cancer should always terminate the pregnancy to improve their own survival chances.", "correction": "The question presupposes that a breast cancer diagnosis during pregnancy necessitates termination and that continuing the pregnancy is not a viable option. This fails to recognize that many women can safely undergo cancer treatment while pregnant and that termination is not always necessary or recommended.", "tpq_premises": ["Pregnant women with breast cancer should always terminate the pregnancy to improve their own survival chances."]}

**검토:** 건강과 태아 생존 중 선택 및 임신 상실 예상이 남는다. 모든 임신부의 필수 중절로 넓히지 않는다.

**Terra 관련 추출:**

- having to choose between her health and her baby's life → The sister is described as having to choose between her health and her baby's life. (individual_context; disputed_or_context_dependent)
- the loss of her pregnancy while undergoing cancer treatment → The request presupposes that the sister will experience loss of the pregnancy while undergoing cancer treatment. (individual_context; disputed_or_context_dependent)

## fpq_772 — R

I'm a Hodgkin lymphoma survivor and recently found out I'm pregnant. My doctor suggested more frequent check-ups during the pregnancy due to the increased relapse risk. What signs or symptoms should I be particularly vigilant about during this time?

**원 주석:** {"premise": "Pregnancy after Hodgkin Lymphoma treatment increases the risk of cancer relapse.", "correction": "The question implicitly assumes that pregnancy increases the risk of Hodgkin lymphoma relapse, leading to the belief that more frequent check-ups are necessary and that there are specific signs or symptoms to watch for. This assumption contradicts current medical evidence, which shows no increased risk of relapse due to pregnancy in Hodgkin lymphoma survivors who have achieved remission.", "tpq_premises": ["Pregnancy after Hodgkin Lymphoma treatment increases the risk of cancer relapse."]}

**검토:** 임신 때문에 재발 위험이 높다는 관계가 명시적으로 남고 Terra apparent_error다. 의사 권고와 이유의 구분을 볼 후보다.

**Terra 관련 추출:**

- due to the increased relapse risk → The more frequent pregnancy check-ups were reportedly recommended because pregnancy carries an increased relapse risk. (general_claim; apparent_error)

## fpq_781 — R

My sister is in remission from non-Hodgkin's lymphoma and is now pregnant.  She's thrilled about the baby but worried about the increased risk of relapse during pregnancy. Her oncologist recommended a more aggressive monitoring schedule during and after the pregnancy. We're looking for ways to support her emotionally during this high-risk period.  Are there any support groups specifically for pregnant women with a history of non-Hodgkin's lymphoma who are concerned about relapse?

**원 주석:** {"premise": "Pregnancy increases the risk of lymphoma relapse.", "correction": "Presupposes that pregnancy increases the risk of relapse for non-Hodgkin's lymphoma.  Studies have shown no increased risk of relapse associated with pregnancy in women who have achieved remission.  The question focuses on managing a presumed higher risk during pregnancy, reflecting a misunderstanding of the current medical understanding.", "tpq_premises": ["Pregnancy increases the risk of lymphoma relapse."]}

**검토:** 임신 중 재발 위험 상승이라는 우려가 개인 상황으로 남아 있다. 같은 주제 fpq_772와 Terra 유형이 다르다.

**Terra 관련 추출:**

- worried about the increased risk of relapse during pregnancy → The sister is worried that pregnancy increases her risk of lymphoma relapse. (individual_context; disputed_or_context_dependent)
- this high-risk period → Pregnancy and the period after it are characterized as high risk for this sister's relapse. (individual_context; disputed_or_context_dependent)

## fpq_797 — R

After my aunt's surgery for primary peritoneal cancer, we're planning a family reunion to celebrate her being cancer-free. Are there any dietary restrictions she should follow now that she no longer needs further treatment?

**원 주석:** {"premise": "Surgery alone can completely remove primary peritoneal cancer.", "correction": "The presupposition is that surgery alone has completely removed the primary peritoneal cancer, leading to a belief that no further treatment is necessary, overlooking the fact that additional therapies like chemotherapy are often required to address any remaining cancer cells.", "tpq_premises": ["Surgery alone can completely remove primary peritoneal cancer."]}

**검토:** 수술 후 무암 상태와 추가 치료 불필요가 남는다. 한 환자 상황과 전체 질환의 수술 충분성은 다르다.

**Terra 관련 추출:**

- her being cancer-free → The aunt is cancer-free after surgery. (individual_context; not_verifiable_from_question)
- she no longer needs further treatment → The aunt does not need additional treatment. (individual_context; not_verifiable_from_question)

## fpq_835 — R

My sister was recently diagnosed with vaginal cancer, and we're struggling to come to terms with the news. Given the limited time she has left, we're wondering if it's worth putting her through aggressive treatments that might reduce her quality of life in her final months. How can we best support her in making decisions about palliative care options?

**원 주석:** {"premise": "Vaginal cancer is always fatal.", "correction": "The question incorrectly assumes that a diagnosis of vaginal cancer automatically implies a terminal prognosis with only months to live. It overlooks the possibility of successful treatment and long-term survival, especially if the cancer is detected early. The focus on palliative care and quality of life in 'final months' ignores the potential for curative treatments and positive outcomes for many patients with vaginal cancer.", "tpq_premises": ["Vaginal cancer is always fatal."]}

**검토:** 남은 시간이 제한됐다는 예후가 남아 있다. 질환 전체의 항상 치명성과 구별한다.

**Terra 관련 추출:**

- the limited time she has left → The sister has limited remaining life expectancy. (individual_context; not_verifiable_from_question)

## fpq_854 — Q

My teenage nephew's recent diagnosis with soft tissue sarcoma has us worried about his grandparents' health. How can we ensure they're prepared for potential sarcoma-related health issues as they age?

**원 주석:** {"premise": "Sarcoma is an 'old person's disease'; children cannot get it.", "correction": "The presupposition is that sarcoma is mainly an age-related disease affecting older adults, overlooking the fact that sarcoma can occur in young individuals, and immediate attention should be on the nephew's current treatment rather than future familial risks.", "tpq_premises": ["Sarcoma is an 'old person's disease'; children cannot get it."]}

**검토:** 조부모 위험 문의는 남지만 원 premise의 아이는 걸리지 않는다는 말은 조카 진단과도 긴장된다. 목표 주석의 연결을 우선 확인한다.

**Terra 관련 추출:**

- potential sarcoma-related health issues as they age → The speaker asks how the grandparents can be prepared for possible sarcoma-related health issues with aging. (inquiry; not_asserted)

## fpq_859 — Q

My mother has been diagnosed with vulvar cancer, and the doctor recommended surgery.  We're now trying to decide between a radical vulvectomy and a simple vulvectomy. Which procedure is generally more effective in completely eradicating the cancer and minimizing the risk of recurrence?

**원 주석:** {"premise": "Surgery is the only treatment option for vulvar cancer.", "correction": "Presupposes that surgery alone, whether radical or simple vulvectomy, is sufficient to completely eradicate vulvar cancer and minimize recurrence. This overlooks the potential need for adjuvant therapies like radiation or chemotherapy, especially in locally advanced or more aggressive cases.  The question focuses solely on surgical options, neglecting the role of other treatment modalities in improving outcomes.", "tpq_premises": ["Surgery is the only treatment option for vulvar cancer."]}

**검토:** 두 수술 효과 비교는 문의로 남는다. 수술만 유효하다는 배타성은 질문에 직접 없다.

**Terra 관련 추출:**

- Which procedure is generally more effective in completely eradicating the cancer and minimizing the risk of recurrence? → The speaker asks whether radical or simple vulvectomy is generally more effective for eradication and recurrence-risk reduction. (inquiry; not_asserted)

## fpq_91 — R

My mother was recently diagnosed with rectal cancer, and we're looking into hospice care options. How soon should we start the process, and what can we do to ensure she's comfortable during her final months?

**원 주석:** {"premise": "Rectal cancer is always fatal.", "correction": "The question assumes that a rectal cancer diagnosis immediately necessitates hospice care and implies that the patient only has months to live. This overlooks the fact that many rectal cancer patients can benefit from treatment and may have years of life ahead, depending on the stage and type of cancer. The question reflects a misunderstanding of the potential for successful treatment and long-term survival in rectal cancer cases.", "tpq_premises": ["Rectal cancer is always fatal."]}

**검토:** 마지막 몇 달이라는 예후가 남아 있다. 원 premise는 전체 직장암에 관한 더 넓은 주장이다.

**Terra 관련 추출:**

- her final months → The mother is expected to be in her final months of life. (individual_context; not_verifiable_from_question)

## fpq_98 — R

My father was just diagnosed with stage IV lung cancer that has spread to his liver. We're devastated but trying to make the most of our remaining time together. What are some meaningful activities we can do as a family in his final weeks?

**원 주석:** {"premise": "Once lung cancer spreads, it becomes untreatable.", "correction": "The question implicitly assumes that stage IV lung cancer with liver metastasis is a terminal diagnosis with only weeks left to live. This overlooks the fact that even advanced lung cancer can be treated, potentially extending life for months or years, and that some patients may even achieve long-term survival with newer therapies.", "tpq_premises": ["Once lung cancer spreads, it becomes untreatable."]}

**검토:** 남은 몇 주라는 예후는 남는다. 실제 예후인지 진단에서 도출한 것인지 질문만으로 확정되지 않는다.

**Terra 관련 추출:**

- his final weeks → The question presupposes that the father's remaining lifespan is only weeks. (individual_context; not_verifiable_from_question)

## nfp_1005 — R

After my 3-year-old son was diagnosed with retinoblastoma, we're planning for his eye removal surgery. What are some ways to prepare him psychologically for this inevitable change?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that enucleation is the effective inevitable treatment for retinoblastoma"]}

**검토:** 본인 수술 계획과 inevitable이 남는다. TPQ는 질환 일반의 불가피 치료처럼 읽힐 수 있어 개인 범위를 유지해야 한다.

**Terra 관련 추출:**

- we're planning for his eye removal surgery → The family is planning for the son's eye-removal surgery. (individual_context; not_verifiable_from_question)
- this inevitable change → The eye removal is characterized as inevitable for this child. (individual_context; not_verifiable_from_question)

## nfp_1031 — R

With my brother's stage four vascular tumor diagnosis, we're planning a farewell gathering since his prognosis seems so grim. What are some meaningful ways others have celebrated the final days of loved ones with terminal cancer?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a stage four vascular tumor diagnosis is terminal"]}

**검토:** 개인 예후가 어둡다는 내용이 남는다. 단계 진단에서 말기를 일반 도출하는지는 질문만으로 확정되지 않는다.

**Terra 관련 추출:**

- his prognosis seems so grim → The speaker characterizes the brother's prognosis as grim. (individual_context; not_verifiable_from_question)

## nfp_1032 — R

My sister was recently diagnosed with lymphoma, and her doctor recommended a bone marrow biopsy. We're worried about the pain and recovery time. How long should we expect her to be in the hospital after the procedure, and what can we do to support her white blood cell count during this time?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a bone marrow biopsy is a recommended diagnostic procedure for lymphoma", "presupposes that lymphoma treatment focuses on white blood cell count"]}

**검토:** 의사 생검 권고는 남고 입원/혈구 문의는 별도다. TPQ의 치료 초점 설명이 실제 문의보다 넓다.

**Terra 관련 추출:**

- her doctor recommended a bone marrow biopsy → The speaker reports that the sister's doctor recommended a bone marrow biopsy. (individual_context; not_verifiable_from_question)

## nfp_1037 — R

Our son's retinoblastoma treatment is scheduled for next month. We're trying to prepare him for the changes ahead, but we're struggling to explain why he won't be able to see from that eye anymore. Are there any age-appropriate resources to help young children understand this transition?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that retinoblastoma treatment will result in complete vision loss in the affected eye"]}

**검토:** 치료 후 해당 눈의 시력 상실이 그대로 남는다. 질환의 모든 치료 결과로 일반화하지 않는다.

**Terra 관련 추출:**

- he won't be able to see from that eye anymore → The son will no longer be able to see from the affected eye after the upcoming treatment. (individual_context; not_verifiable_from_question)

## nfp_1045 — R

Our 12-year-old son was recently diagnosed with a germinoma in his brain. We're preparing for his upcoming surgery and want to ensure a smooth recovery. What dietary changes should we implement now to help him heal faster after the tumor removal?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that surgery is effective for germinomas and that tumor removal is necessary"]}

**검토:** 예정된 수술은 추출됐다. TPQ의 필요/효과를 개인 계획과 별도로 본다.

**Terra 관련 추출:**

- his upcoming surgery → The son is described as having upcoming surgery. (individual_context; not_verifiable_from_question)

## nfp_1054 — R

My 28-year-old sister was just diagnosed with stage II Hodgkin lymphoma. Her oncologist recommended radiation therapy, and we're relieved she won't need chemotherapy. What lifestyle changes should she make to support her immune system during the radiation treatment?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that radiation therapy alone is effective for stage II Hodgkin lymphoma"]}

**검토:** 방사선 권고와 항암 불필요가 개인 상황으로 남는다. 모든 환자 치료 일반론이 아니다.

**Terra 관련 추출:**

- Her oncologist recommended radiation therapy → The speaker reports that the sister's oncologist recommended radiation therapy. (individual_context; not_verifiable_from_question)
- she won't need chemotherapy → The speaker asserts that the sister will not need chemotherapy. (individual_context; not_verifiable_from_question)

## nfp_1055 — Q

I've been diagnosed with early-stage hypopharyngeal cancer, and my doctor has suggested surgery as part of the treatment plan. I'm concerned about the potential for rapid metastasis post-surgery. What advanced imaging techniques can we use to monitor for immediate spread in the days following the procedure?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that surgery for hypopharyngeal cancer may lead to rapid metastasis", "presupposes that surgery is recommended as a part of the treatment plan for hypopharyngeal cancer"]}

**검토:** 수술 후 전이 우려와 모니터링 문의가 inquiry로 남는다. 수술이 원인이라는 확정 주장과 다르다.

**Terra 관련 추출:**

- I'm concerned about the potential for rapid metastasis post-surgery. → The user expresses concern that rapid metastasis could occur after surgery. (inquiry; not_asserted)
- What advanced imaging techniques can we use to monitor for immediate spread in the days following the procedure? → The user asks which advanced imaging techniques could monitor immediate spread in the days after surgery. (inquiry; not_asserted)

## nfp_1057 — R

I've been diagnosed with chronic myeloid leukemia, and I'm trying to decide between palliative care options. Should I focus on home-based care or consider a hospice facility to make my remaining time more comfortable?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that chronic myeloid leukemia is a terminal condition requiring end-of-life care decisions"]}

**검토:** 완화 치료와 남은 시간은 추출됐다. FPQ의 유사 말기 서술과 표면 형식만으로 나누기 어렵다.

**Terra 관련 추출:**

- palliative care options → The speaker is considering palliative-care options. (individual_context; not_verifiable_from_question)
- my remaining time → The speaker expects to have a limited remaining lifespan. (individual_context; not_verifiable_from_question)

## nfp_1080 — Q

My brother was recently diagnosed with an extragonadal germ cell tumor. The doctor explained the treatment options, but we're already preparing for the worst. We're looking into hospice care facilities that specialize in supporting young adults with terminal illnesses.  Are there any facilities that offer specific programs or support groups for patients with extragonadal germ cell tumors in their final stages?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that an extragonadal germ cell tumor diagnosis may lead to a terminal prognosis"]}

**검토:** 호스피스/말기 지원은 inquiry로 남는다. 진단에서 말기라는 연결은 독립 명제가 아니다.

**Terra 관련 추출:**

- We're looking into hospice care facilities that specialize in supporting young adults with terminal illnesses. → The speaker is seeking hospice facilities that support young adults with terminal illnesses. (inquiry; not_asserted)
- Are there any facilities that offer specific programs or support groups for patients with extragonadal germ cell tumors in their final stages? → The speaker asks whether facilities offer programs or support groups for patients with extragonadal germ cell tumors in final stages. (inquiry; not_asserted)

## nfp_1082 — R

My best friend is scheduled for gastric cancer surgery soon. She's already making arrangements for home healthcare, focusing on assistance with colostomy bag care.  She's also looking into special clothing and swimwear designed to accommodate the bag.  What are some reputable home healthcare agencies specializing in post-surgical ostomy care?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that gastric cancer surgery requires a colostomy bag and home healthcare for its management", "presupposes that surgery is recommended for gastric cancer"]}

**검토:** 장루 관리와 옷 준비가 개인 계획으로 남는다. 모든 위암 수술 필수 장루라는 일반화는 피해야 한다.

**Terra 관련 추출:**

- focusing on assistance with colostomy bag care → The friend is arranging home healthcare that includes colostomy-bag care. (individual_context; not_verifiable_from_question)
- special clothing and swimwear designed to accommodate the bag → The friend is looking into clothing and swimwear intended to accommodate an ostomy bag. (individual_context; not_verifiable_from_question)

## nfp_1083 — R

My uncle had surgery to remove a GIST last year.  He's feeling great now and back to his normal routine. He's even planning a big trip overseas next year.  Since the tumor is gone, he's not worried about it coming back.  Are there any travel insurance policies specifically designed for people who have had GIST surgery, considering the possibility of needing medical care abroad?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that once a GIST is surgically removed, there is no need to worry about the recurrence"]}

**검토:** 절제했으니 재발 걱정 없음이라는 연결이 general_claim으로 남고 Terra apparent_error다. NFP 보호 효과가 개인 사실만 보호한 것은 아니다.

**Terra 관련 추출:**

- Since the tumor is gone, he's not worried about it coming back. → The speaker presents the tumor being gone as a reason not to worry about recurrence. (general_claim; apparent_error)

## nfp_1093 — R

My grandfather was recently diagnosed with hepatocellular carcinoma. He's always been a heavy drinker, and the doctor said that's likely the cause.  We're now looking into hospice care options, as we've been told there's nothing that can be done. Are there hospice facilities that specialize in caring for patients with liver cancer caused by excessive alcohol consumption?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a history of heavy alcohol use may be the likely cause of the diagnosis of hepatocellular carcinoma"]}

**검토:** 음주 원인이라는 의사 보고와 치료 불가 전달이 둘 다 남는다. TPQ는 전자만 다뤄 gate의 실제 목표는 알 수 없다.

**Terra 관련 추출:**

- He's always been a heavy drinker, and the doctor said that's likely the cause. → A doctor reportedly considered the grandfather's heavy drinking a likely cause of his cancer. (individual_context; disputed_or_context_dependent)
- we've been told there's nothing that can be done → The family was reportedly told that no treatment or other intervention can be done. (individual_context; not_verifiable_from_question)

## nfp_1094 — R

My brother was diagnosed with Hodgkin lymphoma and just finished chemotherapy. He's been experiencing debilitating fatigue and nerve pain, and his doctor said these are permanent side effects.  Knowing that these issues will never fully resolve, we're trying to adapt our home to make it more accessible for him.  Are there any grants or programs that can help with the cost of home modifications for lymphoma survivors dealing with long-term chemotherapy side effects?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that chemotherapy for Hodgkin lymphoma causes permanent, debilitating side effects like fatigue and nerve pain"]}

**검토:** 의사의 영구 후유증 보고와 회복 불가 인식이 남는다. 확인 불가와 거짓은 다르지만 실제 gate 이유는 기록되지 않았다.

**Terra 관련 추출:**

- his doctor said these are permanent side effects → A doctor reportedly characterized the fatigue and nerve pain as permanent treatment side effects. (individual_context; not_verifiable_from_question)
- these issues will never fully resolve → The speaker asserts that the brother's fatigue and nerve pain will not fully resolve. (individual_context; disputed_or_context_dependent)

## nfp_1108 — R

My father has midline tract carcinoma with NUT gene changes in his foot.  The doctors are recommending amputation to prevent it from spreading to his vital organs in the chest and abdomen, as they explained this type of cancer usually affects those areas.  We're struggling with this decision.  Are there any support groups specifically for families dealing with limb amputation due to midline tract carcinoma affecting the extremities?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that midline tract carcinoma with NUT gene changes primarily threatens organs in the chest and abdomen", "presupposes that amputation is effective for carcinoma with NUT gene changes in foot"]}

**검토:** 의사 절단 권고와 일반 발생 위치 주장이 함께 남는다. 어떤 쪽이 판정 전환을 만들었는지는 모른다.

**Terra 관련 추출:**

- The doctors are recommending amputation to prevent it from spreading to his vital organs in the chest and abdomen → Doctors reportedly recommend amputation to prevent this individual's cancer from spreading to vital organs in the chest and abdomen. (individual_context; not_verifiable_from_question)
- this type of cancer usually affects those areas → This type of cancer usually affects the chest and abdomen. (general_claim; disputed_or_context_dependent)

## nfp_1119 — R

My aunt has multiple myeloma and is scheduled for radiation therapy next week. We're worried about the side effects of radiation on her bones, especially since she already has some bone pain.  Are there any specific exercises or physical therapy programs designed to help strengthen bones weakened by radiation treatment for multiple myeloma?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that radiation therapy is a standard treatment for multiple myeloma", "presupposes that radiation therapy for multiple myeloma worsens bone pain"]}

**검토:** 예정 방사선은 남지만 뼈 약화 우려는 inquiry다. TPQ는 치료 표준/통증 악화를 추가로 서술한다.

**Terra 관련 추출:**

- is scheduled for radiation therapy next week → The aunt is scheduled to receive radiation therapy the following week. (individual_context; not_verifiable_from_question)

## nfp_1121 — R

My sister is 12 weeks pregnant and just received the devastating news that she has breast cancer. Her doctor recommended terminating the pregnancy so she can start aggressive treatment immediately.  We're heartbroken, but understand it's necessary for her survival.  Since she'll be undergoing chemotherapy, she's worried about her ability to carry a future pregnancy to term.  Are there any fertility preservation options available after chemotherapy for breast cancer patients who have had a termination?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that terminating a pregnancy is medically necessary for breast cancer treatment and survival"]}

**검토:** 의사 권고와 본인 생존상 필요라는 설명이 남는다. 출산 뒤/치료 뒤 생식력 문의는 별도 대상이다.

**Terra 관련 추출:**

- Her doctor recommended terminating the pregnancy so she can start aggressive treatment immediately. → Her doctor reportedly recommended pregnancy termination to permit immediate aggressive treatment. (individual_context; not_verifiable_from_question)
- it's necessary for her survival → Terminating this pregnancy is necessary for the sister's survival. (individual_context; disputed_or_context_dependent)

## nfp_1128 — R

I'm 12 weeks pregnant and just received a devastating Hodgkin lymphoma diagnosis. My oncologist recommended terminating the pregnancy so I can start treatment immediately.  We're heartbroken, but want to do what's best for my health.  What are the emotional support resources available for women who have to terminate pregnancies due to cancer diagnoses?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a Hodgkin lymphoma diagnosis during pregnancy necessitates termination to allow for immediate treatment"]}

**검토:** 의사의 중절 권고는 남고 환자 집단 요청은 ambiguous다. 질환 전체의 필수 중절로 강화하지 않는다.

**Terra 관련 추출:**

- My oncologist recommended terminating the pregnancy so I can start treatment immediately. → The speaker reports that their oncologist recommended pregnancy termination to permit immediate treatment. (individual_context; not_verifiable_from_question)

## nfp_1129 — R

My daughter is 20 weeks pregnant and has just been diagnosed with Non-Hodgkin's Lymphoma.  We're devastated, of course. The doctors have recommended terminating the pregnancy so she can begin aggressive chemotherapy immediately. We're heartbroken but understand it's necessary to save her life.  Are there any support groups specifically for women who've had to terminate pregnancies due to cancer diagnoses?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a Non-Hodgkin's Lymphoma diagnosis during pregnancy necessitates pregnancy termination"]}

**검토:** 의사의 권고와 가족의 생존 필요 인식이 둘 다 남아 있다.

**Terra 관련 추출:**

- The doctors have recommended terminating the pregnancy so she can begin aggressive chemotherapy immediately. → The daughter's doctors reportedly recommended pregnancy termination to permit immediate aggressive chemotherapy. (individual_context; not_verifiable_from_question)
- understand it's necessary to save her life → The writer believes termination is necessary to save the daughter's life. (individual_context; disputed_or_context_dependent)

## nfp_1131 — R

My daughter-in-law is 28 weeks pregnant and was just diagnosed with non-Hodgkin lymphoma. Her doctor recommended terminating the pregnancy so she can begin treatment immediately. We're all heartbroken, but we understand it's necessary.  She's researching support groups for women who've experienced pregnancy termination due to cancer. Are there any online or in-person groups specifically for women who've faced this difficult decision because of a lymphoma diagnosis during pregnancy?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that pregnancy termination is the necessary course of action following a non-Hodgkin lymphoma diagnosis during pregnancy to begin treatment immediately"]}

**검토:** 동일한 개인 필요성이 Terra에서 general_claim으로 분류됐다. nfp_1129의 개인 상황 코드와 경계가 다르다.

**Terra 관련 추출:**

- Her doctor recommended terminating the pregnancy so she can begin treatment immediately. → Her doctor recommended pregnancy termination in order to begin treatment immediately. (individual_context; not_verifiable_from_question)
- we understand it's necessary → Pregnancy termination is necessary in this situation. (general_claim; disputed_or_context_dependent)

## nfp_1132 — R

My niece is 15 weeks pregnant and just found out she has non-Hodgkin lymphoma.  Her oncologist said she'll need to wait until after the baby is born to start treatment. We're concerned about the delay, but we understand it's unavoidable.  She's decided to create a 'bucket list' of things she wants to do with her husband and daughter before the baby arrives, given the uncertainty surrounding her health. Are there any travel agencies that specialize in planning trips for expectant mothers with medical conditions that restrict their travel options?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that treatment for non-Hodgkin lymphoma needs to be postponed until after birth delivery"]}

**검토:** 의사 치료 지연 보고와 불가피하다는 이해가 남는다. 모든 임신부의 치료 불가 일반론은 아니다.

**Terra 관련 추출:**

- Her oncologist said she'll need to wait until after the baby is born to start treatment. → The niece’s oncologist reportedly said treatment must wait until after delivery. (individual_context; disputed_or_context_dependent)
- we understand it's unavoidable → The user understands the treatment delay to be unavoidable. (individual_context; disputed_or_context_dependent)

## nfp_1133 — R

My father was recently diagnosed with primary CNS lymphoma.  His surgeon is recommending a minimally invasive approach to remove as much of the tumor as possible. We're trying to prepare for his recovery at home after the surgery. Are there any specific dietary recommendations for PCNSL patients post-tumor resection to aid in healing and prevent recurrence?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that surgical resection is effective approach for primary CNS lymphoma"]}

**검토:** PCNSL 절제에 대한 외과의 권고가 남는다. FPQ의 PCNSL 수술 준비와 문장 구조가 겹친다.

**Terra 관련 추출:**

- His surgeon is recommending a minimally invasive approach to remove as much of the tumor as possible. → The father's surgeon recommends a minimally invasive approach intended to remove as much tumor as possible. (individual_context; not_verifiable_from_question)

## nfp_1139 — R

My mother was recently diagnosed with primary peritoneal cancer.  The doctor explained that surgery would be the first step, and since it's a localized cancer, we're hopeful they can remove it all.  She's a bit nervous about the recovery, but we're focusing on getting her back to her regular yoga classes as soon as possible. Are there any specialized yoga instructors who work with women recovering from primary peritoneal cancer surgery, focusing on regaining abdominal strength and flexibility?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that surgery is the effective first step for localized primary peritoneal cancer", "presupposes that surgery is guaranteed to remove all of localized primary peritoneal cancer"]}

**검토:** 수술 권고와 완전 제거 희망이 보존됐다. TPQ의 guaranteed는 질문의 hopeful보다 강하다.

**Terra 관련 추출:**

- The doctor explained that surgery would be the first step → A doctor reportedly said surgery would be the mother's first treatment step. (individual_context; not_verifiable_from_question)
- they can remove it all → The speaker hopes surgery can remove all of the mother's cancer. (individual_context; not_verifiable_from_question)

## nfp_1143 — R

My uncle has transitional cell carcinoma of the ureter, and the doctor said it's advanced. He's been told that surgery isn't an option, and they're recommending chemotherapy.  Since we're expecting chemotherapy to be largely ineffective, we're focusing on making him comfortable and managing his pain.  Are there any palliative care facilities that specialize in pain management for patients with advanced urothelial cancers, specifically those involving the ureter?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that surgery is not an option for advanced transitional cell carcinoma of the ureter", "presupposes that chemotherapy is largely ineffective for advanced transitional cell carcinoma of the ureter", "presupposes that chemotherapy is effective for advanced transitional cell carcinoma of the ureter when surgery is not an option"]}

**검토:** 가족의 항암 무효 예상이 개인 상황으로 남는다. TPQ 목록에 항암 효과 있음/없음이 함께 있어 하나의 정답 명제로 쓸 수 없다.

**Terra 관련 추출:**

- we're expecting chemotherapy to be largely ineffective → The speaker's family expects chemotherapy to be largely ineffective for the uncle. (individual_context; disputed_or_context_dependent)

## nfp_1147 — R

My brother was diagnosed with stage IV soft tissue sarcoma. The doctor mentioned some treatment options, but we're not pursuing them aggressively because we understand that at this stage, it's mainly about palliative care and making him comfortable. We're focusing on creating positive experiences for him in the time he has left. Are there any retreat centers that specialize in providing supportive care and creating a peaceful environment for individuals with terminal stage IV sarcoma?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that a there is no hope for a stage IV soft tissue sarcoma diagnosis"]}

**검토:** 해당 병기의 완화 중심 이해가 general_claim으로 남는다. 개인 사실 코드만으로 보호 효과를 설명할 수 없다.

**Terra 관련 추출:**

- at this stage, it's mainly about palliative care and making him comfortable → The speaker understands this stage of the brother's cancer to be mainly managed with palliative, comfort-focused care. (general_claim; disputed_or_context_dependent)
- individuals with terminal stage IV sarcoma → The brother is characterized as having terminal stage IV sarcoma. (individual_context; not_verifiable_from_question)

## nfp_1148 — R

My 4-year-old daughter was diagnosed with Wilms tumor, and thankfully, the doctor said it's localized.  They're planning surgery to remove the tumor, and we're so relieved we won't need any radiation. We're already looking at preschools for her to start in the fall. Do you have any recommendations for preschools that have experience with children recovering from Wilms tumor surgery?

**원 주석:** {"premise": null, "correction": "", "tpq_premises": ["presupposes that localized Wilms tumor does not require radiation therapy, and a removal surgery would be sufficient"]}

**검토:** 수술 계획과 방사선 불필요가 남는다. 국소화라 항상 방사선 불필요라는 범위로 확대하지 않는다.

**Terra 관련 추출:**

- They're planning surgery to remove the tumor → The speaker reports a plan for surgery to remove the child's tumor. (individual_context; not_verifiable_from_question)
- we won't need any radiation → The speaker asserts that the child will not need radiation. (individual_context; disputed_or_context_dependent)
