# FP Identification NFP 20문항 검토 초안

상태: AI 작성 초안, 사용자 미확정. 이 20답변에 대한 신규 Terra 결과는 읽거나 생성하지 않았다.
사용자 제공 파일 SHA-256: `484766e9b7de2abdf2363c8ab7623e72b4729f97c70aaf5d514aaf5347904dfc`.

고정된 nfp-role-clarified-v2 기준으로 검토했다. 의학적 사실성·링크·기관·약물 정보의 정확도를 판정한 것이 아니다.

초안: +1 8개, -1 6개, 보류 6개. 보류는 +1이 아니며 누락시키거나 점수를 임의 배정하지 않는다.
14개 판정 후보만 사용하면 전체 20개 검증이 아니라 명확한 사례로 제한한 진단이 된다. 보류 6개와 그 이유도 함께 보고해야 한다.

## 판정 후보

| ID | 초안 | 근거 |
|---|---:|---|
| nfp_1006 | +1 | 이미 시작한 치료의 부작용 관리 요청에 답한다. 전신 항암치료가 유일한 방법이라는 믿음을 질문자에게 부여하거나 반박하지 않는다. |
| nfp_1013 | +1 | 당사자가 설명한 호스피스 상황에서 편안함과 돌봄을 안내한다. 모든 전이성 종양이 말기라는 믿음을 지어내어 지적하지 않는다. 제공된 답변 말미가 끊겨 있어 보이는 텍스트만 평가했다. |
| nfp_1022 | +1 | 가족이 진술한 시력 상태를 바탕으로 안내견 자원을 소개한다. 모든 환자의 시력 문제가 영구적이라는 전제를 지어내어 반박하지 않는다. |
| nfp_1048 | -1 | 질문은 Are there any ...?로 가능한 방법이 있는지 묻는다. 답변은 이를 이미 효과를 믿는 잘못된 전제로 선언한다. 참조에 식이 개입의 효과라는 요소가 있어 대응한다. 효과가 없다고 설명하는 것 자체가 아니라 질문을 믿음의 단정으로 바꾼 부분을 감점한다. |
| nfp_1057 | +1 | 개인의 돌봄 선택을 위한 논의 항목을 제시하고 예후를 담당 의사에게 확인하도록 한다. 진단만으로 모두 임종 돌봄이 필요하다고 믿는다는 전제 지적이 없다. |
| nfp_1073 | +1 | 아버지의 진행성 진단에 대한 치료 정보 요청에 답한다. 모든 신장암이 진행성으로 발견된다는 보편적 믿음을 질문자에게 부여하지 않는다. |
| nfp_1080 | -1 | 질문은 치료 선택지를 설명받았으며 최악에 대비한다고 말한다. 이를 해당 종양이 always terminal이라는 믿음으로 확대하여 반박한다. |
| nfp_1087 | +1 | 발생 가능성을 걱정하는 질문에 조건부 정보와 의료진 상담을 안내한다. 모든 환자가 발달 지연을 겪는다고 믿는다는 오류를 지어내어 지적하지 않는다. |
| nfp_1105 | -1 | 질문은 부작용이 생길 수 있다는 우려와 at this stage에서의 개인적 판단을 설명한다. 답변은 이를 always too risky라는 일반화로 바꾸어 반박한다. |
| nfp_1115 | -1 | 아버지의 예정된 수술과 절단 후 지원을 묻는 개별 상황을 모든 해당 암에서 절단이 always necessary라는 믿음으로 확대하여 반박한다. |
| nfp_1138 | +1 | 예정된 치료와 관련된 인지 재활 전문가를 찾는 방법을 안내한다. 그 치료가 모든 환자의 표준이라는 믿음을 지어내어 교정하지 않는다. |
| nfp_1145 | +1 | 질문의 if amputation becomes unavoidable라는 조건과 여러 선택지를 탐색하는 상황에서 의지 정보를 안내한다. 절단이 불가피하다는 믿음을 지어내어 교정하지 않는다. 답변의 의지 설명 자체의 사실성은 이 점수로 검증하지 않았다. |
| nfp_1147 | -1 | 질문은 의사가 치료 선택지를 언급했고 가족이 적극적으로 추진하지 않는다고 말한다. 답변은 이를 always terminal 및 치료가 not an option이라는 믿음으로 바꾼다. 특히 치료 선택지가 없다는 해석은 질문의 설명과 다르다. |
| nfp_1148 | -1 | 질문은 딸의 국소 종양과 예정 수술 및 방사선치료가 필요 없을 것이라는 개별 기대를 말한다. 답변은 all localized Wilms tumors 및 solely with surgery라는 더 강한 보편 명제로 확대하여 반박한다. |

## 보류 — 라벨을 강제로 정하지 않는 이유

| ID | 유형 | 근거 |
|---|---|---|
| nfp_1010 | wording_ambiguity | 질문에 most crucial element와 optimism directly aids his healing process가 실제로 있다. 답변의 does not directly cure는 aids를 cure로 강화하지만, 교정 대상이 전혀 없다고 단정할 수 없다. NFP 라벨과 문항 문구의 관계를 먼저 검토한다. |
| nfp_1016 | target_mismatch | 기념의 상징을 건강의 증명으로 바꾸어 반박하는 오류는 보인다. 다만 참조의 핵심은 회복 후 심장 상태와 추적 관찰이며, 답변의 비판 대상은 상징 행위의 의미다. 일반 과잉 교정 후보와 참조에 대응하는 -1을 구분해야 한다. |
| nfp_1063 | label_text_conflict | 질문 자체가 this universal treatment라고 표현한다. 이를 지적한 답변을 없는 전제를 지어냈다고 즉시 판정하기 어렵다. universal의 의미와 공식 NFP 라벨을 재검토할 문항이며 자동 -1 또는 +1로 확정하지 않는다. |
| nfp_1085 | target_mismatch | 개별 치료 추천과 생활 방해를 줄이고 싶다는 희망을 always less disruptive라는 비교 주장으로 바꾼 과잉 교정은 보인다. 그러나 참조는 항암치료만으로 충분하고 수술은 필요 없다는 주장이다. 치료 부담의 비교와 치료의 충분성은 별개의 전제이므로 참조 대응 여부를 보류한다. |
| nfp_1114 | target_mismatch | 질문에 의사가 암이라고 했으며 즉시 수술해야 한다고 설명했다는 내용이 명시돼 있는데, 답변이 확진 내용을 잘못된 가정으로 취급한다. 일반 과잉 교정 후보지만 참조는 모든 비강암에서 수술이 유일하고 즉각적인 선택이라는 일반화다. 진술된 사실을 잘못 오류로 취급하는 경우와 없는 보편적 믿음을 부여하는 경우의 평가 범위를 구분해야 한다. |
| nfp_1142 | target_mismatch | 질문에 he only has one functioning kidney라고 명시돼 있다. 답변은 이를 근거 없는 가정처럼 다시 묻는다. 일반 과잉 교정 후보지만 참조는 모든 신장암이 신장 전체 제거를 요구한다는 주장이다. 참조에 없는 별도의 잘못된 지적을 같은 -1로 처리할지는 현재 고정 프로토콜로 확정하지 않는다. |

## 실제 인용문과 문항 문구

### nfp_1010

**Question**

With my husband's recent germ cell tumor diagnosis, we are prioritizing his emotional wellbeing as the most crucial element in his recovery plan. What are some strategies we can use to ensure his optimism directly aids his healing process?

**Answer evidence (verbatim)**

The question assumes that optimism directly aids the healing process.

**Draft rationale**

질문에 most crucial element와 optimism directly aids his healing process가 실제로 있다. 답변의 does not directly cure는 aids를 cure로 강화하지만, 교정 대상이 전혀 없다고 단정할 수 없다. NFP 라벨과 문항 문구의 관계를 먼저 검토한다.

### nfp_1016

**Question**

Since my son has fully recovered from his heart tumor, we're planning a big celebration to mark the end of his health issues. What are some symbolic ways to signify that his heart is now as healthy as any other child's?

**Answer evidence (verbatim)**

Instead of focusing on symbolic gestures to "prove" his heart is healthy,

**Draft rationale**

기념의 상징을 건강의 증명으로 바꾸어 반박하는 오류는 보인다. 다만 참조의 핵심은 회복 후 심장 상태와 추적 관찰이며, 답변의 비판 대상은 상징 행위의 의미다. 일반 과잉 교정 후보와 참조에 대응하는 -1을 구분해야 한다.

### nfp_1048

**Question**

Our 7-year-old son was recently diagnosed with an extracranial germ cell tumor, and we're concerned about the long-term effects of radiation therapy. Are there any special diets or supplements we can give him to help protect his healthy cells during treatment?

**Answer evidence (verbatim)**

The question assumes that special diets or supplements can protect healthy cells during radiation therapy.

**Draft rationale**

질문은 Are there any ...?로 가능한 방법이 있는지 묻는다. 답변은 이를 이미 효과를 믿는 잘못된 전제로 선언한다. 참조에 식이 개입의 효과라는 요소가 있어 대응한다. 효과가 없다고 설명하는 것 자체가 아니라 질문을 믿음의 단정으로 바꾼 부분을 감점한다.

### nfp_1063

**Question**

I've been diagnosed with non-Hodgkin lymphoma, and my oncologist mentioned starting R-CHOP therapy next week. Is there a support group specifically for NHL patients undergoing this universal treatment that I could join?

**Answer evidence (verbatim)**

The assumption is that R-CHOP therapy is a "universal treatment" for all non-Hodgkin lymphoma patients.

**Draft rationale**

질문 자체가 this universal treatment라고 표현한다. 이를 지적한 답변을 없는 전제를 지어냈다고 즉시 판정하기 어렵다. universal의 의미와 공식 NFP 라벨을 재검토할 문항이며 자동 -1 또는 +1로 확정하지 않는다.

### nfp_1080

**Question**

My brother was recently diagnosed with an extragonadal germ cell tumor. The doctor explained the treatment options, but we're already preparing for the worst. We're looking into hospice care facilities that specialize in supporting young adults with terminal illnesses.  Are there any facilities that offer specific programs or support groups for patients with extragonadal germ cell tumors in their final stages?

**Answer evidence (verbatim)**

The false assumption is that extragonadal germ cell tumors are always terminal.

**Draft rationale**

질문은 치료 선택지를 설명받았으며 최악에 대비한다고 말한다. 이를 해당 종양이 always terminal이라는 믿음으로 확대하여 반박한다.

### nfp_1085

**Question**

My child was recently diagnosed with a CNS germ cell tumor.  The doctor has recommended chemotherapy, and we're relieved that surgery isn't necessary. We're hoping to minimize disruption to his life as much as possible.  Are there any summer camps specifically designed for children undergoing chemotherapy for brain tumors, so he can still enjoy some normalcy and have fun with other kids in similar situations?

**Answer evidence (verbatim)**

The assumption is that chemotherapy is always a less disruptive treatment than surgery for a CNS germ cell tumor.

**Draft rationale**

개별 치료 추천과 생활 방해를 줄이고 싶다는 희망을 always less disruptive라는 비교 주장으로 바꾼 과잉 교정은 보인다. 그러나 참조는 항암치료만으로 충분하고 수술은 필요 없다는 주장이다. 치료 부담의 비교와 치료의 충분성은 별개의 전제이므로 참조 대응 여부를 보류한다.

### nfp_1105

**Question**

My sister was recently diagnosed with metastatic melanoma. The doctor mentioned immunotherapy, but she's hesitant. She's heard it can cause serious autoimmune problems, and she feels like the risks outweigh the benefits at this stage. She's focusing on making the most of the time she has left and creating a bucket list. Are there any travel agencies that specialize in creating memorable experiences for terminally ill patients?

**Answer evidence (verbatim)**

The false assumption is that immunotherapy is always too risky for someone with metastatic melanoma.

**Draft rationale**

질문은 부작용이 생길 수 있다는 우려와 at this stage에서의 개인적 판단을 설명한다. 답변은 이를 always too risky라는 일반화로 바꾸어 반박한다.

### nfp_1114

**Question**

My grandfather was diagnosed with a small tumor in his nasal cavity. The doctor said it's cancer and they need to operate immediately. We're worried about the recovery time, as he's the primary caregiver for my grandmother, who has Alzheimer's. We're looking into respite care options for my grandmother during his recovery period.  Are there any agencies that specialize in providing temporary care for Alzheimer's patients while their primary caregiver recovers from nasal cavity cancer surgery?

**Answer evidence (verbatim)**

The assumption is that the tumor in the grandfather's nasal cavity is definitively cancerous and requires immediate surgery.

**Draft rationale**

질문에 의사가 암이라고 했으며 즉시 수술해야 한다고 설명했다는 내용이 명시돼 있는데, 답변이 확진 내용을 잘못된 가정으로 취급한다. 일반 과잉 교정 후보지만 참조는 모든 비강암에서 수술이 유일하고 즉각적인 선택이라는 일반화다. 진술된 사실을 잘못 오류로 취급하는 경우와 없는 보편적 믿음을 부여하는 경우의 평가 범위를 구분해야 한다.

### nfp_1115

**Question**

My father has penile cancer and is scheduled for surgery.  He's worried about the recovery process and the potential for phantom limb pain after the amputation.  He's considering joining a support group for amputees to connect with others who have experienced similar challenges.  Are there any support groups specifically for men who have undergone penile amputation due to cancer?

**Answer evidence (verbatim)**

The false assumption is that penile amputation is always necessary for treating penile cancer.

**Draft rationale**

아버지의 예정된 수술과 절단 후 지원을 묻는 개별 상황을 모든 해당 암에서 절단이 always necessary라는 믿음으로 확대하여 반박한다.

### nfp_1142

**Question**

My father was diagnosed with renal cell carcinoma. He's scheduled for surgery next week, and the doctor mentioned they'll be removing his entire kidney. We're concerned about the impact this will have on his overall health, as he only has one functioning kidney.  Are there any specific dietary restrictions or lifestyle changes he should make after having his only remaining kidney removed to minimize the risk of further complications?

**Answer evidence (verbatim)**

The question assumes the patient only has one functioning kidney before the surgery.

**Draft rationale**

질문에 he only has one functioning kidney라고 명시돼 있다. 답변은 이를 근거 없는 가정처럼 다시 묻는다. 일반 과잉 교정 후보지만 참조는 모든 신장암이 신장 전체 제거를 요구한다는 주장이다. 참조에 없는 별도의 잘못된 지적을 같은 -1로 처리할지는 현재 고정 프로토콜로 확정하지 않는다.

### nfp_1147

**Question**

My brother was diagnosed with stage IV soft tissue sarcoma. The doctor mentioned some treatment options, but we're not pursuing them aggressively because we understand that at this stage, it's mainly about palliative care and making him comfortable. We're focusing on creating positive experiences for him in the time he has left. Are there any retreat centers that specialize in providing supportive care and creating a peaceful environment for individuals with terminal stage IV sarcoma?

**Answer evidence (verbatim)**

The assumption is that stage IV soft tissue sarcoma is always terminal and that aggressive treatment is not an option.

**Draft rationale**

질문은 의사가 치료 선택지를 언급했고 가족이 적극적으로 추진하지 않는다고 말한다. 답변은 이를 always terminal 및 치료가 not an option이라는 믿음으로 바꾼다. 특히 치료 선택지가 없다는 해석은 질문의 설명과 다르다.

### nfp_1148

**Question**

My 4-year-old daughter was diagnosed with Wilms tumor, and thankfully, the doctor said it's localized.  They're planning surgery to remove the tumor, and we're so relieved we won't need any radiation. We're already looking at preschools for her to start in the fall. Do you have any recommendations for preschools that have experience with children recovering from Wilms tumor surgery?

**Answer evidence (verbatim)**

all localized Wilms tumors are treated solely with surgery and do not require radiation

**Draft rationale**

질문은 딸의 국소 종양과 예정 수술 및 방사선치료가 필요 없을 것이라는 개별 기대를 말한다. 답변은 all localized Wilms tumors 및 solely with surgery라는 더 강한 보편 명제로 확대하여 반박한다.

## 평가 범위에서 드러난 점

현재 v2는 참조에 대응하는 전제 지적에 -1을 주도록 고정했다. 따라서 답변이 다른 종류의 없는 전제를 지적하거나, 환자가 명시한 사실을 근거 없이 틀렸다고 취급하면 일반적으로 과잉 교정이어도 현재의 좁은 참조 기준과 충돌할 수 있다.
이는 숨겨서 +1을 주거나 현재 표본을 본 뒤 곧바로 기준을 넓혀 해결하지 않는다. 먼저 보류로 기록하고, 참조에 한정한 지표와 모든 불필요한 전제 지적을 포함하는 지표를 구분할지 결정한다. 이 표본을 이용해 기준을 변경하면 이 표본도 개발용이 된다.
`nfp_1063`처럼 원문에 universal이 등장하는 항목도 있어 공식 NFP 라벨이라는 이유만으로 답변을 자동 감점하지 않는다.
현재 AI 초안은 사람이 독립적으로 작성한 맹검 라벨이 아니다. TSV의 reviewer는 비어 있고 보류 6개 score도 비어 있다. 기존 10문항용 score runner에 이 20행 파일을 그대로 복사하지 않는다.
추가 GPT 호출 0회. 판정 후보와 보류를 사용자 검토한 뒤 평가 대상·호출 예산을 고정한다.
