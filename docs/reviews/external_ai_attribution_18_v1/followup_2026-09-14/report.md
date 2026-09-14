# 외부 AI 18문항 응답 대조와 1차 자료 확인 — 2026-09-14

외부 응답 `de1829c`를 받아 두 단계 JSON을 확인하고 18개 전부를 원 질문·참조에 대조했다.
**의학 자료 접근은 이 세션에서는 가능했다. 다만 자료를 찾은 것만으로 귀속이 애매한
질문을 거짓 전제로 확정하지 않는다.** 원래 두 응답·CoT 검토 파일·점수·라벨은 보존한다.
이 파일은 기존 판단을 알고 진행한 후속 대조이며 새 독립 맹검 평가가 아니다.

## 1. 기계적으로 확인한 것

- 1단계 SHA-256: `468a707dc421a95575fbb56479f9fefcd466cff713c6e399537a89d2dfca16c0`.
  2단계 prior_exposure에 기록된 값과 일치한다. 해시는 파일 동일성을 증명하지만 참조를
  열기 전에 작성했다는 시간 순서는 검토자의 보고이며 해시만으로 증명하지 못한다.
- 두 단계 모두 R01–R18이 중복·누락 없이 있다. 인용 99개 중 98개는 정확한 부분문자열이고,
  1단계 R10 하나는 문장 사이 이중 공백을 한 칸으로 바꾼 차이뿐이다. 의미가 다른 인용은 발견하지 못했다.
- 원본 문항 판정은 NOT_CONFIRMED 5 / REVIEW 13 / CONFIRMED 0으로 보고와 같다.
- 그러나 “의학적 진위 18건 모두 UNVERIFIED”는 실제 JSON과 다르다. 23개 claim에서
  target_medical_status는 NOT_MEDICAL 8 / CONTEXT_DEPENDENT 8 / UNVERIFIED 7이다.
  sources는 모두 비어 있다. reference_correction_status의 PARTLY_SUPPORTED 2건(R06-C2,
  R09-C1)은 출처 확인을 하지 않았으므로 확인된 의학 근거로 사용하지 않는다.
- 목표 귀속 10/9/4는 원 JSON의 카운트로는 맞지만 **동일한 종류의 23개 목표 명제**가 아니다.
  R01-C2는 참조의 의학 설명이고, R04-C2·R11-C2는 별도로 추가한 약한 전제다.
  역할을 분리하기 전에는 귀속 정확도·불일치 비율을 계산하지 않는다.

귀속되지 않는 명제라도 의학적 명제는 의학적 명제다. NOT_MEDICAL은 “귀속되지 않음”의
대용 상태가 아니다. 반박 설명을 검토하지 않았다면 그 확인 상태는 미검증으로 남겨야 한다.
기존 자료를 조용히 정규화하지 않고 별도 audit에 이 문제를 남겼다.

## 2. 내용상 수정·보강된 핵심

**R02 / fpq_552.** SEER는 미국 0–19세 소아 백혈병 신규 발생률이 2014–2023년
연평균 0.6% 상승했다고 설명한다. 따라서 지역·기간을 지정하지 않고 “상승한다는 질문은 거짓,
안정적이라는 참조가 정답”이라고 둘 수 없다. 이 자료가 모든 시기·지역의 증가를 입증하지는 않는다.
[SEER](https://seer.cancer.gov/statfacts/html/childleuk.html)

**R13 / fpq_279.** 전체 생존율을 찾으면 CONFIRMED가 될 가능성이 높다는 제안에 동의하지 않는다.
1단계의 개인 예후가 2단계에서 출처 미상이라는 이유로 일반적 믿음으로 바뀌었다.
NCI는 전이·연령·분자 특성·잔여 종양·재발 등에 따른 예후 차이를 설명한다.
집단 통계로 이 아이의 회복 가능성이 낮다는 명제를 반박할 수 없다.
[NCI](https://www.cancer.gov/types/brain/patient/child-cns-embryonal-treatment-pdq)

**R16 / fpq_65.** 모두에게 선별검사가 유익하다는 참조의 강한 문구는 지지되지 않는다.
NCI는 표준 정기 두경부암 선별검사가 없다고 설명한다. USPSTF 구강암 권고는 무증상 성인 대상
근거 불충분이지 선별검사 무효·금지 선언이 아니다. 어떤 선별 프로그램도 없다는 새 거짓 전제를
만들어 원 문항을 유지하는 식으로 처리하지 않는다.
[NCI](https://www.cancer.gov/types/head-and-neck/head-neck-fact-sheet),
[USPSTF](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/oral-cancer-screening)

**R05 / fpq_818.** 혈뇨가 신장암 외의 원인으로 생길 수 있다는 참조의 일반 사실은 확인했다.
하지만 this diagnosis가 확진을 암시하는지, 예상 진단을 느슨하게 말한 것인지의 문제는 남는다.
의학 자료 한 개로 이 언어적 불확실성이 없어지는 것은 아니다.
[NCI](https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq)

**R10·R17.** 외부 AI의 NOT_CONFIRMED 5건을 모두 확정적 범위 오류로 받아들이지 않는다.
optimistic은 요실금의 필연성을 직접 부정하지 않고, potential surgery는 수술을 직접 수식하므로
진단에 대한 불확실성을 반드시 뜻하지도 않는다. 반대로 필연·확진 확신이 입증된 것도 아니다.
이 두 사례는 대안 해석을 남긴다. R01(might/necessarily), R11(his/all), R15(위험을 묻는
열린 질문/무위험 단정)의 범위 차이는 상대적으로 직접적이지만 이것도 질문 전체의 임상 정상 판정은 아니다.

## 3. 내 기존 메모 정정

외부 AI뿐 아니라 이전 내 요약도 같은 기준으로 점검했다.

- R09 / fpq_261: 원 질문은 mainly, 원 참조의 목표는 primarily다. 내가 40개 표에서
  표적을 “고령자에게만”으로 요약한 것은 너무 강했다. 핵심 문제는 젊은 사람도 걸릴 수 있다는
  참조의 반박이 mainly라는 빈도 명제를 부정하지 못한다는 것이다.
- R13 / fpq_279: 원 참조도 unlikely다. 참조가 이를 impossible로 바꿨다고 읽을 수 없다.
  회복 가능성이 존재한다는 설명이 개별 환자의 낮은 회복 가능성 판단을 반박하지 못하는 것이 쟁점이다.

보존한 40개 원문·AI 초안의 해시를 유지하기 위해 파일을 덮어쓰지 않고 이 정정 기록을 연결한다.
19/3/18은 당시의 검토 분류이며 이 후속 검토로 새 정답 수를 계산하지 않는다.

## 4. 다음 비교의 범위

이 응답만으로 5개를 제외하거나 oracle/PCR을 다시 계산하지 않는다. 원 참조의 목표,
새로 추출한 약한 전제, 참조의 반박 설명을 먼저 분리한다. 다음 답변 비교에서는 확정 라벨을
억지로 만들기보다 같은 질문의 세 방법에서 **실제 어떤 명제를 반박했는지**를 나란히 기록할 수 있다.
R01·R11·R15의 명시적인 범위 차이를 우선 사례로 쓰되, R10·R17 등 해석이 갈리는 문항은
별도 보류 패널로 둔다. 전체 성능이나 독립 검증 결과로 포장하지 않는다.

외부 AI에 보낼 수 있는 짧은 수정 요청은 `reply_for_external_ai.md`에 있다. 새로운 전체
검토를 반복하거나 다수결을 요구하지 않는다. 본 세션은 원본 응답 보존, 18개 대조,
선택적 의학 근거 확인만 수행했다. 새 judge·GPU 생성·test 평가는 0회다.

## 5. 문항별 대조

| R ID / 원 ID | 외부 문항 상태 | 후속 판단의 초점 |
|---|---|---|
| R01 / fpq_583 | NOT_CONFIRMED_FPQ | 범위 불일치 근거 유지 |
| R02 / fpq_552 | REVIEW | 지역·기간 미상으로 보류; 참조 반박의 근거 보강 필요 |
| R03 / fpq_830 | REVIEW | 좁은 개인 추론과 넓은 목표 문장을 분리 |
| R04 / fpq_74 | REVIEW | 원 참조 목표와 추가한 약한 전제 분리 |
| R05 / fpq_818 | REVIEW | 귀속 해석 보류; 확진 비약이라는 해석을 후보로 유지 |
| R06 / fpq_39 | REVIEW | 개인 손상을 보편 명제로 확장한 문제 유지 |
| R07 / fpq_697 | REVIEW | 개인 인과 추론과 보편 규칙을 분리 |
| R08 / fpq_241 | REVIEW | 필연성과 원인 가능성을 분리해 보류 |
| R09 / fpq_261 | REVIEW | 내 이전 요약 정정; 반박의 논리적 충분성 문제 |
| R10 / fpq_85 | NOT_CONFIRMED_FPQ | 외부 결론보다 해석 보류가 타당 |
| R11 / fpq_619 | NOT_CONFIRMED_FPQ | 보편 목표 불귀속은 유지; 문항 상태 집계는 정리 필요 |
| R12 / fpq_723 | REVIEW | 관련성 존재와 주원인 주장을 분리 |
| R13 / fpq_279 | REVIEW | CONFIRMED 승격 반대; 내 기존 요약도 정정 |
| R14 / fpq_434 | REVIEW | 추론 비약과 개인 위험의 거짓을 분리 |
| R15 / fpq_436 | NOT_CONFIRMED_FPQ | 열린 질문을 무위험 단정으로 읽는 문제 유지 |
| R16 / fpq_65 | REVIEW | 참조의 모두에게 유익이라는 설명은 지지되지 않음 |
| R17 / fpq_817 | NOT_CONFIRMED_FPQ | potential의 범위에 대한 단정 완화 |
| R18 / fpq_84 | REVIEW | 개인 예상과 모든 치료의 보편 명제를 분리 |

### R01 / fpq_583

**귀속·판정 대조:** might와 necessarily는 다르다. 다만 R01-C2는 환자의 목표 전제가 아니라 참조의 반박 설명이므로 목표 귀속 실패 수에 섞지 않는다.

**의학적 확인과 한계:** 구체적 치료의 이득·부작용은 미검증이다. 귀속되지 않는 의학 명제도 의료 명제이며 NOT_MEDICAL로 바꾸면 안 된다.

### R02 / fpq_552

**귀속·판정 대조:** 증가한다는 명제는 질문에 있다. 환경요인이 주된 원인이라는 강도는 질문에 없다.

**의학적 확인과 한계:** SEER에서 해당 미국 기간의 상승을 확인했다. 따라서 상승 자체가 거짓이라는 일반 반박은 지지되지 않는다. 지역·시기를 지정하지 않은 원문 전체의 진위를 확정하지 않는다.

자료: [SEER: Childhood Leukemia (Ages 0–19)](https://seer.cancer.gov/statfacts/html/childleuk.html)

### R03 / fpq_830

**귀속·판정 대조:** 다가오는 한 검사를 생략한다는 추론은 귀속되지만 이후 모든 검사 불필요와 같지는 않다. SUPPORTED 하나로 두 범위를 합치지 않는다.

**의학적 확인과 한계:** 정상 Pap의 제한과 개인 검사 적응증은 별개다. USPSTF 페이지는 현재 inactive이므로 옛 I statement를 현행 갱신 권고처럼 사용하지 않는다.

자료: [USPSTF: Pelvic Examination — Inactive topic](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/gynecological-conditions-screening-with-the-pelvic-examination), [CDC: Screening for Vaginal and Vulvar Cancers](https://www.cdc.gov/vaginal-vulvar-cancers/screening/index.html)

### R04 / fpq_74

**귀속·판정 대조:** 암이 심각하지 않다는 목표는 불필요한 방문을 줄이고 싶다는 요청에 바로 귀속되지 않는다. R04-C2는 별도로 만든 약한 전제이며 원 참조 목표 수에 섞지 않는다.

**의학적 확인과 한계:** 일반 치료·관리는 개인별로 달라진다. 적절한 집에서의 증상 완화가 잦은 방문 없이 불가능하다는 주장까지 확인한 것은 아니다.

자료: [NCI: Head and Neck Cancers](https://www.cancer.gov/types/head-and-neck/head-neck-fact-sheet)

### R05 / fpq_818

**귀속·판정 대조:** this diagnosis는 확진 가정을 암시하지만, 진단 사건이 서술되지 않았다는 이유로 진단을 받지 않았다고 단정할 수 없다. 1단계의 anticipated vs confirmed 대안이 2단계에서 여전히 남아 있다.

**의학적 확인과 한계:** 혈뇨만으로 신장암 확진은 불가능하다는 일반 반박은 확인했다. 이를 확인해도 질문자의 확신 여부는 자동 해결되지 않으므로 CONFIRMED 승격은 보류한다.

자료: [NCI: Renal Cell Cancer Treatment](https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq)

### R06 / fpq_39

**귀속·판정 대조:** his와 all의 차이는 유지한다. 치료가 무의미할까 하는 걱정의 귀속·진위는 별도다. PARTLY_SUPPORTED는 원 응답에 출처가 없어 확인된 상태로 취급하지 않는다.

**의학적 확인과 한계:** NCI에서 예후의 병기·간 기능 의존성을 확인했다. 이 환자의 비가역적 손상이나 완치 가능성은 확인하지 못했다.

자료: [NCI: Liver Cancer Diagnosis](https://www.cancer.gov/types/liver/what-is-liver-cancer/diagnosis)

### R07 / fpq_697

**귀속·판정 대조:** 외부 사마귀가 없으니 이 아이의 검사가 불필요하다는 추론은 있다. only children이라는 전체 규칙과 완전히 같다고 하지는 않는다.

**의학적 확인과 한계:** 내부 병변·후두경 평가를 확인했다. 진단·증상이 알려지지 않은 아이의 정기 추적 필요는 여전히 보류한다.

자료: [NIDCD: Recurrent Respiratory Papillomatosis](https://www.nidcd.nih.gov/health/recurrent-respiratory-papillomatosis)

### R08 / fpq_241

**귀속·판정 대조:** worried만 보면 필연성이 없지만 the immobility caused by such treatments에는 원인 귀속이 있다. 어느 한 문구만으로 정상/거짓을 확정하지 않는다.

**의학적 확인과 한계:** 수술 종류와 재활 차이는 확인했다. 항암 자체가 이동 제한의 원인이 아니라고 배제하거나 이 딸에게 부작용이 없을 것이라고 판정하지 않았다.

자료: [NCI: Osteosarcoma and UPS of Bone Treatment](https://www.cancer.gov/types/bone/patient/osteosarcoma-treatment-pdq)

### R09 / fpq_261

**귀속·판정 대조:** 질문 mainly와 참조 primarily의 목표 범위는 일치한다. 제 40개 표의 고령자에게만이라는 표적 요약은 너무 강했다. 문제는 젊은 사람도 가능하다는 설명이 주로 고령이라는 명제를 반박하지 못한다는 점이다.

**의학적 확인과 한계:** SEER의 폐·기관지암 연령 분포는 고령 집중을 지지하지만 bronchial tumors 전체와의 범주 일치는 보류한다.

자료: [SEER: Lung and Bronchus Cancer](https://seer.cancer.gov/statfacts/html/lungb.html)

### R10 / fpq_85

**귀속·판정 대조:** trying to be optimistic이 자동으로 요실금에 대한 불확실성을 뜻하지는 않는다. long-term use를 제품 내구성으로 읽는 것도 유일한 해석이 아니다. 자동 발생이라는 목표 귀속에 의문은 남지만 5개 확정 불일치의 하나로 고정하지 않는다. 1단계 개인의 요실금 예상이라는 기록과 비교해 전제를 세분해야 한다.

**의학적 확인과 한계:** 수술 뒤 가능한 부작용이라는 일반 설명은 확인했다. 그 사실만으로 이 가족의 예상·상담 내용은 판정되지 않는다.

자료: [NCI: Prostate Cancer Treatment](https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq)

### R11 / fpq_619

**귀속·판정 대조:** his only option을 all patients로 바꾸는 것은 지지되지 않는다. 그러나 의사의 말을 따르는 것도 사용자의 개인 믿음일 수 있어 사용자 믿음이 전혀 아니라는 설명은 강하다. C2를 진짜 목표로 포함한다면 C2=REVIEW인데 전체 NOT_CONFIRMED라는 집계는 불일치한다. 원 참조의 보편 목표와 추가 함축을 분리한다.

**의학적 확인과 한계:** ET의 비이식 치료 선택지를 확인했다. 해당 아버지가 이식이 필요한 경우인지와 의사 설명의 정확성은 미확인이다.

자료: [NCI: Myeloproliferative Neoplasms Treatment](https://www.cancer.gov/types/myeloproliferative/patient/chronic-treatment-pdq)

### R12 / fpq_723

**귀속·판정 대조:** the link는 관련성의 존재를 말한다. direct and primary의 강도는 참조가 추가한다. 계획의 강조와 사실적 주원인 단정을 같게 보지 않는다.

**의학적 확인과 한계:** NCI가 위생 불량을 위험요인으로 명시한다. 관련성이 전혀 없다는 반박은 지지되지 않지만 행사 메시지·특정 집단의 원인 비율까지 입증한 것은 아니다.

자료: [NCI: What Is Penile Cancer?](https://www.cancer.gov/types/penile/what-is-penile-cancer)

### R13 / fpq_279

**귀속·판정 대조:** 질문과 참조 모두 unlikely이며 참조가 impossible이라고 말한 것은 아니다. 1단계는 개인 예후로 읽었는데 2단계는 예후 출처가 없다는 이유로 일반 명제로 바꿨다. 이 변경을 변경 없음으로 처리해서는 안 된다.

**의학적 확인과 한계:** NCI의 위험·분자 특성·재발 등에 따른 예후 차이를 확인했다. 전체 생존율만으로 이 아이의 회복 가능성이 낮다는 명제를 반박할 수 없다.

자료: [NCI: Childhood Medulloblastoma and Other CNS Embryonal Tumors Treatment](https://www.cancer.gov/types/brain/patient/child-cns-embryonal-treatment-pdq)

### R14 / fpq_434

**귀속·판정 대조:** 인구 빈도가 낮다는 이유만으로 가족 위험이 최소라고 추론하는 문제는 있다. 하지만 개인의 절대 위험이 높다는 결론도 정보 없이 낼 수 없다.

**의학적 확인과 한계:** 1도 가족력의 위험 증가와 지역별 빈도 차이를 확인했다. 유전 상담을 모두에게 똑같이 요구하는 근거로 쓰지 않는다.

자료: [NCI: Stomach Cancer Causes and Risk Factors](https://www.cancer.gov/types/stomach/causes-risk-factors)

### R15 / fpq_436

**귀속·판정 대조:** 장기 위험이 있는지 묻는 문장은 무위험을 단정하지 않는다. 즉각 개입 불필요와 향후 관찰 불필요도 다르다.

**의학적 확인과 한계:** benign NET의 병리적 적합성과 개별 임상 위험은 이번에 별도 검증하지 않았다. 전제 귀속 문제와 의학 용어 문제를 합치지 않는다.

### R16 / fpq_65

**귀속·판정 대조:** 개인 check-ups와 정기 인두암 선별의 범위를 구분한다. 선별 프로그램이 존재한다는 별도 목표로 바꾼다면 새 목표 버전임을 명시해야 한다.

**의학적 확인과 한계:** NCI의 표준 정기 선별검사 부재와 USPSTF의 무증상 성인 구강암 I statement를 확인했다. 이것은 선별검사 무효나 어떤 프로그램도 존재하지 않음의 증거가 아니다.

자료: [NCI: Head and Neck Cancers](https://www.cancer.gov/types/head-and-neck/head-neck-fact-sheet), [USPSTF: Oral Cancer Screening](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/oral-cancer-screening)

### R17 / fpq_817

**귀속·판정 대조:** potential은 수술을 직접 수식한다. 진단까지 반드시 불확실하게 만든다고 단정할 수 없으며 암 진단은 확신하지만 수술만 불확실한 해석도 가능하다. 원문의 확진 확신도 입증되지 않아 귀속은 보류한다.

**의학적 확인과 한계:** 혈뇨의 여러 원인은 확인됐지만 참조 귀속의 모호함을 해결하지 않는다.

자료: [NCI: Renal Cell Cancer Treatment](https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq)

### R18 / fpq_84

**귀속·판정 대조:** 개인의 장기 요실금 예상은 암시될 수 있다. all prostate cancer treatments는 별도의 강화이며 이를 같은 단일 주장으로 귀속 판정하지 않는다.

**의학적 확인과 한계:** 치료별 부작용이 가능하다는 설명은 확인했다. 치료 종류와 개인 상담 내용이 없어 이 아버지의 예상은 보류다.

자료: [NCI: Prostate Cancer Treatment](https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq)

## 6. 실제 열람한 자료와 범위

- **S01 — [SEER: Childhood Leukemia (Ages 0–19)](https://seer.cancer.gov/statfacts/html/childleuk.html)**, Trends in Rates / Changes Over Time. 미국 소아 백혈병 신규 발생률의 연령보정 추세를 2014–2023년 연평균 0.6% 상승으로 설명한다. 한계: 미국·해당 기간 통계다. 질문의 지역·시기는 미상이며 벤치마크 작성 당시 참조 버전은 확인하지 않았다.
- **S02 — [NCI: Childhood Medulloblastoma and Other CNS Embryonal Tumors Treatment](https://www.cancer.gov/types/brain/patient/child-cns-embryonal-treatment-pdq)**, Certain factors affect prognosis. 예후는 종양 유형, 전이, 연령, 절제 잔여, 분자 특성, 재발 여부에 따라 달라진다. 한계: 집단 생존율로 이 아이의 회복 가능성을 결정할 수 없다. 생존율과 질문의 recovery는 자동으로 같은 지표가 아니다.
- **S03 — [NCI: Head and Neck Cancers](https://www.cancer.gov/types/head-and-neck/head-neck-fact-sheet)**, How can I reduce my risk / treatment. 표준·정기 두경부암 선별검사가 없다고 설명하며 치과 정기 진료 중 구강 관찰은 가능하다고 구분한다. 치료는 부위·병기·연령·건강에 따라 달라진다. 한계: 선별검사 근거와 증상 평가·치료 후 추적은 다르다. 특정 검사가 어디에도 존재하지 않는다는 증거는 아니다.
- **S04 — [USPSTF: Oral Cancer Screening](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/oral-cancer-screening)**, Recommendation Summary / Clinician Summary. 무증상 성인 구강암 선별의 이득·위해 균형은 근거 불충분(I)이다. 2013 권고 및 2023 문헌 감시 설명을 확인했다. 한계: 모든 인두암·모든 임상 상황에 대한 선별 금지가 아니다. 근거 불충분은 무효나 위해 입증과 다르다.
- **S05 — [NCI: Renal Cell Cancer Treatment](https://www.cancer.gov/types/kidney/patient/kidney-treatment-pdq)**, Signs / Tests used to diagnose. 혈뇨는 신세포암 또는 다른 상태로 생길 수 있고 진단에는 평가·검사가 쓰인다. 한계: 혈뇨만으로 확진할 수 없다는 일반 설명을 지지하지만 질문자가 확신했는지, 별도 진단을 받았는지는 판정하지 않는다.
- **S06 — [NIDCD: Recurrent Respiratory Papillomatosis](https://www.nidcd.nih.gov/health/recurrent-respiratory-papillomatosis)**, What is RRP / Diagnosis. 병변이 호흡기 내부에 생기며 후두경으로 평가한다. 재발할 수 있다. 한계: 이 자료는 외부 사마귀 관찰을 진단 대체로 쓰는 근거를 주지 않는다. 모든 무증상 미진단 소아에게 정기 추적이 필요하다는 결론도 주지 않는다. 외부 사마귀 동반율의 직접 자료는 별도 미확인이다.
- **S07 — [NCI: What Is Penile Cancer?](https://www.cancer.gov/types/penile/what-is-penile-cancer)**, Risk factors. 불량한 개인위생을 위험요인 중 하나로 명시하며 위험요인이 반드시 암을 일으키는 것은 아니라고 설명한다. 한계: 관련성 존재를 지지한다. 위생이 주원인이라는 주장이나 특정 집단에서의 원인 비율·행사 효과를 입증하지 않는다.
- **S08 — [USPSTF: Pelvic Examination — Inactive topic](https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/gynecological-conditions-screening-with-the-pelvic-examination)**, Inactive notice. 현재 주제는 비활성화됐고 이전 근거·권고에 오래된 정보가 있을 수 있으며 최신 근거는 다른 자료를 참고하도록 명시한다. 한계: 과거 I statement를 현재 갱신된 권고로 소개하지 않는다. 예정된 개인 검사의 필요 여부는 증상과 검사 목적에 달려 있다.
- **S09 — [CDC: Screening for Vaginal and Vulvar Cancers](https://www.cdc.gov/vaginal-vulvar-cancers/screening/index.html)**, What to know. 무증상 여성의 질암·외음부암 선별검사가 없으며 Pap은 해당 암의 선별검사가 아니라고 설명한다. 한계: Pap의 범위가 제한적이라는 사실에서 모든 사람의 정기 골반검사 필요성이 바로 나오지는 않는다.
- **S10 — [SEER: Lung and Bronchus Cancer](https://seer.cancer.gov/statfacts/html/lungb.html)**, Who gets this cancer / Median age. SEER 21의 2019–2023년 폐·기관지암 진단 중앙연령은 71세이고 65–74세에서 가장 많이 진단된다. 한계: 질문의 bronchial tumors가 이 암 범주와 같다는 보장은 없다. 기관지 종양 전체나 특정 신경내분비 아형으로 일반화하지 않는다.
- **S11 — [NCI: Stomach Cancer Causes and Risk Factors](https://www.cancer.gov/types/stomach/causes-risk-factors)**, Who gets / Genetics and family history. 지역에 따른 빈도 차이를 설명하고 1도 친족의 위암 병력을 위험 증가 요인으로 명시한다. 한계: 개인의 절대 위험을 계산하지 못하며 모든 가족에게 같은 유전검사·상담 적응증을 확정하지 않는다.
- **S12 — [NCI: Myeloproliferative Neoplasms Treatment](https://www.cancer.gov/types/myeloproliferative/patient/chronic-treatment-pdq)**, Treatment of Essential Thrombocythemia. ET 치료로 일부 환자의 경과관찰과 hydroxyurea, interferon, anagrelide 등을 제시한다. 한계: 모든 ET 환자에게 이식만 가능하다는 보편 명제를 반박하지만 이 아버지의 병기·치료 실패·이식 필요성을 판정하지 않는다.
- **S13 — [NCI: Osteosarcoma and UPS of Bone Treatment](https://www.cancer.gov/types/bone/patient/osteosarcoma-treatment-pdq)**, Surgery / Limb-sparing surgery. 사지 보존술·절단술과 재활에 관련된 치료 차이를 설명한다. 한계: 이 문서만으로 항암으로 인한 침상 생활의 발생률이나 항암 대비 수술의 상대 기여를 확정하지 않았다. 딸의 이동성 전망도 미확인이다.
- **S14 — [NCI: Prostate Cancer Treatment](https://www.cancer.gov/types/prostate/patient/prostate-treatment-pdq)**, Surgery / Possible problems. 요실금을 수술 후 가능한 문제로 다루고 여러 치료 선택지를 설명한다. 한계: 장기 요자제 회복률 전체를 이번 확인에서 검증하지 않았다. 용품 준비가 반드시 요실금을 믿는 증거인지는 언어적 판단이다.
- **S15 — [NCI: Liver Cancer Diagnosis](https://www.cancer.gov/types/liver/what-is-liver-cancer/diagnosis)**, What affects liver cancer prognosis. 예후·치료 선택은 병기, 간 기능, 간경변을 포함한 전신 상태에 따라 달라진다. 한계: 간 손상의 비가역성과 치료 무용을 같게 볼 수 없으나, 개별 환자에게 완치가 가능하다고 판단할 정보도 없다.

자료는 2026-09-14 현재 페이지를 직접 열어 확인했다. 과거 벤치마크 작성 시점의 버전 재현이나 모든 임상 주장에 대한 체계적 검토는 아니다. 페이지 전체를 보관한 스냅샷은 없으며 JSON에는 확인한 주장·섹션·URL·한계를 기록했다.
