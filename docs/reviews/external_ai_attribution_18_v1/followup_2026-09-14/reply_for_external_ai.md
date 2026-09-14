# 외부 AI에게 보낼 후속 대조 요청

원 응답은 보존하고 아래 항목만 정정·해명하는 별도 응답을 주세요. 이전 판정에 동의하라는
요청이 아니며 새 단계는 기존 결과와 근거를 본 후의 대조입니다.

1. stage1 SHA는 일치합니다. 원 JSON의 의학 상태는 전부 UNVERIFIED가 아닙니다.
   NOT_MEDICAL 8 / CONTEXT_DEPENDENT 8 / UNVERIFIED 7이며 R06-C2·R09-C1의
   reference_correction_status는 PARTLY_SUPPORTED입니다. sources가 빈 상태에서 이 값을
   어떻게 해석해야 하는지 정리해 주세요. 귀속 불성립과 NOT_MEDICAL은 구분해 주세요.
2. R01-C2는 환자 목표가 아닌 참조의 반박 설명이고, R04-C2·R11-C2는 약한 전제 후보입니다.
   원 참조 목표·추가 추론·반박 설명을 분리해 주세요. R11-C2가 미해결인데 전체 문항은
   NOT_CONFIRMED라는 집계가 어떤 범위에 대한 것인지 밝혀 주세요.
3. R13은 stage1에서 개인 예후였으나 stage2에서는 출처가 없다는 이유로 일반 명제로 바뀝니다.
   변경 없음으로 기록해도 되는지 검토해 주세요. NCI는 개인의 예후 요인을 구분합니다.
   https://www.cancer.gov/types/brain/patient/child-cns-embryonal-treatment-pdq
   집단 생존율만 확인하면 CONFIRMED로 갈 수 있다는 결론을 재검토해 주세요.
4. R10의 optimistic이 요실금의 필연성을 직접 부정하는지, R17의 potential이 수술만이 아니라
   암 진단까지 반드시 수식하는지 대안 해석을 남겨 주세요. R05의 확진 사건이 서술되지 않았다는
   것과 확진되지 않았다는 것은 다릅니다. 1단계의 불확실성이 2단계에서 사라진 이유를 점검해 주세요.
5. 이번 세션에서 SEER 원문을 열었고 미국 0–19세 발생률이 2014–2023년 연평균 0.6% 상승한다는
   설명을 확인했습니다. 질문의 지역·기간 미상이라는 한계와 함께 R02에 반영할 수 있습니다.
   https://seer.cancer.gov/statfacts/html/childleuk.html
6. R16: NCI는 표준 정기 두경부암 선별검사가 없다고 설명하지만 USPSTF I는 무효나 금지가 아닙니다.
   어떤 프로그램도 존재하지 않는다는 새 전제로 바꾸지는 말아 주세요. R03의 USPSTF 골반검사
   주제는 현재 inactive이며 옛 권고가 오래됐을 수 있다고 안내합니다.
   https://www.cancer.gov/types/head-and-neck/head-neck-fact-sheet
   https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/oral-cancer-screening
   https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/gynecological-conditions-screening-with-the-pelvic-examination

저의 이전 40개 메모도 정정했습니다. R09 원 참조는 primarily이고 R13 원 참조는 unlikely입니다.
이를 각각 고령자만·회복 불가능으로 요약한 것은 너무 강했습니다. 해당 사례의 핵심 쟁점은
목표 문장 자체의 강화보다 반박 설명의 충분성과 개인 예후를 인구 통계로 대체하는 문제입니다.

각 쟁점에 `동의 / 반대 / 보류`, 원문 인용, 짧은 이유를 남겨 주세요. 위 출처를 직접 열지 못하면
직접 검증했다고 쓰지 말고 후속 검토자가 제공한 자료 내용으로 표시하세요. 새로운 gold label이나
PCR을 만들지 말고 원 응답은 그대로 보존해 주세요.
