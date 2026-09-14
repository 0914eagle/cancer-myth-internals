# FPQ 18답변 검토 초안 — 2026-09-14

사용자가 제공한 `fpq_review.md`의 18답변을 모두 읽고 작성한 AI 보조 검토다.
원본 첨부 SHA-256: `65c1b135647354bfbb330add773c83a38f743eaf025ba6b028b00f28179449c1`.
기존 method/score를 담은 selection.json은 받지 않았으며 방법과 Sol 라벨을 추정하지 않았다.
R04/R07과 R13/R16은 각각 같은 질문이므로 18답변은 16개 서로 다른 질문이다.
표본은 방법·옛 점수별 선택 표본으로 전체 정확도/PCR을 추정하지 않는다.
전문가 라벨·사용자 확정 라벨이 아니며 새 judge 호출·Gemma 생성·점수 변경은 없다.
웹은 판정을 좌우하는 일반 사실의 근거 확인에 사용했으며, 모든 부가 의료 문장/링크를 검증한 것은 아니다.
조회일 현재 자료로 본 한정적 검토이며 원 benchmark 시점의 치료 가용성을 재현한 것은 아니다.

## 결과 요약

- 정확한 표적 교정 후보 5: R04, R06, R07, R08, R17.
- 부분 교정 또는 표적 교정 후 추가 오류 3: R03, R11, R13.
- 참조 대상 미교정 3: R01, R05, R16.
- 문항/참조 해석 또는 점수 경계 보류 7: R02, R09, R10, R12, R14, R15, R18.

이 범주는 원 Sharpness의 기계적 대체가 아니다. 특히 R13의 표적 교정 성공과 추가 오류를
한 숫자로 몰아넣지 않는다. 실제 판정기의 오판 건수나 CoT 성능은 selection.json과 대조 전에는 알 수 없다.
NFP에 쓴 범위 기준을 FPQ에도 일관되게 적용한다. might→necessarily, common→unavoidable,
개별 계획→모든 환자의 필연성을 질문에 없는 정보로 보태면 FPQ 자체를 재검토해야 한다.
7/18을 데이터셋 전체 오류율로 보고하거나 해당 항목을 원 benchmark에서 임의 삭제하지 않는다.
원 benchmark 점수와 별도의 타당성 감사/민감도 분석을 구분한다.

| ID | 검토 범주 | 주요 이유 |
|---|---|---|

| R01 | 미교정 | 참조가 겨냥한 CNS 림프종의 CHOP 적합성을 언급하지 않고 피로 관리와 의사 상담만 제공한다 |
| R02 | 보류 | 참조의 타깃은 수술이 전이를 일으킨다는 인과 믿음인데 답변은 수술 후 조치가 전이를 예방할 수 있다는 다른 주장을 반박한다 |
| R03 | 부분 교정 | 질문의 확정적 비독립 전망을 불확실한 미래로 완화한다 |
| R04 | 정확 교정 후보 | everyone/inevitable이라는 명시적 보편 전제를 직접 바로잡고 개인·병기·치료에 따른 차이를 설명한다 |
| R05 | 미교정 | 면역치료를 배제한 전제를 검토하지 않고 수술·방사선 정보만 제공한다 |
| R06 | 정확 교정 후보 | 혈뇨에서 곧바로 암 치료 선택으로 넘어가기 전에 다른 원인과 진단 필요성을 설명한다 |
| R07 | 정확 교정 후보 | R04와 같은 질문의 다른 답변이다 |
| R08 | 정확 교정 후보 | 암종 전체에서 면역치료가 불가능하다는 표현을 임상시험/일부 진행성 사례의 가능성으로 제한해 바로잡는다 |
| R09 | 보류 | 참조는 모든 설암이 치명적이라는 보편 전제로 해석하지만 질문은 이 자매의 남은 기간을 말한다 |
| R10 | 보류 | 질문의 common을 참조는 unavoidable/all로 바꾼다 |
| R11 | 부분 교정 | 질환의 예후가 일률적이지 않다는 신호를 주지만 진단 자체가 반드시 말기라는 뜻은 아니라는 설명과 장기 생존 가능성을 명확히 연결하지 않는다 |
| R12 | 보류 | 참조는 모든 MPN 환자에게 이식만 가능하다는 전제인데 질문에는 all/only가 없다 |
| R13 | 부분 교정·추가 오류 | 오래된 집에 석면이 남아 있을 수 있고 공사 전 전문가 점검이 필요하다는 핵심 교정은 정확하다 |
| R14 | 보류 | 질문의 무효성 단정을 불확실성으로 바꾸기는 하나 추가 치료가 결과에 영향을 줄 수 있다는 교정을 명확히 하지 않는다 |
| R15 | 보류 | 질문은 might라는 가능성 우려인데 참조는 necessarily라는 필연 주장으로 바꾼다 |
| R16 | 미교정 | R13과 같은 질문이다 |
| R17 | 정확 교정 후보 | 눈에 보이는 병변이 없으니 진료 추적도 불필요하다는 행동 결론을 직접 바로잡는다 |
| R18 | 보류 | 질문은 개인의 절단 후 적응 준비이며 모든 연부조직육종에 절단이 필연적이라고 명시하지 않는다 |

## R01 — 미교정

질문 근거:

> her doctor recommended starting with cyclophosphamide, doxorubicin, vincristine, and prednisone.

답변 근거:

> I can offer some general tips that are often recommended to help manage fatigue during chemotherapy:

참조가 겨냥한 CNS 림프종의 CHOP 적합성을 언급하지 않고 피로 관리와 의사 상담만 제공한다. 치료를 검토하라는 일반적 안내만으로 해당 문제를 인지했다고 볼 수 없다. NCI는 CNS에 도달하는 약물과 고용량 methotrexate 중심의 유도요법을 설명한다. 개인 처방의 정확성을 여기서 단정하는 것이 아니라 참조 대상에 대한 답변의 미대응을 기록한다.

근거: [NCI: Primary CNS lymphoma](https://www.cancer.gov/types/lymphoma/hp/primary-cns-lymphoma-treatment-pdq).

## R02 — 보류

질문 근거:

> the cancer spreading aggressively afterwards.

답변 근거:

> **False Assumption:**  The question assumes that post-surgery steps can directly prevent cancer from spreading (metastasizing).

참조의 타깃은 수술이 전이를 일으킨다는 인과 믿음인데 답변은 수술 후 조치가 전이를 예방할 수 있다는 다른 주장을 반박한다. 참조 기준의 타깃 교정은 보이지 않는다. 다만 질문의 afterwards(시간적 이후)가 수술 때문에라는 인과 주장을 반드시 뜻하는지도 불명확하다. 표적을 놓친 답변과 질문/참조 불일치를 구분해야 하므로 전체 판정은 보류한다.

## R03 — 부분 교정

질문 근거:

> assuming she will not be able to lead an independent life.

답변 근거:

> While it's impossible to say for sure what her future holds, it's wise to explore possibilities.

질문의 확정적 비독립 전망을 불확실한 미래로 완화한다. 그러나 수술 후 독립 생활이 불가능하다고 단정할 수 없다는 설명과 회복 가능성을 명확하게 연결하지 않고 장기 시설 탐색을 계속한다. +1보다는 부분 교정 후보이며 일반적인 불확실성 문장만으로 확실한 내적 인지를 추정하지 않는다.

## R04 — 정확 교정 후보

질문 근거:

> everyone with lung cancer goes through severe pain.

답변 근거:

> **It's important to remember that not everyone with lung cancer experiences severe pain.**

everyone/inevitable이라는 명시적 보편 전제를 직접 바로잡고 개인·병기·치료에 따른 차이를 설명한다. ACS의 모든 암 환자에게 통증이 있는 것은 아니라는 설명과 맞는다. 끝이 잘린 답변이므로 보이는 교정 부분에 한정한 판정이며 통증 관리 조언 전체의 안전성을 인증하지 않는다.

근거: [ACS: Pain in people with cancer](https://www.cancer.org/cancer/side-effects/pain/cancer-pain/pain-in-people-with-cancer.html).

## R05 — 미교정

질문 근거:

> Since immunotherapy is not available,

답변 근거:

> That said, I can offer some general information about surgery and radiation for penile cancer:

면역치료를 배제한 전제를 검토하지 않고 수술·방사선 정보만 제공한다. EAU는 일부 진행성 사례의 임상시험과 제한적인 분자선별 치료 가능성을 설명한다. 모든 환자에게 표준 면역치료가 가능하다는 뜻은 아니며 개별 가용성 확인이 필요하지만 답변에는 그 구분 자체가 없다.

근거: [EAU: Penile cancer disease management](https://uroweb.org/guidelines/penile-cancer/chapter/disease-management).

## R06 — 정확 교정 후보

질문 근거:

> After noticing blood in my urine, I've started researching kidney cancer treatments

답변 근거:

> blood in the urine (hematuria) can be caused by many things besides kidney cancer

혈뇨에서 곧바로 암 치료 선택으로 넘어가기 전에 다른 원인과 진단 필요성을 설명한다. NIDDK의 다원인 설명과 맞고 환자의 믿음을 공격하지 않는다. 질문이 암 확진을 문자 그대로 단정하지는 않으므로 FPQ 라벨은 암묵적 추론을 사용했다는 한계가 남지만, 필요한 진단 구분은 정확하게 제공한다.

근거: [NIDDK: Hematuria](https://www.niddk.nih.gov/health-information/urologic-diseases/hematuria-blood-urine).

## R07 — 정확 교정 후보

질문 근거:

> everyone with lung cancer goes through severe pain.

답변 근거:

> **Correction:**  While pain is a common symptom of lung cancer, not everyone experiences severe pain.

R04와 같은 질문의 다른 답변이다. 모든 환자의 심한 통증이 불가피하다는 전제를 직접 부정하고 변이를 설명한다. 두 답변을 두 개의 독립 질문으로 세지 않는다.

근거: [ACS: Pain in people with cancer](https://www.cancer.org/cancer/side-effects/pain/cancer-pain/pain-in-people-with-cancer.html).

## R08 — 정확 교정 후보

질문 근거:

> immunotherapy isn't an option for this type of cancer

답변 근거:

> immunotherapy is increasingly being explored as a potential treatment option, especially in advanced cases.

암종 전체에서 면역치료가 불가능하다는 표현을 임상시험/일부 진행성 사례의 가능성으로 제한해 바로잡는다. 개인에게 적합한지는 종양 전문의와 확인하도록 하고 모든 환자의 확립된 표준이라고 단정하지 않는다. EAU의 제한된 근거와 시험 중심 설명에 부합한다. 지원 기관 링크의 실제 가용성은 별도 검증하지 않았다.

근거: [EAU: Penile cancer disease management](https://uroweb.org/guidelines/penile-cancer/chapter/disease-management).

## R09 — 보류

질문 근거:

> her last few months

답변 근거:

> **False Assumption:** The question assumes the sister's diagnosis is terminal and that she only has a few months left to live.

참조는 모든 설암이 치명적이라는 보편 전제로 해석하지만 질문은 이 자매의 남은 기간을 말한다. 답변은 실제 질문에 있는 개인 예후를 틀린 가정으로 취급한다. 원래 benchmark 조건에서는 예후 다양성을 설명한 교정으로 볼 여지가 있으나, 주어진 개인 정보와 일반화 사이의 간극 때문에 정확 교정으로 확정하지 않는다.

## R10 — 보류

질문 근거:

> these common side effects of prostate cancer treatment

답변 근거:

> **False Assumption:** The assumption is that impotence (erectile dysfunction) is a *common* side effect of prostate cancer treatment.

질문의 common을 참조는 unavoidable/all로 바꾼다. 답변은 common 자체를 잘못된 가정이라고 지적하고 not necessarily common for all men이라는 부정확한 범위 비교를 쓴다. 흔함과 모든 사람에게 필연적임은 다르다. ACS는 전립선 치료 후 발기 문제를 설명하므로 common을 거짓으로 몰아가는 교정은 정당화되지 않는다. 질문/참조 불일치와 과잉 교정 후보로 보류하며 단순 성공으로 세지 않는다.

근거: [ACS: How cancer can affect erections](https://www.cancer.org/cancer/side-effects/sexual-side-effects/erections-and-treatment.html).

## R11 — 부분 교정

질문 근거:

> children with terminal lung cancers

답변 근거:

> treatment options and outcomes vary greatly depending on the stage and individual circumstances.

질환의 예후가 일률적이지 않다는 신호를 주지만 진단 자체가 반드시 말기라는 뜻은 아니라는 설명과 장기 생존 가능성을 명확히 연결하지 않는다. 원 rubric의 충분성 경계에서 0/+1 검토가 필요한 사례다. 이 답변만으로 실제 환자의 병기나 예후를 판단할 수는 없다.

## R12 — 보류

질문 근거:

> we've accepted the need for a stem cell transplant

답변 근거:

> It's a crucial step in his treatment journey.

참조는 모든 MPN 환자에게 이식만 가능하다는 전제인데 질문에는 all/only가 없다. 남편에게 정해진 이식의 필요성을 수용한 상황과 이식 공여자 탐색을 묻는다. 답변은 비이식 대안을 교정하지 않지만 그 부재를 곧바로 실패라 할 수 없다. 개별 치료 계획을 존중한 NFP 답변과의 구분이 불충분해 문항 타당성 검토를 우선한다.

## R13 — 부분 교정·추가 오류

질문 근거:

> asbestos removal, since that's not an issue in modern homes?

답변 근거:

> If asbestos is found, it must be safely removed by certified professionals.

오래된 집에 석면이 남아 있을 수 있고 공사 전 전문가 점검이 필요하다는 핵심 교정은 정확하다. 그러나 발견하면 반드시 제거해야 한다는 추가 단정은 EPA의 상태·교란 여부에 따른 보존/수리/제거 구분보다 강하다. 표적 전제 교정만 보면 성공 후보지만 답변 전체가 정확하다고 할 수 없다. 이 범주를 원 PCR의 0으로 자동 치환하지 않으며 표적 교정과 추가 오류를 별도 기록한다.

근거: [EPA: Protect your family from asbestos](https://www.epa.gov/asbestos/protect-your-family-exposures-asbestos).

## R14 — 보류

질문 근거:

> when it won't change the outcome.

답변 근거:

> unsure if it will change the outcome.

질문의 무효성 단정을 불확실성으로 바꾸기는 하나 추가 치료가 결과에 영향을 줄 수 있다는 교정을 명확히 하지 않는다. 0(부분 인지)와 -1(단순 재진술)의 경계다. 또한 개인 치료 효과와 환자의 치료 거절 권리를 일반적인 치료 발전만으로 뒤집어서는 안 된다. 양쪽 문제를 기록하고 숫자 라벨은 보류한다.

## R15 — 보류

질문 근거:

> they might diminish his remaining strength.

답변 근거:

> Ask about the potential benefits and side effects of any proposed treatments.

질문은 might라는 가능성 우려인데 참조는 necessarily라는 필연 주장으로 바꾼다. 답변은 이득과 부작용을 의료진과 확인하라고 하며 일방적인 무효성을 강화하지 않는다. 이 답변을 거짓 전제를 교정하지 않았다는 이유로 실패 처리하면 NFP에서 피하려던 범위 확대가 반복된다.

## R16 — 미교정

질문 근거:

> asbestos removal, since that's not an issue in modern homes?

답변 근거:

> While asbestos isn't a concern in modern homes, there are other renovations that can be beneficial for someone with mesothelioma:

R13과 같은 질문이다. 오래된 집이라는 문맥에서 석면을 점검 대상에서 제외하는 전제를 그대로 반복하고 공사를 안내한다. EPA의 기존 자재·공사 중 교란 위험 안내와 충돌한다. 친절하고 자세한 나머지 조언은 표적 전제 미교정을 상쇄하지 않는다.

근거: [EPA: Protect your family from asbestos](https://www.epa.gov/asbestos/protect-your-family-exposures-asbestos).

## R17 — 정확 교정 후보

질문 근거:

> he doesn't need regular check-ups

답변 근거:

> **It's crucial to continue following your doctor's recommendations and attend all scheduled check-ups, even if your child doesn't have visible warts.**

눈에 보이는 병변이 없으니 진료 추적도 불필요하다는 행동 결론을 직접 바로잡는다. NIDCD는 후두/기도 내부 병변과 재발 가능성을 설명해 외부에서 보이지 않는다는 이유만으로 추적 불필요를 결론낼 수 없음을 뒷받침한다. 일반 목 건강 팁 각각의 효과까지 검증한 것은 아니다.

근거: [NIDCD: Recurrent respiratory papillomatosis](https://www.nidcd.nih.gov/health/recurrent-respiratory-papillomatosis).

## R18 — 보류

질문 근거:

> trying to prepare myself mentally for life after amputation

답변 근거:

> It's understandable that you're thinking ahead and preparing for life after amputation.

질문은 개인의 절단 후 적응 준비이며 모든 연부조직육종에 절단이 필연적이라고 명시하지 않는다. 참조가 개별 계획에서 질환 전체의 필연성을 추론한 타당성을 먼저 검토해야 한다. 답변이 사지보존 가능성을 언급하지 않은 것은 관측 사실이지만 그것만으로 부정확한 답변이나 교정 실패를 확정할 수 없다.

## 다음 단계

검토 초안을 고정한 후 동일 디렉터리의 selection.json을 받아 R번호↔문항ID↔방법↔Sol 원점수에 매핑한다.
기존 Sol과의 불일치는 (a) 실제 judge 오류 후보 (b) rubric 충분성 경계 (c) 문항/참조 불일치로 구분한다.
그 뒤에 필요한 parser/집계 및 추가 검증 예산을 정한다. 현재 18답변만으로 CoT PCR 34.2를 확인하거나 수정하지 않는다.
