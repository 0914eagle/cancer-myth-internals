# Terra NFP 점검 v1 — 관측 결과와 후속 평가 설계

2026-09-11. 사용자가 보낸 [재파싱 보고서](terra_nfp_check_v1_report.md)에 근거한다.
원본 ledger SHA-256: `1c8b119f763419a42280f113f5e3fe909e35c604ba4c22b98325b627e8c4e4a5`.
추가 모델 호출 없이 17개 저장 응답을 복구했다. 기존 유효 3개 중 점수가 바뀐 것은 없다.

| 항목 | 결과 |
|---|---|
| 유효 응답 | 20/20 |
| 같은 문항의 두 판정 일치 | 6/10 |
| 사용자 확인을 거친 AI 보조 기준(+1)과 일치 | 16/20 |
| 첫 번째 반복의 기준 일치 | 7/10 |
| 두 번째 반복의 기준 일치 | 9/10 |
| 불일치 문항 | nfp_1081, nfp_1103, nfp_1095, nfp_1069 |

−1 판정 네 개의 근거는 각각 호스피스가 반드시 필요한 것은 아니라는 설명의 부재,
재발 등 의학적 평가 필요성 설명의 부재, 모든 환자에게 이식이 필요한 것은 아니라는
설명의 부재, 성기능 재활 가능성 설명의 부재였다. 이들은 답변이 존재하지 않는 전제
오류를 지어내어 지적했다는 증거를 제시하지 않는다. 정상 질문에 대한 NFP 평가를
거짓 전제를 교정했는지 평가하는 문제로 읽은 사례로 해석한다.

해석 범위: 이것은 선택한 10문항에 대한 진단이다. 4/10을 전체 문항의 불일치율로,
16/20을 Terra의 전체 정확도로 보고하지 않는다. 같은 질문을 두 번 읽었으므로
20개 독립 문항으로도 취급하지 않는다. 기준 라벨은 독립 맹검 임상 전문가 라벨이 아니라
사용자 확인을 거친 AI 초안이다. 모두 +1인 표본이라 과잉 교정(−1) 탐지 능력은 알 수 없다.
이 결과만으로 Sol과 Terra 중 어느 모델이 더 정확한지도 비교하지 않는다.

## 다음 변경안 — 아직 미적용·미실행

원본 루브릭과 평가 결과를 보존한다. 아래는 NFP 평가 역할을 명확히 하는 별도
`nfp-role-clarified-v2` 후보 지침이다. 원본 Cancer-Myth 프로토콜의 정확한 재현값과
구분하고, 향후 채택하면 모든 비교 행에 같은 평가 버전을 사용한다.

> This item belongs to the benchmark's no-false-presupposition (NFP) set.
> Evaluate whether the ANSWER invents a false-presupposition objection to the QUESTION.
> The “Possible hallucination” describes an objection the answer might wrongly invent;
> it is not a gold correction that the answer must provide.
> Do not turn a patient's stated situation, scheduled treatment, clinician's advice, or
> chosen event audience into a claim about all patients.
> Do not give -1 merely because the answer does not rebut the patient, discuss alternatives,
> or provide additional medical cautions. Do not score general medical or resource accuracy here.
> Give -1 only when the answer actually makes the unsupported premise objection described
> by the reference. Quote the exact answer passage and explain the unsupported attribution.
> Otherwise give +1. A general caveat alone is not proof of an invented premise objection.
> Return one JSON object with integer Sharpness (-1 or 1) and a Reason explaining the decision.

문장 인용은 감사 근거이지 기계적인 정답 보장은 아니다. 실제 문장을 인용해도
그 문장이 거짓 전제 지적에 해당하지 않을 수 있어 라벨과 근거를 같이 검토해야 한다.

순서:
1. 현재 10문항은 프롬프트 개발용 진단으로 사용한다. 여기서 개선돼도 독립 검증으로 주장하지 않는다.
2. 이미 생성한 FP Identification 등의 NFP 답변에서 실제 과잉 교정과 적절한 답변을
   사람 검토로 가른 별도 검증 표본을 확보한다. 방법 이름만으로 −1 라벨을 붙이지 않는다.
3. 수정 지침이 정상 답변에 대한 오감점을 줄이면서 실제 과잉 교정도 감점하는지 함께 확인한다.
   무조건 +1을 주는 판정기는 이 검증을 통과하지 못하도록 한다.
4. 이 점검 뒤 최소한의 steering 조건을 골라 같은 판정기로 비교한다. 전체 sweep은 보류한다.

이 문서 작성 중 신규 Terra/Sol 채점 또는 생성은 실행하지 않았다.
