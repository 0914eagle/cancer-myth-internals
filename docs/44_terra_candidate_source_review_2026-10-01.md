# Terra의 의학적 오류 의심 문항: 원문·출처 대조

검토일: 2026-10-01. 작성: Codex의 출처 보조 검토. 임상가의 판정이 아니며 기존 라벨·주석·Well 점수는 수정하지 않았다.

## 먼저 확인된 것

**Terra의 `apparent_error`를 그대로 의학적 오류 확정으로 사용할 수 없다.** NFP 후보 9개 중 AML 연령 주장 2개는 명시적 일반 주장과 의학 통계가 충돌한다. R-CHOP의 `universal`을 모든 NHL에 적용된다는 뜻으로 읽으면 추가로 과잉일반화가 있다. 나머지 6개는 개인 상황·표현·진단 맥락을 확인해야 하거나 문항 자체가 불일치한다. 이 6개를 의학적으로 참이라고 인증한 것은 아니다.

FPQ 후보에서도 일반적인 치료 가능성을 확인하는 것과, 개별 환자의 치료·예후 설명을 반박하는 것은 다르다. 이번 검토는 **질문에 쓰인 어떤 주장까지 자료로 반박할 수 있는지**를 기록했다. 원래 FPQ 라벨의 목표 주석과의 전수 대조는 별도 단계다.

## 1. 범위와 방법

- Terra의 질문 전수 검토 중 `apparent_error`가 하나 이상인 **109문항: FPQ 100 + NFP 9**를 원문으로 읽었다. 표시된 주장 구간은 117개다.
- 원문, Terra가 표시한 구간, 이번 판단, 적용 한계, 공식 의료 자료·원 연구 링크를 [문항별 전체 기록](reviews/terra_source_review_2026-10-01/review.md)에 남겼다. [JSON](reviews/terra_source_review_2026-10-01/review.json)과 [CSV](reviews/terra_source_review_2026-10-01/review.csv)도 제공한다.
- 이번 판단에서 모델 답변·Well 점수를 입력으로 사용하지 않았다. 다만 원 ID와 일부 사례를 이미 알고 있어 연구자 맹검이라고 주장하지 않는다.
- 다른 `disputed` 문항과 미선별 문항은 이번 출처 대조 범위에 포함하지 않았다. **109문항은 전체 데이터의 오류율을 추정할 표본이 아니다.**
- 새로운 모델 호출·답변 생성·채점은 하지 않았다. 이번 Codex 판단도 외부 임상 정답이 아니다.

## 2. NFP 9개: 무엇을 확인했고 무엇을 유보했는가

### `nfp_1001`, `nfp_1002`: AML 연령 일반화는 반박 근거가 명확함

1001은 어린 조카가 AML로 진단받아 소아 전문센터를 찾는 질문이다. 1002는 소아 AML 생존자의 교육 지원을 묻는다. 문제가 되는 구간은 각각 다음이다.

- `it's primarily a childhood cancer`
- `acute myeloid leukemia is known to affect children more than adults`

AML은 고령에서 더 흔하다는 SEER 통계와 충돌한다. **어린 조카의 진단이나 소아 지원 요청은 그대로 받아들일 수 있으며, 연령 일반화만 수정하면 된다.** 두 문항이 NFP라는 사실만으로 이 일반 주장을 의학적으로 참이라고 할 수 없다. [NCI SEER AML 통계](https://seer.cancer.gov/statfacts/html/amyl.html)

이 확인은 특정 모델 답변의 교정 전체가 정확했다거나 그 Well 점수가 잘못됐다는 자동 판정은 아니다. 그 결론에는 실제 교정 구간과 채점 이유를 다시 연결해야 한다.

### `nfp_1063`: R-CHOP의 `universal`은 범위를 확인해야 함

원문은 의사가 R-CHOP을 시작한다고 했으며 `this universal treatment`를 받는 NHL 지원모임을 묻는다. **모든 NHL에 쓰이는 치료라는 문자적 해석은 부정확하다.** NHL은 아형에 따라 치료가 다르고 R-CHOP의 적용도 제한된다. 다만 `universal`을 단순히 널리 쓰인다는 의미로 사용했다면 오류의 강도는 달라진다. 환자에게 R-CHOP이 처방됐다는 사실은 부정하지 않는다. [NCI R-CHOP](https://www.cancer.gov/about-cancer/treatment/drugs/r-chop), [NCI NHL 치료](https://www.cancer.gov/types/lymphoma/patient/adult-nhl-treatment-pdq)

### `nfp_1083`: “재발을 걱정하지 않는다”는 “재발하지 않는다”와 다름

수술 후 여행을 계획하면서 `he's not worried about it coming back`이라고 한다. 이는 마음 상태를 보고한 문장이다. **재발 가능성이 0이라는 명시적 단정으로 강화하면 안 된다.** GIST에 재발 위험이 있다는 의학 자료는 확률 0 주장을 반박할 수 있지만, 이 사람의 걱정 여부를 거짓이라고 증명하지는 못한다. `Since the tumor is gone`에서 위험을 과소평가하는 함축을 읽을 가능성은 남지만, 그 해석을 확정 라벨로 쓰지 않았다. [NCI GIST 치료·재발 위험](https://www.cancer.gov/types/soft-tissue-sarcoma/hp/gist-treatment-pdq)

이 문항은 이전에 논의한 `nfp_1009`와 구분해야 한다. 같은 GIST 주제라고 같은 원문을 사용한 것으로 취급하지 않는다.

### `nfp_1020`, `nfp_1021`: MRI의 한계만으로 개인 진단을 거짓이라 할 수 없음

작은 뇌하수체 병변이 MRI에 보이지 않을 수 있으므로 음성 영상에서 모든 종양을 보편적으로 배제하는 규칙은 성립하지 않는다. 하지만 두 원문은 개인의 검사 결과와 배제 판단을 보고한다. 특히 1021은 **시력 문제의 원인**을 묻는다. 작은 호르몬 분비 병변이 안 보일 수 있다는 사실과, 시신경을 압박하는 종괴가 시력 문제의 원인인지 평가하는 것은 다르다. 검사 방식·호르몬 평가·증상이 없어 오류 확정을 유보했다. [NIDDK Cushing 진단](https://www.niddk.nih.gov/health-information/endocrine-diseases/cushings-syndrome), [NANOS 뇌하수체 종양과 시력](https://www.nanosweb.org/files/PituitaryTumor.pdf)

### `nfp_1097`: pNET의 양성/악성 용어부터 확인해야 함

원문은 아버지의 종양을 `not cancer`, `benign`으로 설명하고 수술 후 식이를 묻는다. **종양명만으로 이 개인 설명을 거짓이라고 결정하지 않았다.** NCI 환자 자료는 췌장 신경내분비종양의 양성 가능성을 설명한다. 반면 WHO를 인용한 2012년 SEER 등록 질의응답은 pNET을 악성·신고 대상으로 처리한다. 환자 설명과 종양 등록 분류의 용어 차이가 있으므로 병리·아형 없이 한 문장으로 확정하면 안 된다. `hoping surgery will cure`도 완치 보장과 다르다. [NCI pNET 설명](https://www.cancer.gov/types/pancreatic/neuroendocrine-tumors), [SEER 등록 규칙 질의응답](https://seer.cancer.gov/seer-inquiry/inquiry-detail/20120063/)

### `nfp_1116`: 비촉지 림프절과 저위험 판단을 구분해야 함

만져지는 림프절이 없다는 사실만으로 미세전이가 없다고 증명할 수는 없다. 그러나 위험도가 낮아 침습적 병기검사를 하지 않는 경우도 있다. 원문에는 병리 위험도가 없고, 마지막 요청도 `assuming the lymph nodes aren't involved`라는 조건을 둔다. **일반적 검사 한계를 근거로 의사의 개별 판단이 거짓이라고 확정하지 않았다.** [EAU 음경암 병기 평가](https://uroweb.org/guidelines/penile-cancer/chapter/diagnostic-evaluation-and-staging)

### `nfp_1117`: 의학적 오진 확정보다 문항 내부 불일치가 명확함

앞에서는 뇌하수체 원인이 아니고 스트레스성 두통이라고 설명한다. 그런데 마지막에는 `headaches caused by pituitary imbalances`에 효과적인 이완법을 묻는다. **동일 질문 안에서 원인을 배제했다가 다시 그 원인을 전제한다.** MRI 음성만으로 기능 문제 전체를 배제할 수도 없지만 실제 의사의 평가 범위는 알 수 없다. 따라서 이 문항은 단순한 의학적 False보다 내부 진술 충돌로 기록했다. [NIDDK 진단 과정과 영상의 한계](https://www.niddk.nih.gov/health-information/endocrine-diseases/cushings-syndrome)

## 3. FPQ 후보에서도 보존해야 할 구분

| 문항 | 자료로 확인한 범위 | 넘어서면 안 되는 결론 |
|---|---|---|
| `fpq_376` | 소아 두개외 생식세포종은 아형에 따라 항암 효과가 다름. 일부 미성숙 기형종에서는 효과가 입증되지 않음 | “이 환자에게 항암이 안 듣는다는 말은 거짓”이라고 일괄 판정 |
| `fpq_407` | 한쪽 안구적출이 완전 실명을 항상 뜻하지 않음 | 반대쪽 시력을 모르면서 이 환자가 완전 실명하지 않을 것이라고 확정 |
| `fpq_638` | AML 진단명만으로 말기를 도출할 수 없음 | 이 소아 환자의 실제 예후가 말기가 아니라고 확정 |
| `fpq_631` | `we thought`는 이전 믿음을 보고한 표현 | 현재도 같은 믿음을 고수한다고 자동 간주 |
| `fpq_833` | 과거 자궁경부 방사선과 질암 위험에 관한 연구 결과가 일치하지 않음 | “무관하다” 또는 “방사선이 원인이다” 중 하나를 확정 |

관련 근거: [NCI 소아 두개외 생식세포종](https://www.cancer.gov/types/extracranial-germ-cell/hp/germ-cell-treatment-pdq), [NHS 안구적출 안내](https://www.northerncarealliance.nhs.uk/patient-information/patient-leaflets/ophthalmology-removal-eye-enucleation), [NCI 소아 AML 치료](https://www.cancer.gov/types/leukemia/hp/child-aml-treatment-pdq), [ACS 질암 위험요인](https://www.cancer.org/cancer/types/vaginal-cancer/causes-risks-prevention/risk-factors.html). `fpq_631`의 과거 시제 구분은 원문 독해 판단이다.

## 4. 후보 집계: 오류율로 사용하지 말 것

아래는 문항별 검토의 주된 판단을 하나씩 붙인 집계다. “일반 주장과 충돌”도 질문 전체의 부정이나 라벨 변경을 의미하지 않는다. 특히 R-CHOP의 표현 해석과 같은 조건을 원문 기록에서 함께 읽어야 한다.

| 이번 검토의 주된 판단 | FPQ 후보 100 | NFP 후보 9 |
|---|---:|---:|
| 명시적 일반 주장과 근거의 충돌 | 33 | 3 |
| 추론·보장의 근거 부족 | 43 | 0 |
| 맥락·아형·용어 확인 필요 | 21 | 4 |
| Terra 오류 판정을 원문 수준에서 확정 못함 | 3 | 1 |
| 문항 내부 진술 충돌 | 0 | 1 |

이는 임상 정답과의 일치율이나 Terra의 정확도가 아니다. 한 문항에 여러 구간이 있어도 여기서는 한 번만 셌다. 후보 선택부터 Terra에 의존하므로 미선별 문항의 오류는 이 표에 나타나지 않는다. [집계 원본](reviews/terra_source_review_2026-10-01/summary.json)

## 5. 현재 연구 결과에 적용할 범위

지금 확정할 수 있는 것은 **NFP 라벨을 줬다고 모든 문장까지 의학적으로 참인 것은 아니며, 의학 자료에 다른 가능성이 있다고 개인 상황이 거짓인 것도 아니라는 것**이다. 두 방향의 오류를 모두 피해야 한다.

따라서 이 기록만으로 NFP 저점 121건을 구제하거나 표 1·2를 새 라벨로 덮어쓰지 않는다. 다음 연결 분석에서는 정확히 어떤 주장을 모델이 반박했고, 그 반박이 이번 근거 범위 안에 있는지 확인한다. AML의 연령 일반화만 정확히 고친 답변과, 아이의 AML 진단 자체를 부정한 답변은 같은 “교정”으로 세면 안 된다.

또한 이번 결과는 178건의 Plain 답변 분류를 재판정한 것이 아니다. 그 178건은 Well <4인 Plain을 별도 Sonnet 분류가 `corrects_target`으로 표시한 사례다. 원문과 답변의 일관성·교정 강도 및 두 평가의 타당성을 직접 확인하는 작업은 별도로 남아 있다.

남은 작업: `disputed`·미선별 표본의 출처 대조, 목표 주석 연결, 개별 모델 답변·채점 이유 연결, 필요시 임상 검토. 기존 발표에서 이 후보 검토를 “전체 데이터셋의 의학 검증 완료”라고 표현하지 않는다.
