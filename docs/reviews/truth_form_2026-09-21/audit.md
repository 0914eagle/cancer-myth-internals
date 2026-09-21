# 진위 × 표현 control audit — 2026-09-21

## 결론

현재 50개 묶음을 깨끗한 진위×표현 2×2의 확정적 검정 자료로 사용하지 않는다. 탐색적 원본은 보존하며, 결과가 나오더라도 생성 조건에 대한 점수 변화로만 해석한다. 이 파일은 새 FPQ/NFP 라벨도, 의료 검증도 아니다. GPU·LLM 호출 0; 원 파일 변경 없음.

## 범위와 기계적 검사

원본은 아래 로컬 경로와 SHA로 식별하며 Git에는 포함하지 않는다. 실행 상태·다음 절차는 [35번 문서](../../35_truth_form_audit_and_next_steps.md)를 따른다.

- 원본: `/Users/heejae/Downloads/truth_form_review/variants.jsonl`
- SHA-256: `3a5bd9803caa042e3df81bd14ed289c7c2da69da51154ee1b898cd47a244f459`
- 전체 320행 = 자동 통과 200행(50개 원문 × 4조건), 자동 탈락 120행(30개 원문 × 4조건).
- 자동 통과 50개 모두 네 조건이 있고 span 문자열과 인덱스가 일치하며, 구간 밖 prefix/suffix가 네 조건 사이에서 동일하다.
- 그러나 구간 밖 고정은 구간 안의 환자 사실·행동 계획·여러 명제까지 고정했다는 뜻이 아니다.
- 50개 통과 묶음의 네 span 및 최종 접합 질문을 검토했다. 탈락 30개는 재심사하지 않았다. 임상 진위는 전부 NOT_ASSESSED이며 일부 의학 주장에 문제가 의심되어도 여기서 참/거짓을 새로 확정하지 않는다.

## 공통 설계 문제

1. FH/TH 100개 전부에 read/heard/told/understand 계열 출처 표지가 있고 FA/TA 100개에는 없다. 의도된 조작이지만 이를 일반적인 문체 또는 단정성만의 효과라고 부르기 어렵다. 문구 인식, 출처 귀속, 화자의 믿음 강도가 함께 변한다. 이 사실만으로 F/T 비교가 자동 무효인 것은 아니다.
2. A/H 내부에서도 will→might, never→rarely, 단정→probably 등 대상 명제의 강도·범위가 변한다. 별도로 각 문장을 FALSE/TRUE라고 검사하는 것만으로 두 문장의 명제 동일성이 확인되지 않는다.
3. F/T 간에는 전제뿐 아니라 치료·여행·홍보 대상·가족력 등의 사실과 요청 목적이 바뀌거나, 참 조건에 약물·의료 조언이 더해진다.
4. 구간만 고친 뒤 구간 밖 요청은 그대로여서 의미 연결이 어색하거나 원 오해가 남은 문항이 있다. 호스피스·부작용 지원 요청 자체를 잘못됐다고 판정하지는 않는다.
5. 기존 CHECK_PROMPT는 단일 변형의 CLAIM/STANCE만 검사한다. 네 변형의 내용 동등성, 개인 사실 보존, 원 참조의 타당성을 검증하지 않는다. TRUE는 제공 참조에 따른 자동 라벨이며 임상적으로 확인된 진위가 아니다.

## 자동 집계

| 조건 | n | 출처 표현 있음 | 구간 평균 단어 수 |
|---|---:|---:|---:|
| FA | 50 | 0 | 14.14 |
| FH | 50 | 50 | 16.64 |
| TA | 50 | 0 | 16.42 |
| TH | 50 | 50 | 18.92 |

## 항목별 검토

아래 표시는 서로 겹치는 검토 사유다. 오류율, 의료 오답률, 사용 가능한 최종 표본 수로 해석하지 않는다. 표본 제외·새 학습·재생성은 수행하지 않았다.

ACTION=개인 상황/행동/요청 대상 변경; EXTRA=추가 정보·조언; AH_DRIFT=같은 진위의 A/H 내용 차이; CONTEXT=구간 밖과의 연결/잔여 전제; SCOPE=일반·조건부·개인 명제 범위; TARGET=서로 다른 목표 주장; SYNTAX=접합 문법.

| ID | 검토 사유 | 근거 |
|---|---|---|
| fpq_81 | ACTION | keeping him comfortable → getting him into treatment: 치료 계획도 바뀜. OR 원문의 whether to tell him or just focus on 구조에 들어가므로 요청의 배경도 달라짐. |
| fpq_112 | ACTION, CONTEXT, SYNTAX | last few months → treatment and recovery로 개인 예후가 바뀜. as her ability to eat declines는 밖에 남음. make her treatment and recovery, since ..., as comfortable 문장 접합도 부자연스러움. |
| fpq_54 | ACTION, SYNTAX | cancer treatment → recovery from an ovarian cyst로 치료 대상·상황 변경. which is usually benign and not cancer treatment의 병렬 관계가 부자연스러움. |
| fpq_313 | EXTRA, AH_DRIFT | incurable의 반대를 직접 만들기보다 targeted drugs/control/normal lifespan/supportive care를 추가함. TA can control와 TH can often control도 동일 명제에 wrapper만 붙인 형태가 아님. |
| fpq_834 | AH_DRIFT | 주된 위험 명제는 비교 가능성이 있으나 TA can raise → TH can sometimes raise로 내용 안의 양태도 바뀜. 의학적 위험 관계는 본 검토에서 확인하지 않음. |
| fpq_730 | EXTRA | rare ↔ common에 더해 참 조건에 treatable이라는 별도 속성 추가. |
| fpq_158 | AH_DRIFT, SCOPE | patients whose leukemia라는 개인/부분집합 표현과 일반 leukemia 교정의 범위를 확인해야 함. TH에 or put into remission이 추가되어 TA와 내용이 다름. |
| fpq_789 | EXTRA, CONTEXT | 참 조건에 high-dose methotrexate라는 약물 추가. 효과가 좋다는 이유로 alternative therapies as our primary approach를 택한다는 given 연결이 남아 원인 관계가 이상해짐. |
| fpq_848 | ACTION, EXTRA | feel reassured → stay alert로 목표·태도도 바뀜. 참 조건에 imaging/biopsy가 추가되어 단순 진위 차이 이상. |
| fpq_865 | ACTION, EXTRA, AH_DRIFT, CONTEXT | avoid → not fear 행동 변화와 다른 부인암 위험 감소 정보 추가. raise → may raise, can lower → may lower도 변함. safer natural family planning 요청이 남음. |
| fpq_622 | CONTEXT | 앞은 possible progression으로 수정했지만 뒤에는 as they transition to the terminal phase of acute leukemia가 유지됨. 전체 질문을 깨끗한 참 전제 음성으로 보기 어려움. |
| fpq_122 | EXTRA, AH_DRIFT | 참 조건에 HPV 정보 추가. FA will not face → FH probably will not face로 자신의 위험 전망도 달라짐. |
| fpq_749 | CONTEXT, SYNTAX | like quitting smoking, since/though ... 삽입이 중첩됨. 참 조건도 smoking과 무관하다고 하면서 금연이 예후 개선의 예로 남아 있음. |
| fpq_55 | EXTRA, SCOPE | 참 조건에 hers may not have been[ caught early ]라는 개인 병기 추측을 새로 추가함. 원 질문의 치료 선호를 정당화하는 별도 정보가 됨. |
| fpq_245 | SCOPE, SYNTAX | 일부가 수년 생존 가능하다는 일반 명제에서 이 환자의 여행을 non-terminal로 재분류함. 원 참조만으로 그 추론은 성립하지 않음. specifically for non-terminal, since ..., travel 접합도 부자연스러움. |
| fpq_13 | ACTION, CONTEXT | 개인의 수술을 whole kidney removed → just the tumor removed로 변경. 뒤의 living with a single kidney post-surgery 요청이 남아 질문의 정합성이 달라짐. |
| fpq_695 | ACTION | 홈스쿨링 이유를 감염 예방에서 진료 일정으로 바꿈. 기존 믿음의 진위뿐 아니라 결정의 근거도 새로 부여함. |
| fpq_261 | AH_DRIFT | FA never really occur → FH rarely occur는 단정의 출처만 바꾼 것이 아님. only older people와 rarely younger가 한 문장 안에 혼재함. |
| fpq_95 | SCOPE | 개인이 영구 장루를 가지리라는 전망을 if I do end up needing으로 바꿈. 보편 필요성과 개인 예정 사실을 별개로 추적해야 함. |
| fpq_216 | CONTEXT | needs prompt medical treatment가 참 조건에 들어가지만 Since ... we are planning a long vacation의 인과 연결과 피부관리 요청이 남음. 임상적 오류 확정이 아니라 문맥 정합성 검토 대상. |
| fpq_552 | TARGET, EXTRA, AH_DRIFT, SYNTAX | FA/FH는 소아 중심 발생 연령을 다루지만 TA/TH는 소아 발생률의 시간 추세를 다룸. 같은 명제의 진위 대조가 아님. TA에 genetic/chance causes도 추가되고 문법도 어색함. |
| fpq_302 | AH_DRIFT, CONTEXT | TA individual patient ↔ TH watchful waiting으로 내용 변경. 뒤의 initial treatment phase 및 significant side effects 전제는 그대로 남음. |
| fpq_160 | ACTION, CONTEXT | other kids → other adults로 기금모금 대상 변경. children activities와 kid-friendly request는 계속 유지. FALSE 조건은 40-year-old sister와 only children이 직접 충돌하여 모순 단서가 됨. |
| fpq_399 | ACTION, AH_DRIFT | 참 조건에 임상시험 참여 목적 to compare approaches가 새로 들어감. TA standard treatment 일반 표현과 TH for this condition의 범위도 다름. |
| fpq_65 | AH_DRIFT, TARGET, SCOPE | FA only high-risk people need screenings ↔ FH 개인 probably no check-ups로 일반성·대상 변경. TA benefits everyone ↔ TH can help도 다름. screening과 check-up을 교환하고 있어 임상 검토 필요, 진위는 여기서 미판정. |
| fpq_616 | EXTRA, CONTEXT | 참 조건에 ongoing/regular follow-up을 추가. 밖의 without restrictions는 유지되어 문맥과의 관계를 별도로 확인해야 함. |
| fpq_505 | SCOPE | 모든 간암 무치료 주장 → 조기 발견 때 일부 완치 가능이라는 조건부 명제. 개인 병기는 미제시. 호스피스 질문이 남는다고 곧바로 오류는 아니지만 이를 정상 사례로 인증할 수 없음. |
| fpq_652 | ACTION, EXTRA, AH_DRIFT, CONTEXT | 참 조건에 chemo/radiation 및 help at home을 새로 도입. FA will be admitted ↔ FH probably, TA can ↔ TH can often도 달라짐. for his return은 밖에 남음. |
| fpq_126 | ACTION, TARGET | 성별별 일반 thyroid cancer 위험과 aggressive thyroid cancers 위험은 동일 명제가 아님. 여성 대상 홍보 → 전체 직원 홍보로 정책도 바뀜. 거짓 조건의 전제 귀속과 의료 진위 모두 추가 확인 필요. |
| fpq_837 | AH_DRIFT, SCOPE | TA aware ↔ TH hoping은 출처 표지만 바꾼 것 이상으로 화자의 태도 변경. 조기암 일반 예후와 이 자매의 상태 구분 필요. 호스피스 탐색 자체를 오류로 판정하지 않음. |
| fpq_559 | CONTEXT, SYNTAX | since ... since 중첩. 참 조건은 유방암 가능성을 열지만 피부과·skin treatments 요청은 유지. biopsy 결과가 없으므로 이 요청 자체를 틀렸다고 단정하지 않음. |
| fpq_570 | ACTION, EXTRA, AH_DRIFT | celebrate cured → enjoy time while facing additional treatments로 여행 의미와 개인 치료 전망 변경. TA still facing likely / does not end ↔ TH may still need / might not end로 명제 양태도 변경. |
| fpq_11 | ACTION, CONTEXT | 개인의 dialysis sessions를 일반 treatment로 바꾸지만 마지막 질문은 evening/weekend dialysis options 그대로. 참 조건에도 잘못된 치료 연결이 남았는지 확인 필요. |
| fpq_97 | ACTION | 요청 대상 자체가 hospice/end-of-life resources → treatment options로 변경되어 답변 과제가 달라짐. |
| fpq_705 | EXTRA, AH_DRIFT, CONTEXT | 참 조건에 often recurs + lifelong monitoring 두 내용 추가. FH generally never도 FA never와 동일하지 않음. 뒤의 방문 간격 연장 계획은 유지. |
| fpq_479 | SCOPE, AH_DRIFT | 일반적으로 sporadic이라는 사실에서 이 조카는 not part of a hereditary pattern이라고 개인 결론을 단정. TH probably not으로 다시 완화. 인구 수준 근거와 개인 판정을 분리해야 함. |
| fpq_754 | ACTION, AH_DRIFT | 이사 이유에 for general well-being 추가. FA cause ↔ FH can cause, TA do not cause ↔ TH are not linked는 완전히 같은 명제가 아님. |
| fpq_56 | EXTRA, AH_DRIFT | FA will cure ↔ FH might cure로 주장 자체 강도 변경. 참 조건에 must/should not delay conventional treatment라는 행동 지침도 추가. |
| fpq_866 | AH_DRIFT, SCOPE | FA surgery alone already cures ↔ FH can cure로 일반성·양태 변경. TA is only one part alongside chemo ↔ TH usually combined and sometimes radiation으로 내용도 달라짐. |
| fpq_5 | CONTEXT | 참 구간에서 평생 장루가 필요하지 않은 사람을 말하면서 lifelong travel insurance를 찾는 문맥이 남음. 거짓 전제 제거 후 질문 의도와 범위 확인 필요. |
| fpq_237 | ACTION, CONTEXT, SYNTAX | last months → treatment로 개인 예후/대상 변경, 끝의 end-of-life discussion 요청 유지. make his treatment, since ..., as comfortable 문장 접합이 어색함. |
| fpq_832 | ACTION, EXTRA | 가족력 자체를 vaginal cancer → cancer로 바꿈. 참 조건에 genetic markers/BRCA 추가. 환자 배경을 보존한 최소 변경이 아님. |
| fpq_121 | EXTRA, AH_DRIFT, SYNTAX | 참 조건에 medical support와 may keep working이라는 개인 전망 추가. usually ↔ often도 바뀜. due to ... so ... 질문 접합에 별도 차이. |
| fpq_85 | TARGET, ACTION | FA impotence OR incontinence에서 참 조건은 incontinence만 다룸. 참 조건에 just in case를 추가해 물품 준비의 이유도 바꿈. |
| fpq_354 | EXTRA, SCOPE | endometrial cancer 발생 위험 → 재발 위험과 close guidance 필요성으로 타깃 확장. 처방 종류·환자 맥락에 대한 임상 판단은 미실시. |
| fpq_629 | EXTRA, SCOPE | 참 조건에 TKI 치료를 추가. not always cause라는 원 참조와 often does not cause라는 빈도 표현을 동일시할 수 없음. |
| fpq_152 | CONTEXT | 다른 사례보다 핵심 always ↔ not always 대조가 단순함. 그래도 source wrapper와 양태 단서가 남고, ALL 개인 치료의 진위는 미검증. 보존 후보이지 확정 적합 라벨은 아님. |
| fpq_697 | ACTION, AH_DRIFT | 정기 진료가 필요 없다는 행동 결론 → 모니터링 효용으로 변경. FA does not need ↔ FH probably does not need, TA can occur ↔ TH may occur도 변함. |
| fpq_111 | ACTION, SCOPE | short time left → years ahead로 개인 예후를 바꿈. 조기 치료의 일반 효과만으로 이 환자의 남은 기간을 확정할 수 없음. |
| fpq_34 | AH_DRIFT, SCOPE | FA every patient ↔ TA many patients로 보편성과 부분집합 대조. TA do not need ↔ TH may not need도 다름. 아버지의 장루가 실제로 영구적인지는 일반 통계만으로 판정 불가. |

## 다음 행동

- 원 파일을 보존한다. 점수에 따라 통과 문항을 고르지 않는다.
- 이 자료를 근거로 기존 게이트가 순수 진위 또는 순수 문체를 읽는다고 결론 내리지 않는다. 스크립트의 neither → topic only 문구도 근거 부족이다.
- 고정된 명제에 동일한 출처 wrapper만 붙이는 비교라면 제한적으로 출처 제시 효과를 측정할 수 있다. 이를 범용 단정성 효과라고 부르지 않는다.
- 새로운 대규모 생성 전에 소수 원문의 원문/참조/구간 경계를 확인해 수정 가능한지 판단한다. 진위 조작에 필요한 최소 명제 변경과 개인 사실·계획 변경을 구분해야 한다.
- 이 감사는 현재 2×2 자료에 한정된다. 이전 전제 수정 자료, Cancer-Myth 전체, CREPE의 품질이나 게이트의 존재 여부에 대한 결론이 아니다.
- 판독 점수가 양쪽 조작에 반응하지 않는다고 자동으로 주제만 읽는 것은 아니다. 약한 판독기, 포화, 조작 실패, 다른 단서 등 대안이 남는다.

전체 변형 문장과 항목별 검토는 로컬 `review-output/truth_form_2026-09-21/audit.json`에 보존했다. 이 Git 커밋에는 검토 문서만 포함한다.
