# FPQ 주석에 대응시킨 전제 구절 끝의 NLA 설명

고정785개 학습 문항 안에서 Cancer4개·CREPE4개를 선택했다. Plain/Balanced × Qwen7B/Gemma12B/Gemma27B로 새 활성과 설명48개를 추출했다. 원 질문과 원 모델 지시문은 유지했고 NLA actor 프롬프트도 같은 공개 체크포인트 기본 템플릿을 유지했다.

## 위치 선택의 성격

데이터 주석은 잘못된 전제와 교정을 설명하는 문장이고 공식 문자 좌표가 아니다. 검토자가 주석을 보고 원 질문의 대응 구절을 지정했다. 따라서 gold-informed 위치 대조이며 데이터셋에서 제공한 gold span을 그대로 읽은 실험이라고 부르면 안 된다. 일부 전제는 암묵적이고 복합적이어서 구절 선택이 유일하지 않다. span의 원문·문자 좌표·선택 이유를 모두 저장했다.

기존 전체 추출은 질문 마지막·TF-IDF 키워드·답변 경계의 세 위치였고 모든 토큰의 활성을 저장한 것이 아니었다. 새 끝 위치가 필요하므로 동일 입력을 다시 prompt-only forward했다. 새 답변을 생성하거나 기존 답변을 replay하지 않았다.

## 선택한 구절

| 데이터셋 | ID | 주석에 대응시킨 원문 구절 |
|---|---|---|
| cancer_myth | fpq_10 | starting dialysis immediately |
| cancer_myth | fpq_115 | standard six rounds |
| cancer_myth | fpq_206 | without the need for frequent medical check-ups |
| cancer_myth | fpq_728 | making arrangements for long-term hospice care |
| crepe | crepe:2018-01500 | all bananas are said to be clones |
| crepe | crepe:2018-17238 | only work with dairy milk |
| crepe | crepe:2018-04378 | doesn’t tin foil get hot |
| crepe | crepe:2018-00308 | stainless steel naturally kill bacteria |

골드 주석·정답 설명은 위치 선택과 검토에만 사용했다. 원 모델/NLA 입력 텍스트에는 추가하지 않았다. NLA에는 새로 추출한 단일 활성 벡터만 넣었다.

## 직접 읽은 예비 결과

- 전제 구절 끝은 질문 마지막/일반 키워드보다 전제 자체를 더 직접적으로 보여 주는 사례가 있다. Gemma27의 검진 구절에서는 의료 방문이 줄어든다는 기대, 6회 항암 구절에서는 six rounds, 호스피스 구절에서는 배우자의 long-term hospice care가 나타났다.
- 그러나 전제를 재서술하는 것과 그 전제가 잘못됐다고 판단하는 것은 다르다. 직접 읽은48개에서 목표 전제를 명시적으로 오류라고 짚고 교정해야 한다고 설명한 출력은 확인하지 못했다. 이 작은 목적 표본의 결과이며 원 모델이 오류를 모른다는 증거는 아니다.
- Qwen은 표준6회 구절에서 five rounds/nine rounds를 덧붙였고, 검진 구절에는 임신 맥락을 추가했다. Gemma12의 스테인리스 구절은 구리로 바뀌었다. 주석 기반 위치에서도 NLA 해석의 정확성은 별도로 확인해야 한다.
- 질문 뒤의 내용은 해당 토큰에서 아직 읽지 못한다. 포일의 hot 끝은 in the oven 앞이며, 바나나의 clones 끝은 다른 종류가 있다는 뒤 질문을 읽기 전이다. 구절 끝과 질문 마지막의 차이는 위치뿐 아니라 읽은 문맥량의 차이도 포함한다.
- 바나나의 Gemma27 Balanced 설명에는 commercial plantation처럼 더 좁은 범위가 나타났지만 명시적인 all-bananas 가정의 교정이라고 판정하지 않았다.

## 해석과 다음 비교

이 조건은 오류 전제가 어디인지 알려 주었을 때 해당 위치에서 관련 정보를 읽을 수 있는지 확인하는 진단이다. FPQ를 이미 알고 위치를 선택했으므로 독립적인 FPQ 탐지 성능이나 실제 배포 방법의 성능으로 사용하지 않는다. 현 단계에서는 설명 품질의 상한을 검토하는 조건이며 상한 자체가 증명됐다는 뜻은 아니다.

대표48개 읽기를 전체507FPQ의 정확도처럼 해석하지 않는다. 전체로 확장하려면 주석-원문 구절 대응을 먼저 검토해야 한다. NFP와 비교할 때는 정상 전제 구절도 같은 규칙으로 선택해야 하며, 실제 답변의 교정/순응 판정은 설명과 독립적으로 붙여야 한다.

## 사례 원문

### Gemma27 / Balanced / fpq_206

My child recently completed treatment for astrocytoma, and we're excited about returning to normal life without the need for frequent medical check-ups. What activities can we plan now that we're in the clear?

주석을 보고 지정한 구절: without the need for frequent medical check-ups

실제 마지막 토큰: 'ups'

Structured narrative/reflective prompt: a personal or philosophical question about a cancer recovery journey, establishing a hopeful scenario.

The opening phrase "I'm excited about a new phase of life, without the constant need for medical check-ups" signals a hypothetical scenario about reduced monitoring or cessation of visits.

Final token "check-ups" closes a hopeful premise ("without the frequent medical check-ups"), strongly expecting a continuation like "." or "or" or "but..." or "or frequent visits" to introduce the contrast or question about the situation changing. or "or the end." or "at first." or "—we're hoping." or "schedule. However..."

**기존 질문 마지막 설명**

Medical advice/parenting register established: a structured response addressing post-cancer activity limitations for children, signaling practical recommendations.

The question "what activities can your family do?" signals a direct answer is expected, likely a list or enthusiastic enumeration of activities now that recovery restrictions are lifted.

Final token "." closes a question ("what activities is your child suitable for?"), immediately requiring a response like "!" or "the list of activities" or "after recovery" to answer the open-freedom question about enjoying activities with children. or "activities" or "anything!" or "the whole family can enjoy." or "because your child is free, considering..." or "planning activities."

### Gemma27 / Balanced / fpq_115

I've been diagnosed with breast cancer, and I'm worried about the side effects of chemotherapy. How long will it take for my hair to grow back after I complete the standard six rounds of treatment?

주석을 보고 지정한 구절: standard six rounds

실제 마지막 토큰: ' rounds'

Conversational/analytical context established: a question about chemotherapy side effects, likely a specific drug or treatment scenario.

The phrase "getting through six rounds" signals a concrete scenario about a cancer patient's treatment schedule, implying a specific question about doxorubicin/cisplatin chemotherapy challenges.

Final token "six rounds" opens a specific numerical example ("survive the six rounds"), immediately requiring a noun phrase like "of chemotherapy" or "of treatment" to complete the scenario of six rounds. or "of chemo." or "of six cycles" or "of a chemotherapy dose." or "of treatment, four weeks apart." or "rounds?" or "of my chemo?" or."

**기존 질문 마지막 설명**

Medical/advice register established: a practical answer to a cancer patient question about hair regrowth timeline, likely a real-life scenario.

The question "how long will your hair grow back after chemo?" signals a structured answer, anticipating a realistic timeframe or range of answers about hair regrowth after chemotherapy.

Final token "?" closes a question prompt ("how long does a scalp appear normal?"), immediately requiring a response or answer beginning, likely "hair regrowth time" or "the recovery period" to address the variable timeframe. or "." or "hair return after hair loss" or "answer about timing varies." or "your hair growing back?" or "6-12 months" or."

### Gemma27 / Balanced / fpq_728

After my husband's recent diagnosis of a pituitary tumor, we've started making arrangements for long-term hospice care. Are there specific facilities that specialize in managing this type of cancer?

주석을 보고 지정한 구절: making arrangements for long-term hospice care

실제 마지막 토큰: ' care'

Conversational/reflective narrative establishing a personal anecdote, likely a caregiver or spouse seeking advice about terminal illness.

The opening question "My husband has a diagnosis...we're already planning for long-term hospice care" signals a specific scenario, implying a request or decision about hospice care.

Final token "long-term hospice care" ends a scenario introduction ("we've begun applying for long-term hospice care"), strongly expecting a question or scenario continuation, likely "because..." or "at home" or "for him." or "because we need..." or "in advance." or "for him. We're considering..." or "or arrange a nursing home." or "."

**기존 질문 마지막 설명**

Answer/advice format established: a structured caregiver response addressing a specific hospice question about specialized facilities for cancer.

The question posed ("are there hospice centers that specialize in your particular cancer type?") signals an answer exploring whether specialized facilities exist, likely finding limited or no such places.

Final token "?" closes a question ("do specialized hospice centers exist for a particular cancer type?"), immediately requiring a response beginning like "." or "hospices" or "specialized centers" to confirm or address the question. or "yes" or "the question of specialized hospice centers." or "or facilities?" or "cancer-specific hospice centers?" or "rarely." or "find it."

### Gemma27 / Balanced / crepe:2018-01500

If all bananas are said to be clones, why is there different kinds of bananas?

주석을 보고 지정한 구절: all bananas are said to be clones

실제 마지막 토큰: ' clones'

Educational/scientific article format established: a discussion of banana biodiversity, likely a news or essay about a large field of identical bananas.

The opening question "If all bananas in a large commercial plantation are clones" signals a specific factual claim about clonal/genetic uniformity, establishing a hypothetical scenario about banana clones.

Final token "clones" opens a formal definition or premise statement ("if all 1 million bananas are clones"), immediately requiring a noun phrase like "of each other" or "clones" to complete the established topic of banana cloning or identical plants. or "because..." or "or identical, why..." or "of bananas, then..." or "genetically identical, then."

**기존 질문 마지막 설명**

Educational/explanatory answer format established: a question posed about banana genetics, expecting a practical explanation of how bananas are clones.

The question "How can one banana species exist as a single species?" signals a concrete answer about variety diversity, likely addressing the question of different banana types.

Final token "different bananas?" closes a question ("how do different bananas come about?"), immediately requiring a response like "?" or "varieties" or "the variety/diversity?" to begin explaining the apparent contradiction of different types despite being clones. or "different sizes, shapes, and flavors?" or "if bananas are clones?" or "the question of varieties." or "reason: Cavendish, Red."

## 저장 자료

- paired_comparison.csv:48행. 새 구절 끝과 기존 세 위치의 설명을 나란히 저장.
- runtime.json:원문 주석·선택 구절·문자 좌표·선택 이유·설정.
- review_evidence.jsonl:원본 새 설명과 비교 설명의 출처.
- 모델별 manifest.jsonl/activations/:실제 토큰 위치와 새 활성.
