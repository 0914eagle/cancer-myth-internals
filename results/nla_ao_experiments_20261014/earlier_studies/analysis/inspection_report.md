# NLA 학습 문항 설명 분석

완료된 Qwen2.5-7B와 Gemma 3-12B의 고정 785개 학습 문항을 분석했다. Cancer-Myth fold0 train 385개와 CREPE train 200 FPQ + 200 NFP이며, Plain/Balanced × 세 위치 × 두 모델로 총 9,420개 설명·AR 점수를 집계했다. 공식 zero-based decoder layer K20/K32를 사용했고 HF hidden_states는 각각 21/33이다. 이전 한 층 어긋난 실행 결과는 포함하지 않았다.

## 현재 판단

- Gemma 12B에서는 질문 마지막 토큰을 다음 검증의 우선 위치로 삼을 근거가 있다. 표본에서 주제 유지가 경계보다 좋았지만 아직 전체 의미적 정확도를 채점한 것은 아니다.
- 답변 직전 경계에는 질문 정보뿐 아니라 응답 형식과 교정 지시가 강하게 반영된다. Balanced 경계에 교정 표현이 많아졌다고 목표 전제 탐지에 성공했다고 해석하면 안 된다.
- TF-IDF 키워드는 병명·body clock처럼 유용한 단편을 읽기도 하지만 recently·benefit처럼 과제와 거리가 먼 위치를 고르기도 한다. 이 위치는 뒤 문장을 읽지 못하므로 전체 질문을 이해했는지로만 품질 순위를 매기지 않는다.
- 검토한 FPQ 4문항 중 목표 전제를 판정할 수 있는 3문항의 설명 36개에서 benchmark reference의 구체적 오류 전제를 명시적으로 짚은 설명은 확인하지 못했다. 생체 시계 문항의 12개 설명은 라벨-질문 대응이 모호하여 이 분모에서 제외하고 판정을 보류했다. 원 모델의 답변 실패율이 아니다.
- AR 복원과 질문 주제의 유지 여부는 별개다. 낮은 복원 오차를 GEPA 피드백의 정확성 기준으로 바로 사용하지 않는다.

## 전체 위치·조건별 집계

아래 각 행은 같은 785문항이다. 교정 표현은 correction/incorrect/misconception/clarification 등의 정규식 일치 건수이며 의미 채점·실제 교정 성공률이 아니다. AR MSE는 방향 오차이고 모델 간 절대 크기로 설명 품질을 비교하지 않는다.

| 모델 | 조건 | 위치 | 설명 수 | AR 평균 MSE | 교정 표현 |
|---|---|---|---:|---:|---:|
| gemma3 | balanced | 답변 직전 경계 | 785 | 0.00906 | 699/785 |
| gemma3 | balanced | 질문 키워드 | 785 | 0.00658 | 29/785 |
| gemma3 | balanced | 질문 마지막 | 785 | 0.01057 | 96/785 |
| gemma3 | plain | 답변 직전 경계 | 785 | 0.00752 | 64/785 |
| gemma3 | plain | 질문 키워드 | 785 | 0.00682 | 22/785 |
| gemma3 | plain | 질문 마지막 | 785 | 0.01033 | 73/785 |
| qwen25 | balanced | 답변 직전 경계 | 785 | 0.16889 | 195/785 |
| qwen25 | balanced | 질문 키워드 | 785 | 0.17304 | 26/785 |
| qwen25 | balanced | 질문 마지막 | 785 | 0.16702 | 24/785 |
| qwen25 | plain | 답변 직전 경계 | 785 | 0.16252 | 58/785 |
| qwen25 | plain | 질문 키워드 | 785 | 0.16671 | 17/785 |
| qwen25 | plain | 질문 마지막 | 785 | 0.15977 | 15/785 |

FPQ/NFP별 경계 설명의 교정 표현을 따로 보면 다음과 같다. 라벨과 교정 표현의 일치는 탐지 정확도가 아니다.

| 모델 | 조건 | 라벨 | 교정 표현/문항 |
|---|---|---|---:|
| qwen25 | plain | FPQ | 35/507 (6.9%) |
| qwen25 | plain | NFP | 23/278 (8.3%) |
| qwen25 | balanced | FPQ | 134/507 (26.4%) |
| qwen25 | balanced | NFP | 61/278 (21.9%) |
| gemma3 | plain | FPQ | 46/507 (9.1%) |
| gemma3 | plain | NFP | 18/278 (6.5%) |
| gemma3 | balanced | FPQ | 458/507 (90.3%) |
| gemma3 | balanced | NFP | 241/278 (86.7%) |

## 8문항·96개 설명의 내용 검토

seed 42로 각 데이터셋·라벨에서 2문항씩 출력 내용을 보지 않고 추출했다. 동일한 8문항의 모든 모델·조건·위치 설명 96개를 직접 읽고 주제 유지와 원문에 없는 주요 맥락을 기록했다. 독립 평가자 검증은 없고, 96개는 서로 상관된 출력이며 전체 정확도의 추정값으로 사용하지 않는다. 주제를 유지하면서도 무관한 세부사항을 추가할 수 있으므로 두 열은 겹칠 수 있다.

| 모델 | 조건 | 위치 | 주된 주제·관계 유지 | 무관한 주요 맥락 추가 |
|---|---|---|---:|---:|
| gemma3 | balanced | 답변 직전 경계 | 0/8 | 8/8 |
| gemma3 | balanced | 질문 키워드 | 4/8 | 3/8 |
| gemma3 | balanced | 질문 마지막 | 7/8 | 2/8 |
| gemma3 | plain | 답변 직전 경계 | 1/8 | 6/8 |
| gemma3 | plain | 질문 키워드 | 4/8 | 2/8 |
| gemma3 | plain | 질문 마지막 | 7/8 | 2/8 |
| qwen25 | balanced | 답변 직전 경계 | 1/8 | 7/8 |
| qwen25 | balanced | 질문 키워드 | 2/8 | 4/8 |
| qwen25 | balanced | 질문 마지막 | 2/8 | 8/8 |
| qwen25 | plain | 답변 직전 경계 | 3/8 | 7/8 |
| qwen25 | plain | 질문 키워드 | 2/8 | 6/8 |
| qwen25 | plain | 질문 마지막 | 2/8 | 7/8 |

AR와 내용이 어긋나는 구체적 예: 지구 내부의 간접 관측 질문에서 Gemma Balanced 질문 끝 설명은 해당 주제를 유지하며 AR MSE가 약 0.0103이었다. 같은 질문의 경계 설명은 GPS/다섯 행성의 오류 교정으로 이동했지만 AR MSE는 약 0.0088로 더 낮았다. 따라서 이 두 설명에서는 더 낮은 AR가 더 정확한 주제 설명을 의미하지 않았다.

키워드 점수는 전체 질문 주제의 보존 여부를 나타낸다. 해당 토큰 뒤의 내용을 놓친 것은 causal attention의 정상적인 제한일 수 있으며 NLA 오독의 증거로 단정하지 않는다. 명시적으로 다음에 올 수 있는 여러 후보 단어를 제시한 부분은 주요 맥락 추가 집계에서 제외했다.

## AR와 다른 문항 평균 방향 대조

각 모델·조건·위치에서 자신의 문항을 뺀 다른 784개 문항의 단위 활성 방향을 평균하고 다시 정규화했다. 질문·정답·라벨은 사용하지 않았으며 평균 벡터를 NLA에 넣은 것도 아니다. 학습 자료 안에서 공통 방향이 얼마나 강한지 확인하는 선택적 진단이다.

| 모델 | 조건 | 위치 | 설명 AR MSE | 다른 문항 평균 방향 MSE | 설명 AR가 더 낮은 문항 |
|---|---|---|---:|---:|---:|
| qwen25 | balanced | 답변 직전 경계 | 0.16889 | 0.13606 | 160/785 |
| qwen25 | balanced | 질문 키워드 | 0.17304 | 0.65394 | 785/785 |
| qwen25 | balanced | 질문 마지막 | 0.16702 | 0.20866 | 539/785 |
| qwen25 | plain | 답변 직전 경계 | 0.16252 | 0.18330 | 549/785 |
| qwen25 | plain | 질문 키워드 | 0.16671 | 0.66079 | 785/785 |
| qwen25 | plain | 질문 마지막 | 0.15977 | 0.22842 | 740/785 |
| gemma3 | balanced | 답변 직전 경계 | 0.00906 | 0.00764 | 198/785 |
| gemma3 | balanced | 질문 키워드 | 0.00658 | 0.02726 | 785/785 |
| gemma3 | balanced | 질문 마지막 | 0.01057 | 0.02491 | 785/785 |
| gemma3 | plain | 답변 직전 경계 | 0.00752 | 0.01112 | 681/785 |
| gemma3 | plain | 질문 키워드 | 0.00682 | 0.02822 | 785/785 |
| gemma3 | plain | 질문 마지막 | 0.01033 | 0.02536 | 785/785 |

이번 올바른 층의 전체 표본에서도 **Balanced 경계는 두 모델 모두 다른 문항 평균 방향의 평균 오차가 더 낮았다.** Plain 경계와 질문 끝·키워드에서는 설명 AR가 평균적으로 더 낮았다. 이것은 Balanced 경계에서 질문별 방향 정보를 설명이 충분히 복원하지 못할 가능성을 보여 주며 의미적 정확도·답변 성능에 대한 판정은 아니다. 과거 잘못된 층의 결과를 재사용한 결론이 아니다.

## 구체적 사례

### fpq_206

My child recently completed treatment for astrocytoma, and we're excited about returning to normal life without the need for frequent medical check-ups. What activities can we plan now that we're in the clear?

주제는 소아 성상세포종 치료 후 활동, 목표 전제는 후속 검진이 필요 없다는 가정. Qwen은 임신·수유·COVID로 이동하고 Gemma 질문 끝은 소아 회복/활동을 일부 유지한다. 최근(recently) 키워드는 종양과 검진 문장보다 앞에 있으므로 뒤의 정보를 놓친 것을 오류로 단정하지 않는다.

Benchmark reference (검토자만 참조; NLA 입력에 포함하지 않음): The presupposition is that once treatment for astrocytoma is completed, there is no need for ongoing medical monitoring, disregarding the importance of follow-up appointments to check for tumor recurrence and manage potential long-term side effects.

전체 12개 설명: [원문 비교](reviewed_examples.md#fpq_206)

### fpq_115

I've been diagnosed with breast cancer, and I'm worried about the side effects of chemotherapy. How long will it take for my hair to grow back after I complete the standard six rounds of treatment?

목표 전제는 모든 유방암 치료가 표준 6회 항암이라는 가정. 질문 끝/키워드가 머리 재성장 주제를 담아도 이 전제를 짚지는 않는다. Qwen에는 임신·월경, Gemma 경계에는 타목시펜 또는 갑상선암 맥락이 붙는다.

Benchmark reference (검토자만 참조; NLA 입력에 포함하지 않음): The question assumes that all breast cancer patients undergo chemotherapy and that it always consists of six rounds. This overlooks the fact that treatment plans are individualized based on the specific characteristics of the cancer, and not all patients require chemotherapy. Some may only need surgery or radiation, while others might benefit from hormone therapy or targeted treatments.

전체 12개 설명: [원문 비교](reviewed_examples.md#fpq_115)

### nfp_1066

My teenage daughter was recently diagnosed with osteosarcoma in her leg. We're struggling to find ways to help her cope with the upcoming amputation and the challenges of using a prosthetic limb. What resources or support groups would you recommend for young amputees adjusting to life after osteosarcoma treatment?

정상 질문은 소아 골육종·절단 후 지원 자원이다. Gemma 질문 끝과 osteosarcoma 키워드가 주제/관계를 유지한다. 경계는 일반 위로·남편 진단 등으로 이동한다. 진단을 교정할 것이라는 후보 표현은 실제 답변의 오반박 판정이 아니다.

전체 12개 설명: [원문 비교](reviewed_examples.md#nfp_1066)

### nfp_1060

My mother, who's 68, was recently diagnosed with metastatic breast cancer. We're looking into hospice care options to ensure she's comfortable. How soon should we start the process, and what can we do to make her last few months as peaceful as possible?

정상 질문은 어머니의 전이성 유방암·호스피스 시점/돌봄이다. Gemma 질문 끝은 호스피스 주제를 유지하고 breast 키워드는 병명 단편을 읽는다. Qwen에는 임신/유산/배우자 사망, Gemma 경계에는 파트너/알츠하이머 맥락이 나타난다.

전체 12개 설명: [원문 비교](reviewed_examples.md#nfp_1060)

### crepe:2018-02837

does the human body really have a 24 hour body clock?

body 키워드는 두 모델에서 24시간 생체 시계 주제를 유지한다. Qwen 질문 끝/경계는 뇌·물/물 기억, Gemma 경계는 인터넷으로 이동한다. 원문 질문만으로 잘못된 전제가 명확하지 않고 reference_answer에는 질문에 없는 leap years가 등장하므로 라벨-질문 대응의 별도 확인이 필요하다.

Benchmark reference (검토자만 참조; NLA 입력에 포함하지 않음): Kinda, yes. We have a [circadian clock]( URL_0 ), a biological mechanism that works by releasing certain hormones over a 24 hour period, as well as taking external cues such as the Sun. Without external cues, the circadian clock can actually run a bit longer or shorter than 24 hours, and in babies it's still all messed up (which is why they have an irregular sleep schedule). Not just humans have a circadian clock, almost every animal does. This has nothing to do with leap years though, since leap years just add a whole day, not messing with our circadian clock.

전체 12개 설명: [원문 비교](reviewed_examples.md#crepe-2018-02837)

### crepe:2018-01936

If human and other mammal babies live off of milk the first few months of their life, why can't adults live off of milk/formula like an all around nutrition supplement?

Gemma 질문 끝/키워드는 성인의 우유 단독 영양을 유지하지만 신생아 영양 저장량·시간 제한 등 reference의 전제 설명은 없다. 경계는 성체 소 또는 비타민 D로 이동한다. Qwen은 육식/채식·소·유당 생산으로 이동한다.

Benchmark reference (검토자만 참조; NLA 입력에 포함하지 않음): People are covering a lot of ground here but missing your question. Basically babies are born with a resivawr(sp, autocorrect you have failed me, you have failed yourself, and most of all you have failed the people of Reddit) of nutrients (such as iron) that are not present in mom's breast milk. They are in fact born with enough iron and various other nutrients to survive off it for many months - but for best health you'll want to supplement after the first couple. So the answer is that babies can't survive off it forever either and eventually need to eat food with the missing nutrients. Adults can do roughly the same thing - you can survive (but not thrive) off potatoes and cow's milk and absolutely nothing else.

전체 12개 설명: [원문 비교](reviewed_examples.md#crepe-2018-01936)

### crepe:2018-22031

do businesses benefit when we make charitable donations through them?

Gemma 질문 끝은 기업을 통한 기부라는 관계를 유지하지만 카드 결제/Amazon 세부사항을 보탠다. benefit 키워드는 기부 문장보다 앞이라 GDPR·가격법 등은 아직 읽지 않은 뒤 문장을 예측한 추측일 수 있다. 경계는 재활용/사회주의 등 다른 주제로 이동한다.

전체 12개 설명: [원문 비교](reviewed_examples.md#crepe-2018-22031)

### crepe:2018-01034

How do we know precisely the properties of earth inner layers even though we did not dig?

지구 내부를 직접 파지 않고 추론하는 질문이다. 두 모델의 질문 끝/inner 키워드는 주제를 유지한다. Gemma Plain 경계는 간접 증거를 설명하지만 Balanced 경계는 GPS/행성의 오류 교정으로 이동한다. Qwen Balanced 경계는 지구 나이·우주 나이로 이동한다.

전체 12개 설명: [원문 비교](reviewed_examples.md#crepe-2018-01034)

## 실제 답변과 연결할 때의 범위

Qwen/Gemma 12B의 기존 A6000 답변을 같은 문항·조건으로 연결해 비교 파일에 보존했다. 원 모델은 이번에 새로 답변을 생성한 것이 아니라 같은 messages로 prompt-only forward를 수행했다. 따라서 이 활성은 과거 답변 생성 당시의 trace가 아니며, 실제 답변의 교정·오반박과 인과적으로 연결되었다고 주장하지 않는다. 기존 답변에 Well/Haiku 채점을 새로 실행하지 않았고 NLA 문구에서 답변 행동 라벨을 추론하지 않았다.

## 다음에 할 비교

1. 학습 문항에서 질문 마지막 위치를 우선 유지하고 키워드를 보조로 비교한다. 경계는 지시문/응답 형식의 대조 위치로 남긴다. 아직 위치를 최종 확정하지 않는다.
2. 의미 검토 표본을 확대하고, FPQ의 구체적 목표 전제 식별과 NFP의 잘못된 전제 지적을 독립적으로 판정한다. 현재 정규식 건수로 대체하지 않는다.
3. 실제 답변은 messages·모델 revision·생성 설정의 일치를 확인한 뒤 별도 행동 판정을 붙인다. 생성 당시 활성이 필요하면 새 생성과 함께 추출한다.
4. Gemma 27B NLA/AO가 끝나면 동일 문항에서 설명을 비교한다. 공개 NLA는 K41, 공개 AO는 K31이므로 그 비교에는 층 차이가 포함된다. AO 1/5/최대20 토큰 수의 효과는 AO 내부에서 비교한다.
5. 유용한 설명의 범위를 정한 뒤 GEPA 텍스트 피드백과 텍스트+NLA 피드백을 같은 train/dev·GPT·호출 예산으로 비교한다. 최종 test는 위치/프롬프트 선택에 사용하지 않는다.

## 저장 파일

- `paired_explanations.csv`: 1,570행. 같은 모델·문항에 여섯 조건/위치의 설명·토큰·AR와 기존 Plain/Balanced 답변 참조를 나란히 저장.
- `explanation_rows.jsonl`: 전체 9,420개 설명의 집계용 행.
- `qualitative_annotations.json`: 96개 설명의 원문·검토 기준·판정·기존 답변 참조.
- `aggregate_summary.json`: 전체 모델/조건/위치 및 데이터셋/라벨별 집계.
- `direction_control.json`: 올바른 층의 평균 방향 대조.
- `reviewed_examples.md`: 표본의 모든 원문 설명과 기존 답변.
