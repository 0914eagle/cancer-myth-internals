# 전제 교정의 부작용, 원인 규명은 아직 비어 있다

거짓 전제 질문(FPQ) 처리를 개선하는 방법이 정상 전제 질문(NFP/TPQ) 성능을 떨어뜨린다는 보고는 2023–2026년 문헌에 **최소 6편** 있다. FalseQA, FreshLLMs, CoCoNot, Cancer-Myth(ICLR 2026), "Don't 'Well, Actually' Me"(2608.06539), "Two Axes of LLM Abstention"(2607.08456)이다. 일반적인 "전제를 점검하라" 계열 개입에서 오탐 규모는 **41%(Cancer-Myth-NFP), 57%(건전한 질문에 대한 반박), "참 전제의 절반 이상 기각"**으로 서로 독립적인 세 보고가 비슷한 크기를 보인다. 그런데 **이 트레이드오프의 원인을 관찰보다 한 단계 아래에서 증거와 함께 규명한 논문은 이번 탐색에서 찾지 못했다.** 원인에 가장 가까이 간 두 편도 원인의 *위치*를 짚는 데서 멈춘다. Well Actually는 "검증 단계가 검증할 수 없는 참 전제를 기각한다"는 단계를 지목했고, Two Axes는 "모델 내부(프로브)는 전제를 판별하지만 출력은 판별하지 못한다"는 해리를 보였다. 이 둘은 위치이지 기제가 아니다. 반면 인접 분야(안전 과잉거부, 아첨 억제, 기권 학습)는 판단-응답 분리, 공유 기질, 비특이적 조향 방향, 과제 의존 부분공간, 표면 단서 의존, 기준(criterion) 이동 같은 기제를 인과 개입 증거로 제시했다. 거짓 전제 연구가 빌려 와 검증할 수 있는 원인 후보가 이미 갖춰져 있다는 뜻이다. 단, **모든 출처는 검색 결과 스니펫과 초록 수준으로만 읽었다.** 이 세션에서 arXiv, ACL Anthology, OpenReview 등 전문 접근이 모두 차단되었다(GitHub 저장소 두 곳만 직접 읽음). 따라서 "찾지 못했다"는 "존재하지 않는다"와 같은 말이 아니며, 아래 수치 상당수는 전문 대조가 필요하다.

## 교정 강화가 정상 질문을 깎는다는 보고는 쌓였지만 대부분 관찰에서 멈춘다

트레이드오프가 처음 분명하게 드러난 곳은 학습 개입이다. **FalseQA**(Hu et al., ACL 2023)는 FalseQA만으로 파인튜닝하면 일반 질문까지 반박하게 된다고 보고했다. 저자들은 이를 "catastrophic forgetting"이라 부르고, ARC-DA 같은 일반 질문을 섞는 데이터 재생(replay)으로 완화해 **FPQ 86.7% 판별, 일반 질문 반박 1.4%**를 얻었다 ([FalseQA ACL PDF](https://aclanthology.org/2023.acl-long.309.pdf)). 1.4%는 재생을 *적용한 뒤*의 수치다. 재생이 없을 때의 반박률은 스니펫에서 확보하지 못했다. **CoCoNot**(Brahman et al., NeurIPS 2024 D&B)은 거짓 전제를 포함한 비순응 학습에서 직접 SFT가 "과잉거부와 일반 성능 저하"를 낳고, LoRA와 대조셋 DPO가 이를 줄인다고 보고했다 ([CoCoNot NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/58e79894267cf72c66202228ad9c6057-Paper-Datasets_and_Benchmarks_Track.pdf)). 두 논문이 붙인 이름("망각", "양성 요청 거부에 대한 과적합")은 현상의 명칭이지 기제가 아니다.

프롬프트 개입에서도 같은 패턴이 반복된다. **FreshLLMs**(Vu et al.)는 "답하기 전에 전제가 타당한지 확인하라"는 지시가 거짓 전제 정확도를 strict 기준 **+23.4%p(GPT-3.5), +6.4%p(GPT-4)** 올리지만 "타당한 전제 질문의 정확도를 해칠 수 있다"고 적었다 ([FreshLLMs ACL Findings PDF](https://aclanthology.org/2024.findings-acl.813.pdf)). 하락 폭은 스니펫에서 확보되지 않았다. **Cancer-Myth** ICLR 2026판은 예방적 프롬프트에 GEPA 최적화를 더해 Cancer-Myth 정확도를 **80%**까지 올렸지만 그 대가로 **Cancer-Myth-NFP의 41%를 잘못 지적**했고 다른 의료 벤치마크에서 **상대 10% 하락**이 생겼다 ([Cancer-Myth OpenReview](https://openreview.net/forum?id=fOXLhZIaUj)). 이 수치는 OpenReview 판의 검색 요약에서 나왔고 v1 arXiv 초록에는 없다. **Two Axes**(Wagner, arXiv 2607.08456)는 전제 점검 지시가 "건전한 전제와 거짓 전제를 똑같이 반박하게 만든다(**57% false challenges**)"고 보고했다 ([Two Axes arXiv](https://arxiv.org/abs/2607.08456)). **Well Actually**(Wang, Shwartz & Gonen, arXiv 2608.06539)는 4개 벤치마크, 5개 모델군, 여러 RAG 설정 전반에서 "FPQ에서 나은 방법일수록 TPQ에서 나쁘다"는 일반화를 제시했다. 검증 단계는 거짓 전제를 거의 다 기각하는 동시에 **참 전제의 절반 이상도 기각**했다 ([Well Actually arXiv](https://arxiv.org/abs/2608.06539)).

반례와 주변 사례도 있다. Cheng, Hawkins & Jurafsky(ACL 2026)는 "wait a minute" 같은 화용적 개입이 Cancer-Myth 등에서 성능을 크게 올리면서도 **낮은 오탐률을 유지한다**고 주장한다. 이 논문은 150문항 NFP 세트로 FPR을 빼는 보정 점수를 쓴다 ([Cheng et al. arXiv](https://arxiv.org/abs/2601.04435)). PreWoMe도 정상 질문에서 손실이 없다고 주장한다 ([PreWoMe arXiv](https://arxiv.org/abs/2310.16147)). 기제 연구인 FAITH와 Knowing-but-Not-Correcting은 거짓 전제 *수용* 쪽을 다루고, 정상 전제 대조군을 측정했는지는 확인되지 않았다.

### 표 1. FPQ 개선 방법과 NFP 측 손실 (스니펫 기준)

| 논문 | 도메인 | 방법(개입) | NFP 측 측정 | 트레이드오프 크기 | 원인 제시 수준 | 증거 유형 |
|---|---|---|---|---|---|---|
| FalseQA (Hu et al., ACL 2023) | 상식 FPQ | FalseQA 파인튜닝, 일반 질문 재생 | 예 (일반 질문 반박률) | 재생 적용 시 FPQ 86.7% / 일반 반박 1.4%. 재생 없을 때 수치 미확보 | 관찰+완화 ("catastrophic forgetting" 명명) | 행동 (학습 개입) |
| FreshLLMs (Vu et al., ACL Findings 2024) | 시의성 QA | "premise check" 지시 | 예 (valid-premise 정확도 별도 보고) | FP +23.4%p/+6.4%p (strict). valid 하락은 정성 서술만, 크기 미확보 | 관찰+완화 (기본 프롬프트에서 점검 제외) | 행동 (프롬프트 ablation) |
| CoCoNot (Brahman et al., NeurIPS 2024) | 비순응 전반 (거짓 전제 포함) | 직접 SFT, LoRA, 대조셋 DPO | 예 (대조셋 379개, XSTest 양성) | 방향만 확인, 수치 미확보 | 관찰+완화 ("과적합" 명명) | 행동 (학습 개입) |
| Cancer-Myth (Zhu et al., ICLR 2026) | 암 환자 질문 | 예방적 프롬프트+GEPA | 예 (NFP 150문항) | 정확도 80%, NFP 41% 오탐, 타 의료 벤치 상대 10% 하락 | 관찰만 | 행동 |
| Well Actually (2608.06539, 2026.8) | 사실지식, CREPE, 의료 (4개) | 추출→팩트체크→응답 파이프라인 | 예 (TPQ) | 모델군·크기·RAG 전반에 일관. 참 전제 절반 이상 기각 | **위치** (검증 단계) + 결정 규칙 ("검증 불가 = 기각") | 파이프라인 단계별 행동 분석. 내부 분석 없음 |
| Two Axes (2607.08456, 2026.7) | CREPE 등, 2B–14B 5개 모델 | 전제 점검 지시, 프로브 라우팅 | 예 (건전한 질문) | 건전한 질문 57% 반박. 라우팅으로 반박 정밀도 약 3배 | **위치** (내부 판별, 출력 비판별) + "기준 이동" 서술 | 선형 프로브 + 행동 |
| Syn-(QA)² (Daswani et al., 2024) | 합성 Wikidata/HotpotQA 쌍 | 개입 없음 (모델 비교) | 예 (쌍 설계) | Llama-2가 내용과 무관하게 "거짓 가정 있음" 응답 | 관찰 ("언어 구조" 추측) | 행동 (trivial probe 대조) |
| Cheng, Hawkins, Jurafsky (ACL 2026) | Cancer-Myth, SAGE-Eval, ELEPHANT | "wait a minute" 등 화용 개입 | 예 (NFP 150, FPR 차감) | "낮은 FPR 유지" 주장. 조건별 FPR 미확보 | 과소교정 쪽 원인 (화용적 수용). 과잉교정 원인은 없음 | 행동 (단서 조작) |
| PreWoMe (Han et al., EMNLP 2023) | 장문 QA | 전제 추출 working memory | 예 | 손실 없다고 주장, 수치 미확보 | 해당 없음 | 행동 |
| Premise verification via RALR (2504.06438) | 사실 QA | 논리형 변환+검색 검증 | 예 (음성 정확도) | 약한 설정에서 정확도 91.11% / TPR 37.78%. 과소탐지 방향 | 관찰 | 행동 |
| Kim 2021, (QA)², CREPE (2021–2023) | NQ, 검색 질의, Reddit | 전제 생성·검증, 탐지 벤치마크 | 암묵적 (macro-F1, 균형 설계) | 교정 유발 손실로 다루지 않음 | CREPE는 탐지 약세를 검색 병목으로 설명 | 행동/오류 분석 |
| Knowing but Not Correcting (2605.05957) | 과제 요청 속 거짓 주장 | CDS, DPA | 확인 못함 | Qwen3.5-9B 교정 0→58.2%. 참 진술 대조군 미확인 | **억제 쪽 기제** (초기층 주의 전환, 중간층 순응 의도). 과잉교정 기제 아님 | 은닉상태, 주의, 불확실성 |
| FAITH (Yuan et al., EMNLP 2024) | Movie/Prize, Llama-2 | false-premise head 제약 | 측정 안 된 것으로 보임 | Llama-2-7b Movie 46.53→77.74% | 거짓 전제 *환각*의 기제 (헤드). 과잉교정 아님 | 헤드 식별·제약 |
| FPCO-Dialog (2609.03331), RPCBench (2609.00918) | VLM 대화, 추천 | 벤치마크 | 예 (CorrFP@K, 깨끗한 질의 FPR) | RPCBench 0.55% (24/4,400, 귀속 불확실). CorrFP@K 수치 미확보 | 관찰 | 행동 |

출처: FalseQA ([ACL](https://aclanthology.org/2023.acl-long.309.pdf)), FreshLLMs ([ACL Findings](https://aclanthology.org/2024.findings-acl.813.pdf)), CoCoNot ([arXiv](https://arxiv.org/pdf/2407.12043)), Cancer-Myth ([OpenReview](https://openreview.net/forum?id=fOXLhZIaUj)), Well Actually ([arXiv](https://arxiv.org/html/2608.06539)), Two Axes ([arXiv PDF](https://arxiv.org/pdf/2607.08456)), Syn-(QA)² ([arXiv](https://arxiv.org/html/2403.12145)), Cheng et al. ([arXiv PDF](https://arxiv.org/pdf/2601.04435)), PreWoMe ([arXiv](https://arxiv.org/abs/2310.16147)), RALR ([arXiv PDF](https://arxiv.org/pdf/2504.06438)), (QA)² ([GitHub](https://github.com/najoungkim/QAQA)), CREPE ([ACL](https://aclanthology.org/2023.acl-long.583.pdf)), Knowing but Not Correcting ([arXiv HTML](https://arxiv.org/html/2605.05957v2)), FAITH ([ACL](https://aclanthology.org/2024.emnlp-main.155.pdf)), FPCO-Dialog ([arXiv](https://arxiv.org/html/2609.03331)), RPCBench ([arXiv](https://arxiv.org/pdf/2609.00918)).

## 원인에 가장 가까운 두 편도 '위치'를 짚었을 뿐 '기제'는 아니다

질문의 핵심에 대한 답은 이렇다. 이번 탐색 범위에서 **FPQ↔NFP 트레이드오프의 기제를, 관찰보다 한 단계 아래의 설명으로 증거와 함께 제시한 논문은 찾지 못했다.** 여기서 구분을 분명히 해 둔다. "어느 단계·어느 신호에서 실패가 일어나는가"는 *위치(location)*다. "왜 그 단계가 참 전제에 대해 그렇게 동작하는가"가 *기제(mechanism)*다. 기제의 예로는 두 판단이 같은 내부 구성요소를 공유한다, 응답 정책이 판단이 아닌 표면 단서에 묶여 있다, 분해 과정이 원래 주장을 더 강한 주장으로 바꾼다 같은 것이 있다.

**Well Actually**의 원인 진술은 "약한 팩트체크 모듈이 참 전제까지 기각한다", 즉 "검증할 수 없는 참 전제를 기각한다"이다 ([Well Actually arXiv](https://arxiv.org/abs/2608.06539)). 이 설명은 손실을 추출·응답 단계가 아닌 *검증 단계*에 국소화하고, "검증 불가를 거짓으로 처리하는" 결정 기본값을 서술한다. **이것은 위치이지 기제가 아니다.** 검증이 참 전제에서 왜 실패하는지는 스니펫 수준에서 설명되지 않는다. 증거 부족 때문인지, 추출된 하위 주장이 원래 질문의 전제보다 강하거나 축자적이기 때문인지(분해가 주장을 바꾸는 가설), 모델의 기저 사전확률 때문인지가 남아 있다. 이 논문은 RAG 변형을 시험했다. 만약 전문에서 "증거를 늘리면 TPQ 기각이 줄어든다"는 결과가 나오면 원인은 "증거 가용성"이라는 한 단계 아래 설명으로 이동한다. 그 결과는 스니펫에서 보이지 않았다. 현실적 FPQ 비율(약 13%)로 가중하면 단순 직접 QA가 가장 낫고 가장 공격적인 FPQA 방법이 가장 나쁘다는 진술도 스니펫에 있다 ([Well Actually HTML](https://arxiv.org/html/2608.06539)).

**Two Axes**는 한 걸음 더 들어가 내부 증거를 제시한다. CREPE에서 답변 신뢰도, P(IK), P(True), 직접 질문은 모두 우연 수준에 머물지만 은닉상태 선형 프로브는 **AUROC 0.69–0.77**에 이른다. 저자는 "지시는 모델의 임계값을 옮길 뿐 지식을 옮기지 않는다"고 해석한다 ([Two Axes arXiv PDF](https://arxiv.org/pdf/2607.08456)). 이는 신호탐지 이론 수준에서 "판별력(d′)은 그대로 두고 기준(c)만 이동한다"는 서술이며, 판별 정보가 내부에는 있고 출력에는 없다는 *위치* 증거다. **"모델은 내부적으로 판별하지만 출력에서는 판별하지 못한다"는 것은 기제가 아니다.** 생성 경로가 왜 그 신호를 읽지 않는지, 즉 어떤 층·헤드·방향에서 연결이 끊기는지는 패칭이나 소거 분석 없이 남아 있다. 프로브 라우팅으로 반박 정밀도를 약 3배 높인 것은 우회책(workaround)이다. 이 논문은 단일 저자의 2026년 7월 프리프린트이고 프로브 AUROC도 중간 수준이라는 점을 함께 고려해야 한다.

나머지 논문들은 관찰과 완화에 머문다. FalseQA의 "망각", CoCoNot의 "과적합", FreshLLMs의 "점검을 기본값에서 뺀다"는 모두 현상에 이름을 붙이거나 설계 선택으로 피한 것이다. Cancer-Myth는 41%를 보고했을 뿐 원인 설명은 과소교정 쪽("의학 사실을 알지만 적용하지 않는다")에만 있다. Cheng et al.의 화용적 수용 이론은 *왜 교정하지 않는가*를 설명하는 진짜 원인 이론이다. 그러나 과잉교정 쪽 원인은 다루지 않고, 오히려 단서 수준 개입이 큰 오탐 없이 교정을 늘렸다고 주장해 비교 기준점이 된다. 추론을 하나 덧붙이면, "wait a minute"이 전역 반박 임계값을 낮추는 대신 전제를 덜 배경화(at-issue화)해 *처리 방식*을 바꾸기 때문일 수 있다. 다만 이 해석은 어느 논문도 내부적으로 검증하지 않았다.

세 독립 보고(41%, 57%, 절반 이상)가 비슷한 크기라는 점은 하나의 SDT 수준 패턴을 시사한다. 일반적인 "경계하라" 개입은 판별력보다 기준을 훨씬 크게 움직인다. GEPA처럼 FPQ 세트만으로 프롬프트를 최적화하면 반박에 관대한 기준을 찾는 것이 목적함수상 예측 가능한 결과다. 이 역시 필자의 추론이며, 기제 수준 설명이 되려면 아래 인접 분야의 후보 중 하나를 거짓 전제 과제에서 직접 검증해야 한다.

**"찾지 못했다"와 "없다"의 구분.** 2021–2024년 고전 문헌(Kim 2021, (QA)², CREPE, FalseQA, FreshLLMs, CoCoNot, Syn-(QA)², FAITH)에 대해서는 여러 검색 경로가 일관되게 기제 부재를 가리키므로 "없다"에 가깝다고 본다. 확신도는 중상이다. 2025–2026년 문헌은 사정이 다르다. 관련 논문이 2026년 7–9월에 집중적으로 나오고 있고(2607.08456, 2608.06539, 2609.03331, 2609.00918), 전문을 읽지 못했기 때문에 본문의 오류 분석 절에 기제 수준 분석이 있을 가능성을 배제할 수 없다. 특히 Well Actually의 "거짓 판정 대 검증 불가 판정" 분리 여부, Two Axes의 층별 분석 여부, Knowing-but-Not-Correcting의 참 진술 대조군 여부는 전문으로 확인해야 한다. "분해가 주장을 바꾼다"는 가설은 어느 논문에서도 직접 검증된 흔적을 찾지 못했다. 비영어권, 워크숍, 미색인 논문은 탐색하지 않았다.

## 인접 분야는 여섯 갈래의 기제를 인과 증거로 확보했다

안전 과잉거부 문헌은 "X를 억제하면 이웃 Y가 손상된다"를 가장 성숙하게 설명한다. **Arditi et al.**은 13개 채팅 모델에서 거부가 단일 차이평균 방향으로 매개되며, 이 방향을 *더하면* 무해한 지시도 거부한다는 것을 인과적으로 보였다 ([Arditi et al. NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)). **Zhao et al.**은 유해성이 마지막 지시 토큰에, 거부가 그 이후 토큰에 따로 부호화되고, 모델이 "내부적으로 무해하다고 알면서도 과잉거부할 수 있다"는 *판단-응답 분리*를 조향으로 입증했다 ([Zhao et al. arXiv](https://arxiv.org/abs/2507.11878)). 이것이 Two Axes의 위치 증거를 기제로 끌어올리는 데 필요한 종류의 분석이다. 판단 표상과 응답 표상을 따로 찾고 각각을 조향해 인과적 역할을 분리했기 때문이다. **Maskey et al.**은 유해 거부 방향은 과제 무관한 단일 벡터인 반면 과잉거부 방향은 양성 과제 군집 안에 있는 과제 의존적 고차원 부분공간이라서 전역 방향 소거만으로는 고칠 수 없다고 설명한다 ([2603.27518](https://arxiv.org/pdf/2603.27518)). 같은 맥락의 조건부 조향(SafeConstellations)은 과잉거부를 **최대 73%** 줄였다 ([SafeConstellations](https://arxiv.org/abs/2508.11290)). **OverKill**은 "kill" 같은 유해 단어에 대한 과도한 주의라는 *표면 단서 지름길*을 지목했다. 안전 강조 프롬프트가 이를 악화시키며, 강조 프롬프트 유무를 대비하는 Self-CD는 거부율을 평균 **20%** 낮췄다 ([OverKill ACL 2024](https://aclanthology.org/2024.acl-long.253/)). NASA는 예/아니오 과제의 "No" 지름길을 부정 토큰에 주의하는 특정 헤드로 국소화하고 그 헤드만 미세조정했다 ([NASA NAACL 2025](https://aclanthology.org/2025.naacl-long.503/)).

아첨 억제 문헌은 *공유 기질*과 *비특이적 방향*을 보여 준다. **"Sycophancy Suppression Can Impair Rational Updating"**(2608.26511)은 근거 없는 굴복(UY)을 줄이면 근거 있는 수정(RU)도 줄어드는 트레이드오프가 공동 최적화에서도 지속된다고 보고했다. 원인으로는 두 행동을 움직이는 MLP 뉴런과 어텐션 헤드가 크게 겹치고 조향 방향 코사인이 **+0.40~+0.84**로 양의 정렬을 이룬다는 점을 들었다. 직교화 조향은 선택적 설정을 **36개 중 5개에서 10개**로 늘리는 데 그쳤다 ([2608.26511](https://arxiv.org/abs/2608.26511); [GitHub](https://github.com/dependentsign/sycophancy-rational-updating)). **Dual-Stance**(2606.11205)는 Llama-3-8B-Instruct가 아첨적 동의와 사실적 동의를 기하학적으로 구별된 영역에 표상하지만, 중심차 조향 방향이 둘에 똑같이 투영되어 아첨 동의를 **89%**, 참 진술 동의를 **14%** 줄인다는 것을 보였다 ([Dual-Stance arXiv](https://arxiv.org/abs/2606.11205)). 손상 크기가 항목의 이중 입장 일관성으로 예측된다는 점은, 거짓 전제 과제라면 모델이 불확실한 정상 전제 항목이 가장 크게 손상될 것이라는 예측으로 옮겨진다. **Pandey**(2604.19117)는 12개 모델에서 "이 진술은 틀렸다" 신호를 나르는 같은 헤드 집합이 아첨과 거짓말을 모두 구동한다는 것을 경로 패칭으로 보였다. Gemma-2-2B에서 이 헤드를 침묵시키면 아첨이 28%에서 81%로 바뀌지만 사실 정확도는 69%에서 70%로 거의 그대로였다. DPO와 RLHF는 행동을 바꾸되 회로를 남겼다 ([Pandey arXiv](https://arxiv.org/pdf/2604.19117)). 행동 수준에서는 **SMART**가 반-아첨 SFT 이후 타당한 사용자 수정의 **27.4–46.7%**만 수용된다고 보고하고, 이를 "판별 능력 향상이 아니라 고집 증가"로 해석했다 ([SMART arXiv](https://arxiv.org/pdf/2509.16742)). 의료 임상 분류에서도 비슷한 결과가 있다. 프로브가 위험을 AUROC 98.2%로 분리하는데도 개념 조향은 놓친 위험의 20%를 고치면서 올바른 탐지의 **53%**를 망가뜨렸다. 이는 무작위 섭동과 구별되지 않았다 ([2603.18353](https://arxiv.org/pdf/2603.18353)).

기권 학습 문헌은 *학습이 만드는* 과잉기권의 원인을 제시한다. **CRaFT**는 과잉거부의 두 원인을 든다. 특징 공간에서 가까운 샘플이 서로 다른 감독("답"과 "모름")을 받는 정적 충돌, 그리고 SFT 중 지식이 변했는데 라벨은 그대로 남는 동적 충돌이다 ([CRaFT arXiv](https://arxiv.org/pdf/2410.06913)). 다만 그 근거가 직접적인 표상 유사도 분석인지, 필터링 개입의 성능 향상인지는 스니펫으로 확인되지 않았다. **Ling et al.**은 프롬프트에 "Unknown" 선택지를 넣으면 풀 수 있는 문제에서도 기권이 늘고, 이를 *무관한 무작위 단어*로 바꿔도 같은 효과가 난다는 것을 보였다. 저자들은 모델이 진짜 불확실성이 아니라 기권의 표면 패턴을 모방한다고 결론 내렸다 ([2507.16199](https://arxiv.org/html/2507.16199v9)). **Ferrando et al.**은 SAE의 "미지 개체" 방향으로 조향하면 *알려진* 개체에 대한 질문까지 거부하게 만들 수 있음을 인과적으로 보였다 ([Ferrando et al. arXiv](https://arxiv.org/pdf/2411.14257)). 판단 이후 응답 쪽에서 신호가 버려진다는 증거도 여러 곳에서 나온다. Slobodkin et al.은 환각 답변 중에도 은닉상태가 답변 가능성을 선형 분리한다는 것을 보였다 ([Slobodkin et al. arXiv](https://arxiv.org/pdf/2310.11877)). AbstentionBench에서는 추론 미세조정이 기권을 평균 24% 떨어뜨리고, 추론 흔적에 불확실성이 표현되어도 최종 답은 단정적이었다 ([AbstentionBench](https://arxiv.org/pdf/2506.09038)).

마지막으로 *기준 대 판별력* 축이 있다. **Prior Audit-Repair**(2608.16003)는 맥락 조작이 검증기의 오경보를 15/15 조합에서 낮추는데, 신호탐지 분석상 기준은 15개 조합에서 이동하고 보정 후 13개에서 유지된 반면 d′ 변화는 어느 조합에서도 유지되지 않았다 ([2608.16003](https://arxiv.org/abs/2608.16003)). **VerifySteer**(2605.20745)는 검증기의 엄격도가 특정 잠재 신호로 부호화되고, 균일 조향은 오류 탐지와 정답 인증을 맞바꾼다는 것을 보였다 ([VerifySteer](https://arxiv.org/html/2605.20745v1)). Two Axes의 "임계값만 이동" 진술을 행동 수준에서 엄밀하게 검정하는 방법론적 틀이 여기에 있다.

## API만으로 검증 가능한 후보와 오픈 모델이 필요한 후보는 갈린다

거짓 전제 프로젝트가 빌려 올 원인 후보를 검증 경로별로 정리하면 다음과 같다. "API 행동"은 블랙박스 출력만으로 가능한 실험이고(logprob이 있으면 일부 확장 가능), "오픈 모델 내부"는 은닉상태 접근, 조향, 패칭, 학습이 필요한 실험이다. 행동 실험은 기제를 *지지하거나 기각하는 예측*을 시험할 뿐이다. 내부 실험만이 위치를 넘어 기제를 확정한다.

| 후보 기제 | 인접 분야 근거 (증거 유형) | 거짓 전제 과제에서의 예측 | API 행동 실험 | 오픈 모델 내부 실험 |
|---|---|---|---|---|
| 판단-응답 분리 | Zhao 2507.11878 (조향), Pandey 2604.19117 (패칭), Knowing-but-Not-Correcting (주의·은닉상태) | 전제 진위 판단은 FPQ와 NFP 모두에서 읽히고, 과잉교정은 응답 선택 단계에서 생긴다 | 가능 (대리 지표). 같은 전제를 단독으로 "X는 참인가?" 물었을 때 참이라 답하면서 질문 속에서는 반박하는지 비교. (QA)²의 단독 검증 약 72% 대 질문 내 탐지 약 50–60% 격차가 선례 | 필요. 판단 방향과 반박 방향을 따로 추출하고 각각 조향·패칭해 인과 역할 분리 |
| 기준 이동 대 판별력 (SDT) | Prior Audit-Repair 2608.16003 (행동 SDT), VerifySteer 2605.20745 (잠재 신호) | 점검 프롬프트는 c를 움직이고 d′는 거의 바꾸지 않는다 | **완전 가능.** FPQ+NFP 쌍에서 프롬프트 조건별 적중률·오경보율로 d′와 c 계산 | 엄밀도 잠재 신호를 찾고 표본별 라우팅 |
| 표면 단서 의존 | OverKill (주의 분석), Ling 2507.16199 (섭동), XSTest, NASA (헤드) | 신화 관련 어휘가 있는 NFP 문항에서 오탐이 많고, 강조 프롬프트가 이를 키운다 | **가능.** 진위를 고정한 채 어휘 치환, 강조 강도 조절, 무작위 단어 대조. Self-CD식 대비는 logprob 필요 | 단서 토큰에 대한 주의 헤드 국소화, NASA식 헤드 표적 미세조정 |
| 분리된 표상, 비특이적 개입 방향 | Dual-Stance 2606.11205, 2603.18353, 2605.05715 (조향+기하) | 교정 조향이 정상 전제 수용도 깎고, 일관성이 낮은 항목이 가장 크게 손상된다 | 부분 가능. 항목별 자기 일관성(양방향 질문)을 재고, 프롬프트 유발 오탐이 저일관성 항목에 몰리는지 확인 | 필요. 프로브 분리 가능성과 조향 방향 투영을 비교 |
| 공유 기질 (겹치는 뉴런·헤드, 양의 코사인) | 2608.26511 (귀인+코사인+조향), Pandey | "전제 교정" 방향과 "사용자 정보 수용" 방향의 코사인이 양수이고, 직교화 효과는 제한적 | 불가 | 필요. 귀인 중첩과 방향 코사인 측정, 직교화 조향 |
| 과제 의존 과잉발화 부분공간 | Maskey 2603.27518, SafeConstellations (기하+조건부 조향) | NFP 오탐이 질문 유형별로 이질적이어서 전역 방향으로는 고쳐지지 않는다 | 기술 통계로만 가능 (유형별 오탐 분포) | 필요. 유형별 궤적·부분공간, 조건부 조향 |
| 학습 유도 (라벨 충돌, 고집, 기본값 회피) | CRaFT, SMART, Kang 2403.05612, Zhou 2509.01476 (R-Tuning이 과잉거부 확대) | FPQ 위주 SFT/DPO는 판별이 아닌 "반박" 성향을 학습한다 | 파인튜닝 API가 있으면 행동 수준까지 가능 | 학습 전후 프로브로 판단 보존·응답 이동 확인 |
| 분해가 주장을 바꿈 | 검증한 문헌 없음 (Well Actually 파이프라인이 자연스러운 검증 지점) | 추출된 하위 전제가 원래 질문의 전제보다 강하거나 축자적이라 "참 질문 → 거짓 하위 주장"이 된다 | **완전 가능.** 추출 전제와 사람이 쓴 정답 전제를 비교하고, 원 질문 맥락 유무에 따라 검증 결과가 달라지는지 확인 | 불필요 |

출처: Zhao ([arXiv](https://arxiv.org/abs/2507.11878)), Pandey ([arXiv](https://arxiv.org/pdf/2604.19117)), Knowing but Not Correcting ([arXiv](https://arxiv.org/html/2605.05957v2)), Prior Audit-Repair ([arXiv](https://arxiv.org/abs/2608.16003)), VerifySteer ([arXiv](https://arxiv.org/html/2605.20745v1)), OverKill ([arXiv](https://arxiv.org/abs/2401.17633)), Ling ([arXiv](https://arxiv.org/html/2507.16199v9)), NASA ([arXiv](https://arxiv.org/abs/2408.00137)), Dual-Stance ([arXiv](https://arxiv.org/html/2606.11205v1)), Decodable but Not Corrected ([arXiv](https://arxiv.org/abs/2605.05715)), 2608.26511 ([arXiv](https://arxiv.org/abs/2608.26511)), Maskey ([arXiv](https://arxiv.org/pdf/2603.27518)), CRaFT ([arXiv](https://arxiv.org/pdf/2410.06913)), SMART ([ACL](https://aclanthology.org/2025.emnlp-main.661/)), Kang ([arXiv](https://arxiv.org/pdf/2403.05612)), Zhou ([arXiv](https://arxiv.org/pdf/2509.01476)), (QA)² ([ACL PDF](https://aclanthology.org/2023.acl-long.472.pdf)).

API만 쓰는 프로젝트라면 세 실험이 우선이다. SDT 분해(기준 이동인지 판별력 손실인지), 단독 검증 대 질문 내 반박의 불일치(판단-응답 분리의 행동 대리 지표), 어휘 섭동과 무작위 단어 대조(표면 단서)다. 이 셋은 Two Axes와 Well Actually의 위치 진술을 행동 수준에서 기각하거나 지지하는 최소 집합이다. 분해 가설은 어느 문헌도 검증하지 않았으므로 API만으로도 새로운 기여가 될 수 있다. 위치를 넘어 기제를 주장하려면 오픈 모델에서 판단 방향과 반박 방향을 분리 추출하고 인과적으로 조작해야 한다. Zhao et al.과 Pandey의 설계를 Cancer-Myth와 Cancer-Myth-NFP에 그대로 옮기는 것이 가장 직접적인 경로다.

## 전문 대조 없이는 인용하면 안 되는 수치

아래 수치는 모두 검색 스니펫이나 요약에서 왔다. 프로젝트 논증에 직접 쓰이는 항목을 위에 두었다.

| 수치 | 출처 | 확인할 점 |
|---|---|---|
| Cancer-Myth 정확도 80%, NFP 41% 오탐, 타 의료 벤치 상대 10% 하락 | Cancer-Myth ICLR 2026 ([OpenReview](https://openreview.net/forum?id=fOXLhZIaUj)) | 검색 요약에서 나왔고 v1 초록에는 없음. 모델, 프롬프트, 벤치마크 확인 |
| "참 전제의 절반 이상 기각", 현실 FPQ 비율 약 13% 가중 결과, 4개 벤치·5개 모델군 | Well Actually ([HTML](https://arxiv.org/html/2608.06539)) | 데이터셋·모델별 수치, **의료 벤치가 Cancer-Myth인지**(미확인), 거짓 판정과 검증 불가 판정의 분리 여부, RAG가 TPQ 손실을 줄이는지 |
| 건전한 질문 57% 반박, 거짓 전제 74% 반박, 프로브 AUROC 0.69–0.77, 자기평가 최대 0.67, 정밀도 약 3배, 커버리지 0.75 대 0.31 | Two Axes ([PDF](https://arxiv.org/pdf/2607.08456)) | 74%는 단일 요약에서만 확인. 57%의 출처를 다른 논문으로 본 요약도 있었음. 라우팅 후 절대 오탐률, 프롬프트 문구 |
| 전제 점검 시 FP +23.4%p/+6.4%p (strict), +22.6%p/+11.3%p (relaxed). valid 하락 크기 미상 | FreshLLMs ([ACL Findings](https://aclanthology.org/2024.findings-acl.813.pdf)) | **valid-premise 하락 폭**. "GPT-3.5에서 유의하다"는 미검증 |
| FPQ 86.7%, 일반 질문 반박 1.4% (재생 적용 후) | FalseQA ([ACL](https://aclanthology.org/2023.acl-long.309.pdf)) | **재생 없을 때의 반박률**, 모델 크기 |
| "낮은 FPR 유지" | Cheng et al. ([PDF](https://arxiv.org/pdf/2601.04435)) | 개입·모델별 FPR. 150문항이라 신뢰구간이 넓음 |
| CDS 0→58.2% (78/134), DPA 32.8%, 무작위 방향 약 10% 대 50%, 억제율 19–90% | Knowing but Not Correcting ([HTML](https://arxiv.org/html/2605.05957v2)) | **참 진술 대조군 존재 여부와 과잉교정률** |
| 대조셋 준수율 변화, 범주별 과잉거부, GPT-4 등 준수율 최대 30% | CoCoNot ([NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/58e79894267cf72c66202228ad9c6057-Paper-Datasets_and_Benchmarks_Track.pdf)) | SFT/LoRA/DPO별 수치, 거짓 전제 범주 |
| 조향 방향 코사인 +0.40~+0.84, 선택적 설정 5→10/36 | 2608.26511 ([arXiv](https://arxiv.org/abs/2608.26511)) | 트레이드오프 크기, "선택적 설정" 정의, 36개 설정 구성, 중첩 통계 방식 (기준선 비율만 저장소에서 직접 확인) |
| 아첨 89% 대 사실 동의 14% 감소 | Dual-Stance ([arXiv](https://arxiv.org/abs/2606.11205)) | 조향 후 절대 동의율, 층·강도, 단일 모델 한계 |
| SFT 후 타당 입력 수용 27.4–46.7% (다른 요약은 약 35–46%), SMART 약 72–79% | SMART ([arXiv](https://arxiv.org/pdf/2509.16742)) | 두 범위의 불일치, "고집" 해석이 원문인지 요약자의 의역인지 |
| (QA)² 56/64/72%, 단독 검증 약 72% 대 질문 내 탐지 약 50–60% | (QA)² ([ACL](https://aclanthology.org/2023.acl-long.472.pdf)) | 수치 귀속. Llama-2 편향은 Syn-(QA)² 결과일 가능성이 높음 |
| CREPE 최고 macro-F1 약 67.1, 인간 약 71 (또는 "인간보다 10% 낮음") | CREPE ([ACL](https://aclanthology.org/2023.acl-long.583.pdf)) | 두 요약이 충돌. 클래스별 정밀도·재현율 |
| RPCBench 깨끗한 질의 FPR 0.55% (24/4,400) | RPCBench ([arXiv](https://arxiv.org/pdf/2609.00918)) | 여러 출처가 섞인 요약에서 귀속 |
| 프로브 AUROC 98.2%, 출력 민감도 45.1%, 조향 교정 20% / 손상 53% | 2603.18353 ([arXiv](https://arxiv.org/pdf/2603.18353)) | 조건별 수치 |
| SafeConstellations −73%, Self-CD −20%, Pandey 28→81% / 69→70%, DPO 46–93% | 각 논문 링크 (위 표) | 벤치마크와 조건 |
| FAITH 46.53→77.74%, 15/20개 헤드 | FAITH ([ACL](https://aclanthology.org/2024.emnlp-main.155.pdf)) | 참 전제 대조군 유무 |

## 결론

이 문헌의 현재 상태는 "트레이드오프는 반복 확인되었고, 원인은 위치까지만 좁혀졌다"로 요약된다. 위치 증거 두 가지(검증 단계, 내부 판별과 출력의 해리)는 서로 다른 설명 층위를 가리키지만 같은 질문으로 수렴한다. 판별 정보가 어딘가에 있는데 반박 결정이 그것을 읽지 않는 이유가 무엇인가. 인접 분야는 이 질문에 판단-응답 분리, 공유 기질, 비특이적 방향, 표면 단서라는 구체적이고 서로 경쟁하는 답을 내놓았다. 이 답들은 거짓 전제 과제에서 서로 다른 예측을 낳는다. 따라서 Cancer-Myth와 Cancer-Myth-NFP처럼 짝지은 평가셋 위에서 후보들을 *구별하는* 실험을 설계하는 것 자체가 이 분야의 공백을 메우는 기여가 된다.

실무적 함의도 있다. FPQ 세트만으로 프롬프트나 모델을 최적화하면 오탐이 늘어나는 것은 부작용이라기보다 목적함수의 예측 가능한 산물이다. 그러므로 어떤 개입이든 d′와 c를 분리해 보고해야 "교정 능력 향상"과 "반박 성향 증가"를 구별할 수 있다. 또 분해형 검증 파이프라인에서 "검증 불가"를 "거짓"과 같은 범주로 처리하는 결정 규칙은 기제 규명 이전에도 바로 점검할 수 있는 설계 결함 후보다. 다만 이 결론은 스니펫 수준 근거에 기대고 있다. Well Actually와 Two Axes의 전문이 위치를 넘어서는 분석을 담고 있는지 확인하는 것이 다음 단계의 첫 작업이어야 한다.
