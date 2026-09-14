# L21 gate 첫 결과와 후속 오프라인 진단 — 2026-09-14

## 출처와 범위

아래 숫자는 사용자가 공유한 서버 콘솔 보고서에서 전사했다. 이 문서를 작성한 로컬 환경에는
서버의 원본 feature/score ledger가 없다. 새 진단의 실제 결과는 아직 받지 않았다.
새 스크립트 검증에 사용한 합성 데이터는 연구 결과가 아니다.

서버 원본: `$PILOT_DIR/gate_L21_v1/report.md` 및 `report.json`.
Gemma-2-9B-it, L21 마지막 prefill 토큰, gate train 292 / calibration 146 / dev 147.
FPQ 117은 기존 Sol, NFP 30은 Terra v2 참조 전제 대상 루브릭을 사용한다.
동일 답변 donor 공유로 바뀐 점수는 0개, NFP 근거 형식 flag는 1개다.
주석의 모호함, judge 검증 규모, dev 반복 관찰 한계는 그대로 남는다.

이번 실행은 **hidden/text classifier → FP Identification 답변 선택**이다.
선택되지 않으면 Plain 답변을 쓴다. ON 시 FP Identification 자체의 Yes/No 단계도 포함한다.
저장된 최종 답변을 재조합했으며 activation steering이나 새 답변 생성을 실행하지 않았다.
Two Axes와 같은 probe 기반 프롬프트 라우팅 계열을 의료 데이터에 적용한 파일럿이며,
원 논문의 데이터·모델·프롬프트·평가를 그대로 재현했다는 의미는 아니다.
문서에서 표 번호가 바뀌어 왔으므로 “Table 3 실행 완료”보다 **hidden gate × FP Identification**으로 식별한다.

## 관측된 결과

| 신호 | dev AUROC | dev TPR | dev FPR | 총 선택 | calibration FPR |
|---|---:|---:|---:|---:|---:|
| hidden | 0.638 | 6/117 | 1/30 | 7/147 | 1/30 |
| TF-IDF text | 0.766 | 19/117 | 1/30 | 20/147 | 1/30 |

| 방법 | PCR % | PCS | NFP pass % | FPQ rescue/harm | NFP rescue/harm |
|---|---:|---:|---:|---:|---:|
| Plain | 6.0 | −0.718 | 100.0 | 0/0 | 0/0 |
| 항상 FP Identification | 59.0 | 0.410 | 60.0 | 64/2 | 0/12 |
| Premise CoT | 34.2 | −0.162 | 100.0 | 34/1 | 0/0 |
| hidden gate × FP Identification | 11.1 | −0.615 | 100.0 | 6/0 | 0/0 |
| text gate × FP Identification | 16.2 | −0.538 | 96.7 | 13/1 | 0/1 |

7/147 고정 무작위 선택 500회: PCR 평균 8.6, draw 5–95% 범위 6.8–10.3;
NFP 평균 98.1, 범위 93.3–100.0. 이 범위는 신뢰구간이 아니다.

## 해석을 바로잡을 부분

### 1. Oracle은 두 종류이며 개입 자체가 완벽하다는 증거가 아니다

현재 pass를 `score == +1`로 정의하면, 요약표에서 다음 두 행을 산술적으로 도출할 수 있다.
스크립트는 원본 문항별 점수로 이를 다시 확인한다.

| 비배포 진단 | 선택 규칙 | FPQ 성공 | PCR % | NFP pass % |
|---|---|---:|---:|---:|
| FPQ-label oracle | FPQ에 FP ID, NFP에 Plain | 7 + 64 − 2 = 69/117 | 59.0 | 100.0 |
| Observed-benefit oracle | Plain이 실패하고 FP ID가 +1인 문항에만 FP ID | 7 + 64 = 71/117 | 60.7 | 100.0 |

두 번째는 **저장된 두 답변과 현재 pass 지표 아래에서만** 문항별 좋은 답변을 고르는 상한이다.
의학적 품질 전반, 새로운 생성, 새로운 개입 방식의 상한이 아니다. 첫 번째 oracle은 FPQ 내 harm을
피하지 못하므로 이 두 답변 사이에서도 최적 선택 상한이 아니다.

FPQ-label oracle이 CoT의 PCR을 넘는다는 것은 **현재 답변 후보들에 활용 가능한 이득이 있다**는 뜻이다.
“선택만 문제이며 개입은 문제없다”는 결론은 안 된다. FP ID는 FPQ 48개에서 +1에 못 미치며,
Plain이 맞힌 FPQ 2개를 망치고, NFP에 적용하면 12개를 부당하게 교정한다.
완벽한 gate는 이런 개입에 노출되는 것을 줄인다. 개입의 오류 자체가 없어지는 것은 아니다.

### 2. 구제, 순이득, 답변 성공은 다른 라벨이다

`Δ_i = 1[FP ID_i = +1] − 1[Plain_i = +1]`로 정의한다.
`Δ=+1`은 rescue, `Δ=−1`은 harm, `Δ=0`은 pass 여부 변화 없음이다.
이는 −1에서 0으로의 개선 등 PCS 변화까지 담는 효용은 아니다.

- hidden: 선택 FPQ 중 rescue **6/6**, 전체 선택 중 rescue **6/7**.
- text: 선택 FPQ 중 rescue **13/19**, harm **1/19**, 순이득 **12/19**.
  전체 선택에서는 rescue 13/20, harm 2/20, 순이득 11/20이다.
- 항상 FP ID: FPQ rescue **64/117**, harm **2/117**, 순이득 **62/117**.

따라서 `12/19`를 구제율이라고 쓰지 않는다. `FP ID가 성공하는가`도 `개입으로 새로 구제되는가`와 다르다.
새 보고서는 원 FPQ/NFP AUROC와 함께 FPQ 내 FP ID 성공, FPQ 내 rescue,
Plain 실패 FPQ 내 rescue의 AUROC를 따로 낸다. 마지막 두 라벨은 기존 점수로 만드는
**사후 진단**이며, 이 라벨로 새 probe를 학습하거나 test 성능을 주장하지 않는다.
전체 dev에서 harm을 예측하는 AUROC도 방향을 명시해 별도 표시한다.

E1의 0.82–0.88은 다른 모델·분할·readout에서 **Plain의 PCR 성공 여부**를 예측한 결과다
([원 기록](../experiments/01-e1-prediagnostic.md)). FP ID 개입의 순이득 예측 성능으로 전용할 수 없다.
과거 전제 구간 AUROC 역시 정렬·입력 가용성 문제를 가진 별도 설정이다.

### 3. 실제 곡선으로 확인할 수 있으므로 선택된 NFP의 피해율을 추정하지 않는다

Plain의 성공은 7/117, CoT는 40/117이므로 **동률에는 순구제 33개, 초과에는 34개**가 필요하다.
선택된 NFP의 harm 비율이 전체 NFP에서의 40%와 같다는 보장은 없다.
“NFP 8–10개를 열면 3–4개가 망가질 것”은 결과가 아니라 독립성에 가까운 가정이다.
저장된 문항별 점수로 정확한 dev 곡선을 계산할 수 있으므로 이 가정을 사용하지 않는다.

새 코드는 두 비교를 구분한다.

1. **같은 k개 선택 곡선:** hidden/text 점수 내림차순으로 dev top-k를 선택한다.
   점수 동점은 문항 ID 오름차순으로 처리한다. 같은 k의 uniform random 정확한 기대값을 함께 낸다.
   이는 관찰한 dev 묶음의 사후 순위 진단이며 배포용 고정 문턱과 다르다.
   k=0은 Plain, k=147은 항상 FP ID다. PCR–NFP와 k–PCR, k–NFP를 함께 출력한다.
2. **calibration 문턱 곡선:** 미리 적은 calibration FPR 목표
   `{0, .01, .05, .10, .20, .30, .50, .75}`마다 기존 strict `score > threshold` 규칙을 적용한다.
   dev에서 문턱을 고르지 않고 각 문턱의 dev 결과를 전부 보고한다.
   음성 30개이므로 서로 다른 목표가 같은 문턱이 될 수 있다.

CoT보다 PCR이 높고 NFP가 낮지 않은 **dev 곡선의 k 전체**를 표시하되 최적 k를 선택·확정하지 않는다.
여기서 유망한 결과가 나와도 별도의 동결 절차와 독립 평가가 필요하다.
test에는 현재 답변/점수가 준비됐다는 확인이 없으므로 “test까지 GPU/GPT 없이 완료”라고 하지 않는다.

### 4. 정규화 확인의 목적과 한계

차원 3,584, 학습 292개, C=1이라는 사실만으로 과적합을 확정하지 않는다.
선택 사항인 CPU 진단은 C `{.001, .01, .1, 1, 10}`을 **train 292개 안의 grouped 3-fold CV**로 비교한다.
매 fold의 StandardScaler도 fold train으로만 학습한다. 평균 AUROC 최대 C를 선택하며 동률이면 작은 C다.
선택한 C로 train 전체를 다시 학습하고, 기존 calibration으로 5% 문턱을 정한 뒤 dev를 보고한다.
이번 체크의 C 선택 기준은 탐지 AUROC이며 낮은 FPR에서의 효용을 직접 최적화하는 새 방법은 아니다.
calibration에서 C와 문턱을 모두 반복 선택하는 것보다 역할을 분리한 절차다.
처음 dev 결과를 본 뒤 추가한 탐색인 점은 바뀌지 않는다.

## 실행 — GPU/GPT 0회, test 미사용

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix

python scripts/gate_diagnostics.py \
  --pilot-dir "$PILOT_DIR" \
  --gate-dir "$PILOT_DIR/gate_L21_v1" \
  --nfp-dir "$PILOT_DIR/nfp_baselines_terra_v2_once" \
  --out-dir "$PILOT_DIR/gate_L21_diagnostics_v1" \
  --regularization-check --plot

cat "$PILOT_DIR/gate_L21_diagnostics_v1/report.md"
```

`--regularization-check`는 작은 logistic classifier를 CPU로 다시 학습한다(5 C × 3 folds + 최종 1회).
Gemma 추출·답변 생성·GPT 판정은 0회다. 이 옵션을 빼면 저장된 점수의 재집계만 한다.
`--plot`는 프로젝트 의존성 matplotlib으로 PNG/PDF를 만든다. 항상 CSV와 JSON도 저장한다.
옵션을 바꿔 재실행할 때는 별도 out-dir을 사용한다. 같은 설정 재실행은 원본을 바꾸지 않는다.
기존 `pilot_gate.py`와 core generation 파일은 수정하지 않아 이전 cache/generation identity를 보존한다.

검증: 합성 signed artifacts의 재조합, 원본 변조 거부, 동일 답변 donor 공유,
구제/피해/순이득 구분, 곡선 끝점·동점·무작위 기대값, dev feature를 변경해도
C 선택·학습 파라미터·문턱이 같음을 확인했다. 실제 추가 AUROC/곡선/C 결과는 서버 실행 후 기록한다.

## 현재 주장할 수 있는 범위

현재 고정된 L21 gate × FP ID는 CoT보다 좋은 교정·보존 결과를 보이지 않았다.
hidden이 text보다 높은 탐지 AUROC를 보인 것도 아니다. 다만 표본이 작고 설정이 하나이므로
내부 방법 전체에 대한 기각이나 모든 steering 방법의 실패로 확대하지 않는다.

“S2가 입증됐다”거나 “Findings 수준으로 성립한다”도 현재 결과로 판단하지 않는다.
22의 모의 패널은 여러 결과를 가정한 시나리오 평가였다. 투고 가치에는 기존 연구 대비
새 발견, 평가 신뢰성, 독립 검증, 일반성 등의 실제 증거가 더 필요하다.
이번 후속 분석의 목적은 **이 조합을 더 볼 이유가 있는지, 병목이 선택과 개입 각각에 얼마나 있는지**를
현재 답변 후보와 지표의 범위 안에서 확인하는 것이다.
