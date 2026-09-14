# C 방향 overnight 진단 — 2026-09-14

## 목적과 현재 위치

Quick45에서 기존 L21 C, α=0.1의 PCR은 3/30이고 Plain은 4/30이었다.
이 한 설정에서 뚜렷한 개선을 얻지 못했다는 뜻이다. 답변이 달라졌다는 사실만으로
벡터가 순수한 문체 방향이거나 교정 방향이 아니라고 확정하지 않는다.

다음 질문을 한 번의 GPU 실행으로 확인한다.

1. 기존 C가 교정 참조 답변의 상대적 확률을 바꾸는가? 강도·부호·층·위치에 따라 다른가?
2. 같은 norm의 무작위 방향과 구별되는가? 실제 residual 변경 크기는 얼마인가?
3. 첫 32토큰 대신 응답 전체를 쓰거나, 동일한 Gemma의 지시 조건 대조로 C를 만들면 다른가?
4. 그 설정에서 실제로 어떤 답변이 생성되는가?

**원격 서버 GPU 0·1만 사용한다. 최대 9시간의 누적 실행 예산이며 모두 끝나면 일찍 종료한다.**
이는 9시간 안에 모든 칸의 완료를 보장하는 ETA가 아니다. 문항 길이·GPU 속도·오류에 따라
남은 작업이 있을 수 있고, 완료·실패·시간 초과·의존성 실패·미실행을 모두 보고한다.
종료 시 프로세스 정리와 CPU 보고서 작성에 짧은 추가 시간이 들 수 있다.

새 GPT/Codex 판정 호출은 **0회**다. 기존 Quick45 평가가 완료되어 있어야 하며,
저장된 Plain 판정은 투영 분석의 기술 통계에만 재사용한다. 새로운 생성 답변에는 PCR을 붙이지 않는다.

## 서버에서 한 번에 시작

```bash
cd /home/eagle0914/cancer-myth-internals
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate

export DATA_ROOT=/data1/heejae
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export QUICK_DIR="$PILOT_DIR/quick45_original_terra_v1"
export OVERNIGHT_DIR="$PILOT_DIR/overnight_c_v1"

bash scripts/run_overnight_c.sh start
```

`Started coordinator PID ...`가 나오면 SSH 연결을 닫아도 된다. `nohup`으로 분리하며
GPU마다 Gemma 한 개를 올리고 작업을 나눠 처리한다. 각 프로세스는 물리 GPU 하나만 보며
그 안에서는 `cuda:0`으로 표시된다. 두 번째 프로세스의 `cuda:0`은 물리 GPU 1이다.
두 카드 모두 모델을 올릴 여유 메모리가 필요하다. 서버에서 실제 GPU 실행 시간·결과는 이 코드의 로컬 테스트와 별개다.

기존 `fit/fit.json`, `fit/directions.npz`, `split/manifest.json`과 Quick45 파일을 서버에서
직접 읽는다. 따로 내려받아 전달할 필요가 없다. 모델·참조 데이터·원본 답변·판정 대응과
코드 hash를 확인하고 계획을 고정한다. 실행 중에는 이 저장소 코드를 수정하거나 pull하지 않는다.

확인·중지는 다음 명령이다. 동일한 환경변수 또는 기본 경로를 사용한다.

```bash
bash scripts/run_overnight_c.sh status
tail -n 40 "$OVERNIGHT_DIR/coordinator.log"
# 중지가 필요할 때만:
bash scripts/run_overnight_c.sh stop
```

아침에는 다음 명령만 실행한다. GPU 추론이나 채점은 시작하지 않는다.

```bash
bash scripts/run_overnight_c.sh report
cat "$OVERNIGHT_DIR/report.md"
```

## 고정한 작업 목록

아래는 실행 우선순위다. 두 worker가 다음 실행 가능한 작업을 가져가므로 일부 단계는 겹친다.
참조 쌍이 부족하면 가능한 수만 쓰고 계획에 ID·수를 남긴다. 기본값은 dev 참조 쌍 최대 24개,
자기 모델 답변을 생성할 fit FPQ 최대 64개다.

| 순서 | 실험 | 기본 설정 | 비교할 내용 |
|---|---|---|---|
| 1 | 기존 C 용량–반응 | L21, α=0, 0.1, −0.1, 0.3, 0.6, 1.0, −0.3; dev 최대 24쌍 | 교정/순응 참조 각각의 log probability와 차이 |
| 2 | 기존 참조 쌍 재추출 | 원래 fit 쌍, 가능한 L14·21·28; 첫 32토큰/전체 | 원 벡터와 재추출 벡터의 일치, pooling 변화 |
| 3 | Gemma 자체 답변 생성 | fit FPQ 최대 64개 × Plain/교정 지시 | 같은 모델·같은 질문에서 지시만 다른 대조쌍 확보 |
| 4 | 자기 모델 C 추출 | 3에서 완료된 서로 다른 답변 쌍; 첫 32토큰/전체 | 외부 답변 저자 차이가 없는 지시 대조 방향 |
| 5 | 무작위 방향 대조 | seed 17·29·43, α=0/0.1/0.3/0.6; dev 최대 12쌍 | 동일 unit norm·동일 fit residual scale에서 비교 |
| 6 | 층 비교 | 원래 fit에 있는 L14·L28, α=0/−0.1/0.1/0.3/0.6; 12쌍 | 새로운 probe AUROC 탐색 없이 개입 층 비교 |
| 7 | 위치·축 제거 | L21, 마지막 prompt token/첫 32 predictor 위치, α=0.1/0.3/0.6; 전체 축 제거; 12쌍 | 생성 전 계획 위치와 생성 중 개입의 차이 |
| 8 | 자기 답변 투영 | Quick45 Plain FPQ 최대 30개, L21 C에 첫 32토큰/전체 투영 | 저장된 +1과 −1의 분리도; 0은 AUROC에서 제외 |
| 9 | 새 C 용량–반응 | 참조 전체/Gemma 첫32/Gemma 전체, L21, α=0/−0.1/0.1/0.3/0.6/1; 12쌍 | 방향 추출 방식 변화에 따른 확률 곡선 |
| 10 | 생성 답변 비교 | 고정 dev 12 FPQ+6 NFP × 최대 14설정, 답변 512토큰 | Plain·기존 C·새 C·위치·축 제거의 실제 출력 |

모든 층과 표본이 있을 때 최대 1,144개 작업이다. 이는 GPT 호출 수가 아니다.
하나의 확률 작업은 한 질문의 두 참조를 teacher forcing하는 GPU forward 작업이며,
추출 작업은 여러 fit 문항을 처리한다. 두 개의 추출 작업은 문항별 캐시를 남긴다.

자기 모델 대조의 교정 조건은 `FP_CORRECT` 지시를 항상 주는 것이다.
**기존 FP Identification의 Yes/No → 조건부 답변 전체 파이프라인과 같지 않다.**
별도 채점하지 않으므로 “교정 성공 − 순응 실패” 방향으로 부르지 않고
“Gemma 지시 조건 대조 방향”이라고 부른다. 동일한 답변인 쌍은 제외한다.
fit 데이터만 사용하며 dev의 117개 기존 성공/실패 쌍으로 새 C를 학습하지 않는다.

## 개입·확률의 정확한 의미

방향은 unit vector `u`이며 원래 fit에서 얻은 층별 residual norm scale을 `s`로 사용한다.
추가 개입은 `h' = h + α s u`다. 무작위·새 방향도 같은 층의 `s`를 쓴다.
BF16 반올림 뒤 실제 `||h'−h||`, 원래 `||h||`, 상대 변경 크기를 저장한다.
음수 α는 같은 방향의 반대 부호 대조이며, 원래 논문들의 계수가 다른 정규화와 같다고 해석하지 않는다.

teacher forcing에서는 정답 토큰을 예측하는 **직전 위치**에 개입한다.
`all`은 마지막 prompt token부터 응답 predictor 위치 전체,
`prefill`은 마지막 prompt token 하나, `first_k`는 첫 K개 응답 predictor 위치다.
`ablate`는 `h' = h − (h·u)u`로 전체 축을 제거한다. 부호의 양쪽을 제거하므로
“순응 성분만 제거했다”는 뜻이 아니다. 이 조건의 α=0 표기는 비개입이 아니다.

확률 비교는 각 참조의 답변 토큰별 평균 log probability와 그 차이
`mean log p(positive) − mean log p(negative)`다. 길이와 문체의 영향이 남으므로
합계 log probability, 첫 32토큰/이후 구간, 양·음 참조의 개별 변화도 함께 저장한다.
응답은 최대 512토큰, prompt를 포함한 길이가 4096을 넘으면 조용히 질문을 자르지 않고 실패 처리한다.
같은 문항의 α=0과 짝지은 변화만 집계하며, 정상 비개입과 axis 제거를 섞지 않는다.

**곡선 상승은 참조를 더 선호했다는 진단 결과이고 자유 생성의 정확 교정을 증명하지 않는다.**
둘 다 확률이 낮아지면서 음성 참조만 더 낮아지는 경우도 있어 차이 하나만으로 고르지 않는다.
평평한 곡선도 평가한 층·위치·참조 범위 안의 결과이며 steering 전체를 기각하는 근거가 아니다.
새 방향 간 cos은 방향 유사도이지 문체 혼입의 확정 증거가 아니다.
동일 저자의 정상 질문 문체 대조쌍이 현재 준비되지 않아 순수한 `C_style` 분석은 실행하지 않는다.

## 실패·재개 정책

- 작업별 결과 JSON을 원자적으로 저장한다. 개별 오류·OOM은 기록하고 다음 작업으로 넘어간다.
- 확률/투영 작업은 180초, 생성은 240초, 대규모 추출은 5400초를 상한으로 둔다.
  멈춘 worker는 coordinator가 종료하고 GPU별 총 3회 실행까지 재시작한다.
- 한 선행 작업이 실패하면 그 결과에 의존하는 작업은 `skipped`로 남긴다.
  자체 답변 추출은 완료된 서로 다른 쌍이 2개 이상 있으면 부분 결과로 진행하고 수를 보고한다.
- 시간 한도에 도달하면 실행 중 작업은 `timed_out`, 시작하지 못한 작업은 `pending`으로 남긴다.
  그래프와 보고서에는 확보한 결과만 넣고 빠진 칸을 숨기지 않는다.
- 같은 출력 폴더의 중복 coordinator는 파일 잠금으로 막는다. `start`를 다시 실행하면
  완료/실패 기록을 재실행하지 않고 미실행 작업만 남은 누적 시간 안에서 진행한다.
  9시간을 이미 썼다면 GPU를 새로 실행하지 않는다. 실패 작업은 자동 재채점·무한 재시도하지 않는다.
- 비정상적으로 coordinator 자체가 강제 종료되어 worker가 남으면 중복 실행을 거부한다.
  `run_status.json`의 PID와 `logs/`를 확인한다. 일반 `stop`은 해당 coordinator의 worker만 종료한다.

## 아침에 읽을 파일

| 파일 | 내용 |
|---|---|
| `report.md` | 전체 완료 현황, 확률 비교, 투영 AUROC, 방향 유사도, 생성 현황과 실패 |
| `summary.json` | 같은 집계의 기계가 읽는 형태 |
| `curves.svg` | matplotlib을 사용할 수 있을 때 확률 변화 곡선 |
| `generated_examples.md` | dev의 실제 생성 답변과 기존 Plain; 자동 의료 판정 없음 |
| `plan.json`, `reference_pairs.json` | 실행 전 고정한 설정·문항·참조·입력 hash |
| `tasks/*.json` | 작업별 설정·측정 값·실제 개입 크기·오류·실행 시간 |
| `directions/*.npz`, `extract_cache/` | 새 방향과 fit 문항별 추출 캐시 |
| `run_status.json`, `coordinator.log`, `logs/gpu*.log` | 시간 제한, worker 상태, 오류 원문 |

먼저 실패와 실제 표본 수를 확인하고, 다음으로 α=0 대조와 참조 prefix32 재추출이
원래 방향과 일치하는지 본다. 이어서 무작위 대비 차이·양/음 참조 각각의 변화·출력 붕괴를
함께 확인한다. 이후에 소수 후보의 생성 답변을 읽고 필요한 경우에만 별도 채점 계획을 세운다.
이번 결과 전체는 반복 열람한 dev의 탐색 결과이며 test 성능이나 새 방법의 효과로 보고하지 않는다.
