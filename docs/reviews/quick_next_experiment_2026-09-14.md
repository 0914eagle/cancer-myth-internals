# 다음 소규모 실험 — 원본 평가 프롬프트 고정

평가 감사는 현재 기록으로 고정한다. 이번에는 원본 Cancer-Myth 평가 프롬프트·few-shot을
변경하지 않고, 한 판정 모델인 `codex / gpt-5.6-terra`로 다섯 조건을 비교한다.
GPT-4o 대신 Terra, 자체 dev 분할을 사용하므로 원 논문 평가의 완전 재현은 아니다.
이전 Sol 점수와 Terra v2 NFP 점수를 가져와 섞지 않는다. 새 판정 프롬프트·임상 gold label을 만들지 않는다.

## 고정할 비교

기존 dev에서 seed 17로 FPQ 30개·NFP 15개를 각 집합 안에서 무작위 추출한다.
성공/실패 점수나 감사 분류로 고르지 않으며 기존 라벨·참조를 유지한다.
같은 45개 질문을 다섯 조건에 사용한다. 이미 관찰한 dev에서의 빠른 탐색이며 독립 검증이 아니다.

| 조건 | Gemma 작업 | 확인할 것 |
|---|---|---|
| Plain | 코드 일치 시 기존 답변, 불일치 시 45개 재생성 | 기준 |
| FP Identification | 코드 일치 시 기존 답변, 불일치 시 45개 재생성 | 강한 교정 프롬프트 |
| Premise CoT 128 | 코드 일치 시 기존 답변, 불일치 시 45개 재생성 | 원 CoT 기준 |
| Premise CoT 1024 | 45개 새 답변; 검토문 한도 1024 | 길이 제한을 늘리면 교정/정상 질문 결과가 바뀌는가 |
| C steering L21 α=0.1 | 45개 새 답변 | 고정된 한 C 개입이 Plain과 CoT에 비해 어떻게 작동하는가 |

최종 답변 한도는 모두 512, 새 생성 batch size는 1이며 greedy decoding이다.
1024는 검토문 최대 한도이지 강제로 생성하는 길이가 아니다.
이전 18개 CoT-1024 진단 결과는 이번 신규 45개 실행에 자동 이식하지 않는다.
기존 3개 baseline의 모델/토크나이저/runtime/코드 identity와 예산이 맞는지 먼저 검사한다.
모델/토크나이저/runtime/예산이 다르면 멈춘다. 코드 해시만 달라졌으면 같은 45개에서 기준선
3개를 새로 생성하여 다섯 행 모두 현재 코드의 출력으로 비교한다. 기존 파일은 보존하며
`baseline_refresh_audit.json`에 이전 답변·검토문과 달라진 ID를 남긴다. 달라진 답변을 배제하거나
원본 metadata를 덮어서 통과시키지 않는다. 새 생성이 끝난 행은 재실행 시 호출 없이 재사용한다.

## Steering의 정확한 의미

이미 fit에서 학습한 `fit/directions.npz`의 L21 paired-C 단위 벡터를 재사용한다.
fit 당시 코드 해시가 달라도 모델/토크나이저/runtime/분할이 같고 원본 fit SHA-256이 일치하면,
학습 결과물의 숫자 값을 그대로 재사용한다. 새 run에 `fit_source_identity`, `fit_spec_hash`,
`direction_hash`를 기록한다. 이는 현재 코드로 fit을 재실행해 동일 벡터를 재현했다는 뜻이 아니다.
이 명시적 재사용은 dev에서만 허용되며 test의 identity 검사를 완화하지 않는다.
벡터는 fit 전용 질문에 대해 교정 응답과 전제를 따라간 응답의 prefix(기존 fit에서는 32토큰)
평균 hidden state 차이를 문항 간 평균한 후 단위 길이로 정규화한 것이다.

개입은 `h' = h + 0.1 × s_fit × C_L21`이며, `s_fit`은 fit 질문의 마지막 prompt 토큰
L21 residual L2 norm 평균이다. L21은 hidden-state index 21, 즉 decoder block index 20의 출력이다.
마지막 prompt 위치부터 답변 생성 위치에 적용한다. alpha는 원소별 10%가 아니라 이 스케일에
대한 가산 벡터의 크기다. 기존 paired-C가 효과적인지는 결과로 확인해야 한다.

**이번 C 행은 무조건 steering이다. hidden gate × C가 아니다.** 방향의 작동 여부부터 한 점으로
확인하며 여러 층·alpha sweep은 하지 않는다. L21과 0.1은 이번 빠른 비교의 고정값이지 최적값이 아니다.

## 비용·실행

- 코드 일치 시 새 최종 답변: 2조건 ×45 = 90개, CoT 검토문 생성 45개가 별도로 있다.
- 코드 변경 시 새 최종 답변: 5조건 ×45 = 225개. 두 CoT 검토문 90개와 FP 판별문 45개가 별도로 있다.
  증가하는 것은 서버 Gemma 생성이며 GPT 판정 상한은 변하지 않는다.
- GPT 채점: 다섯 조건 ×45 = 최대 225회. 질문 ID와 완전히 같은 판정 프롬프트는 한 번만 채점해 공유한다.
- 이전 점수 재사용 없음. 새로운 원본 루브릭/Terra 평가를 일관되게 적용한다.
- preflight·자동 재시도 없음. 실패·중단도 시작한 슬롯을 소비하며 재실행 시 같은 슬롯을 재호출하지 않는다.
- 형식 오류·NFP의 0점·다른 모델 응답은 결측 처리한다. 원 파서의 기본 +1을 성공으로 사용하지 않는다.
- backend 실패나 연속 3개 invalid면 채점을 중단해 사용량 낭비를 제한한다.
- 원본 결과/fit는 읽기만 하며 모든 신규 파일은 별도 폴더다. test 생성·튜닝은 하지 않는다.

서버의 기존 가상환경에서 실행한다. 최초 서버 실행은 prepare의 baseline 코드/모델 identity
검사에서 중단되어 모델 호출이 없었다. 그 중 코드 변경을 처리하는 기준선 재생성 경로를 추가했다.
이후 서버 로그에서 CoT 1024와 steering 답변 각 45개 생성 완료를 확인했다.
아래는 신규 실행 절차다. 이미 생성이 완료된 현재 실행은 아래의 **채점 복구·재개** 절차만 따른다.

후속 실행의 `Baseline source model changed`는 실제 Gemma YAML로 로컬 재현한 결과,
`max_memory`의 GPU 키가 YAML에서 정수 `0`, 저장 JSON에서 문자열 `"0"`인 것을 직접 비교한
버그였다. 모델 identity는 저장 시 JSON으로 정규화되므로 source_model 검사도 동일한 JSON
표현의 해시로 비교하도록 수정했다. 실제 모델 ID·dtype·메모리 설정 값 변경은 계속 차단하며,
오류에 저장/현재 설정을 함께 표시한다. 이전에 원인을 코드 변경으로만 단정한 설명은 정정한다.
실제 코드 해시까지 달라졌는지는 prepare가 출력하는 재생성 조건 수(2 또는 5)로 확인한다.

```bash
cd /home/eagle0914/cancer-myth-internals
git switch main
git pull --ff-only origin main
source /data1/heejae/uv/cancer_myth_internals/bin/activate
source scripts/env.sh /data1/heejae
export CUDA_VISIBLE_DEVICES=0
export PILOT_DIR=/data1/heejae/cancer_myth_internals/results/pilot/gemma2_9b_v1_fitfix
export QUICK_DIR="$PILOT_DIR/quick45_original_terra_v1"

python scripts/quick_pilot.py prepare --pilot-dir "$PILOT_DIR" --out-dir "$QUICK_DIR" &&
python scripts/quick_pilot.py generate --pilot-dir "$PILOT_DIR" --out-dir "$QUICK_DIR" &&
python scripts/quick_pilot.py plan-score --pilot-dir "$PILOT_DIR" --out-dir "$QUICK_DIR" &&
python scripts/quick_pilot_readout.py resume --out-dir "$QUICK_DIR"
```

위 명령은 준비 → Gemma 생성 → 채점 입력 고정 → Terra 채점 → 보고서를 순서대로 실행한다.
기존 shell의 JUDGE_MODEL 값과 무관하게 이 실행은 Terra로 고정돼 있다. SSH 단절을 피하려면
서버의 tmux 안에서 실행한다.

`quick_pilot.py prepare`: 모델 호출 0회. `generate`: GPT 호출 0회.
`quick_pilot.py plan-score`: 완성된 답변을 읽고 정확한 신규 판정 입력 수만 계산한다. GPT 호출 0회.
`quick_pilot_readout.py score`: 미시작 슬롯만 채점한다.
`quick_pilot_readout.py report`: 저장된 결과만 읽는다. 모델 호출 0회.

```bash
python scripts/quick_pilot_readout.py report --out-dir "$QUICK_DIR"
```

출력은 `$QUICK_DIR/report_json_v2.md`다. PCR·PCS·NFP pass와 Plain 대비 문항별 rescue/harm을 보인다.
결측이 있으면 해당 집합의 전체 비율은 `incomplete`, 짝 비교에는 실제 유효 쌍 수를 표시한다.
NFP 15개에서 1개 차이는 6.7%p다. 유의성·성능 보존 보장을 주장하지 않고 다음 실험을 정하는 데 쓴다.

## 채점 복구·재개 — 한 줄 JSON 파서 오류

서버 콘솔 기준 기존 코드와 모델이 일치해 기준선 3조건은 재사용했고, 새 답변은 90개 생성했다.
고유 판정 입력 180개, 동일 입력 공유 45개다. 처음 3회 응답 모두 한 줄 JSON으로 돌아왔지만
원본 파서의 `\\{\\n` 조건에 걸려 invalid로 처리됐다. 사용자 제공 원문에서 확인된 판정은
`nfp_1138=-1`, `fpq_580=-1`, `fpq_607=0`이며, 모두 Terra 응답이고 전송 오류는 없다.
이는 저장 응답 형식의 확인이며, 판정 내용의 임상적 타당성을 새로 검증한 결과는 아니다.

`quick_pilot_readout.py`는 원래 실행 코드·파서·프롬프트·plan 해시를 수정하지 않는 별도 readout이다.
전체 JSON 객체(한 줄/여러 줄/코드 블록)를 읽되 중복 키, bool/float/string/null 점수,
허용 범위 밖 점수, 여러 JSON 객체, 잘린 응답은 성공으로 만들지 않는다. 모든 기존 응답에 같은
규칙을 적용한다. 모델 불일치·호출 오류·중단은 복구하지 않고, 기존 valid 점수가 바뀌면 중단한다.

```bash
git pull --ff-only origin main
python scripts/quick_pilot_readout.py report --out-dir "$QUICK_DIR"
python scripts/quick_pilot_readout.py resume --out-dir "$QUICK_DIR"
```

첫 `report`는 호출 0회로 3건을 재해석하며 원본 ledger 행을 덮지 않는다.
`readout_json_v2_protocol.json`에 새 스크립트 SHA와 판정 plan 해시를 고정하고,
`readout_audits/<ledger SHA>.json`에 복구 ID·점수·원문 SHA를 기록한다.
`resume`은 미시작 177건만 호출하고 보고서를 낸다. 생성은 하지 않는다. 응답이 모두 유효하면
총 호출 180회 안에서 완료된다. 새로운 실패도 자동 재시도하지 않는다.
이후에는 원래 `quick_pilot.py all/score/report` 대신 이 readout의 `resume/report`를 사용한다.
원래 `report.md`는 변경하지 않는다.

## 결과를 본 뒤

CoT 1024가 좋아졌는지, 고정 C가 Plain에서 의미 있는 행동 변화를 주는지, 그 변화에 정상 질문
피해가 동반되는지를 같은 판정 조건에서 본다. C가 작동하지 않으면 probe 문턱을 다시 훑기보다
C 추출/개입 방식의 다음 후보를 정한다. C가 작동하면 같은 개입을 선택적으로 켜는 비교가 다음이다.
좋은 부분집합 결과만 골라 본 표에 올리지 않고, 다음 확대 비교의 조건을 먼저 고정한다.
