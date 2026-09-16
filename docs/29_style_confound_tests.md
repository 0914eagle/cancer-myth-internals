# 29. 문체 교란 검정: text 게이트의 우위가 출처 문체인지 전제인지

2026-09-16. 코드와 절차만 추가했다. **실행 결과는 아직 없다.** 실행 후 §7 표를 채운다.

## 1. 질문

Qwen 732문항 crossfit(26, `qwen25_7b_final1024_v1/gates/crossfit_v1`)에서 질문 판별 AUROC는
text 0.755 [0.709, 0.799] > hidden 0.693 > mean(DiM) 0.681 > review 0.587 > direct 0.466이었다.
FPQ는 GPT-4o가 통념에서 써낸 질문이고 NFP는 LLM이 오경보를 낸 사용자 질문이라 **출처가 다르다**.
그래서 text가 읽는 것이 전제의 거짓인지 출처의 문체인지 구분되지 않는다. E1에서도 같은 경고가
있었다(15 §"텍스트만으로도 갈린다", TF-IDF 0.771). 이 문서는 그 가설을 세 가지 통제로 판별하는 절차다.

판별하려는 명제: **"text > hidden은 문체 때문이다."** 참이면 text 게이트는 벤치마크 산물이고 배포
게이트로 부를 수 없다. 거짓이면 text는 진짜 강하고 hidden은 보완 신호다. 어느 쪽이든 12 §17의
방법 주장(같은 오탐에서의 상보적 선택)은 유지되고, 바뀌는 것은 어느 신호를 주 게이트로 쓰는가다.

## 2. 세 검정

| # | 검정 | 무엇을 고정하는가 | 비용 | 코드 |
|---|---|---|---|---|
| 1 | **문체 하한선** (`style`, `masked`) | 내용어를 전부 지운 뒤 FPQ vs NFP 분류. 길이·구두점·문장 형태·인칭·기능어 비율만 | CPU 몇 분, 판정 0 | `src/style_features.py`, gate 신호 `style`/`masked` |
| 2 | **의역 통제** (`para`) | 한 writer가 732문항 전부를 같은 평이한 문체로 다시 씀. 믿음은 유지. FPQ는 전제 유지 확인, 전 행 충실성 확인 | LLM 약 732×2–3회(판정기 아님), Qwen prefill 한 번 | `scripts/make_paraphrases.py`, `src/paraphrase.py` |
| 3 | **참 전제 쌍둥이** (`twins`) | 정렬된 전제 구간만 정답으로 바꿔 끼움. 바깥 텍스트는 바이트 동일. FPQ vs 쌍둥이 | 정렬 성공 294/585만 가능(22 §3). 쌍둥이 생성 LLM 2회/문항, Qwen prefill 한 번 | `scripts/make_true_twins.py`(기존, stage 9), `run_e0_twins.sh` |

셋을 한 번에 돌리는 드라이버가 `scripts/run_style_controls.py`다. 조건은 일곱 개:

| 조건 | 학습 | 평가 | 읽는 것 |
|---|---|---|---|
| natural | FPQ vs NFP | 같은 것 (OOF) | 기준 행. crossfit과 같은 수치가 나와야 함 |
| twins | FPQ vs 쌍둥이 | 같은 것 (OOF) | 문체 고정 시 남는 판별력 |
| para | 의역 FPQ vs 의역 NFP | 같은 것 (OOF) | 출처 문체 제거 시 남는 판별력 |
| natural→twins | FPQ vs NFP | FPQ vs 쌍둥이 | 자연 데이터로 배운 게이트가 문체 없이 버티는가 |
| natural→para | FPQ vs NFP | 의역 | 위와 같음, 의역 쪽 |
| para→natural | 의역 | FPQ vs NFP | 문체 없이 배운 게이트가 자연 데이터에서 얼마나 되는가 |
| twins→natural | FPQ vs 쌍둥이 | FPQ vs NFP | 전제 진위만 배운 게이트의 전이 |

신호는 text(TF-IDF 1–2gram), style, masked, hidden(층별 logistic), mean(DiM). 조건마다 신호 ×
fold로 내부 3-fold CV로 C·층을 고르고(crossfit과 같은 규칙), OOF 점수로 fold 가중 AUROC를 낸다.

## 3. 누출 규칙

- fold는 자연 문항 732개에 한 번만 배정한다(`StratifiedGroupKFold`, group_id, seed 17). **쌍둥이와
  의역은 원문의 fold를 물려받는다.** 원문이 학습에 있으면 그 쌍둥이·의역이 평가에 오지 않는다(27 §150).
- 쌍둥이·의역 행의 id는 `<원문id>_true`, `<원문id>_para`로 원문과 겹치지 않고, 특성 캐시도
  `model_variants/<name>/`에 따로 둔다. 같은 id가 두 소스에 있으면 드라이버가 거부한다.
- writer(쌍둥이·의역을 쓴 LLM)는 행마다 기록된다. **판정기는 바뀌지 않는다.** 보고 표의 판정은 여전히
  Terra(26)이며, writer는 질문 텍스트를 만들 뿐 답변 점수에 관여하지 않는다.

## 4. 통계

- 주 통계량은 조건별 **(신호 − text) AUROC 차이**와 짝 지은 group bootstrap CI(같은 재표집으로 두 신호를
  같이 계산; 22 §6의 검정력 계산과 같은 이유로 주변 CI 두 개를 겹쳐 보지 않는다).
- 문체 하한선은 max(style, masked). 각 신호의 "하한선 위 여분"이 natural에서 twins·para로 갈 때 어떻게
  변하는지가 판독 대상이다.
- 쌍둥이는 294개 이하라 반폭이 넓다(150개 정상에서 ±4–7pp였으므로 비슷하거나 약간 좁다). 의역은 732개
  전부라 natural과 같은 정밀도다.

## 5. 판독 기준 (실행 전에 적어 둔다)

| 결과 | 해석 | 논문 서술 |
|---|---|---|
| 하한선(style/masked)이 natural에서 hidden 0.69에 근접 | hidden도 문체를 읽음 | 게이트 전반이 출처를 읽는다고 보고. 방법 A는 매칭 FPR의 보완성으로만 주장 |
| twins·para에서 text 급락, hidden 상대적으로 유지 (delta hidden−text가 양으로 전환) | text 우위는 출처 문체 | text 게이트는 벤치마크 산물. 배포 게이트는 hidden/DiM, text는 상보 신호 |
| twins·para에서 둘 다 비슷하게 하락, 순서 유지 | 둘 다 문체를 일부 읽고, 순서는 진짜 | text가 주 게이트. hidden은 보완 신호(21개 DiM-only 등)로만 |
| twins·para에서 둘 다 유지 | 문체 아님 | text가 진짜 강함. 12 §17의 A(text+DiM 결합) 그대로 |
| natural→para가 para(OOF)보다 크게 낮음 | 자연 게이트가 문체에 과적합 | 배포 시 분포 이동 위험을 한계로 명시 |

쌍둥이에서 text가 0.5까지 떨어지지 않아도 문체 가설이 기각되는 것은 아니다. 쌍둥이도 부정어·단정어
(only/always/cure)가 같이 바뀌고(27 §141), 그 단어들은 style/masked에서 **내용어로 가려진다**. 그래서
절대값이 아니라 hidden−text 차이의 변화를 본다.

## 6. 실행 순서 (실험 세션)

판정 배치(2,965회)와 독립이므로 병렬로 돌린다. 1은 지금, 2·3은 이번 주.

```bash
# 0. 오프라인 검증
python -m pytest -q tests/test_style_controls.py tests/test_baseline_gates.py tests/test_baseline_cli.py

# 1. 문체 하한선: crossfit에 style/masked 행을 더한 새 이름 (기존 crossfit_v1 plan.json은 동결이라 새 이름 필수)
python scripts/run_baseline_suite.py gate --out-dir "$SUITE_DIR" --scheme crossfit --name crossfit_v2_style \
  --signals text style masked hidden mean direct review
#    report.md에 "Matched operating points" 부록 표가 같이 나온다 (§8).

# 2. 의역 (writer는 claude 또는 codex; 판정기 아님)
python scripts/make_paraphrases.py --suite-dir "$SUITE_DIR" --backend claude --limit 20   # 눈으로 확인
python scripts/make_paraphrases.py --suite-dir "$SUITE_DIR" --backend claude              # 732 전부, 재개 가능
cat "$SUITE_DIR/variants/para/para_audit.md"; less "$SUITE_DIR/variants/para/para_sample.md"
CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline_suite.py extract --config configs/qwen25_7b.yaml \
  --out-dir "$SUITE_DIR" --variant-file "$SUITE_DIR/variants/para/questions_para.jsonl" --variant-name para

# 3. 쌍둥이 (E0b 산출물이 있으면 재사용; 없으면 생성)
DATA_ROOT=/data1/heejae JUDGE_BACKEND=claude bash scripts/run_e0_twins.sh     # -> $DATA/e1_rows_v1/questions_twins.jsonl
CUDA_VISIBLE_DEVICES=1 python scripts/run_baseline_suite.py extract --config configs/qwen25_7b.yaml \
  --out-dir "$SUITE_DIR" --variant-file "$DATA/e1_rows_v1/questions_twins.jsonl" --variant-name twins
#    E1 행 id와 suite id 체계가 다르면 --twins-questions로 원문 텍스트 기준 매핑

# 4. 세 검정 한 번에 (CPU)
python scripts/run_style_controls.py --suite-dir "$SUITE_DIR" \
  --twins "$DATA/e1_rows_v1/questions_twins.jsonl" --twins-questions "$DATA/e1_rows_v1/questions.jsonl" \
  --paraphrases "$SUITE_DIR/variants/para/questions_para.jsonl" --name v1
cat "$SUITE_DIR/style_controls/v1/report.md"
```

`extract --variant-*`는 suite의 `model/`을 건드리지 않고 `model_variants/<name>/`에 쓴다. 쌍둥이 특성
없이 드라이버를 돌리면 hidden/mean 행은 그 조건에서 건너뛰고 보고서에 이유가 적힌다. text/style/masked는
언제나 돈다.

**역할 고정 (2026-09-16 사용자 결정).** 판정기 = Terra(`--backend codex`, 26 그대로). writer(쌍둥이·의역) = **Claude Sonnet 5**
(`configs/default.yaml` `judge.claude_model: claude-sonnet-5`; 처음 Opus 5로 정했다가 구독 사용량을 아끼려 같은 날 Sonnet 5로 내림.
writer는 판정에 관여하지 않으므로 타당성에 영향 없음). `--backend claude`만 주면 Sonnet 5가 쓰고, `run_judge.py`는
backend를 주지 않는 한 codex/Terra다. 한 표 안에서 판정기를 섞지 않는다.

**첫 실행에서 난 오류와 수정.** `claude -p failed (1): ... "Not logged in · Please run /login"`. 원인은 모델이 아니라 백엔드가
`--bare`를 붙였기 때문이다: `--bare`는 ANTHROPIC_API_KEY만 읽고 구독 로그인(OAuth)은 절대 읽지 않는다. `--bare`를 빼고 짧은
고정 시스템 프롬프트(`--system-prompt`)를 주도록 고쳤다. 같이 고친 것: 백엔드 실패(인증·타임아웃·한도)는 문항의 "empty"로
기록하지 않고 건너뛰며, 3회 연속이면 중단한다. 시작 전에 스모크 호출 1회로 로그인을 확인한다. 예전 체크포인트의 "empty" 행은 재시도한다.

**의역 눈 검사 결과와 프롬프트 수정 (2026-09-16, 첫 100문항 샘플).** 전제 유지·내용 보존은 통과했으나 **문체가 바뀌지
않았다**: 일곱 샘플 전부 원문을 거의 그대로 베꼈다(fpq_111은 단어 하나 삭제). FPQ의 긴 1인칭 서사 틀이 그대로면 통제가
무력하다. 수정: (1) 프롬프트가 고정 틀을 지시한다: 정확히 두 문장, 25–45단어, 1문장 = 상황과 모든 믿음, 2문장 = 질문,
원문 구조 재사용 금지, 4단어 이상 구절 복사 금지. 상황이 없는 짧은 NFP는 "I have a question about <주제>."로 1문장을 채운다.
(2) 원문과의 단어 Jaccard가 0.6을 넘으면 한 번 재작성시키고, 그래도 넘으면 `near_copy`로 버린다. (3) 프롬프트 버전 `p2`가
writer 태그에 들어가 옛 체크포인트는 거부된다(`para.jsonl` 삭제 후 재실행). (4) `--limit N`은 FPQ·NFP 반반을 뽑아 샘플에
두 문체가 다 보이게 한다. 눈 검사 항목은 셋: 전제가 남았는가, 사실이 더해지거나 빠졌는가, **문체가 실제로 바뀌었는가**(길이·
문장 수·서사 틀이 두 클래스에서 비슷해졌는가). 셋째가 이 통제의 존재 이유다.

**서버에서 claude 로그인.** 브라우저 없는 서버에서는 두 방법 중 하나.
1. 구독 계정(권장): 서버에서 `claude auth login` → 터미널에 URL이 뜨면 노트북 브라우저로 열어 로그인 → 표시된 코드를
   서버 터미널에 붙여넣기. 확인은 `claude auth status`. 로그인 정보는 그 계정 홈에 남으므로 tmux 세션마다 다시 할 필요 없다.
   장시간 배치는 `claude setup-token`으로 장기 토큰을 만들어 `export CLAUDE_CODE_OAUTH_TOKEN=...`으로 넣어도 된다(토큰은 git에 넣지 않는다).
2. API 키(사용량 과금): `export ANTHROPIC_API_KEY=sk-ant-...`. 구독 한도와 무관하게 돈다.
스모크: `echo "Reply with OK." | claude -p --tools "" --no-session-persistence --output-format json --model claude-sonnet-5`
→ `"is_error":false`이고 `result`가 `OK`면 된다. 인증 실패면 `"result":"Authentication error..."`가 보인다.
호출 수는 의역 732×2–3 ≈ 2,000, 쌍둥이 294×2 ≈ 600으로 짧은 호출 2,600회 안팎이다. 구독 5시간 한도에 걸리면 스크립트가
체크포인트에서 재개하므로 같은 명령을 다시 돌리면 된다.

**Claude 백엔드.** `--backend claude`는 Claude Code CLI의 print 모드(`claude -p --tools "" --disable-slash-commands
--no-session-persistence --system-prompt <고정 문장> --output-format json`)로 codex exec와 같은 자리에서 쓴다. stdin으로 프롬프트를 넣고 JSON의 `result`를 읽으며
서빙 모델은 `modelUsage` 키로 행마다 기록한다. `configs/default.yaml`의 `judge.claude_model`이 비어 있으면
CLI 기본 모델이다. `run_judge.py`에도 선택지는 열려 있지만 **보고 표의 판정기는 Terra 그대로**다. 판정기를
바꾸면 표 전체를 한 판정기로 다시 채점해야 한다(25 §"Well 판정").

## 7. 결과 (실행 후 채움)

| 조건 | text | style | masked | hidden | mean | hidden − text [CI] |
|---|---|---|---|---|---|---|
| natural | | | | | | |
| twins (n=) | | | | | | |
| para (n=) | | | | | | |
| natural→twins | | | | | | |
| natural→para | | | | | | |
| para→natural | | | | | | |
| twins→natural | | | | | | |

의역 감사: 수락 /732, 상태별 수, 원문·의역 단어 Jaccard 중앙값(높으면 거의 복사라 약한 통제).
쌍둥이 감사: `twins_audit.md`의 상태별 수, 눈으로 본 30개 중 잘못된 것.

## 8. 부록: 같은 오탐에서의 TPR (실험 세션 지적, 2026-09-16)

calibration에서 5%로 잡은 문턱이 OOF에서는 text 8.1%, hidden 8.7%, review 10.1%로 나왔다. 행마다
오탐이 달라 TPR 열을 그대로 비교하면 안 된다. `gate_report`가 이제 "Matched operating points" 표를
붙인다: 각 held fold의 **평가 정상 문항**에서 같은 비율(5/8/10%)이 잡히도록 사후 문턱을 잡고 TPR을 읽는다.
이건 순위 비교용 oracle 운영점이지 배포 문턱이 아니며, 표 머리말에 그렇게 적힌다. 같은 표에 FPR ≤ 0.1
구간의 표준화 partial AUROC를 넣어 "단일 운영점" 반론도 같이 막는다. CI는 fold 안 group bootstrap.
149개에서 오탐 1개가 0.67pp이므로 매칭점은 거칠고, 차이는 CI로 읽는다.

## 9. 한계

- 의역 writer 자체의 문체가 두 클래스에 같이 들어간다. 그건 의도이지만, writer가 FPQ의 단정어를 부드럽게
  바꾸면 전제 신호가 약해진다. 전제 유지 확인(YES)이 그 손실을 막지만 완전하지 않다. `para_sample.md`
  30개를 사람이 본다.
- 쌍둥이는 정렬 성공 문항만이라 표본이 치우칠 수 있다(정렬이 쉬운 = 전제가 명시적인 문항).
- style/masked는 영어 기능어 목록에 의존한다. 하한선을 낮게 잡을 수는 있어도 높게 잡지는 않는다.
- 어느 결과도 "hidden이 진위를 표상한다"의 증거는 아니다. 문체 가설의 기각 또는 채택까지만 말한다.
