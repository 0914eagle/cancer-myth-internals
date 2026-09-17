# 33. Baseline 목록: 선행 논문의 방법과 우리가 실제로 돌린 것 (2026-09-17)

> 25(방법 정의)·06(계획)·02·07(선행 연구)·23·26(Gemma 파일럿)·28 §10.1(Qwen 판정)에서 옮김. "✓"는 결과가 있는 것,
> "부분"은 돌렸으나 지금 기준으로는 해석 불가, "✗"는 안 돌린 것.

## 1. 선행 논문이 제시한 방법

| 출처 | 방법 | 무엇을 하나 | 그 논문의 결과 (Qwen2.5-7B 기준이면 표기) |
|---|---|---|---|
| Cancer-Myth | Plain | 질문에 바로 답 | 공개 모델 PCR 한 자리 수 |
| Cancer-Myth | GEPA | 프롬프트 최적화 | 닫힌 모델 중심 |
| Cancer-Myth | Monitor | 별도 검사기 | 코드 없음 |
| Well | **FP Identification (무조건)** | 모든 질문에 "거짓 전제를 찾아 교정하라" | Qwen FPQ S5 75 / TPQ 0 |
| Well | Extract + FactCheck (LLM / MiniCheck) | 전제 추출 → 각 전제 진위 판별 → 결과를 넣어 답변 | TPQ 손실 큼 |
| Well | Self-Dual-Critique | 답변 후 자기 비판 두 방향 | 6개 방법군 중 하나 |
| Well | PreWoMe | 전제 가중 답변 | 〃 |
| Well | Question-to-Statement | 질문을 진술로 바꿔 검증 | 〃 |
| Well | FAITH | 특정 head 차단 | Qwen FPQ 0. 02 §: head를 영화 데이터에서 찾아 무효 |
| Well | LoRA fine-tuning | FPQ로 미세조정 | Qwen FPQ 62 / TPQ 0 |
| Well | GEPA (FPQ+TPQ) | 양쪽 목적으로 프롬프트 최적화 | gemma FPQ 19 / TPQ 96 |
| Two Axes | BoW 텍스트 분류기 | 질문 단어로 FPQ 판별 | CREPE 0.59 |
| Two Axes | hidden probe / DiM | 마지막 토큰 잔차로 판별 | CREPE 0.69–0.73 |
| Two Axes | **probe 게이트 → "전제를 의심하라" 프롬프트** | 골라진 질문에만 프롬프트 | 일반 도메인에서 이득 |
| Tripathi | probe 게이트 → ITI head steering | 의료 EHR 맥락 | 게이트·강도 결합 |
| CAST 등 | 조건 벡터 → 행동 벡터 | 조건부 steering | 일반 도메인 |
| Kojima | Zero-shot CoT | "step by step" | 일반 |

## 2. 우리가 돌린 것

| 방법 | Gemma 파일럿 (dev 117/30, 원 척도·Terra) | Qwen suite (732, Well 0–5·Sonnet, 판정 중) | 비고 |
|---|---|---|---|
| Plain | ✓ 6.0 / 63.3 (v2 정상 100) | ✓ ≥4 1.5% / 과교정 0 | Well Direct QA 1%와 일치 |
| **FP Identification (무조건, Well식)** | ✓ 59.0 / 30.0 (v2 정상 60) | **✗ 없음** | 파일럿의 `FP_CORRECT`가 이것. Qwen suite에는 빠졌다 |
| **Self-gated FP Identification (우리)** | ✗ | ✓ 7.0% / 과교정 5% | `direct` Yes/No 게이트 → 양성만 교정 지시. 25에서 이름 정정 |
| Premise CoT / Premise Review | ✓ 34.2 / 53.3 (v2 정상 100) | ✓ 12.8% / 5% | 파일럿 `premise_cot` = suite `premise_review` |
| Zero-shot CoT | ✗ | ✓ 0.5% / 0 | |
| Extract + Verify (LLM, RAG 없음) | ✗ | ✓ 3.6% / 0 | Well Extract+FactCheck의 LLM 판 adaptation. MiniCheck 판 없음 |
| Two Axes식 probe 게이트 → FP 프롬프트 | 부분: 낮은 오탐 문턱에서 117 중 6개 선택, CoT를 못 넘음 | ✗ | 29로 게이트 자체가 무효 판정. 다시 돌릴 이유 없음 |
| Oracle 게이트 → FP 프롬프트 (상한) | ✓ 59 / 100 | ✗ (무조건 행이 있으면 계산으로 나옴) | FPQ는 무조건 교정, NFP는 Plain |
| 무작위 게이트 (대조) | ✗ | ✗ (계산으로 나옴) | |
| 무조건 steering C (α sweep) | ✓ Quick45 α=.1: 10.0 / 53.3, 구제 0/26 | ✗ | C0 외부 참조 쌍. 저자 불균형 |
| 조건부 steering (게이트 × C) | ✗ | ✗ | 원래 제안 방법. 게이트도 C도 검증 전이라 미실행 |
| Self-Dual-Critique / PreWoMe / Q2S | ✗ | ✗ | Well 코드 있음 |
| GEPA / LoRA / FAITH | ✗ | ✗ | GEPA·LoRA는 학습 비용. FAITH는 무효 |
| MiniCheck | ✗ | ✗ | |

## 3. 여기서 드러나는 문제 둘

1. **Gemma 파일럿과 Qwen suite의 "FP Identification"이 다른 방법이다.** 파일럿은 Well식 무조건 교정(59/30), suite는
   self-gated(7/5). 30·28의 서사에서 둘을 같은 행으로 읽으면 "Qwen은 프롬프트가 안 먹힌다"는 잘못된 결론이 나온다. 정확한 문장은
   "Qwen은 지시하면 고치지만(Well 75%) 스스로 판별하지 못한다(direct 0.47)". 28 §10.1에 정정 반영.
2. **Qwen suite에 상한 행이 없다.** 무조건 교정 행이 있어야 FPQ 상한(교정 능력)과 NFP harm(무조건의 대가)이 같은 판정기로 나오고,
   oracle 게이트·무작위 게이트 행은 그 둘과 Plain을 섞어 계산만으로 만들어진다.

## 4. 추가 실행 후보와 비용

| 후보 | 생성 | 판정 (Sonnet) | 얻는 것 | 판단 |
|---|---|---|---|---|
| **무조건 FP Identification (Well식)** | 732 (GPU 30분) | 732 (약 2시간) | 상한·harm·oracle·무작위 행 넷 | **필수** |
| 31 지식 검사 | 2,332 짧은 생성 (GPU 1시간) | 0 | 문항별 "안다" 라벨, 2×2 | **필수** |
| Self-Dual-Critique, PreWoMe, Q2S | 각 732×2–3단계 | 각 732 | Well 방법군 완성 | 선택. 표의 폭은 늘지만 결론은 안 바뀜 |
| Extract+FactCheck MiniCheck 판 | 732 + MiniCheck | 732 | 〃 | 선택 |
| Terra 겹침 40 | 0 | 40 (codex 복귀 후) | 판정기 일치율 | 필수, 시점 미정 |
| 조건부 steering | C 재구성 후 | FPQ+NFP | 원래 방법 | 31 결과와 C 검증 뒤 |

## 5. 실행 (9/17)

`scripts/run_followups_0917.sh` 하나가 tmux 한 창에서 nohup 두 개를 띄운다.

| 작업 | GPU | 단계 | 산출물 |
|---|---|---|---|
| A | 0 | `fp_unconditional` 생성 732 → answers 내보내기 → Well 판정 계획(`well_judge_claude_fpu/`) → Sonnet 채점 루프(15분 재시도) → report | `answers/fp_unconditional.jsonl`, `well_judge_claude_fpu/report.md` |
| B | 1 | 지식 검사(31) → CREPE clone·inspect·추출 → CREPE↔Cancer-Myth 전이 평가 | `knowledge/qwen25_7b/report.md`, `crepe/transfer/v1/report.md` |

CREPE 라벨 필드는 자동 감지(`labels`에 /false/, 또는 비어 있지 않은 `presuppositions`)이며 감지 실패 시 B가 inspect 출력을 남기고 멈춘다.
그때는 `--label-key/--positive-regex`를 넣어 `extract`와 `eval`을 손으로 잇는다. A의 채점은 본 배치(`well_judge_claude/`)와 같은
구독 한도를 나눠 쓰므로 둘 다 느려질 수 있다.
