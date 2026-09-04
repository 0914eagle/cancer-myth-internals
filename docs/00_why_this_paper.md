# 00. 왜 Cancer-Myth인가

## 지도 조건

> ICML/ICLR 2026 의료 논문 중 하나를 찍는다. 코드·데이터 공개, 가능하면 SAE 공개 백본(Gemma Scope / Llama Scope / Qwen-Scope). 그 논문의 문제점을 찾아 해결한다. **메인 표를 그대로 가져와 한 줄 추가**한다 — baseline·dataset 고민 없이.

## 후보 풀

ICLR 2026 + ICML 2026 accepted 11,699편 → 의료 × 언어 필터 119편 → 수동 검토 후 core 92편 + borderline 14편.
그중 (a) GitHub 저장소가 **실제로 살아 있고** (b) 방법론을 제안하며 baseline과 경쟁하는 논문 22편 ([appendix](appendix/candidates_screening.md)).

## 탈락한 유력 후보와 이유

| 후보 | 왜 탈락 |
|---|---|
| **MedREK** (ICML, 의료 지식 편집) | 표 구조가 이상적(MEND/MEMIT/MedLaSA/RECIPE × 편집 수)이었으나 **논문이 명시한 저장소가 존재하지 않음** (`gh`로 확인) |
| **Med-Scout** (ICML) | "Model weights and Med-Scout-Bench are coming soon" |
| **MEDA** (ICML, Med-LVLM activation editing) | preprint·코드 없음 |
| **Medical Interpretability & Knowledge Maps** (ICLR) | 백본 좋음(Llama3.3-70B, Gemma-27B)이나 분석 논문 — 행을 추가할 표가 없음 |
| **MA-RAG** (ICML, Qwen3-8B) | 조건 충족. 다만 약점이 저자 인정이 아닌 우리 가설. 차선 |
| **AnesSuite** (ICLR, Llama-3.1-8B) | 인프라 리스크 최저. 다만 "마취과 추론이 약하다"는 격차가 무딤. 차선 |
| **SAE racial bias** (ICLR, Gemma-2) | 저자가 실패를 선언한 negative result. MIMIC-IV 인증 필요, 분석 논문 |

## Cancer-Myth이 남은 이유

1. **격차가 압도적이고 정량화돼 있다** — 프런티어 ≤43%, 공개 모델 최고 17.3% PCR
2. **저자들이 해결을 시도했고 실패를 정확히 기록했다** — GEPA·Monitor가 NFP를 무너뜨리는 Table 1이 곧 문제 정의
3. **통제군이 있다** — Cancer-Myth-NFP 150. 꼼수(무조건 의심)를 잡아내는 유일한 자산
4. **평가 프로토콜이 다 짜여 있다** — Table 1의 열이 채점표. 데이터·판정기·코드 공개
5. **Appendix Table 3에 SAE 공개 백본이 이미 행으로 있다** — Gemma-2-27B(Gemma Scope), LLaMA-3.1-8B(Llama Scope)

## 프로토콜이 한 번 꺾이는 지점

Cancer-Myth 본문 결과표는 **Table 1 하나**이고, 그 행은 전부 닫힌 모델(GPT-4o, Gemini-2.5-Pro, GPT-4o+MDAgents)이다. 내부 표상을 읽는 방법은 여기에 "+Ours"를 붙일 수 없다. 따라서 "한 줄 추가"는 실제로 **공개 모델 한 블록 추가**가 되고, 그 블록의 Plain·완화책 행은 우리가 채워야 한다. 데이터·지표·판정기는 그대로 쓰되 baseline은 재실행이 필요하다. 이 사실은 처음부터 말하고 시작한다.
