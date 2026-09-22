# 브랜치 통합 기록 — 2026-09-23

사용자 요청에 따라 활성 연구 문서를 main으로 통합하고, main에 포함된 작업 브랜치를 정리했다. 실험 원본 결과와 미커밋 작업은 삭제하거나 이 커밋에 임의로 포함하지 않았다.

| 기존 원격 브랜치 | 정리 시 tip | 처리 |
|---|---|---|
| `claude/cancer-myth-internals-context-cccnkd` | `5af75675ce0eb90b07e1bbfa5a074494d6dd9267` | `archive/claude-context-20260909` 태그로 원본 보관; 브랜치 삭제 대상 |
| `claude/document-session-o2temh` | `b226ce368b47d9b268e461e8c657d91993a75668` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/advisor-presentation-20260923` | `2438770485dc1400d3898fe9702e1b7984ae9a83` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/gate-transfer-discussion-20260922` | `a6db41e3f123e67e5939a8cb3299dbe0e8d1915d` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/gemma-steering-pilot` | `879370c79ea50e34c5baa3e6d15fb429fbddbbde` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/paper-storyline-2026-09-09` | `0f9a1009ae8d999b6614f823c6f08fbf5d04db57` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/research-discussion-2026-09-08` | `29932de2d5d1f12d39858278f4afb1f3c1c16e6b` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/review-context-2026-09-07` | `e16910408a4a8391b147c9b9f4afa7956d803561` | main에 포함됨; 브랜치 삭제 대상 |
| `codex/truth-form-audit-20260921` | `5047149c6bd7b8f371656499b58e510f3ba5b564` | main에 포함됨; 브랜치 삭제 대상 |

36번 논의 문서와 37번 발표 원고·근거 파일은 main에 병합했다. README에서 현재 문서로 바로 이동할 수 있게 연결했고, 삭제할 브랜치를 가리키던 37번 문서의 링크를 main 내부 상대 경로로 바꿨다.

오래된 Claude 브랜치의 고유 커밋 `5af7567`은 길이·부정어 제약을 포함한 true twin/false paraphrase 생성 실험이다. 현재 main의 후속 backend 인증·실패 처리와 다른 구현이므로, 브랜치 정리만을 위해 현재 생성 동작에 끼워 넣지 않았다. 원본 코드·테스트·기록은 보관 태그로 그대로 조회할 수 있다. 이 커밋을 main에 병합했다고 표시하지 않는다.

브랜치 삭제는 main 및 보관 태그를 원격에 먼저 push한 뒤에 수행한다. 원격 tip이 바뀌었으면 삭제가 거부되도록 각 브랜치에 expected-tip lease를 사용한다. main에는 force push하지 않는다.
