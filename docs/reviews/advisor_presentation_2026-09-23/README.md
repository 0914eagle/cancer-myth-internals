# 발표 원고의 수치 근거 — 2026-09-23

[발표 원고 37](../../37_advisor_presentation_2026-09-23.md)의 동결 근거다. 모델 호출·학습·채점 없이 기존 결과를 읽었다. 표 2-B의 routing과 무작위 선택 기대값은 이번에 저장된 점수로 계산했다. 기존 탐색 자료를 새로운 독립 test처럼 해석하지 않는다.

| 파일 | 용도 |
|---|---|
| [answer_scores.csv](answer_scores.csv) | 생성 모델·방법·문항별 유효 Well 점수. 모델 출력 원문은 포함하지 않음 |
| [answer_summary.json](answer_summary.json) | 표 A·2-A의 개수, 분모, 0–5 분포, ≥4, S5, 평균, 누락 |
| [routing_scores.csv](routing_scores.csv) | 공통 731개 × 5정책. Plain/교정/선택 점수와 gate 결정 |
| [routing_summary.json](routing_summary.json) | 표 2-B, 구제·손실, 선택 수를 맞춘 무작위 정책 기대값, 제외 ID |
| [gate_summary.json](gate_summary.json) | 표 1의 기존 AUROC·CI·탐지·오탐 집계 |
| [transfer_summary.json](transfer_summary.json) | 표 3의 조건별 전이 AUROC·CI와 분모 |
| [style_summary.json](style_summary.json) | 별도 style_controls/v2 프로토콜의 원본·의역·수정 조건 비교 |
| [positions_tables.md](positions_tables.md) | 위치×층 및 어휘 대조 수치 발췌 |
| [token_scan_tables.md](token_scan_tables.md) | 토큰 스캔 gate 수치 발췌 |
| [signal_flow_tables.md](signal_flow_tables.md) | 구간/마지막 토큰 방향, patching, 문항별 출력 변화 수치 발췌 |
| [diagnostic_summary.json](diagnostic_summary.json) | 선택지 판단·직접 탐지·검토·교정 응답의 기존 문항별 연결 |
| [annotated_summary.json](annotated_summary.json) | 기존 전제 주석 제공 진단. 실제 배포 gate 결과가 아님 |
| [replacement_summary.json](replacement_summary.json) | 별도 Sonnet 보완 전제 725개 집계. 전문가 정답이 아님 |
| [lora_scores.csv](lora_scores.csv), [lora_summary.json](lora_summary.json) | 원 NFP와 수정 질문을 분리한 미세조정 탐색 점수 |
| [nfp_examples.json](nfp_examples.json) | 호스피스·통계 요청의 두 사례 원 질문, Qwen/Sonnet 답변·판정문 |
| [run_status.json](run_status.json) | 동결 당시 상태. Luna Plain 생성 완료, 채점 156/732 중단 |
| [provenance.json](provenance.json) | 동결 UTC 시각, 원장·계획·집계·추출 스크립트 SHA-256 |

`positions_tables.md` 등은 원 자동 보고서의 표만 발췌했다. 과거 자동 보고서에 남은 “진위 정보가 사라진다” 같은 설명은 채택하지 않으며, 해석 범위는 원고에 명시했다. 새 의료 사실 검증이나 전체 실패 원인 분류를 수행한 것은 아니다.

Luna Plain의 부분 점수는 원자료 보존을 위해 CSV·JSON에는 포함되지만 완료 모델과 비교하는 발표 표에는 넣지 않았다. 누락을 0점으로 취급하지 않는다. 본문 비율은 유효 채점 분모이며, 기존 보고서의 `S5/all expected` 필드와 혼합하지 않는다.

재생성 명령은 [원고 부록 C](../../37_advisor_presentation_2026-09-23.md#부록-c-이번-문서의-수치-검증과-재현)에 있다. 서버 원장 없이도 CSV에서 개수와 비율을 다시 계산할 수 있다. 원장 무결성 검증에는 원본 서버 파일이 필요하다.
