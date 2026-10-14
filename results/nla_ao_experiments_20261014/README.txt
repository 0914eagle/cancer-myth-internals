NLA·AO 실험 결과 묶음 — 2026-10-14 스냅샷

먼저 볼 파일: gepa_pilot/cpu_analysis_20261013/report.html
점수 CSV: gepa_pilot/cpu_analysis_20261013/selected_dev_scores.csv
새 대조 계획: registered_controls_20261014/protocol.json
실제 토큰·레이어 검증: registered_controls_20261014/actual_protocol_verification.json

포함: 고정785문항, Qwen2.5-7B/Gemma3-12B/27B의 NLA 설명 각4710건,
Gemma27 AO 설명4710건, Qwen/Gemma12 AR 스칼라 점수, 당시 답변과 탐색 분석,
GEPA 답변·채점·후보 프롬프트·진행 상태, 재현용 코드 스냅샷.

주의: Gemma27의 초기785문항×2조건은 hidden-state-only 추출이어서
models/gemma27/source_answers.jsonl의 answer가 비어 있다. 실제 생성 답변은
gepa_pilot/native_answers.jsonl 및 별도 소규모 검증 결과에 있다.

dev 점수는 후보 선택 자료이며 최종 test 결과가 아니다. 보고서의2026-10-13
분석은 당시6조건 완료 시점의 기록이다. 현재 상태는 SNAPSHOT.json을 본다.
미완료 AO/Haiku/새 대조는 완료된 결과로 해석하지 않는다.
AR 값은 활성 방향 복원 지표이며 답변 교정이나 설명 사실성 점수가 아니다.
GPT 탐색 판정과 주석 불일치 표시는 사람 검증된 정답이 아니다.

hidden state·모델 가중치는 external_tensor_index.json의 원본 위치에 보존했다.
코드 실행에는 원본 저장소와 공식 NLA/AO 의존성 및 고정 캐시 경로가 필요하다.
이 묶음에는 로그인 정보·API 인증 파일을 포함하지 않았다.

서버에서 갱신:
cd /home/eagle0914/cancer-myth/external/paper_baselines_b81ace2
python3 scripts/package_reader_experiments.py
