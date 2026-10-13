# 모델 자체 추출 후속 실행 — 2026-10-13

사용자 결정: 진행 중인 GEPA를 마친 뒤 PreWoMe와 Extract+Verify를 모델 자체 추출로 다시 평가한다. 기존 Sol 공통 추출 결과는 추가 비교용으로 보존한다.

## 변경과 통제

- Qwen2.5 7B → Qwen 추출·검토·답변, Gemma3 12B → Gemma 추출·검토·답변, GPT-6 Luna → Luna 추출·검토·답변.
- 각 모델 안에서는 동일 추출 프롬프트/질문을 잠금 캐시에 저장해 두 방법이 같은 추출 목록을 재사용한다. 모델 간 공유하지 않는다.
- 고정 bundle, Cancer 3-fold 및 CREPE train/dev/test 명단, Well 프롬프트·few-shot, 토큰 제한, greedy/CLI 설정, Haiku 최종 채점과 4점 기준은 유지한다.
- GPU Qwen 0번/최대 batch16, Gemma 1·2·3번/최대 batch32. GPT 동시 호출 4개 슬롯을 기존 GEPA와 공유한다.
- Table 1은 영향 없음. Plain·Balanced·CoT·gate·GEPA 답변은 재생성하지 않는다.
- 원본 `main/`과 `haiku/`는 덮어쓰지 않는다. 새 생성은 `results/paper_baselines_v4/self_extraction/`, 채점은 `self_extraction_haiku/`에 저장한다.

## 자동 실행 순서

`queue_paper_self_extraction.py`가 각 모델을 30초마다 확인한다. 모델별로 GEPA 3종의 모든 Table 2 평가가 끝나 `table3` 단계로 진입했을 때만 STOP 파일을 만들어 기존 워커를 정상 종료시킨다. 실패 상태는 완료로 간주하지 않는다.

1. 자체 추출 PreWoMe·Extract+Verify의 Table 2를 생성한다. 별도 Haiku 큐가 생성된 답변을 계속 채점한다.
2. 기존 실행기를 재개해 나머지 방법의 Table 3를 진행한다. 여기서는 Sol 기반 두 파이프라인을 제외한다.
3. 나머지 Table 3가 끝나 GPU가 비면 두 파이프라인의 Table 3도 자체 추출로 실행한다.

중단/실패는 기존의 문항별 캐시·실패 원장 및 systemd 재시도로 복구한다. 무한 출력·빈 생성은 정답으로 만들지 않으며 실패 분모를 보존한다. 새 Haiku 거부는 자동 점수 부여 없이 원장에 남는다.

## 표 반영

표 갱신기는 모델×방법×데이터셋의 FPQ/NFP 평가가 모두 끝난 후 자체 추출 결과로 해당 묶음을 교체한다. 완료 전에는 기존 Sol 결과를 보존한다. `pipeline_result_origin`에 각 칸의 추출 조건을 기록한다. Table 3는 해당 평가 칸의 두 출처 설정이 모두 처리된 후 교체한다. 일부 문항의 새 결과와 기존 Sol 결과를 합쳐 하나의 점수를 만들지 않는다.

이 비교는 추출기 변경의 영향에 대한 조건 비교이다. 반복 실행 변동까지 통제한 인과 추정 또는 Well 원 논문의 완전한 재현이라고 주장하지 않는다.
