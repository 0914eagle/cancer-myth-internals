# NLA·AO 병행 실행

고정된 학습 문항 785개(Cancer fold0 train 385개 + CREPE 공식 train 200 FPQ·200 NFP)를 모든 모델에 공통 적용한다. 공식 test는 추출·위치 선택·설명 발견에 사용하지 않는다. TF-IDF·Probe·GEPA의 공식 train/dev 크기를 변경하는 작업은 아니다.

| GPU | 모델·작업 | 처리 범위 |
|---|---|---|
| 0 | Qwen2.5-7B NLA | Plain·Balanced × 질문 끝·키워드·답변 경계 = 4,710개 |
| 1 | Gemma3-12B NLA | 같은 범위 4,710개 |
| 2·3 | Gemma3-27B 활성 추출 → AO | 같은 문항·조건; AO 1·5·최대20토큰 |
| 4·5 | Gemma3-27B NLA 설명 | 같은 문항·조건·세 위치 4,710개; 가중치 준비 후 자동 실행 |

활성 추출은 모두 새 prompt-only forward다. 답변 생성·저장 답변의 재입력은 하지 않는다. Qwen/Gemma12의 예전 A6000 답변은 문맥 확인용 참조로 보존했고, Gemma27B 원모델의 답변은 아직 생성하지 않았다. 새 활성을 예전 답변 생성 때의 활성이라고 주장하지 않는다.

## 레이어와 공개 체크포인트

공식 NLA의 layer_index K는 zero-based decoder block K의 출력이다. HF hidden_states[K+1]과 첫 문항의 모든 선택 위치에서 직접 일치함을 검사했다.

| 모델 | NLA block / HF index | AO block / HF index |
|---|---|---|
| Qwen2.5-7B | 20 / 21 | 공개 호환 AO 없음 |
| Gemma3-12B | 32 / 33 | 공개 호환 AO 없음 |
| Gemma3-27B | 41 / 42 | 주 비교 31 / 32 |

Gemma27B 공개 AO는 block 15·31·46에서 학습했다. 주 비교는 31을 사용한다. NLA와 AO는 같은 원모델·문항·조건·질문 끝 위치를 사용하지만 **학습된 레이어가 다르다**. 이를 설명 방식만의 차이로 해석하지 않는다. 추가 확인을 위해 block41의 AO용 창도 저장했지만, 현재 주 AO 실행에는 사용하지 않는다.

AO는 질문 본문 안에서 끝나는 연속 토큰 창을 사용한다. 긴 창에 assistant 헤더나 답변 토큰을 섞지 않는다. 짧은 질문은 실제 길이만 사용하고 requested_window_size와 window_size를 구분해 기록한다.

## 실행과 저장

- 원모델·NLA·AO 버전: download_specs.json 및 runtime_*.json에 고정.
- 모든 활성 파일과 학습 ID·레이어·창 길이 검증: capture_verification.json.
- 모델별 manifest.jsonl: NLA 벡터 위치와 원래 활성 파일.
- gemma27/ao_manifest.jsonl: block31·41의 최대20토큰 원시 활성 창.
- 모델별 av_results.jsonl: NLA 설명.
- 모델별 scored.jsonl: 공개 AR 검증(Qwen·Gemma12만; Gemma27 AR는 이번에 받지 않음).
- gemma27/ao_results.jsonl: 공식 AO 어댑터의 설명.

공식 AO 논문 브랜치 커밋은 9816813552346b519f7e5c924a3e3204d5b723c6이다. 공식 create_training_datapoint/run_evaluation/활성 주입 함수를 수정 없이 사용한다. 추가 설치는 독립된 AO 환경에서만 했으며 기존 NLA·기준선 환경은 변경하지 않았다. Torch2.6.0, Transformers4.55.2, PEFT0.17.1을 사용하고 모델 병렬 훅과 충돌하는 자동 컴파일은 껐다. 생성은 greedy, 최대256토큰으로 기록한다.

Gemma27 NLA 공개 가중치는 FP32 약108GB이므로, 샤드를 하나씩 받은 후 BF16 추론용 파일로 변환한다. 공개 revision·변환 dtype·텐서 수·크기를 conversion_status.json에 기록한다. 변환 후 이 작업에서 받은 임시 FP32 샤드만 정리하며, 기존 체크포인트·실험 결과는 삭제하지 않는다.

## 평균 활성 방향 대조의 역할

평균 방향 대조는 답변 품질 평가나 GEPA 학습 목표가 아니다. 문항별 설명 없이 다른 문항의 공통 방향만으로 활성 복원 오차가 낮아지는지 확인하는 보조 진단이다. AR의 낮은 오차만으로 설명의 정확성이나 FPQ 교정 효과를 주장하지 않기 위한 기준이다. 이번 병행 작업의 우선순위는 올바른 활성 확보와 실제 설명 비교이며, 평균 방향 대조 때문에 GPU 생성 작업을 지연시키지 않는다.

NLA·AO 설명의 사실적 정확도, 원모델의 교정·오반박 행동, 설명을 GEPA에 넣었을 때의 성능 변화는 별도 평가가 필요하다. 이번 실행에는 외부 GPT·Haiku·Sonnet 채점을 호출하지 않았다.
