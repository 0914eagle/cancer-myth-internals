# Table 2 빈칸 보완

2026-10-01 사용자 요청에 따른 추가 집계와 Luna 생성 실행. 기존 수치를 덮어쓰거나 균형 지시를 일반 CoT로 바꾸지 않는다.

## 새 모델 호출 없이 완료한 선택 평가

- Qwen2.5 Direct gate, 전제 검토 CoT (2-step) gate로 기존 Plain/무조건 교정 답변을 선택했다.
- 동일한 crossfit TF-IDF text gate 결정을 Luna, Sonnet, Gemma, Qwen3.8 OFF/ON의 기존 답변에 적용했다. 모델마다 새 텍스트 분류기를 학습한 것이 아니다.
- 질문 ID로 저장된 판정과 Well 점수를 연결했고, 추가 공개 모델의 원 질문 일치를 검증했다. Qwen의 기존 Text/Hidden 선택 결과와 재집계가 일치하는지도 확인했다.
- 결과: [집계](routing_tables.md), [문항별 선택](routing_scores.csv), [수치와 원본 해시](routing_summary.json).
- Qwen은 FPQ 582/NFP 149, 나머지는 FPQ 583/NFP 149다. Qwen2.5 이외 Hidden gate는 해당 모델의 hidden state로 학습·평가한 결과가 아니므로 채우지 않았다.

재실행: `python3 -m scripts.complete_table2_routing`

## Luna 추가 답변 두 조건

| 항목 | 설정 |
|---|---|
| 모델 | `gpt-5.6-luna`, reasoning effort medium |
| 문항 | FPQ 583 + NFP 149 = 732 |
| 일반 CoT (1-step) | 원 질문 뒤 `Let's think step by step.`; Gemma·Qwen3.8의 기존 user template과 동일 |
| 전제 검토 CoT (2-step) → 답변 | 기존 Luna의 732개 검토문을 재사용하고 `src.pilot.COT_ANSWER`로 질문+검토문에서 답변 생성 |
| 검토문 재사용 검증 | 문항 ID, 원 질문으로 구성한 COT_REVIEW 프롬프트 일치, 완료 상태, 파일 해시 |
| 새 생성 호출 | 732 × 2 = 1,464회; 검토문 재생성 없음 |
| 생성 제한 | 독립 세션, 도구 금지, 호출당 240초, 추가 출력 토큰 상한 없음 |
| 병렬 | 생성 10개, Well 채점 5개 |
| 채점 | 기존 Sonnet `claude-sonnet-5` Well 프롬프트·주석, 원 평가 732문항 |
| 실패 처리 | 완료 결과 보존, 실패/중단 레코드를 조용히 재호출하지 않음; STOP 파일 지원 |

답변 system prompt는 기존 Luna Plain과 동일하다. 생성에는 라벨·목표 전제 주석·교정 참조를 제공하지 않는다. 모델 간 user template을 맞춘 조건도 CLI/native runtime 차이와 생성 설정 차이는 남는다.

실행 디렉터리: `results/frontier_cli/luna_table2_completion_20261001_v1/`.
`manifest.json`에 프롬프트·모델·호출 옵션을, `jobs.json`에 문항별 입력과 검토문 해시를 저장한다. `status.json`과 `run.log`가 현재 진행 상태다. nohup으로 실행하며 별도 작업 관리자를 사용하지 않는다.

```bash
python3 -m scripts.run_luna_table2_completion prepare \
  --out-dir results/frontier_cli/luna_table2_completion_20261001_v1

nohup python3 -u -m scripts.run_luna_table2_completion run \
  --out-dir results/frontier_cli/luna_table2_completion_20261001_v1 \
  --workers 10 --judge-workers 5 \
  > results/frontier_cli/luna_table2_completion_20261001_v1/run.log 2>&1 < /dev/null &
```

**2026-10-02 저장 원장 확인:** 두 조건 각각 732개, 총 1,464개 생성 완료. Sonnet 유효 채점은 857개이며 607개 미완료다. [조건별 중간 수치·분모](luna_answer_progress.md)를 표 2에서 ‡로 구분해 반영했다. 마지막 채점 중단은 세션 한도다. 현재 한도 확인이나 재호출은 하지 않았다.

초기 실행 기록: 스모크 테스트 4개 답변이 정상 생성됐고 전체 생성을 시작했다. 시작 시 Sonnet 접근 확인은 세션 한도로 실패했다(메시지의 Perth 18:20 = 한국시간 19:20 갱신). 생성 완료 뒤에도 접근이 막혀 있으면 Well 단계가 멈추고 기록을 보존한다. 채점 재개 시 실패 원장을 먼저 확인해야 한다.

원래 Direct gate의 저장 답변 선택과 Self-gated FP Identification은 이름만으로 같은 실행으로 취급하지 않는다. 이번 Luna 추가 생성은 일반 CoT·전제 검토 CoT 답변 두 조건이며 별도 Self-gated 답변 실험을 새로 실행한 것은 아니다.
