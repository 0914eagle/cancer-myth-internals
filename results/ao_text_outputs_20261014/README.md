# AO가 실제 생성한 설명 텍스트

Gemma 3 27B의 내부 활성을 AO로 읽은 실제 영어 출력과 대응 질문·답변을 담았다.
생성 텍스트를 GPT로 다시 쓰거나 요약하지 않았다.

먼저 [실제 예시](EXAMPLES.md)를 보면 된다.
[GEPA 설명 전체67개](GEPA_AO_TEXTS.md)와 [초기 Plain100 설명 전체](PLAIN100_AO_TEXTS.md)도 GitHub에서 바로 읽을 수 있다.

| 파일 | 내용 |
|---|---|
| `fixed785_ao_texts.jsonl` / `.csv` | 고정785문항 × Plain/Balanced × 1/5/최대20토큰 = 4,710개 AO 설명. 각 행에 질문과 AO 원문 포함 |
| `plain100_ao_question_answer_pairs.jsonl` / `.csv` | 같은100문항의 Plain AO 설명과 실제 답변·GPT 채점. 기존 활성과 바이트 일치 확인 후 설명 연결 |
| `gepa_ao_explanations.jsonl` / `.csv` | AO-GEPA 중 저장한67개 설명. 질문·후보 ID·AO 원문·실제 답변·점수·초기 설명을 나란히 저장. 완료된 평가 피드백에 포함된 것은66개 |
| `fresh_ao_raw_outputs.jsonl` | 그중 새로 디코딩한43개 AO의 원본 결과와 실제 reader 프롬프트 |
| `candidate_prompt_catalog.json` | 후보 ID별 원 모델 입력 프롬프트 |
| `selected_ao_prompts.json` | Cancer-Myth·CREPE 최종 선택 프롬프트 |
| `ao_heldout_answers_haiku.jsonl` | 고정 프롬프트의 별도 보류16문항 답변과 Haiku 채점 |
| `heldout_comparison_summary.csv` | 기본 GEPA·텍스트·NLA·AO의 공통 보류 문항 점수 |
| `protocol.json` / `token_layer_verification.json` | AO 질문·레이어·토큰 위치 및 실제 저장값 검증 |

AO 질문은 `Describe the semantic content represented by these activations.`였다.
source decoder block31(0기반), HF hidden_states[32]를 사용했다.
GEPA에서는 질문 마지막 최대20토큰을 읽었다. 질문·정답·실제 답변 텍스트를 AO에 주지 않았다.

4,710개 초기 출력의 원 활성 추출에서는 원 모델 답변을 생성하지 않았다.
이후 Plain100 및 GEPA 답변은 별도 실제 생성에서 얻고 활성 대응을 확인했다.
67개 저장 설명에는 초기 설명 재사용24개와 새 디코딩43개가 포함된다.
이 중1개는 중단 당시 저장됐지만 완료된 평가 피드백 기록에는 없다.
`supplied_in_completed_gepa_evaluation` 필드로66개 완료 기록과 구분한다.
같은 질문이 여러 후보에서 반복되므로67문항으로 해석하면 안 된다.
보류 문항에서는 AO 설명을 추가 생성하지 않고, 고정된 프롬프트로 답변만 생성했다.

AO 출력은 모델의 믿음이나 사실의 정답이 아니다. 긍정 사례만 골라 성공률을 계산하지 않았다.
새 수용·의심·거부 질문 비교와 다른 문항 설명 대조는 별도 진행 중이며 이 파일들의 완료 결과와 구분한다.
