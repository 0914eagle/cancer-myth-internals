# Gemma27 Plain/NLA/AO → 네 조건 GEPA 예비 비교

같은 fixed785 학습 풀에서 seed42로 Cancer FPQ25/NFP25, CREPE FPQ25/NFP25를 고정했다. 질문 원문·label·모델 출력에 따른 선택은 하지 않았으며 group은 중복하지 않는다. 기존8개 예비 문항과는 우연히 겹치지 않았다. 먼저 실제 Plain 답변의 prefill에서 NLA K41 질문끝1벡터, AO K31 질문끝최대20벡터를 저장한다. 이전 공개 reader 출력은 해당 활성의 float32 바이트가 동일할 때만 재사용한다.

GPU0/1은 원 Gemma3-27B BF16/greedy/batch1/최대8192를 유지하며 실제 답변과 활성을 함께 저장한다. GPU2/3은 필요할 때 고정 NLA 또는 AO를 로드한다. Source에는 질문과 현재 후보 system prompt만 넣으며, 주석/정답/reader 설명을 추가하지 않는다. Reader에는 해당 생성의 활성만 주입하며 원 질문/정답/실제 답변을 텍스트로 주지 않는다.

4조건은 GEPA, GEPA+텍스트 진단 대조, GEPA+NLA, GEPA+AO다. 모두 같은 Plain seed, GPT6.1sol 요청/medium, 같은 train50/dev8(4FPQ4NFP), 최대100문항 단위 metric evaluations/데이터셋/조건, 최대4후보 제안,512토큰 prompt 제한, seed42를 사용한다. 반성 미니배치는 FPQ1+NFP1 쌍2개다. 목적함수는 두 Well0–5/5 점수의 평균이다. 텍스트 대조는 질문/답변/점수/이유만 보고 같은 GPT로 추가 진단을 생성한다. 추가 진단은512 공통 토크나이저 토큰으로 제한하고 원문/제한 여부를 모두 저장한다. 추가 설명은 학습 반성 피드백에만 넣는다.

설치된 gepa0.1.1에서 candidate_hash+integer example_id 캐시가 train/dev의0기반 번호에 충돌하는 것을 실제 smoke에서 재현했다. 따라서 이번4조건은 engine cache_evaluation=False로 통일했다. 동일 답변·judge 재사용은 질문/메시지/candidate/model을 포함한 별도 content-addressed 캐시에만 허용한다. 개선train1.0/dev0.1 후보가 seeddev0.2를 이기지 못하는 회귀 검증을 통과했다. 원래 frozen4090 실행기는 변경하지 않았으므로 그 설정과 결과는 별도 감사가 필요하다.

모든 후보를 GPT candidate-dev 점수로 고정한 뒤, 후보 선택에 쓰지 않은 dev4FPQ4NFP/데이터셋에서64개의 최종 답변을 생성하고 고정 servedID claude-haiku-5-5로 채점한다. 이16개 독립 문항은 본표의 official test가 아니라 연결 예비 검증 표본이다. Haiku 결과로 후보를 다시 선택하지 않는다. 주 실험의 CREPE100+100dev 제안이나 Cancer3fold/fulltest 규약을 이 작은 pilot로 대체하지 않는다.

현재 GPT/Haiku 짧은 접속 확인은 통과했다. 실제 연구 payload의 외부 전송/GEPA 실행은 최초 자동 승인 검토가 구체적 전송 승인이 부족하다는 이유로 거절했으며, 이후 사용자가 정리한 범위를 명시적으로 승인했다(export_consent.json). GPU 로컬 생성/바이트 대응/export는 계속된다. 전송 범위132문항 원문·주석과 provider별 fields는 export_source_rows.json/export_manifest.json에 저장했다.

이 비교의 분석자는 GPT이며 topic/특정 오류/행동 일치/추가 단서 판정은 정답 라벨이나 인과적 믿음 증거가 아니다. malformed 분석 JSON은 실패로 기록하고 채점 성공으로 바꾸지 않는다. 후보 선택은 목적함수/dev score이고 분석에서 유리한 사례만 선별하지 않는다. 학습 주석/참조를 답변 생성 모델에 제공하지 않지만 reflection의 학습 피드백에는 채점 이유를 통해 그 정보가 포함된다. 이 비용과 일반 지시 개선/학습 지식 주입을 구분해 보고한다.

실행은 tmux detached 관리 프로세스에서 유지한다. Provider 제한/일시 실패는 결과를0점으로 바꾸지 않고 저장 결과에서10분 간격으로 재개한다. 구체적quota는 해제까지, 상세 원인이 감춰진 일반CLI 실패는 최대3회 재시도 후 점검 대상으로 기록한다. 원 모델과 채점 결과 캐시는 동일 입력의 재호출을 줄이며 model fallback은 하지 않는다.
