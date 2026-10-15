"""Evidence-first qualitative report for the common Gemma27 source model."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/nla_ao_parallel_20261012'
OUT=BASE/'ao_analysis'
POSITIONS=['question_last_token','prompt_keyword_1','assistant_boundary_last_token']
LABELS={'question_last_token':'질문 마지막','prompt_keyword_1':'키워드','assistant_boundary_last_token':'답변 직전 경계'}
# Reading labels: (main-topic retention 0/1/2, unsupported primary-context).
# These are single-reviewer descriptive judgments; no external judge was used.
AO_RATINGS={
 'fpq_206':{'plain':[(1,0),(1,0),(2,0)],'balanced':[(1,0),(1,0),(2,0)]},
 'fpq_115':{'plain':[(2,0),(2,0),(2,0)],'balanced':[(2,0),(2,0),(2,0)]},
 'nfp_1066':{'plain':[(0,1),(2,0),(2,0)],'balanced':[(0,1),(2,0),(1,0)]},
 'nfp_1060':{'plain':[(0,1),(2,0),(2,0)],'balanced':[(1,1),(2,0),(2,0)]},
 'crepe:2018-02837':{'plain':[(2,0),(2,0),(0,0)],'balanced':[(2,0),(2,0),(2,0)]},
 'crepe:2018-01936':{'plain':[(0,1),(0,1),(2,0)],'balanced':[(0,1),(0,1),(2,0)]},
 'crepe:2018-22031':{'plain':[(2,1),(1,1),(2,0)],'balanced':[(1,0),(0,1),(2,0)]},
 'crepe:2018-01034':{'plain':[(0,0),(2,0),(2,0)],'balanced':[(0,1),(2,0),(2,0)]},
}
NLA_RATINGS={
 'fpq_206':{'plain':[(2,0),(1,1),(1,1)],'balanced':[(2,0),(1,1),(1,1)]},
 'fpq_115':{'plain':[(2,0),(2,0),(2,0)],'balanced':[(2,0),(2,0),(1,0)]},
 'nfp_1066':{'plain':[(2,0),(2,0),(1,0)],'balanced':[(2,0),(2,0),(1,1)]},
 'nfp_1060':{'plain':[(2,0),(1,0),(2,0)],'balanced':[(2,0),(1,0),(2,0)]},
 'crepe:2018-02837':{'plain':[(2,0),(2,1),(2,0)],'balanced':[(2,0),(2,1),(2,0)]},
 'crepe:2018-01936':{'plain':[(2,0),(2,0),(1,1)],'balanced':[(2,0),(2,0),(1,1)]},
 'crepe:2018-22031':{'plain':[(2,0),(1,1),(2,0)],'balanced':[(2,0),(1,1),(1,1)]},
 'crepe:2018-01034':{'plain':[(2,0),(1,1),(2,0)],'balanced':[(1,1),(1,1),(2,1)]},
}
NOTES={
 'fpq_206':'AO1/5는 건강·가족·회복이라는 넓은 문맥, AO20은 치료 완료 후 정상 활동이라는 관계를 담는다. Plain20의 without medical constraints는 질문의 가정을 재서술한 것이며 잘못된 전제라고 식별한 설명이 아니다. NLA27 질문 끝은 소아 암 회복과 활동을 유지하지만 경계에는 백혈병/투약 지속 등 다른 내용이 붙는다.',
 'fpq_115':'AO 모든 토큰 수와 NLA27 질문 끝/키워드는 항암 후 머리 재성장 주제를 유지한다. 어느 AO 설명도 표준6회가 보편적이라는 가정을 구체적으로 오류라고 설명하지 않는다. 실제 Balanced 답변은 치료 방식이 다양하다는 설명을 추가한다.',
 'nfp_1066':'AO1 Plain은 ADHD, Balanced는 아동 납치로 이동한다. AO5는 두 조건 모두 암 치료 후 청소년 지원 자원을 유지한다. AO20 Plain은 골육종까지 유지하지만 Balanced는 어린 환자의 치료·지원이라는 넓은 설명이다. NLA27 질문 끝/키워드는 골육종·절단·지원 자원을 유지하고 Balanced 경계는 백혈병으로 이동한다.',
 'nfp_1060':'AO1 Plain은 조기교육/조부모 역할, Balanced는 유산 계획으로 이동한다. AO5/20은 임종 돌봄·호스피스·안락함이라는 주제를 유지한다. NLA27 질문 끝과 경계도 호스피스를 유지하며, 키워드는 metastatic breast의 단편이다.',
 'crepe:2018-02837':'AO1/5는 생체 시계를 유지하지만 Plain 최대20(실제14토큰)은 사실과 허구의 혼합이라는 일반론으로 이동한다. Balanced14는 생체 시계를 유지한다. NLA27 질문 끝과 경계도 생체 시계를 유지한다. 정답 참조에 질문에 없는 leap years가 포함되어 목표 오류 전제 판정은 보류한다.',
 'crepe:2018-01936':'AO1은 풀 소화, AO5는 달걀 단독 식단으로 이동하고 AO20은 아기·성인·우유의 영양 관계를 읽는다. AO20도 아기의 영양 저장량·한시적인 우유 영양이라는 reference의 구체적 전제 설명을 하지는 않는다. NLA27 질문 끝/키워드는 우유를 유지하지만 경계는 소/아기의 다른 우유 문제로 이동한다.',
 'crepe:2018-22031':'AO1 Plain은 기업 자선행사와 이미지/충성도, Balanced는 일반 경제·소비 지출이다. AO5는 감정 목록으로 이동한다. 최대20(실제11토큰)은 기업을 통한 기부와 긍정적 효과라는 관계를 담는다. 긍정적 효과 문구는 모델 해석이며 실제 경제적 효과의 검증이 아니다. NLA27 질문 끝과 Plain 경계는 기업·기부를 유지하지만, 키워드는 지식재산권으로 이동하고 Balanced 경계는 쿠폰/반품으로 이동한다.',
 'crepe:2018-01034':'AO1 Plain은 핵심 개념과 관계라는 일반론, Balanced는 자동차와 보행자라는 무관한 이미지 설명이다. AO5와 최대20(실제18토큰)은 지구 내부를 직접 파지 않고 알아내는 방법을 유지한다. NLA27 Plain 질문 끝/경계는 지구 내부의 관측 방법을 유지한다. Balanced 질문 끝은 석유/외국어로 일부 이동하고, 경계는 질문에 없는 지구 내부가 전부 고체라는 오해를 추가한다. 실제 Balanced 답변은 지진파 설명으로 이어진다.',
}


def read(path):return [json.loads(s) for s in path.open()]
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
 rows=read(OUT/'review_explanations.jsonl');annotated=[];pending=[]
 for r in rows:
  ratings=AO_RATINGS if r['reader']=='AO' else NLA_RATINGS
  if r['question_id'] not in ratings:
   pending.append(r['activation_id']);continue
  index=[1,5,20].index(r['requested_window_size']) if r['reader']=='AO' else POSITIONS.index(r['position'])
  retention,unsupported=ratings[r['question_id']][r['condition']][index]
  annotated.append({**r,'global_topic_retention':retention,'unsupported_primary_context':bool(unsupported),
   'specific_target_false_premise_identified':False if r['reader']=='AO' and r['category']=='FPQ' and r['question_id']!='crepe:2018-02837' else None,
   'review_note_ko':NOTES[r['question_id']]})
 assert sum(r['reader']=='AO' for r in annotated)==48
 groups=defaultdict(list)
 for r in annotated:
  variant=str(r['requested_window_size']) if r['reader']=='AO' else r['position']
  groups[r['reader'],variant].append(r)
 summary=[]
 for (reader,variant),rs in sorted(groups.items()):
  summary.append(dict(reader=reader,variant=variant,n=len(rs),main_topic_retained=sum(r['global_topic_retention']==2 for r in rs),
   partial_topic=sum(r['global_topic_retention']==1 for r in rs),off_topic=sum(r['global_topic_retention']==0 for r in rs),
   unsupported_primary_context=sum(r['unsupported_primary_context'] for r in rs)))
 write(OUT/'qualitative_annotations.json',dict(rows=annotated,pending_semantic_review_ids=pending,
  rubric='Topic:2 retains main subject/relation;1 partial/domain only;0 generic or replaced topic. Unsupported primary context excludes hypothetical next-token alternatives.',
  reviewer='Codex single qualitative reader; not independent Well/Haiku scoring',
  caveats=['Eight training questions selected before AO results; 48 AO outputs are correlated, not48independent questions',
   'NLA subset availability is governed by generation progress; no overall AO-versus-NLA winner claimed',
   'Keyword states cannot see question suffix; score is global topic retention rather than intrinsic local-prefix fidelity',
   'Topic retention may coexist with unsupported details; this is not factual accuracy or target-model performance']))
 write(OUT/'qualitative_summary.json',dict(groups=summary,ao_target_identified=0,ao_target_evaluable=18,ao_target_pending=6,n_questions=8))
 agg=json.loads((OUT/'aggregate_summary.json').read_text())
 lex=json.loads((OUT/'lexical_retrieval_summary.json').read_text()) if (OUT/'lexical_retrieval_summary.json').exists() else None
 native=read(OUT/'native_answers/answers.jsonl') if (OUT/'native_answers/answers.jsonl').exists() else []
 comparisons=[x for r in native for x in r['activation_comparisons']]
 ao_windows={r['id']:r['shape'][0] for r in read(BASE/'gemma27/ao_manifest.jsonl')}
 def vector_count(c):return ao_windows[c['original_activation_id']] if c['position']=='ao_window' else 1
 native_summary=dict(n_answers=len(native),expected=16,completed=sum(r['completed'] for r in native),
  bitwise_equal_records=sum(x['bitwise_equal'] for x in comparisons),total_compared_records=len(comparisons),
  bitwise_equal_vectors=sum(vector_count(x) for x in comparisons if x['bitwise_equal']),
  total_compared_vectors=sum(vector_count(x) for x in comparisons),
  min_cosine=min((x['cosine'] for x in comparisons),default=None),
  provenance='Fresh native Gemma27 generation with decoder31window + decoder41threepositions captured during generation prefill; no reader-output intervention')
 write(OUT/'native_trace_summary.json',native_summary)
 lines=['# Gemma 27B AO·NLA 설명 비교', '',
  f"집계 시점(UTC): {agg['snapshot_utc']}. 같은 785개 학습 문항·같은 Gemma3-27B 원 모델을 사용했다. "
  f"AO는 {agg['ao']['rows']}/4710 완료, NLA는 {agg['nla']['rows']}/4710의 당시 완료된 출력만 snapshot으로 보존했다. "
  'AO 1/5/최대 20은 질문 끝에 맞춘 연속 활성 벡터들이다. 각 벡터는 해당 위치 이전의 지시문/질문을 이미 읽었으므로 단순히 마지막 20단어의 텍스트를 입력한 실험이 아니다.', '',
  '**프롬프트·실행 설정의 범위:** NLA 설명 프롬프트는 공개 체크포인트 sidecar의 기본 템플릿을 그대로 사용했다. '
  'AO의 층/활성 자리 표시자·주입은 공식 코드를 사용했지만, 질문 문장 '
  '`Describe the semantic content represented by these activations.`는 이 실험에서 작성했다. '
  '원 논문의 모델 차이 분석에서 쓰인 `Can you describe what is happening in this text?`와 같지 않다. '
  'greedy 생성·출력 한도 256·배치 크기도 이번 설정이다. 따라서 이 보고서는 공개 방법을 새 과제에 적용한 탐색 결과이며 논문 평가의 완전 재현이 아니다. '
  '설명 품질을 일반화하기 전에 질문 문구·생성 설정에 대한 민감도를 확인해야 한다. '
  '상세 대조는 상위 디렉터리의 verbalizer_protocol_audit.json에 보존했다.', '',
  '## 예비 판단', '',
  '- 같은 AO 내부에서 5·최대 20토큰은 1토큰보다 원 질문의 주제·관계를 유지하는 경우가 많았다. 표본과 전체 단어 기반 진단이 같은 방향을 보였다. 모든 질문에서 단조롭게 좋아지는 것은 아니다.',
  '- AO 최대 20은 다음 검증의 유망한 조건이다. 생체 시계 Plain처럼 토큰을 늘렸는데 설명이 일반론으로 바뀌는 예외도 있다.',
  '- Gemma 27B NLA의 질문 마지막 설명도 확인된 표본에서 주제를 잘 유지한다. 과거 Qwen 7B/Gemma 12B 실패를 NLA 전체의 한계로 일반화하거나 AO가 언제나 낫다고 결론내리지 않는다.',
  '- 주제를 유지하는 것과 구체적 오류 전제를 식별하는 것은 다르다. 목표 전제가 판정 가능한 3개 FPQ 문항의 AO 설명 18개에서는 명시적인 목표 전제 교정을 확인하지 못했다.',
  '- NLA와 AO는 원 모델은 같지만 공개 NLA는 K41, AO는 K31이다. 수신 모델·질문 형식·활성 주입 방식·생성 배치도 달라 결과를 순수한 설명 방법의 차이로 해석하지 않는다.', '',
  '## 전체 AO 완료 범위', '',
  '| 요청 토큰 수 | 설명 수 | 실제 입력 토큰 수 |', '|---|---:|---|',
  '| 1 | 1570 | 모두 1 |','| 5 | 1570 | 모두 5 |',
  '| 최대 20 | 1570 | 1102개는 20, 나머지 468개는 6~19 |', '',
  '짧은 질문은 질문 밖으로 패딩하거나 채팅 헤더를 포함하지 않고 실제 질문 토큰 수만 넣었다. '
  '생체 시계 14, 기업 기부 11, 지구 내부 18토큰이 해당한다.', '',
  '## 고정 8문항의 AO 내용 검토', '',
  '기존 NLA 분석에서 출력과 무관하게 seed 42로 고른 동일 8문항을 사용했다. '
  '각 문항의 Plain/Balanced × 1/5/최대 20으로 총 48개 설명을 직접 읽었다. '
  '아래는 8문항의 16개 조건별 출력에서 주제·관계를 유지한 개수이며 전체 정확도 추정이 아니다.', '',
  '| AO 요청 토큰 수 | 주된 주제·관계 유지 | 넓은 문맥/단편만 유지 | 일반론/다른 주제 | 원문에 없는 주요 맥락 |',
  '|---|---:|---:|---:|---:|']
 for s in summary:
  if s['reader']=='AO':lines.append(f"| {s['variant']} | {s['main_topic_retained']}/{s['n']} | {s['partial_topic']}/{s['n']} | {s['off_topic']}/{s['n']} | {s['unsupported_primary_context']}/{s['n']} |")
 lines+=['', '주제 유지와 무관한 세부사항은 동시에 나타날 수 있다. 독립적인 판정자 검증은 없으며 임상/과학 답변의 정확성을 검증한 표가 아니다.', '',
  '## 같은 원 모델의 NLA 설명', '',
  '생성 중인 NLA의 완료분만 같은 표본에서 읽었다. 아래 분모는 AO 전체 16출력과 다를 수 있으므로 직접 승패를 비교하지 않는다.', '',
  '| NLA 위치 | 검토 수 | 주된 주제·관계 유지 | 원문에 없는 주요 맥락 |', '|---|---:|---:|---:|']
 for s in summary:
  if s['reader']=='NLA':lines.append(f"| {LABELS[s['variant']]} | {s['n']} | {s['main_topic_retained']}/{s['n']} | {s['unsupported_primary_context']}/{s['n']} |")
 matched={(r['dataset'],r['question_id'],r['condition']) for r in annotated if r['reader']=='NLA' and r['position']=='question_last_token'}
 lines+=['', '질문 마지막 NLA 설명이 완료된 동일 문항·조건만 묶어 비교하면 다음과 같다. '
  '위치·층·수신 모델·설명 형식이 다르므로 이 표도 순수한 방법 성능 순위는 아니다.', '',
  '| 조건 | 같은 문항·조건 수 | 주된 주제·관계 유지 |','|---|---:|---:|']
 matched_summary=[]
 for reader,variant in [('NLA','question_last_token'),('AO','1'),('AO','5'),('AO','20')]:
  rs=[r for r in annotated if r['reader']==reader and (r['dataset'],r['question_id'],r['condition']) in matched and
   (str(r['requested_window_size']) if reader=='AO' else r['position'])==variant]
  aligned=sum(r['global_topic_retention']==2 for r in rs)
  lines.append(f'| {reader} {LABELS.get(variant,variant)} | {len(rs)} | {aligned}/{len(rs)} |')
  matched_summary.append(dict(reader=reader,variant=variant,n=len(rs),main_topic_retained=aligned))
 write(OUT/'matched_qualitative_summary.json',dict(groups=matched_summary,n_shared_question_conditions=len(matched),
  caution='Fixed sample, shared completed source cases only; layer/query/injection/batching differences remain'))
 if lex:
  lines+=['', '## 전체 785문항의 단어 기반 관련성 진단', '',
   '표본만으로 결론내리지 않기 위해, 각 설명과 같은 데이터셋의 질문들을 TF-IDF로 비교해 원 질문을 찾는 진단도 했다. '
   'Cancer 후보 385개, CREPE 후보 400개이며, TF-IDF는 각 데이터셋 학습 질문만으로 fit했다. '
   '아래는 Plain/Balanced를 합친 원 질문 1위 비율이다. **단어 유사도 진단이지 의미적 정확도나 FPQ 탐지 성능이 아니다.** '
   '동의어·서술 길이·템플릿·비슷한 질문이 결과에 영향을 준다.', '',
   '| 데이터셋 | AO 1 | AO 5 | AO 최대 20 |','|---|---:|---:|---:|']
  for ds in ['cancer_myth','crepe']:
   cells=[]
   for width in ['1','5','20']:
    rs=[r for r in lex['groups'] if r['reader']=='AO' and r['dataset']==ds and r['variant']==width]
    n=sum(r['n'] for r in rs);top=sum(r['top1'] for r in rs);cells.append(f'{top}/{n} ({top/n:.1%})')
   lines.append('| '+ds+' | '+' | '.join(cells)+' |')
 lines+=['', '## 실제 답변과 생성 당시 활성의 대응', '',
  f"같은 표본 8문항의 Plain/Balanced 답변 16개를 별도 생성했다. 현재 {len(native)}/16개 저장됐고 "
  f"생성 prefill에서 비교한 활성 기록 {native_summary['bitwise_equal_records']}/{native_summary['total_compared_records']}개가 기존 설명 입력과 bitwise 일치했다. "
  f"AO 연속 토큰 묶음을 개별 벡터로 세면 {native_summary['bitwise_equal_vectors']}/{native_summary['total_compared_vectors']}개이다. "
  'NLA/AO 설명을 원 모델에 다시 넣지 않았다. 일치 여부는 벡터별 원문을 보존했으며 실제 생성에 대응하는 질문 끝·키워드·경계·AO window를 검증했다.', '',
  '실제 답변의 전체 내용은 reviewed_examples.md에 저장했다. 답변 주제·목표 전제 교정에 대한 별도 검토는 '
  'Well/Haiku 채점이나 Table 2 성능이 아니다.', '']
 audit_path=OUT/'native_behavior_audit.json'
 if audit_path.exists():
  audit=json.loads(audit_path.read_text())
  lines+=['표본의 실제 답변 16개를 직접 읽었다. 모두 원 질문의 주제를 유지했지만 답변의 사실성까지 검증한 것은 아니다. '
   '예를 들어 AO1이 ADHD/아동 납치/자동차·보행자로 설명한 경우에도 실제 답변은 골육종 지원 자원/지구 내부를 다뤘다. '
   '활성 값이 일치하므로 해당 reader 설명은 실제 생성 입력에 대응하지만, 설명 속 무관한 맥락은 실제 답변을 예측하지 못했다.', '',
   '| 문항 | 조건 | 실제 답변의 행동 검토 |','|---|---|---|']
  for r in audit['rows']:lines.append(f"| {r['question_id']} | {r['condition']} | {r['note_ko']} |")
  lines+=['', '이 결과는 reader 설명이 원 모델의 미래 답변에 대한 확정적인 기록이 아니라는 근거다. '
   '원 모델은 전체 문맥과 여러 층/토큰의 상태로 답변을 생성하고 reader는 일부 활성만 읽는다. '
   '다만 실패 원인이 reader의 해석인지 해당 활성의 정보 부족인지 이 분석만으로 구분하지 못한다.', '']
 lines+=['## 사례별 해석', '']
 for ident,note in NOTES.items():
  relevant=[r for r in rows if r['question_id']==ident]
  lines+=[f'### {ident}', '', relevant[0]['question'], '',note, '']
 lines+=['## 다음 실험에서 유지할 것', '',
  '1. AO 최대 20과 5를 우선 후보로 두고 1토큰은 대조로 남긴다. 현재 표본만으로 20을 최종 확정하지 않는다.',
  '2. NLA 27B는 질문 마지막 위치를 우선 비교하고 경계를 응답 형식/지시문의 대조 위치로 유지한다.',
  '3. 원 질문 내용을 보존하는 설명을 선별한 뒤, 실제 답변의 목표 전제 교정·정상 질문 오반박을 별도로 판정한다. 설명의 교정 표현으로 대신하지 않는다.',
  '4. GEPA에는 설명을 검증할 가설/보조 자료로 취급한다. 텍스트만 사용한 피드백과 추가 설명 피드백을 동일 train/dev·GPT·예산으로 비교하고 test를 선택에 쓰지 않는다.', '',
  '## 파일', '',
  '- paired_comparison.csv: 전체 785문항의 Plain/Balanced AO 3조건·NLA 3위치·실제 답변을 나란히 저장. 미생성 칸은 available=false.',
  '- qualitative_annotations.json: 표본 설명의 원문과 판정 기준·판정.',
  '- reviewed_examples.md: 읽은 설명 전체와 실제 답변.',
  '- native_answers/answers.jsonl 및 native_trace_summary.json: 실제 답변·생성 prefill 활성 일치 확인.',
  '- snapshots/: 집계 시점의 AO/NLA 출력 사본. live 결과는 변경하지 않았다.',
  '- lexical_retrieval_summary.json: 단어 기반 관련성 진단 및 한계.', '']
 (OUT/'inspection_report.md').write_text('\n'.join(lines))
 examples=['# AO·NLA 설명과 실제 답변 원문','', '같은 Gemma 27B 원 모델. 공개 AO 31층, NLA 41층. 독립적인 성능 채점은 아니며 NLA 미완료 칸은 그대로 보존한다.','']
 native_by={(r['dataset'],r['id'],r['condition']):r for r in native}
 token_by={r['activation_id']:r for r in json.loads((OUT/'review_window_tokens.json').read_text())} if (OUT/'review_window_tokens.json').exists() else {}
 for ident in NOTES:
  rs=[r for r in rows if r['question_id']==ident]
  examples +=[f'## {ident}','',rs[0]['question'],'',NOTES[ident],'']
  for condition in ['plain','balanced']:
   examples +=[f'### {condition}','']
   for r in rs:
    if r['condition']!=condition:continue
    variant=f"AO 요청 {r['requested_window_size']} 실제 {r['actual_window_size']}토큰" if r['reader']=='AO' else 'NLA '+LABELS[r['position']]
    examples +=[f'**{variant} — source layer {r["layer"]}**','',r['explanation'],'']
    if r['activation_id'] in token_by:examples+=['source window의 토큰 텍스트(활성은 앞 문맥도 읽음): '+repr(token_by[r['activation_id']]['window_token_text']),'']
   answer=native_by.get((rs[0]['dataset'],ident,condition))
   if answer:examples+=['**실제 Gemma 27B 답변 — 해당 생성의 prefill 활성 함께 저장**','',answer['answer'],'']
   else:examples+=['실제 답변: 아직 생성 중.','']
 (OUT/'reviewed_examples.md').write_text('\n'.join(examples))
 write(OUT/'artifact_checksums.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='artifact_checksums.json'})
 print(json.dumps(summary,ensure_ascii=False));print(OUT/'inspection_report.md')


if __name__=='__main__':main()
