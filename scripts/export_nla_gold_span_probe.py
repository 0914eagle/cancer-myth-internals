"""Export a gold-informed training-only endpoint comparison without scoring claims."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/nla_ao_parallel_20261012'
OUT=BASE/'gold_span_probe'
POSITIONS=['question_last_token','prompt_keyword_1','assistant_boundary_last_token']


def read(path):return [json.loads(line) for line in path.open()]


def main():
 config=json.loads((OUT/'runtime.json').read_text())
 cohort={(r['dataset'],r['id']):r for r in config['cohort']}
 paired=[];evidence=[]
 for model in ['qwen25','gemma3','gemma27']:
  new=read(OUT/model/'av_results.jsonl');assert len(new)==16
  assert len({r['id'] for r in new})==16
  old={(r['source'],r['base_id'],r['condition'],r['position_family']):r for r in read(BASE/model/'av_results.jsonl')}
  for row in new:
   q=cohort[row['source'],row['base_id']];span=q['gold_aligned_span']
   assert row['partition']=='train' and row['category']=='FPQ'
   assert row['prompt']==q['question'] and q['question'][slice(*span['char_span'])]==span['text']
   assert row['hidden_state_index']==row['layer']+1
   originals=[old[row['source'],row['base_id'],row['condition'],p] for p in POSITIONS]
   assert all(r['query']==row['query'] for r in originals)
   wide=dict(model=model,dataset=row['source'],question_id=row['base_id'],condition=row['condition'],question=row['prompt'],
    annotated_premise=json.dumps(q['gold_presuppositions'],ensure_ascii=False),reference=q['gold_reference'],
    selected_span=span['text'],span_char_start=span['char_span'][0],span_char_end=span['char_span'][1],
    span_alignment='Reviewer aligned annotation to original text; not official gold character offsets',
    gold_end_token=row['selected_token_text'],gold_end_position=row['position'],gold_end_explanation=row['nla_output'],
    source_layer=row['layer'],hf_hidden_state_index=row['hidden_state_index'],
    diagnostic_scope='FPQ train only; oracle position, no final test or NFP specificity score')
   for p,r in zip(POSITIONS,originals):
    wide[p+'_token']=r['selected_token_text'];wide[p+'_position']=r['position'];wide[p+'_explanation']=r['nla_output']
   paired.append(wide);evidence.append(dict(new=row,comparison=originals,gold_question=q))
 assert len(paired)==48
 with (OUT/'paired_comparison.csv').open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
 (OUT/'paired_comparison.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in paired))
 (OUT/'review_evidence.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evidence))
 lines=['# FPQ 주석에 대응시킨 전제 구절 끝의 NLA 설명', '',
  '고정785개 학습 문항 안에서 Cancer4개·CREPE4개를 선택했다. Plain/Balanced × Qwen7B/Gemma12B/Gemma27B로 '
  '새 활성과 설명48개를 추출했다. 원 질문과 원 모델 지시문은 유지했고 NLA actor 프롬프트도 같은 공개 체크포인트 기본 템플릿을 유지했다.', '',
  '## 위치 선택의 성격', '',
  '데이터 주석은 잘못된 전제와 교정을 설명하는 문장이고 공식 문자 좌표가 아니다. '
  '검토자가 주석을 보고 원 질문의 대응 구절을 지정했다. 따라서 gold-informed 위치 대조이며 '
  '데이터셋에서 제공한 gold span을 그대로 읽은 실험이라고 부르면 안 된다. '
  '일부 전제는 암묵적이고 복합적이어서 구절 선택이 유일하지 않다. span의 원문·문자 좌표·선택 이유를 모두 저장했다.', '',
  '기존 전체 추출은 질문 마지막·TF-IDF 키워드·답변 경계의 세 위치였고 모든 토큰의 활성을 저장한 것이 아니었다. '
  '새 끝 위치가 필요하므로 동일 입력을 다시 prompt-only forward했다. 새 답변을 생성하거나 기존 답변을 replay하지 않았다.', '',
  '## 선택한 구절', '', '| 데이터셋 | ID | 주석에 대응시킨 원문 구절 |','|---|---|---|']
 for q in config['cohort']:lines.append(f"| {q['dataset']} | {q['id']} | {q['gold_aligned_span']['text']} |")
 lines+=['', '골드 주석·정답 설명은 위치 선택과 검토에만 사용했다. 원 모델/NLA 입력 텍스트에는 추가하지 않았다. '
  'NLA에는 새로 추출한 단일 활성 벡터만 넣었다.', '',
  '## 직접 읽은 예비 결과', '',
  '- 전제 구절 끝은 질문 마지막/일반 키워드보다 전제 자체를 더 직접적으로 보여 주는 사례가 있다. Gemma27의 검진 구절에서는 의료 방문이 줄어든다는 기대, 6회 항암 구절에서는 six rounds, 호스피스 구절에서는 배우자의 long-term hospice care가 나타났다.',
  '- 그러나 전제를 재서술하는 것과 그 전제가 잘못됐다고 판단하는 것은 다르다. 직접 읽은48개에서 목표 전제를 명시적으로 오류라고 짚고 교정해야 한다고 설명한 출력은 확인하지 못했다. 이 작은 목적 표본의 결과이며 원 모델이 오류를 모른다는 증거는 아니다.',
  '- Qwen은 표준6회 구절에서 five rounds/nine rounds를 덧붙였고, 검진 구절에는 임신 맥락을 추가했다. Gemma12의 스테인리스 구절은 구리로 바뀌었다. 주석 기반 위치에서도 NLA 해석의 정확성은 별도로 확인해야 한다.',
  '- 질문 뒤의 내용은 해당 토큰에서 아직 읽지 못한다. 포일의 hot 끝은 in the oven 앞이며, 바나나의 clones 끝은 다른 종류가 있다는 뒤 질문을 읽기 전이다. 구절 끝과 질문 마지막의 차이는 위치뿐 아니라 읽은 문맥량의 차이도 포함한다.',
  '- 바나나의 Gemma27 Balanced 설명에는 commercial plantation처럼 더 좁은 범위가 나타났지만 명시적인 all-bananas 가정의 교정이라고 판정하지 않았다.', '',
  '## 해석과 다음 비교', '',
  '이 조건은 오류 전제가 어디인지 알려 주었을 때 해당 위치에서 관련 정보를 읽을 수 있는지 확인하는 진단이다. '
  'FPQ를 이미 알고 위치를 선택했으므로 독립적인 FPQ 탐지 성능이나 실제 배포 방법의 성능으로 사용하지 않는다. '
  '현 단계에서는 설명 품질의 상한을 검토하는 조건이며 상한 자체가 증명됐다는 뜻은 아니다.', '',
  '대표48개 읽기를 전체507FPQ의 정확도처럼 해석하지 않는다. 전체로 확장하려면 주석-원문 구절 대응을 먼저 검토해야 한다. '
  'NFP와 비교할 때는 정상 전제 구절도 같은 규칙으로 선택해야 하며, 실제 답변의 교정/순응 판정은 설명과 독립적으로 붙여야 한다.', '',
  '## 사례 원문', '']
 for ident in ['fpq_206','fpq_115','fpq_728','crepe:2018-01500']:
  row=next(r for r in paired if r['model']=='gemma27' and r['question_id']==ident and r['condition']=='balanced')
  lines +=[f'### Gemma27 / Balanced / {ident}', '',row['question'],'',
   '주석을 보고 지정한 구절: '+row['selected_span'],'',
   f"실제 마지막 토큰: {row['gold_end_token']!r}", '',row['gold_end_explanation'],'',
   '**기존 질문 마지막 설명**','',row['question_last_token_explanation'],'']
 lines+=['## 저장 자료', '',
  '- paired_comparison.csv:48행. 새 구절 끝과 기존 세 위치의 설명을 나란히 저장.',
  '- runtime.json:원문 주석·선택 구절·문자 좌표·선택 이유·설정.',
  '- review_evidence.jsonl:원본 새 설명과 비교 설명의 출처.',
  '- 모델별 manifest.jsonl/activations/:실제 토큰 위치와 새 활성.', '']
 (OUT/'inspection_report.md').write_text('\n'.join(lines))
 hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'runtime.json',OUT/'paired_comparison.csv',OUT/'paired_comparison.jsonl',OUT/'inspection_report.md',OUT/'review_evidence.jsonl']}
 (OUT/'artifact_checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print('Exported48new explanations and144matched old explanations; same canonical NLA prompts verified.')


if __name__=='__main__':main()
