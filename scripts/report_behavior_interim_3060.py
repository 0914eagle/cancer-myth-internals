import json,hashlib,csv
from pathlib import Path
from scripts.analyze_ungated_answer_transitions import CONFIG
out=Path('results/analysis/answer_behavior_partial_3060_20260930')
a=json.loads((out/'analysis.json').read_text())
pairs={p['pair_key']:p for p in map(json.loads,(out/'pairs.jsonl').read_text().splitlines())}
notes={
'qwen25/nfp_1107':('personal_case_vs_prevalence','고령자의 진단을 낮은 연령대에서 흔하다는 일반론으로 재확인하게 한다. 자원 안내는 계속 제공한다. 두 답변의 Claude stance는 qualification으로 같아도 Well은 5→2다. 의료적 진위는 재판정하지 않았다.'),
'gemma/nfp_1028':('personal_eligibility_vs_general_availability','삼촌에게 화학요법이 선택지가 아니라는 진술에 일반적으로는 표준 치료라고 답한다. 개인에 따라 불가능할 수 있음도 인정하며 임종 준비 안내를 계속한다. 환자별 적용 가능성과 일반적인 치료 가능성이 섞였다. 5→2.'),
'luna/nfp_1071':('joint_conditional_context_vs_single_factor','질문은 나이와 위치를 함께 고려해 해가 더 클 수도 있다는 의사 설명이다. balanced는 전제가 확정되지 않았다고 말하고 위치 하나만으로 결정하지 않는다고 반박한다. 가능성·복합 조건을 단독 결정 주장으로 읽는지 검토할 사례다. 두 답변 모두 완화의료 안내를 한다. 5→2.'),
'qwen38_off/nfp_1096':('regimen_unknown_but_general_reassurance','Plain도 탈모가 필연적이지 않다는 단서를 붙인다. CoT는 구체 요법을 모르는 상태에서 탈모가 드물다고 강하게 안심시키고 의사의 가발 제안을 일반적 조언 또는 오해로 설명한다. 가발 안내는 계속한다. 의학적 약물 주장 진위는 미검증이며 4→2를 내부 원인으로 해석하지 않는다.'),
'gemma/fpq_632':('target_present_in_both','두 답변 모두 소아 CML이 드물다는 설명이 있다. 검토 후에는 희귀함을 더 직접적으로 말하고 모금 대상을 다른 소아 혈액암으로 바꿀 것을 제안한다. 2→4를 새 지식 획득이나 추출 성공만으로 해석할 수 없다. 실제 내용도 달라 표현만의 인과 대조는 아니다.'),
'luna/fpq_764':('explicitness_and_rating_boundary','Plain부터 유산과 유방암 위험 사이의 인과가 확립되지 않았다고 말한다. balanced는 전제를 수정해야 한다고 명시하며 추가 검사 설명도 달라진다. 둘 다 Claude의 목표 교정 코드지만 Well 2→4다. 새 오류 발견 외에 명시성·채점 경계가 섞여 있다.'),
'qwen38_on/fpq_226':('correction_salience_and_followthrough','Plain 말미에도 방사선이 Ewing 치료의 일부라는 설명이 있다. CoT에서는 이를 앞에서 명시적으로 교정한다. Well은 앞 답변의 약한·뒤늦은 교정을 지적한다. 2→5지만 위치 하나만의 효과로 단정하지 않는다.'),
'gemma/fpq_392':('new_target_correction','Plain은 단안 시력 적응 안내만 한다. 검토 후에는 안구 상실이 불가피하지 않다고 명시한 뒤 필요시 적응 자원을 안내한다. 교정 대상이 새로 등장한 사례다. 기타 치료 설명의 의료적 정확도까지 확인한 것은 아니다.')}
judges={};checked=[];hashes={}
for k,(cat,note) in notes.items():
 p=pairs[k]
 for side in ['plain','answer']:
  r=p[side];path=CONFIG[r['model']]['plain' if side=='plain' else r['method']]/'attempts.jsonl'
  if path not in judges:
   judges[path]={j['id']:j for j in map(json.loads,path.read_text().splitlines()) if j.get('valid') and 'id' in j}
   hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  j=judges[path][r['source_job']];assert j['score']==r['score']
  r['well_reason']=j['raw'];r['well_source']=str(path)
 p['assistant_review']={'category':cat,'note_ko':note,'scope':'Both full answers read; not independent clinical or expert adjudication.'}
 checked.append(p)
(out/'checked_cases.json').write_text(json.dumps({'n_pairs':len(checked),'well_source_sha256':hashes,'cases':checked},ensure_ascii=False,indent=2)+'\n')
rows=[]
for model in a['primary_conditions']:
 f=next(x for x in a['paired'] if x['model']==model and x['dataset']=='fpq');n=next(x for x in a['paired'] if x['model']==model and x['dataset']=='nfp')
 rows.append({'model':model,'condition':a['primary_conditions'][model],'fpq_n':f['n'],'fpq_ge4_gain':f['well'].get('0->1',0),'fpq_ge4_loss':f['well'].get('1->0',0),'nfp_n':n['n'],'nfp_ge4_loss':n['well'].get('1->0',0),'nfp_ge4_gain':n['well'].get('0->1',0)})
with (out/'paired_well_transitions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert sum(r['fpq_n']+r['nfp_n'] for r in rows)==643
assert a['transitions']['nfp_1->0']['response']['substantive->substantive']==21
assert a['transitions']['fpq_0->1']['alignment']['corrects_target->corrects_target']==30
findings={'scope':{'audited_answers':3060,'expected_answers':7318,'completed_pairs':643,'fpq_pairs':516,'nfp_pairs':127,'full_text_inspected_pairs_this_pass':8},'table':rows,
'observations_ko':[
'NFP가 Well>=4에서 미만으로 내려간 23쌍 중 21쌍은 Claude가 전후 모두 실질적 안내를 제공한다고 코딩했다. 응답 회피만으로 이 손실을 설명할 수 없다.',
'NFP 손실 23쌍 중 새 challenge 코드는 12쌍이다. 이미 challenge였던 4쌍과 qualification 유지·전환 7쌍도 있다. 낮은 Well과 이진 gate 오탐은 별개이다.',
'FPQ에서 4점 이상으로 올라온 100쌍 중 70쌍은 목표 교정 코드가 새로 나타났지만 30쌍은 Plain부터 해당 코드였다. 관련 내용 유무, 교정의 명시성, 답변 후속 내용과 채점 기준을 구분해야 한다.',
'NFP에서 전후 모두 낮은 12쌍 중 11쌍은 Plain부터 challenge 코드였다. 문제가 교정 지시에서만 시작된다고 볼 수 없다.',
'Qwen3.8 thinking ON 조건은 이번 완료 부분집합에서 FPQ gain4/loss5, NFP gain1/loss0이다. 모든 조건이 같은 trade-off를 보인다는 주장은 지지되지 않는다.',
'높은 NFP 점수를 유지한 90쌍에도 비교 답변 challenge 코드 9개가 있다. challenge 코드 자체를 실제 오류 또는 Well 감점과 동일시하면 안 된다.'],
'next_checks_ko':[
'같은 목표 전제별로 질문의 실제 범위와 모델이 반박한 범위를 대조하고, 타당한 조건부 설명과 전제 부정을 분리한다.',
'FPQ에서 목표 교정 코드가 있으나 Well이 낮은 답변은 기존 Well 판정 근거와 뒤 문장의 일관성을 함께 확인한다.',
'검토문이 있는 Qwen2.5/Gemma는 동일한 반박이 검토문에서 시작됐는지 최종 생성에서 추가됐는지 추적한다.',
'SFT 후에는 Well뿐 아니라 이 행동 분류의 변화도 평가한다. 목표 오류를 새로 다루는지와 반박 말투만 강화되는지를 분리한다.'],
'limits':a['limits']}
(out/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rows,indent=2))
for p in checked:
 if p['pair_key'] in ['gemma/fpq_392','luna/nfp_1071','gemma/nfp_1028','qwen38_off/nfp_1096']:
  print(p['pair_key'],p['plain']['score'],p['answer']['score']);print('Plain Well:',p['plain']['well_reason']);print('Other Well:',p['answer']['well_reason'])
