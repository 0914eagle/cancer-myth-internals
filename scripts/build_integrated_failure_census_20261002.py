"""Integrate all primary answer codes, manual reviews and audit scores.
No model calls or new semantic labels. Missing causal coding is NEVER zero.
"""
import csv,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT=ROOT/'docs/reviews/integrated_failure_census_2026-10-02'
DIRECT=ROOT/'docs/reviews/direct_reading_2026-10-02'
SCOPES=ROOT/'docs/reviews/failure_type_denominators_2026-10-02/manual_scope_subtypes57.json'
SCORES=ROOT/'docs/reviews/well64_adjudication_2026-10-02/score_corrections.csv'
MODELS=['qwen25','gemma','luna','qwen38_off','qwen38_on']
NAMES=dict(qwen25='Qwen2.5',gemma='Gemma',luna='Luna',qwen38_off='Qwen3.8 OFF',qwen38_on='Qwen3.8 ON')
METHODS=dict(qwen25='전제 검토→답변',gemma='전제 검토→답변',luna='균형 지시',qwen38_off='일반 CoT',qwen38_on='일반 CoT')
PATTERNS={
 'target_omitted':'목표 오류 미언급',
 'error_endorsed':'목표 오류 수용',
 'related_only':'관련 설명만 있음',
 'coded_correction':'교정으로 코딩',
 'request_without_target_correction':'미언급/수용/관련 설명이며 요청에 실질 응답',
 'scope_signal':'범위·상황·문의변환 태그 또는 강화/없는 대상 grounding',
 'personal_reopened':'개인 상황 재확인 태그(일반화 오류와 다름)',
 'unverifiable_as_false':'검증 불가를 거짓 근거로 사용했다는 태그',
 'challenge':'반박 코드(부당성 미확정)',
 'qualification':'제한·주의사항 코드',
}

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(name,rows,fields=None):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(k,n):return f'{k}/{n} ({100*k/n:.1f}%)' if n else '— (분모 0)'
def patterns(a,ds):
 c=a['coding'];t={f['tag'] for f in c['flags']};al=c['reference_alignment']['status']
 return dict(target_omitted=ds=='fpq' and al=='misses_target',error_endorsed=ds=='fpq' and al=='endorses_target_error',
 related_only=ds=='fpq' and al=='related_without_correction',coded_correction=ds=='fpq' and al=='corrects_target',
 request_without_target_correction=ds=='fpq' and al in ['misses_target','endorses_target_error','related_without_correction'] and c['response']=='substantive',
 scope_signal=bool(t & {'scope_strengthening','context_shift','request_to_assertion'}) or c['target_grounding'] in ['strengthened_or_shifted','absent'],
 personal_reopened='personal_context_reopened' in t,unverifiable_as_false='unverifiable_as_false' in t,
 challenge=c['stance']=='challenge',qualification=c['stance']=='qualification')

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 pairs=[json.loads(l) for l in SRC.open()];assert len(pairs)==3659
 manual=json.loads((DIRECT/'reviewed_evidence.json').read_text());assert len(manual)==299
 mx={(r['model'],r['id']):r for r in manual};assert len(mx)==299
 scopes=json.loads(SCOPES.read_text());sx={(r['model'],r['id']):r for r in scopes};assert len(sx)==57
 corrections=list(csv.DictReader(SCORES.open()));cx={(r['model'],r['id'],r['method']):r for r in corrections};assert len(cx)==16
 flat=[];evidence=[];quote_warnings=[]
 for pair in pairs:
  for side in ['plain','alternative']:
   a=pair[side];c=a['coding'];ref=c['reference_alignment'];key=(pair['model'],pair['id']);cor=cx.get((*key,a['method']))
   assert hashlib.sha256(a['answer'].encode()).hexdigest()==a['answer_sha256']
   if cor:assert cor['answer_sha256']==a['answer_sha256'] and int(cor['original_score'])==a['score']
   pat=patterns(a,pair['dataset']);m=mx.get(key);ss=sx.get(key)
   row=dict(model=pair['model'],id=pair['id'],dataset=pair['dataset'],side=side,method=a['method'],
    original_score=a['score'],audit_overlay_score=int(cor['audit_effective_score']) if cor else a['score'],
    alignment=ref['status'],stance=c['stance'],response=c['response'],grounding=c['target_grounding'],
    flags=';'.join(f['tag'] for f in c['flags']),manual_pair_reviewed=m is not None,
    manual_pair_category=m['primary_descriptive_category'] if m else '',
    manual_scope_subtype=ss['subtype'] if ss and side=='alternative' else '',
    answer_sha256=a['answer_sha256'],candidate4_status='not_population_measured',candidate5_status='not_population_measured',
    **{k:int(v) for k,v in pat.items()})
   flat.append(row)
   for typ,quote in [('stance',c['stance_quote']),('alignment',ref['answer_quote']),('response',c['response_quote'])]:
    if quote and quote not in a['answer']:quote_warnings.append(dict(model=pair['model'],id=pair['id'],side=side,type=typ))
   if any(pat[k] for k in ['scope_signal','personal_reopened','unverifiable_as_false']):
    evidence.append(dict(model=pair['model'],id=pair['id'],dataset=pair['dataset'],side=side,method=a['method'],score=a['score'],
     question=a['question'],target=c['target_ko'],stance_quote=c['stance_quote'],question_quote=c['question_quote'],
     grounding=c['target_grounding'],grounding_note=c['grounding_note_ko'],flags_json=json.dumps(c['flags'],ensure_ascii=False),answer_sha256=a['answer_sha256']))
 assert len(flat)==len({(r['model'],r['id'],r['side']) for r in flat})==7318
 assert sum(r['original_score']!=r['audit_overlay_score'] for r in flat)==16
 write('all_7318_answer_index.csv',flat);write('candidate_signal_evidence.csv',evidence)
 write('quote_validation_warnings.csv',quote_warnings,fields=['model','id','side','type'])
 groups=defaultdict(list)
 for r in flat:groups[r['model'],r['side'],r['dataset']].append(r)
 totals=[];distributions=[];coverage=[];overlaps=[]
 for model in MODELS:
  for side in ['plain','alternative']:
   for ds in ['fpq','nfp']:
    rr=groups[model,side,ds];low=[r for r in rr if r['original_score']<4];high=[r for r in rr if r['original_score']>=4]
    totals.append(dict(model=model,side=side,method=rr[0]['method'],dataset=ds,n=len(rr),low=len(low),high=len(high),
     audit_low=sum(r['audit_overlay_score']<4 for r in rr),manual_read_low=sum(r['manual_pair_reviewed'] for r in low),
     outside_299_review_low=sum(not r['manual_pair_reviewed'] for r in low)))
    for status,zz in [('low',low),('high',high)]:
     for k in PATTERNS:
      value=sum(r[k] for r in zz) if ds=='fpq' or k not in ['target_omitted','error_endorsed','related_only','coded_correction','request_without_target_correction'] else None
      distributions.append(dict(model=model,side=side,dataset=ds,well_group=status,pattern=k,n=value if value is not None else '',denominator=len(zz),
       percent=100*value/len(zz) if value is not None and zz else '',evidence_kind='existing_AI_code_or_optional_tag',
       absence_interpretation='optional_tag_absence_not_confirmed_negative' if k in ['scope_signal','personal_reopened','unverifiable_as_false'] else 'existing_behavior_code'))
     for k in ['request_without_target_correction','scope_signal','personal_reopened','unverifiable_as_false']:
      for j in ['request_without_target_correction','scope_signal','personal_reopened','unverifiable_as_false']:
       if k>=j:continue
       overlaps.append(dict(model=model,side=side,dataset=ds,well_group=status,pattern_a=k,pattern_b=j,n=sum(r[k] and r[j] for r in zz),denominator=len(zz)))
    for k in ['C1_request_focus','C2_changed_claim','C3_truth_criterion','C4_question_dependency','C5_review_to_answer']:
     code={'C1_request_focus':'request_without_target_correction','C2_changed_claim':'scope_signal','C3_truth_criterion':'unverifiable_as_false'}.get(k)
     coverage.append(dict(model=model,side=side,dataset=ds,candidate=k,low_denominator=len(low),
      population_behavior_proxy_n=sum(r[code] for r in low) if code and (k!='C1_request_focus' or ds=='fpq') else '',
      proxy_interpretation='observable_coding_signal_not_cause_prevalence' if code else 'not_measured_from_final_answers',
      manually_confirmed_scope_n=sum(bool(r['manual_scope_subtype']) for r in low) if k=='C2_changed_claim' and side=='alternative' and ds=='nfp' else '',
      exhaustive_cause_coding='not_complete',causal_absence_count='unknown'))
 write('condition_denominators.csv',totals);write('patterns_by_model_condition_label_score.csv',distributions)
 write('candidate_coverage.csv',coverage);write('pattern_overlaps.csv',overlaps)
 # All same-item transitions, without assuming distinct patterns are independent causes.
 idx={(r['model'],r['id'],r['side']):r for r in flat};trans=[]
 for model in MODELS:
  for ds in ['fpq','nfp']:
   selected=[p for p in pairs if p['model']==model and p['dataset']==ds]
   for k in PATTERNS:
    counts=Counter()
    for p in selected:
     a=idx[model,p['id'],'plain'];b=idx[model,p['id'],'alternative']
     counts[(a[k],b[k],int(a['original_score']>=4),int(b['original_score']>=4))]+=1
    for (a,b,pa,pb),n in counts.items():trans.append(dict(model=model,dataset=ds,pattern=k,plain_pattern=a,alternative_pattern=b,plain_high=pa,alternative_high=pb,n=n))
 write('all_pattern_score_transitions.csv',trans)
 score_transitions=[]
 for model in MODELS:
  for ds in ['fpq','nfp']:
   selected=[p for p in pairs if p['model']==model and p['dataset']==ds]
   counts=Counter((p['plain']['score']>=4,p['alternative']['score']>=4) for p in selected)
   score_transitions.append(dict(model=model,dataset=ds,n=len(selected),both_low=counts[False,False],
    low_to_high=counts[False,True],high_to_low=counts[True,False],both_high=counts[True,True]))
 write('paired_score_transitions.csv',score_transitions)
 # Exhaustive within the pre-selected manual cohorts, not population prevalence.
 manual_rows=[]
 for r in manual:
  assert (r['model'],r['id'],'plain') in idx
  for side in ['plain','alternative']:assert r[side]['answer']==next(p[side]['answer'] for p in pairs if p['model']==r['model'] and p['id']==r['id'])
  ss=sx.get((r['model'],r['id']))
  manual_rows.append(dict(review_index=r['review_index'],model=r['model'],id=r['id'],dataset='fpq' if r['cohort']=='fpq_disagreement' else 'nfp',
   cohort=r['cohort'],category=r['primary_descriptive_category'],scope_subtype=ss['subtype'] if ss else '',
   plain_score=idx[r['model'],r['id'],'plain']['original_score'],alternative_score=idx[r['model'],r['id'],'alternative']['original_score'],
   reading_note=json.dumps(r['assistant_direct_reading'],ensure_ascii=False)))
 # Dataset taken from actual linked index, not guessed cohort label.
 for r in manual_rows:r['dataset']=idx[r['model'],r['id'],'plain']['dataset']
 write('manual_299_pair_registry.csv',manual_rows)
 manual_counts=[];concord=[]
 for model in MODELS:
  f=[r for r in manual_rows if r['model']==model and r['dataset']=='fpq'];n=[r for r in manual_rows if r['model']==model and r['dataset']=='nfp']
  for ds,rr,cats in [('fpq',f,['F1','F2','F3','F4','F5']),('nfp',n,['N1','N2','N3','N4','N5','N6'])]:
   for cat in cats:manual_counts.append(dict(model=model,dataset=ds,category=cat,n=sum(r['category']==cat for r in rr),selected_denominator=len(rr)))
  nn=[r for r in n if r['scope_subtype']]
  concord.append(dict(model=model,manual_n1=len(nn),auto_scope_signal=sum(idx[model,r['id'],'alternative']['scope_signal'] for r in nn),
   auto_personal_reopened=sum(idx[model,r['id'],'alternative']['personal_reopened'] for r in nn),
   auto_grounding_strong_or_absent=sum(idx[model,r['id'],'alternative']['grounding'] in ['strengthened_or_shifted','absent'] for r in nn)))
 write('manual_pair_category_counts.csv',manual_counts);write('manual_vs_auto_scope.csv',concord)
 # NFP primary manual interpretation with complete denominators, personal kept separate.
 manualscope=[]
 for model in MODELS:
  nn=[r for r in manual_rows if r['model']==model and r['dataset']=='nfp'];low=[r for r in groups[model,'alternative','nfp'] if r['original_score']<4]
  z=Counter(r['scope_subtype'] for r in nn)
  manualscope.append(dict(model=model,all_alternative_low=len(low),selected_loss_pairs=len(nn),both_low_outside_selection=len(low)-len(nn),
    personal_to_population=z['personal_to_population'],other_relation_scope=z['relation_or_scope_change'],mixed=z['mixed_or_ambiguous'],referent_time_scenario=z['referent_time_or_scenario_change'],
    non_n1_selected=sum(not r['scope_subtype'] for r in nn)))
 write('manual_nfp_scope_denominators.csv',manualscope)
 # Separate task evidence: never join these as final-answer cause labels.
 gate=json.loads((ROOT/'docs/reviews/luna_claim_gate_recheck_2026-10-02.json').read_text())
 dump(OUT/'luna_gate_separate_task.json',dict(structured=gate['structured'],matched_quotes=gate['matched_quotes'],fixed_claims=gate['fixed_claims']))
 staged=[json.loads(l) for l in (ROOT/'docs/reviews/nonpersonal_stage_trace_2026-10-02/two_step_six.jsonl').open()]
 assert len(staged)==6
 dump(OUT/'stage_reading_coverage.json',dict(selected_review_answer_cases=6,models=dict(Counter(r['model'] for r in staged)),
  directly_documented_omission=[dict(model='qwen25',id=i) for i in ['fpq_762','fpq_839']],
  limitation='Two documented cases among 5 Qwen/1 Gemma selected regressions, not a population screen; unassigned cases not negative labels.'))
 # Render one human-readable report with every requested denominator.
 lines=['# 모델·조건별 실제 답변 실패 통합 집계','',
 '**집계 완성 범위:** 3,659쌍·7,318답변의 기존 코드, 299쌍 직접 독해, 57쌍 범위 세분, 64쌍 채점 감사를 문항·답변 해시로 연결했다. 새로운 전수 원인 판정이나 내부 인과 분석을 수행한 것은 아니다.','',
 '실제 답변만 본편에 집계하며, 별도 Luna gate는 아래 별도 절에 둔다. 저점은 원 Well<4이며 NFP 저점을 실제 오반박 정답으로 부르지 않는다. 사람/임상 검증이 아니라 기존 Sonnet 분류와 Codex의 사후 직접 독해다.','',
 '## 1. 모델·조건별 전체 저점 분모','',
 '각 행의 단위는 답변. Qwen2.5 FPQ는582개, 나머지583개, NFP는149개. OFF/ON은 같은 모델의 두 설정이다.','']
 rows=[]
 for model in MODELS:
  for side in ['plain','alternative']:
   f=next(t for t in totals if (t['model'],t['side'],t['dataset'])==(model,side,'fpq'));n=next(t for t in totals if (t['model'],t['side'],t['dataset'])==(model,side,'nfp'))
   rows.append([NAMES[model],'Plain' if side=='plain' else METHODS[model],pct(f['low'],f['n']),pct(n['low'],n['n']),n['audit_low']])
 lines+=[table(['모델','조건','FPQ 저점','NFP 저점','부분 감사 적용 NFP 저점'],rows),'',
 '**총 FPQ 저점 3,695답변, NFP 저점 230답변**이다. 모델·조건을 합한 답변 수이지 독립 문항 수가 아니다. 감사 점수는 확정 정답이나 전수 재채점이 아니며 기존 점수와 병기했다.','',
 '## 2. FPQ 저점의 행동 구성: 전수 기존 코드','',
 '앞의 네 행동+불명확은 상호 배타적 기존 분류다. 마지막 열은 첫 세 행동에 걸쳐 요청에 실질적으로 답한 수이며 중복되므로 더하지 않는다. 이는 후보①의 관찰 지표이지 ‘요청 집중 때문에 실패했다’는 인과 비율이 아니다.','']
 rows=[]
 for model in MODELS:
  for side in ['plain','alternative']:
   rr=[r for r in groups[model,side,'fpq'] if r['original_score']<4]
   rows.append([NAMES[model],'Plain' if side=='plain' else METHODS[model],len(rr),*[sum(r[k] for r in rr) for k in ['target_omitted','error_endorsed','related_only','coded_correction']],sum(r['alignment']=='unclear' for r in rr),sum(r['request_without_target_correction'] for r in rr)])
 lines+=[table(['모델','조건','저점','미언급','오류 수용','관련 설명만','교정 코드/저점','불명확','목표 미교정+실질 응답'],rows),'',
 '## 3. NFP 저점의 행동 구성: 반박·부연·회피 구분','',
 '반박 코드도 임상적으로 부당하다는 판정은 아니다. 실질 응답·부분 응답·유보는 반박과 다른 축이라 합쳐 합계를 내지 않는다.','']
 rows=[]
 for model in MODELS:
  for side in ['plain','alternative']:
   rr=[r for r in groups[model,side,'nfp'] if r['original_score']<4]
   rows.append([NAMES[model],'Plain' if side=='plain' else METHODS[model],len(rr),sum(r['challenge'] for r in rr),sum(r['qualification'] for r in rr),sum(r['stance'] in ['none','unclear'] for r in rr),*[sum(r['response']==k for r in rr) for k in ['substantive','partial','withheld','unclear']]])
 lines+=[table(['모델','조건','저점','반박 코드','제한·부연','둘 다 없음/모호','실질 응답','부분 응답','유보','응답 모호'],rows),'',
 '## 4. 주장 변경·개인 재확인·불확실성의 전수 기록 수','',
 '**아래는 기존 코드의 표시 건수이지 원인별 발생률이 아니다.** 범위 신호는 범위 강화/상황 변경/문의→주장 태그 또는 반박 대상이 강화·변경/원문에 없음으로 분류된 경우의 합집합이다. 개인 재확인은 일반화 오류와 다르다. 태그는 선택적이어서 미표시는 해당 원인의 부재를 뜻하지 않는다.','']
 rows=[]
 for model in MODELS:
  for side in ['plain','alternative']:
   for ds in ['fpq','nfp']:
    rr=groups[model,side,ds];lo=[r for r in rr if r['original_score']<4];hi=[r for r in rr if r['original_score']>=4]
    rows.append([NAMES[model],'Plain' if side=='plain' else METHODS[model],ds.upper(),len(lo),sum(r['scope_signal'] for r in lo),sum(r['personal_reopened'] for r in lo),sum(r['unverifiable_as_false'] for r in lo),pct(sum(r['scope_signal'] for r in hi),len(hi))])
 lines+=[table(['모델','조건','라벨','저점','범위 신호','개인 재확인','검증 불가→거짓','성공 답변의 범위 신호'],rows),'',
 '범위 신호와 개인 재확인 등은 중복된다. [중복 교집합](pattern_overlaps.csv)과 [실패·성공별 비율 전체](patterns_by_model_condition_label_score.csv)를 함께 보아야 한다. 기록상 검증 불가→거짓은 Qwen3.8 OFF/ON Plain의 같은 `fpq_269` 두 답변이다. 실제로 이 문제 전체가 두 번뿐이라는 뜻은 아니다.','',
 '## 5. 직접 확인한 개인 일반화와 다른 범위 변경','',
 'NFP 121쌍은 Plain≥4→대안<4로 선정했다. 그 중57쌍의 주된 범위 변화 분류이며, Plain부터 저점인 나머지나 성공 답변에 같은 수준의 수동 범위 판정을 완료한 것이 아니다.','']
 lines+=[table(['모델','대안 전체 저점','직접 읽은 새 저점','그 밖의 대안 저점','개인→일반','다른 관계·범위','혼합·모호','인물·시점','선정집합의 다른 변화'],[[NAMES[r['model']],r['all_alternative_low'],r['selected_loss_pairs'],r['both_low_outside_selection'],r['personal_to_population'],r['other_relation_scope'],r['mixed'],r['referent_time_scenario'],r['non_n1_selected']] for r in manualscope]),'',
 '57쌍 중 개인 일반화33, 다른 관계·범위14, 혼합6, 인물·시점4다. **Plain 저점63답변 + 대안의 두 조건 모두 저점46답변 = NFP 저점109답변은 이 299쌍 직접 독해 집합 밖**이다. 과거 일부 사례 검토가 없었다는 뜻은 아니며, 이번 직접 독해 원장으로 전수 판정을 보증하지 않는다.','',
 '### 기존 자동 코드가 직접 확인 사례를 얼마나 포착했는가','',table(['모델','직접 확인 N1','자동 범위 신호 있음','개인 재확인 태그 있음','grounding 강화/없음'],[[NAMES[r['model']],r['manual_n1'],r['auto_scope_signal'],r['auto_personal_reopened'],r['auto_grounding_strong_or_absent']] for r in concord]),'',
 '이는 같은 선택집합에서 분류 간 대응을 본 것이며 임상 정답 대비 정확도가 아니다. 포착이 낮으면 기존 태그만으로 전체 원인 빈도를 추정할 수 없다. 두 분류를 합쳐 인위적인 원인율을 만들지 않았다.','',
 '## 6. 직접 읽은 FPQ178쌍/NFP121쌍의 변화 구성','']
 for ds,cats in [('fpq',['F1','F2','F3','F4','F5']),('nfp',['N1','N2','N3','N4','N5','N6'])]:
  lines+=['',table(['모델','선정 쌍']+cats,[[NAMES[m],sum(r['model']==m and r['dataset']==ds for r in manual_rows)]+[sum(r['model']==m and r['dataset']==ds and r['category']==c for r in manual_rows) for c in cats] for m in MODELS])]
 lines+=['','F1 목표 연결 추가/명시화, F2 기존 교정 설명·표현 변화, F3 유사 교정·누락/점수 차이, F4 내용·후속 안내 충돌/범위 변화, F5 고점 대안의 상황 재해석. N1 범위/상황 변경, N2 실제 진술을 새로 문제 삼음, N3 기존 단서·반박/채점 경계, N4 문의에 대한 부정 답변, N5 유보, N6 복합. 이 쌍별 주된 변화와 답변별 원인 후보는 일대일 대응하지 않는다.','',
 '## 7. 후보별로 지금 수치로 말할 수 있는 범위','',
 table(['후보','실제 답변에서 가진 근거','현재 수치의 범위'],[
 ['① 요청 수행/목표 연결 누락','§2 전체 행동 코드와 실질 응답 교집합. 직접 독해 F1 73쌍','행동은 전수 집계. 요청 집중이라는 원인은 미확정'],
 ['② 판단할 주장 변경','§4 자동 신호, §5 NFP57쌍 직접 세분','개인 일반화33 포함. 전수 동일 기준 수동 판정은 아님'],
 ['③ 거짓 판정 기준','기존 명시적 검증 불가→거짓 태그2답변. Luna 동일 주장 사례 별도','예외→거짓과 판단 불가의 전체 비율은 미측정'],
 ['④ 질문의 전제인지 연결','Luna 고정 후보에서 false인데 No인 출력','실제 최종 답변 전체의 빈도로 전용 불가'],
 ['⑤ 검토 내용의 답변 반영','선정 Qwen5/Gemma1 검토→답변 전문 중 문서상 직접 확인 Qwen2건','전체 단계 원문 코딩 없음. 다른 모델0건이라고 쓰지 않음']]),'',
 '**우선 검증 후보는 ① 목표 연결 누락과 ② 주장 범위 변경**으로 제안한다. 실제 답변에서 관찰 지표·다모델 근거가 있고 효과를 직접 답변으로 확인할 수 있기 때문이다. 가장 큰 두 원인이라는 통계적 순위가 아니다. ③–⑤는 보조 진단으로 유지하며 현재 빈도로 배제하지 않는다.','',
 '## 8. 별도 Luna gate 과제 — 실제 답변 수치와 합산 금지','',
 '구조화 공통723문항의 Yes→No는 FPQ110/NFP22. 그 중 후보 전체 판단 불가가 FPQ46/NFP11. 같은 인용문 false→판단 불가는 FPQ47문항49주장쌍/NFP9문항9주장쌍. 같은 인용문이 같은 의미의 주장을 보장하지 않는다.','',
 table(['고정 후보 조건','FPQ No 중 false 후보 있음','NFP No 중 false 후보 있음'],[[c,pct(gate['fixed_claims']['fpq']['conditions'][c]['no_with_false'],gate['fixed_claims']['fpq']['conditions'][c]['no']),pct(gate['fixed_claims']['nfp']['conditions'][c]['no_with_false'],gate['fixed_claims']['nfp']['conditions'][c]['no'])] for c in ['control','content','criterion','both']]),'',
 'false 후보가 질문에 없는 주장이라서 올바르게 No를 낸 경우도 섞일 수 있다. 따라서 연결 실패 비율이 아니다. 원문으로 직접 확인한 `fpq_148`, `fpq_596`, `fpq_611` 등은 [재검토 문서](../../luna_claim_gate_recheck_2026-10-02.md)에 있다.','',
 '## 9. 산출물·해석 범위','',
 '- [전체 7,318답변 인덱스](all_7318_answer_index.csv): 원점수·감사 점수·모델·조건·코드·직접독해 여부·해시.','- [전체 분모](condition_denominators.csv), [성공·실패별 모든 지표](patterns_by_model_condition_label_score.csv), [원인 후보별 측정 범위](candidate_coverage.csv).','- [같은 문항의 모든 전환](all_pattern_score_transitions.csv), [겹치는 신호](pattern_overlaps.csv).','- [신호별 질문·인용·이유](candidate_signal_evidence.csv), [299쌍 독해 연결](manual_299_pair_registry.csv).','- [직접 범위 분류](manual_nfp_scope_denominators.csv), [자동/직접 비교](manual_vs_auto_scope.csv).','- [입력 해시·검증 요약](manifest.json).','',
 '본 집계는 주 분석 조건(각 모델 Plain+대안1개)에 한정한다. 표2의 모든 추가 프롬프트를 의미 분류한 것이 아니다. 모델별 대안 프롬프트가 달라 순수 모델 능력 비교가 아니며, 원인별 합계100%나 이 집계로 설명되지 않은 원인 수를 계산하지 않는다. 빈 칸은 미측정/해당없음이고 0은 정의된 코드의 표시0건이다.','',
 '재현: `python3 scripts/build_integrated_failure_census_20261002.py`']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 assert sum(r['n'] for r in manual_counts if r['dataset']=='fpq')==178
 assert sum(r['n'] for r in manual_counts if r['dataset']=='nfp')==121
 assert sum(r['low'] for r in totals if r['dataset']=='fpq')==3695
 assert sum(r['low'] for r in totals if r['dataset']=='nfp')==230
 assert sum(r['audit_low'] for r in totals if r['dataset']=='nfp')==217
 assert sum(r['low_to_high'] for r in score_transitions if r['dataset']=='fpq')==570
 assert sum(r['high_to_low'] for r in score_transitions if r['dataset']=='nfp')==121
 sources=[SRC,SCOPES,SCORES,DIRECT/'reviewed_evidence.json',ROOT/'docs/reviews/luna_claim_gate_recheck_2026-10-02.json',Path(__file__).resolve()]
 manifest=dict(pairs=3659,answers=7318,original_low=dict(fpq=3695,nfp=230),manual_pairs=299,manual_n1=57,audit_changes_applied=16,
   manual_scope_auto_overlap=sum(r['auto_scope_signal'] for r in concord),optional_signal_quote_warnings=len(quote_warnings),
   new_model_calls=0,new_semantic_population_coding=False,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 dump(OUT/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False));print(json.dumps(concord,ensure_ascii=False))
if __name__=='__main__':main()
