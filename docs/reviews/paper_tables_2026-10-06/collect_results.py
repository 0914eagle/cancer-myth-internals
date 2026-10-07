"""Collect explicitly mapped existing results; no model calls. Requires original result files.
Use render_tables.py for regeneration from the committed snapshots alone.
"""
import json, csv, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[3]
O=R/'docs/reviews/paper_tables_2026-10-06'
(O/'sources').mkdir(exist_ok=True)
specs=json.loads((O/'table_specs.json').read_text())
Q='Qwen/Qwen2.5-7B-Instruct'; L='gpt-6-luna'; G='google/gemma-4-12B-it'; OFF='Qwen/Qwen3.8-27B-FP8:off'; ON='Qwen/Qwen3.8-27B-FP8:on'
sources={}; cells=[]
def source(name,path):
 p=R/path; b=p.read_bytes(); data=json.loads(b)
 (O/'sources'/f'{name}.json').write_bytes(b)
 sources[name]={'original_path':path,'snapshot':f'sources/{name}.json','sha256':hashlib.sha256(b).hexdigest()}
 return data
ans=source('qwen_answers','docs/reviews/advisor_presentation_2026-09-23/answer_summary.json')
gates=source('default_gates','docs/reviews/default_gate_decisions_2026-09-23/summary.json')
routes=source('routing','docs/reviews/table2_completion_2026-10-01/routing_summary.json')
gemma=source('gemma','results/gemma4/full_20260926_v2/analysis/summary.json')
off=source('qwen38_off','results/qwen38/full_transformers_20260926_v1/thinking_off/analysis/summary.json')
on=source('qwen38_on','results/qwen38/full_transformers_20260926_v1/thinking_on/analysis/summary.json')
pipeline=source('luna_pipelines','results/fpqa_prompting/well_pipelines_luna6_20261004/metrics.json')
test=source('luna_test','results/fpqa_prompting/well_upstream_test_v1_20261003_run/metrics.json')
lora=source('lora','docs/reviews/advisor_presentation_2026-09-23/lora_summary.json')
cohorts={
 'H':{'description':'Historical Cancer-Myth full corpus (583 FPQ / 149 NFP); valid-score denominators vary. Original stored scoring, no retrospective partial audit replacements. Different model prompts/runtimes; not a controlled cross-model ranking.','judge':'claude-sonnet-5 (legacy Well scoring; see original score provenance)','evaluation_status':'historical_exploratory','repeat':'original single generation; not repeat mean'},
 'L':{'description':'GPT-6 Luna Cancer-Myth shared 199 IDs: 99 FPQ / 100 NFP. Excludes fpq_291 overlapping pipeline few-shot examples from every method. Already inspected during development; not a fresh holdout.','judge':'gpt-6-luna / Codex / medium','evaluation_status':'exposed_test_exploratory','repeat':'original single generation; not repeat mean'},
 'R':{'description':'GPT-6 Luna CREPE full stored test: 751 FPQ / 2253 normal questions; frozen Plain and GEPA evaluation.','judge':'gpt-6-luna / Codex / medium','evaluation_status':'stored_test','repeat':'original single generation; not repeat mean'},
 'T':{'description':'Legacy Qwen LoRA evaluation: same 234 FPQ / 149 natural NFP, excluding synthetic twins from reported NFP. Training used FPQ and synthetic corrected twins, not natural NFP.','judge':'legacy Well scoring (original archived scores)','evaluation_status':'legacy_training_evaluation','repeat':'one checkpoint / single generation; not repeat mean'}
}
def add(table,model,method,col,num,den,src,pointer,cohort,note='',expected=None):
 if table=='table_s1' and method=='plain':method='plain_lora_cohort'
 if table=='table_s3':table='table1'
 elif table in ['table_s1','table_s2']:table='table2'
 assert isinstance(num,int) and isinstance(den,int) and 0<=num<=den and den>0
 cells.append(dict(table=table,model_id=model,method=method,metric=col,numerator=num,denominator=den,value_pct=100*num/den,cohort=cohort,source=src,pointer=pointer,status='historical_observed',expected_n=expected if expected is not None else den,missing_n=(expected-den) if expected is not None else 0,note=note))
def response(table,model,method,d,src,prefix,cohort,count='ge4',den='valid',note=''):
 for group in ['fpq','nfp']:
  a=d[group]; add(table,model,method,f'cancer_{group}_well_ge4_pct',a[count],a[den],src,prefix+[group],cohort,note,a.get('expected',a[den]))
# Historical detection at actual default decision rule, NOT calibrated-FPR summaries.
for k,m in [('direct','direct_gate'),('review','premise_review_cot')]:
 a=gates[k]
 add('table1',Q,m,'cancer_fpq_tpr_pct',a['tp'],a['tp']+a['fn'],'default_gates',[k],'H','Yes score > No score; ties negative.')
 add('table1',Q,m,'cancer_nfp_fpr_pct',a['fp'],a['fp']+a['tn'],'default_gates',[k],'H','Yes score > No score; ties negative.')
for model,src,d in [(G,'gemma',gemma),(OFF,'qwen38_off',off),(ON,'qwen38_on',on)]:
 for k,m in [('direct','direct_gate'),('cot','premise_review_cot')]:
  for gr,metric in [('fpq','cancer_fpq_tpr_pct'),('nfp','cancer_nfp_fpr_pct')]:
   a=d['gate'][k][gr]; add('table1',model,m,metric,a['positive'],a['valid'],src,['gate',k,gr],'H','Generated JSON boolean decision.',a['expected'])
 for k,m,t in [('plain','plain','table2'),('premise_review_answer','premise_review_cot','table2'),('zero_shot_cot_one_step','cot','table_s2'),('fp_unconditional','always_correct','table_s2'),('direct_routed','direct_gate_response','table_s2'),('cot_routed','cot_gate_response','table_s2')]:
  response(t,model,m,d['answers'][k],src,['answers',k],'H',count='at_least_4_count',note='Stored Sonnet Well scoring. Routed rows select saved Plain / unconditional-correction answers; no new generation.')
for k,m,t in [('plain','plain','table2'),('premise_review','premise_review_cot','table2'),('zero_shot_cot','cot','table_s2'),('fp_unconditional','always_correct','table_s2')]:
 response(t,Q,m,ans['qwen/'+k],'qwen_answers',['qwen/'+k],'H',note='Original valid-score denominator; absent score is not silently counted as failure.')
for model,key in [(Q,'qwen'),(G,'gemma'),(OFF,'qwen38_off'),(ON,'qwen38_on')]:
 response('table_s2',model,'text_gate_response',routes['results'][key+'/text'],'routing',['results',key+'/text'],'H',den='n',note='Default crossfit text classifier selects stored answers.')
for key,m in [('direct','direct_gate_response'),('review','cot_gate_response')]:
 response('table_s2',Q,m,routes['results']['qwen/'+key],'routing',['results','qwen/'+key],'H',den='n')
for key,model,m in [('hidden',Q,'hidden_probe'),('text','model_independent','tfidf')]:
 a=gates[key]
 for metric,num,den in [('cancer_fpq_tpr_pct',a['tp'],a['tp']+a['fn']),('cancer_nfp_fpr_pct',a['fp'],a['fp']+a['tn'])]:
  add('table_s3',model,m,metric,num,den,'default_gates',[key],'H','Default crossfit classifier decision, not a threshold selected to match FPR.')
# Same 199-item cohort for every Luna Cancer method.
for original,m in [('prewome','prewome_style'),('atomic','extract_verify')]:
 d=pipeline['methods'][original]['primary_excluding_overlap']
 for gr in ['fpq','nfp']:
  a=d[gr]; num=a['scores']['4']+a['scores']['5']
  add('table2',L,m,f'cancer_{gr}_well_ge4_pct',num,a['n'],'luna_pipelines',['methods',original,'primary_excluding_overlap',gr],'L','Well official four-shot / no RAG / shared extraction. Not original PreWoMe complete reproduction.')
for m in ['plain','gepa']:
 d=pipeline['methods']['prewome']['matched_reference_on_completed_primary_ids'][m]
 assert d==pipeline['methods']['atomic']['matched_reference_on_completed_primary_ids'][m]
 for gr in ['fpq','nfp']:
  a=d[gr];add('table2',L,m,f'cancer_{gr}_well_ge4_pct',a['scores']['4']+a['scores']['5'],a['n'],'luna_pipelines',['methods','prewome','matched_reference_on_completed_primary_ids',m,gr],'L','Matched original frozen test answers; zero-shot, unlike pipeline four-shot.')
  a=test['conditions']['crepe/'+m]['metrics'][gr]
  metric='crepe_fpq_well_ge4_pct' if gr=='fpq' else 'crepe_normal_well_ge4_pct'
  add('table2',L,m,metric,a['scores']['4']+a['scores']['5'],a['n'],'luna_test',['conditions','crepe/'+m,'metrics',gr],'R','Full CREPE test; unrounded counts, not optimization/dev score.')
# Scope-preservation variants are archived outside the main tables.
response('table_s2',Q,'hidden_gate_response',routes['results']['qwen/hidden'],'routing',['results','qwen/hidden'],'H',den='n')
# Legacy SFT/DPO and matched Plain are excluded from displayed tables by user request.
# Source runtime manifest hashes / projections: no credentials or huge response dumps.
runtimes={}
for key,path in [('luna_test','results/fpqa_prompting/well_upstream_test_v1_20261003_run/run.json'),('luna_pipelines','results/fpqa_prompting/well_pipelines_luna6_20261004/run.json'),('gemma','results/gemma4/full_20260926_v2/plan.json'),('qwen38_off','results/qwen38/full_transformers_20260926_v1/thinking_off/plan.json'),('qwen38_on','results/qwen38/full_transformers_20260926_v1/thinking_on/plan.json')]:
 b=(R/path).read_bytes();d=json.loads(b);runtimes[key]={'path':path,'sha256':hashlib.sha256(b).hexdigest(),'fields':{k:v for k,v in d.items() if k in ['model','revision','dtype','thinking_enabled','do_sample','protocol','config','configs','test_ids','fewshot_overlap_ids','n_primary_fpq','n_primary_nfp','provenance']}}
(O/'sources/runtime_provenance.json').write_text(json.dumps(runtimes,ensure_ascii=False,indent=2)+'\n')
# Presentation notes and row layout are maintained in table_specs.json.
(O/'table_specs.json').write_text(json.dumps(specs,ensure_ascii=False,indent=2)+'\n')
result={'updated':'2026-10-07','metric_definition':'100 * numerator / valid denominator; display rounded to one decimal; no new inference','cohorts':cohorts,'sources':sources,'cells':cells}
(O/'measured_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('populated',len(cells),'cells')
