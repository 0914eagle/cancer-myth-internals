"""Snapshot current 4090 results, preserving historical tables. No model calls.
python3 update_current_4090.py --collect ; python3 update_current_4090.py
"""
import argparse,collections,csv,datetime,hashlib,json
from pathlib import Path
from copy import deepcopy
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[2]
OUT=BASE/'current_4090'
RUN=ROOT/'results/paper_baselines_v3/main'
MODELS={'qwen25':('Qwen/Qwen2.5-7B-Instruct','Qwen2.5\n7B'), 'gemma3':('google/gemma-3-12b-it','Gemma 3\n12B')}
METHODS={'plain':'plain','balanced':'balanced','cot':'cot','prewome':'prewome_style','extract_verify':'extract_verify','tfidf_gate':'text_gate_response','direct_gate':'direct_gate_response','cot_gate':'cot_gate_response','probe_gate':'hidden_gate_response','gepa_gate':'gepa_gate','gepa':'gepa','gepa_both':'gepa_both'}

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def hashed(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def collect():
    bundle_path=ROOT/'results/paper_baselines_v3/protocol/full_bundle.json'
    b=json.loads(bundle_path.read_text())
    snapshot={'captured_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_run':str(RUN.relative_to(ROOT)), 'bundle_sha256':hashed(bundle_path),
      'qa_sizes':dict(collections.Counter(r['dataset'] for r in b['qa'])),
      'splits':[{'dataset':s['dataset'],'name':s['name'],'n':len(s['test'])} for s in b['splits']],
      'models':{}}
    lookup={r['id']:r for r in b['rows']}
    for model in MODELS:
        root=RUN/model
        result={'identity_sha256':hashed(root/'identity.json'),'fpqa':[],'qa':[]}
        for split in b['splits']:
            for method in METHODS:
                folder=root/'table12'/split['dataset']/split['name']/method
                records=[json.loads(f.read_text()) for f in folder.glob('*.json')]
                for label in [1,0]:
                    scored=[r for r in records if r['label']==label and 'rating' in r]
                    result['fpqa'].append({'dataset':split['dataset'],'split':split['name'],'method':method,
                      'label':label,'expected':sum(lookup[i]['label']==label for i in split['test']),
                      'generated':sum(r['label']==label for r in records),'scored':len(scored),
                      'ge4':sum(r['rating']>=4 for r in scored)})
        # Only shared Plain is fully evaluated at the time of this update.
        for dataset,n in snapshot['qa_sizes'].items():
            records=[json.loads(f.read_text()) for f in (root/'table3/shared/fixed/plain').glob('*.json')]
            records=[r for r in records if r['dataset']==dataset]
            failures=[json.loads(f.read_text()) for f in (root/'failures/table3/shared/fixed/plain').glob('*.json')]
            failures=[r for r in failures if r['dataset']==dataset]
            assert not ({r['id'] for r in records}&{r['id'] for r in failures})
            result['qa'].append({'dataset':dataset,'method':'plain','expected':n,'scored':len(records),
              'correct':sum(r['correct'] for r in records),'failures':len(failures),
              'complete':len(records)+len(failures)==n})
        snapshot['models'][model]=result
    dump(OUT/'current_snapshot.json',snapshot)

def render():
    from render_tables import render_landscape  # also registers original fonts
    snap=json.loads((OUT/'current_snapshot.json').read_text())
    specs=json.loads((BASE/'table_specs.json').read_text());cells={}
    for name,spec in specs.items():
        spec['models']=[{'id':i,'display':d} for i,d in MODELS.values()]
        for method in spec['methods']:
            method.pop('models',None)
            if method['key']=='tfidf':
                method['key']='tfidf_gate'
                method['label']='TF-IDF text (shared)'
        spec['methods']=[m for m in spec['methods'] if m['key']!='ours']
    specs['table1']['title']='Table 1. Detection baselines - current 4090 protocol (pending)'
    specs['table1']['notes']=['Qwen2.5-7B / Gemma 3-12B; no compatible detection results yet.',
       'Dash = pending. Historical model/protocol results are not substituted.',
       'Cancer-Myth: grouped 3-fold; CREPE: official test split.']
    specs['table2']['title']='Table 2. Interim response results - Cancer-Myth fold 0 only'
    progress=[]
    for model, info in snap['models'].items():
        r=[x for x in info['fpqa'] if x['dataset']=='cancer_myth' and x['split']=='fold0' and x['method']=='balanced']
        progress.append(model+': '+str(sum(x['scored'] for x in r))+'/'+str(sum(x['expected'] for x in r)))
    specs['table2']['notes']=['Well score >= 4 (%), Sonnet 5.5. Fold 0: 193 FPQ / 49 NFP; NOT final 3-fold results.',
       'Only fully scored fold rows are shown; partial rows stay blank.',
       'Balanced scored in fold 0 - '+'; '.join(progress)+'.',
       'CREPE and remaining methods pending. No historical measurements mixed into this table.']
    for group in specs['table2']['groups']:
        if group['label']=='Cancer-Myth':group['label']='Cancer-Myth (fold 0)'
    # Do not compare unequal partial cohorts: only display complete fold rows.
    for model,info in snap['models'].items():
        modelid=MODELS[model][0]
        for method in METHODS:
            rr=[r for r in info['fpqa'] if r['dataset']=='cancer_myth' and r['split']=='fold0' and r['method']==method]
            if len(rr)==2 and all(r['scored']==r['expected'] for r in rr):
                for r in rr:
                    key='cancer_fpq_well_ge4_pct' if r['label']==1 else 'cancer_nfp_well_ge4_pct'
                    cells[('table2',modelid,METHODS[method],key)]={'value_pct':100*r['ge4']/r['expected']}
        qa_keys={'medqa':'medqa_accuracy_pct','pubmedqa':'pubmedqa_accuracy_pct','medbullets':'medbullets_accuracy_pct','mmlu':'mmlu_accuracy_pct','gsm8k':'gsm8k_em_pct'}
        for r in info['qa']:
            if r['complete']:cells[('table3',modelid,'plain',qa_keys[r['dataset']])]={'value_pct':100*r['correct']/r['expected']}
    specs['table3']['title']='Table 3. Medical and general QA - current 4090 results'
    specs['table3']['notes']=['Accuracy / numeric EM (%), generated answers; full dataset denominators.',
       'MedQA 1,273; PubMedQA labeled pool 1,000 (not official 500-item test); Medbullets 308.',
       'MMLU 14,042; GSM8K 1,319. MMLU length failures: Qwen 2, Gemma 7, counted unsuccessful.',
       'Plain is shared across adaptation sources/folds. Other methods pending; no Sonnet judge for QA.']
    dump(OUT/'table_specs.json',specs)
    dump(OUT/'displayed_cells.json',[{'table':k[0],'model':k[1],'method':k[2],'metric':k[3],**v} for k,v in cells.items()])
    for name,spec in specs.items():
        render_landscape(OUT,name,spec,cells)
        cols=[c for g in spec['groups'] for c in g['columns']]
        with (OUT/(name+'.csv')).open('w') as f:
            writer=csv.writer(f);writer.writerow(['method']+[m['display'].replace('\n',' ')+' / '+c['key'] for m in spec['models'] for c in cols])
            for method in spec['methods']:
                writer.writerow([method['label']]+[f"{cells[(name,m['id'],method['key'],c['key'])]['value_pct']:.1f}" if (name,m['id'],method['key'],c['key']) in cells else '' for m in spec['models'] for c in cols])
    lines=['# Current 4090 baseline tables (2026-10-12)','',
     'Qwen2.5-7B와 Gemma 3-12B만 포함한다. 과거 Gemma 4·Qwen3.8 결과는 상위 폴더에 그대로 보존한다.',
     'Table 2는 첫 fold 중간 결과이며 최종 3-fold 결과가 아니다. 전체 fold 집계는 원장 완성 후 교체한다.',
     '같은 fold에서 두 클래스가 모두 채점된 행만 표시한다. Balanced의 부분 점수는 비교 표에 넣지 않는다.',
     'Table 3의 Plain은 전체 평가 풀 처리가 끝났다. 생성 실패는 분모에 포함하고 정답으로 세지 않는다.',
     '일반 QA 채점은 정답 문자열/숫자 비교이며 Sonnet 채점이 아니다.','']
    for i in [1,2,3]:lines += [f'## Table {i}',f'![Table {i}](table{i}.png)',f'[PDF](table{i}.pdf) · [CSV](table{i}.csv) · [LaTeX](table{i}.tex)','']
    lines+=['## 완료된 정확한 집계','', '| Model | Dataset | Correct / Total | Generation failures |','|---|---|---:|---:|']
    for model,info in snap['models'].items():
        for r in info['qa']:lines.append(f"| {model} | {r['dataset']} | {r['correct']}/{r['expected']} | {r['failures']} |")
    lines+=['','원장: [current_snapshot.json](current_snapshot.json). 재생성: 상위 폴더의 `update_current_4090.py`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collect',action='store_true');a=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.collect:collect()
    render()
