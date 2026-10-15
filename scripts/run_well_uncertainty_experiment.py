#!/usr/bin/env python3
"""Luna paired response experiment: frozen A, B, C0, C1; resumable, no gold in inference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.fpqa_prompting import freeze, load_rows, response_metrics
from src.fpqa_well_upstream import UncachedCaller, FatalTransportError, load_upstream, make_args, verify_source
from src.fpqa_well_pipelines import load_templates, answer
from src.fpqa_uncertainty import verify_messages, parse_verdict, final_messages, Retriever, VERIFY_SYSTEM, ANSWER_ACTION

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'results/fpqa_prompting/well_pipelines_luna6_20261004')
    p.add_argument('--corpus',type=Path,default=ROOT/'results/fpqa_prompting/nci_common_corpus_20261004_v2')
    p.add_argument('--test',type=Path,default=ROOT/'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl')
    p.add_argument('--well-root',type=Path,default=ROOT/'external/fpqa_20261002/Well')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if not 1<=a.workers<=4: p.error('workers 1..4')
    out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'STOP').exists(): raise RuntimeError('STOP exists')
    original=json.loads((a.source/'run.json').read_text()); config=original['config']
    if sha(a.test)!=original['test_sha256']: raise ValueError('Test file changed')
    rows=[r for r in load_rows(a.test) if r['id'] not in original['fewshot_overlap_ids']]
    if Counter(r['label'] for r in rows)!={1:99,0:100}: raise ValueError('Need matched 99/100')
    sources={}; hashes={}
    for r in rows:
        path=a.source/'items'/r['id']/'atomic/record.json'
        rec=json.loads(path.read_text())
        if rec['question']!=r['question'] or rec['label']!=r['label'] or rec['unparseable_check_indices']:
            raise ValueError('Source mismatch / invalid baseline')
        sources[r['id']]=rec; hashes[r['id']]=sha(path)
    corpus_manifest=json.loads((a.corpus/'manifest.json').read_text())
    if sha(a.corpus/'chunks.jsonl')!=corpus_manifest['chunks_sha256']: raise ValueError('Corpus changed')
    retriever=Retriever(load_rows(a.corpus/'chunks.jsonl'))
    source=verify_source(a.well_root)
    if source!=original['judge_source_sha256']: raise ValueError('Well judge source changed')
    local=threading.local(); stopped=threading.Event()
    balanced,_,_=load_upstream(a.well_root,lambda role,messages:local.judge(role,messages))
    templates=load_templates(a.well_root)
    few=json.loads((a.source/'few_shot_data.json').read_text())
    args=make_args(balanced,'CancerMyth',out/'unused',out,config)
    from constant.constant import MODEL_FINAL_ANSWER_KEY
    repeat_ids=[]
    for label in [1,0]:
        eligible=[r['id'] for r in rows if r['label']==label]
        repeat_ids+=sorted(eligible,key=lambda s:hashlib.sha256(('repeat-v1:'+s).encode()).hexdigest())[:10]
    plan={'protocol':'uncertainty_response_v1','config':config,'source':str(a.source.resolve()),
          'test_sha256':sha(a.test),'ids':[r['id'] for r in rows], 'counts':{'fpq':99,'nfp':100},
          'baseline_sha256':hashes,'well_source_sha256':source,'corpus_manifest':corpus_manifest,
          'repeat_A_ids':repeat_ids,'methods':['A_saved','B','C0','C1','A_repeat_20'],
          'verifier_system':VERIFY_SYSTEM,'answer_action':ANSWER_ACTION,
          'source_sha256':{f:sha(ROOT/f) for f in ['src/fpqa_uncertainty.py',
              'scripts/run_well_uncertainty_experiment.py','src/fpqa_cli_backend.py','src/fpqa_well_upstream.py',
              'src/fpqa_well_pipelines.py']},
          'retrieval':'BM25-like deterministic top4, <=2 chunks/page, question + 2x claim; common NCI corpus',
          'primary_contrast':'C1 vs C0: same eligible claims, previous B, verifier and response instructions; passages differ',
          'A_B_confound':'B bundles question context, ternary/rationale/scope judgments and response policy. NOT isolated ternary effect.',
          'selection':'199 previously inspected questions; exploratory development, not untouched heldout',
          'no_eligible_policy':'C0/C1 reuse B verdicts, answer and judge when all claims supported; no unnecessary calls',
          'judge':'unchanged Well prompts, zero judge examples, method name hidden; same Luna model as generator',
          'retrieval_limit':'NCI-only fixed corpus may lack relevant evidence; not original Well RAG reproduction'}
    freeze(out/'run.json',plan)
    freeze(out/'prompts.json',{'verification_system':VERIFY_SYSTEM,'answer_action':ANSWER_ACTION})
    for r in rows: freeze(out/'items'/r['id']/'A_saved/record.json',sources[r['id']])
    if a.preflight:
        print(json.dumps({'counts':plan['counts'],'corpus_pages':corpus_manifest['pages'],
                          'claims':sum(len(x['claims']) for x in sources.values()),'repeat_A':20})); return

    def status(state):
        data={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),
              'target_primary_records':597,'methods':{},'errors':len(list((out/'items').glob('*/errors/*.json')))}
        for method in ['A_saved','B','C0','C1','A_repeat']:
            recs=[json.loads(q.read_text()) for q in (out/'items').glob(f'*/{method}/record.json')]
            ids={r['id'] for r in recs}
            counts={}
            for label,name in [(1,'fpq'),(0,'nfp')]:
                rr=[r for r in recs if r['label']==label]
                counts[name]={'n':len(rr),'ge4':sum(r['score']>=4 for r in rr),
                    'gain_vs_A':sum(r['score']>=4 and sources[r['id']]['score']<4 for r in rr),
                    'loss_vs_A':sum(r['score']<4 and sources[r['id']]['score']>=4 for r in rr)}
            data['methods'][method]={'completed':len(recs),'counts':counts,'metrics':response_metrics(recs),
                'matched_A':response_metrics([sources[i] for i in ids])}
        for method in ['C0','C1']:
            transitions=Counter()
            for q in (out/'items').glob(f'*/{method}/record.json'):
                r=json.loads(q.read_text()); b=json.loads((q.parent.parent/'B/record.json').read_text())
                for v,w in zip(b['verdicts'],r['verdicts']):
                    transitions[f"{r['label']}:{v['verdict']}->{w['verdict']}"]+=1
            data['methods'][method]['claim_transitions']=dict(transitions)
        tmp=out/'status.tmp'; tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/'status.json')
        return data

    def process(row):
        folder=out/'items'/row['id']; baseline=sources[row['id']]; claims=baseline['claims']
        def call(stage,role,messages):
            if stopped.is_set() or (out/'STOP').exists(): raise FatalTransportError('STOP requested')
            loc=folder/'stages'/stage
            freeze(loc/'request.json',{'role':role,'messages':messages,'config':config[f'{role}_model']})
            if (loc/'result.json').exists(): return json.loads((loc/'result.json').read_text())['text']
            attempts=loc/'attempts'; attempt=attempts/f'{len(list(attempts.glob("*")))+1:03d}'
            raw=UncachedCaller(config,attempt)(role,messages)
            if not raw.strip(): raise ValueError('Empty output')
            freeze(loc/'result.json',{'text':raw}); return raw
        def score(method,generated,verdicts=None,extra=None):
            local.judge=lambda role,messages:call(method+'/judge',role,messages)
            evaluator=balanced.BalancedDirectQAGEPAEvaluator(args,'gemini')
            evaluator.ScoreTemplateClass=evaluator.template_classes['fpq' if row['label'] else 'tpq']
            dp={'id':row['id'],'question':row['question'],'answer':row['reference_answer'],
                'presuppositions':row['presuppositions'],'few_shot_data':[],'passages':[],
                MODEL_FINAL_ANSWER_KEY:generated['answer']}
            value,explanation=evaluator.score_response(dp)
            rec={'id':row['id'],'label':row['label'],'question':row['question'],'claims':claims,
                 'method':method,**generated,'verdicts':verdicts,'score':value,'score_explanation':explanation,**(extra or {})}
            freeze(folder/method/'record.json',rec)
            print(f"COMPLETE {row['id']} {method} score={value}",flush=True)
            return rec
        def generate(method,vs):
            t=templates['atomic'].FactCheckFinalAnswerTemplate(question=row['question'],few_shot_data=few,
                       factcheck_results=[1]*len(claims),model_detected_presuppositions=claims)
            raw=call(method+'/answer','task',final_messages(row['question'],claims,vs,t))
            return {'raw_answer':raw,'answer':t.ResponseClass.model_validate_plain_text(raw).get()}
        try:
            bpath=folder/'B/record.json'
            if bpath.exists(): b=json.loads(bpath.read_text())
            else:
                vs=[]
                for i,c in enumerate(claims):
                    raw=call(f'B/check_{i:03d}','task',verify_messages(row['question'],c))
                    vs.append(parse_verdict(raw))
                b=score('B',generate('B',vs),vs)
            eligible=[i for i,v in enumerate(b['verdicts']) if v['verdict']!='supported']
            # Counterbalance execution order by ID hash, fixed before any results.
            order=['C0','C1'] if int(hashlib.sha256(row['id'].encode()).hexdigest(),16)%2 else ['C1','C0']
            for method in order:
                path=folder/method/'record.json'
                if path.exists(): continue
                if not eligible:
                    freeze(path,{**b,'method':method,'reused_from':'B','eligible_indices':[]})
                    continue
                vs=[dict(v) for v in b['verdicts']]
                for i in eligible:
                    passages=retriever.retrieve(row['question'],claims[i]) if method=='C1' else []
                    if method=='C1': freeze(folder/'retrieval'/f'{i:03d}.json',
                        {'question':row['question'],'claim':claims[i],'passages':passages})
                    raw=call(f'{method}/check_{i:03d}','task',verify_messages(row['question'],claims[i],b['verdicts'][i],passages))
                    vs[i]=parse_verdict(raw,passages)
                score(method,generate(method,vs),vs,{'eligible_indices':eligible})
            if row['id'] in repeat_ids and not (folder/'A_repeat/record.json').exists():
                generated=answer(row['question'],few,claims,'atomic',templates,
                                 lambda stage,role,messages:call('A_repeat/'+stage,role,messages))
                if generated['unparseable_check_indices']: raise ValueError('Invalid repeated A verdict')
                score('A_repeat',generated)
        except BaseException as e:
            stopped.set()
            errors=folder/'errors'; freeze(errors/f'{len(list(errors.glob("*")))+1:03d}.json',
                                         {'type':type(e).__name__,'message':str(e)})
            raise

    # Deterministic label interleave avoids misleading all-FPQ early progress.
    groups=[[r for r in rows if r['label']==y] for y in [1,0]]
    selected=[r for pair in zip(*groups) for r in pair]+groups[1][99:]
    if a.smoke: selected=[groups[0][0],groups[1][0]]
    status('running')
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(process,r) for r in selected]
            for future in as_completed(futures): future.result(); status('running')
        result=status('smoke_complete' if a.smoke else 'complete')
        if not a.smoke:
            if any(result['methods'][m]['completed']!=199 for m in ['B','C0','C1']): raise ValueError('Incomplete')
            if result['methods']['A_repeat']['completed']!=20: raise ValueError('Incomplete repeat')
            freeze(out/'metrics.json',result)
    except BaseException:
        stopped.set(); status('failed'); raise

if __name__=='__main__': main()
