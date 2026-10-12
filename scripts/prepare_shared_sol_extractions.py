"""Precompute question-only Sol premises; separate namespace from archived Luna extraction."""
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper_sol_extraction import SharedSolExtraction
from src.paper_baselines import WellBridge,write_json
from scripts.run_paper_baselines import check_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    cfg=json.loads(a.config.read_text());b=check_bundle(a.bundle)
    worker=SimpleNamespace(args=a,out=a.out/'shared/sol_extraction',identity=json.loads((a.out/'qwen25/identity.json').read_text()),reg=json.loads(Path(cfg['registry']).read_text()))
    e=SharedSolExtraction(worker);bridge=WellBridge(cfg);lookup={r['id']:r for r in b['rows']}
    items=[(b['splits'][0]['dataset'],lookup[b['splits'][0]['train'][0]])] if a.smoke else [(s['dataset'],lookup[i]) for s in b['splits'] for i in s['test']]
    state={'state':'running','completed':0,'total':len(items),'concurrency':4,'model':'gpt-6-sol','effort':e.config['effort'],'scope':'train smoke' if a.smoke else 'FPQA evaluation; question only, no labels/corrections'}
    status=e.out/('smoke.json' if a.smoke else 'progress.json');write_json(status,state)
    def one(item):
        source,row=item
        stage=bridge(op='render',family='prewome',template='PresuppositionExtractionTemplate',kwargs={'question':row['question'],'passages':[],'few_shot_data':worker.reg['fixed_examples'][source]['examples']})
        raw=e.call(stage['messages']);claims=bridge(op='parse',parser=stage['parser'],text=raw)
        if not isinstance(claims,list):raise ValueError('Expected a premise list')
        return row['id'],len(claims)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            it=iter(items);pending={pool.submit(one,next(it)) for _ in range(min(4,len(items)))}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    state['last_id'],state['last_claim_count']=future.result();state['completed']+=1;state['updated_epoch']=time.time();write_json(status,state)
                    if (worker.out/'STOP').exists():raise RuntimeError('STOP requested')
                    item=next(it,None)
                    if item is not None:pending.add(pool.submit(one,item))
        state['state']='complete';write_json(status,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));write_json(status,state);raise

if __name__=='__main__':main()
