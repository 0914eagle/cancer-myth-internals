"""Precompute fixed FPQA premises once with Luna, independently of local GPUs."""
import argparse,json,sys,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.paper_shared_extraction import SharedLunaExtraction
from src.paper_baselines import WellBridge,write_json
from scripts.run_paper_baselines import check_bundle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());b=check_bundle(a.bundle)
    worker=SimpleNamespace(args=SimpleNamespace(out=a.out),out=a.out/'shared/luna_extraction',
        external=json.loads(Path(cfg['external_config']).read_text()),identity=json.loads((a.out/'qwen25/identity.json').read_text()),reg=json.loads(Path(cfg['registry']).read_text()))
    e=SharedLunaExtraction(worker);bridge=WellBridge(cfg);lookup={r['id']:r for r in b['rows']}
    items=[(s['dataset'],lookup[i]) for s in b['splits'] for i in s['test']]
    state={'state':'running','completed':0,'total':len(items),'concurrency':4,'model':'gpt-6-luna','scope':'FPQA evaluation only; no labels or corrections in extraction prompts'}
    status=e.out/'progress.json';write_json(status,state)
    def one(item):
        source,row=item
        stage=bridge(op='render',family='prewome',template='PresuppositionExtractionTemplate',kwargs={'question':row['question'],'passages':[],'few_shot_data':worker.reg['fixed_examples'][source]['examples']})
        raw=e.call(stage['messages']);claims=bridge(op='parse',parser=stage['parser'],text=raw)
        if not isinstance(claims,list):raise ValueError('Expected a premise list')
        return row['id']
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            it=iter(items);pending={pool.submit(one,next(it)) for _ in range(min(4,len(items)))}
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for f in done:
                    state['last_id']=f.result();state['completed']+=1;state['updated_epoch']=time.time();write_json(status,state)
                    if (worker.out/'STOP').exists():raise RuntimeError('STOP requested')
                    item=next(it,None)
                    if item is not None:pending.add(pool.submit(one,item))
        state['state']='complete';write_json(status,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));write_json(status,state);raise


if __name__=='__main__':main()
