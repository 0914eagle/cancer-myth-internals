"""Resume baseline answers with Sol as the shared extractor only.

Keep frozen generator/probe cache identities. Store the extractor override in its
own immutable identity and each affected output. Incompatible old pipeline rows
must be archived before this worker starts.
"""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_paper_baselines_v4 as runner
from scripts.run_paper_baselines_probe_prefetch import PrefetchWorker
from scripts.paper_sol_extraction import SharedSolExtraction,SolPipeline
from src.paper_baselines import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','smoke'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['qwen25','gemma3','luna6'],required=True);p.add_argument('--max-batch',type=int);p.add_argument('--token-budget',type=int)
    p.add_argument('--detect-first',action='store_true');p.add_argument('--methods');p.add_argument('--phase',default='all');p.add_argument('--generate-only',action='store_true',default=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,cfg['models'][a.model]['gpus']))
    for table in ('table12','table3'):
        for method in ('prewome','extract_verify'):
            for path in (a.out/a.model/table).glob(f'*/*/{method}/*.json'):
                if json.loads(path.read_text()).get('extraction_model')!='gpt-6-sol':raise ValueError('Archive incompatible extraction result first: '+str(path))
    runner.SharedLunaExtraction=SharedSolExtraction;runner.EvaluatedPipeline=SolPipeline
    worker=(PrefetchWorker if a.model=='gemma3' else runner.V4Worker)(a)
    override={'model':'gpt-6-sol','effort':json.loads((a.out.parent/'protocol/extraction_sol.json').read_text())['effort'],
        'scope':['prewome','extract_verify'],'other_methods_unchanged':True,'wrapper_sha256':sha(Path(__file__))}
    write_json(worker.out/'sol_extraction_override.json',override,immutable=True)
    worker.execute()

if __name__=='__main__':main()
