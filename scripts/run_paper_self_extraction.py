"""Separate, resumable same-model extraction control; never overwrites Sol runs."""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import Pipeline,sha,write_json
from scripts.paper_v4_pipeline import EvaluatedPipeline

class SelfExtractionPipeline(EvaluatedPipeline):
    def well_stage(self,family,template,**kwargs):
        # Extraction and downstream stages use the same worker's locked cache.
        return Pipeline.well_stage(self,family,template,**kwargs)
    def run(self,method,*args,**kwargs):
        result=super().run(method,*args,**kwargs)
        if method in ('prewome','extract_verify'):
            result.update(extraction_model=self.worker.config['models'][self.worker.args.model]['model_id'],
                          extraction_protocol='same_model_extraction_v1')
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['init','run'])
    for key in ('config','bundle','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--model',choices=['qwen25','gemma3','luna6'],required=True)
    p.add_argument('--table',choices=['table2','table3'],default='table2')
    a=p.parse_args();a.phase='all';a.methods='prewome,extract_verify';a.generate_only=True;a.detect_first=False
    a.max_batch={'qwen25':16,'gemma3':32,'luna6':None}[a.model]
    a.token_budget={'qwen25':147456,'gemma3':393216,'luna6':None}[a.model]
    cfg=json.loads(a.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES']='1,2,3' if a.model=='gemma3' else ','.join(map(str,cfg['models'][a.model]['gpus']))
    from scripts.run_paper_baselines_recovery import install
    from scripts import run_paper_baselines_v4 as runner
    install();runner.EvaluatedPipeline=SelfExtractionPipeline
    runner.SharedLunaExtraction=lambda worker:SimpleNamespace(config={'model':worker.config['models'][worker.args.model]['model_id']}) # metadata only
    worker=runner.V4Worker(a)
    worker.identity['self_extraction_wrapper_sha256']=sha(Path(__file__))
    write_json(worker.out/'self_extraction_policy.json',{
        'model':a.model,'extraction_protocol':'same_model_extraction_v1',
        'methods':['prewome','extract_verify'],'shared_within_model':True,
        'bundle_sha256':worker.bundle['content_sha256'],
        'wrapper_sha256':sha(Path(__file__)),'max_batch':a.max_batch,
        'preserve_sol_output':True},immutable=True)
    if a.command=='init':return
    if a.table=='table2':worker.bundle={**worker.bundle,'qa':[]}
    else:
        # Skip the completed Table 2 stage in memory, preserving the frozen bundle.
        worker.bundle={**worker.bundle,'splits':[{**s,'test':[]} for s in worker.bundle['splits']]}
    worker.execute()

if __name__=='__main__':main()
