"""Explicit Sol extraction override; keep existing model/probe identities reusable."""
import fcntl,json
from pathlib import Path
from src.paper_baselines import sha,write_json
from scripts.paper_batch_inference import LockedCachedTask
from scripts.paper_cli_pool import PooledCLI
from scripts.paper_v4_pipeline import EvaluatedPipeline


class SharedSolExtraction:
    def __init__(self,worker):
        path=worker.args.out.parent/'protocol/extraction_sol.json'
        self.config={**json.loads(path.read_text()),'exact_well_messages':True}
        if self.config.get('backend')!='codex' or self.config.get('model')!='gpt-6-sol':
            raise ValueError('This protocol requires the explicitly requested GPT-6 Sol')
        self.out=worker.args.out/'shared/sol_extraction';self.out.mkdir(parents=True,exist_ok=True)
        identity={'purpose':'shared_premise_extraction_sol_v1','extractor':self.config,
            'registry_sha256':worker.identity['registry_sha256'],'well_source_hashes':worker.identity['well_source_hashes'],
            'backend_sha256':sha(Path(__file__).resolve().parents[1]/'src/fpqa_cli_backend.py')}
        with (self.out/'.identity.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);write_json(self.out/'identity.json',identity,immutable=True)
        self.call=LockedCachedTask(PooledCLI(self.config,worker.args.out,worker.out/'STOP'),self.out,identity,worker.reg['transport_common'])


class SolPipeline(EvaluatedPipeline):
    def run(self,method,*args,**kwargs):
        result=super().run(method,*args,**kwargs)
        if method in ('prewome','extract_verify'):
            result['extraction_protocol']='shared_premise_extraction_sol_v1'
        return result
