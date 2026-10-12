"""One fixed Luna extractor shared across answer models and both Well methods."""
import fcntl
import json
from pathlib import Path
import time
from src.paper_baselines import Pipeline,sha
from scripts.paper_batch_inference import LockedCachedTask


class SharedLunaExtraction:
    def __init__(self,worker):
        from src.fpqa_cli_backend import CLIBackend
        self.config={**worker.external['task_model'],'exact_well_messages':True}
        if self.config.get('backend')!='codex' or self.config.get('model')!='gpt-6-luna':
            raise ValueError('Shared extraction must be explicitly GPT-6 Luna')
        self.out=worker.args.out/'shared/luna_extraction'
        self.out.mkdir(parents=True,exist_ok=True)
        identity={'purpose':'shared_premise_extraction_v1','extractor':self.config,
                  'registry_sha256':worker.identity['registry_sha256'],
                  'well_source_hashes':worker.identity['well_source_hashes'],
                  'backend_sha256':sha(Path(__file__).resolve().parents[1]/'src/fpqa_cli_backend.py')}
        from src.paper_baselines import write_json
        # No model-specific identity: Qwen/Gemma/Luna share exactly this artifact.
        with (self.out/'.identity.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            write_json(self.out/'identity.json',identity,immutable=True)
        cli=CLIBackend(self.config)
        def limited(messages):
            # Global, cross-process cap of four CLI requests, not GPU batch size.
            while True:
                if (worker.out/'STOP').exists():raise RuntimeError('STOP requested')
                for i in range(4):
                    lock=(self.out/f'.slot-{i}').open('a')
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock.close();continue
                    try:return cli(messages)
                    finally:lock.close()
                time.sleep(.1)
        self.call=LockedCachedTask(limited,self.out,identity,worker.reg['transport_common'])


class SharedExtractionPipeline(Pipeline):
    def __init__(self,registry,call,bridge,extractor):
        super().__init__(registry,call,bridge)
        self.extractor=extractor

    def well_stage(self,family,template,**kwargs):
        if template!='PresuppositionExtractionTemplate':
            return super().well_stage(family,template,**kwargs)
        stage=self.bridge(op='render',family=family,template=template,kwargs=kwargs)
        raw=self.extractor.call(stage['messages'])
        return raw,self.bridge(op='parse',parser=stage['parser'],text=raw)

    def run(self,method,question,source,candidate=None,gate=None):
        result=super().run(method,question,source,candidate,gate)
        if method in ('prewome','extract_verify'):
            result.update(extraction_model=self.extractor.config['model'],
                          extraction_protocol='shared_premise_extraction_v1')
        return result
