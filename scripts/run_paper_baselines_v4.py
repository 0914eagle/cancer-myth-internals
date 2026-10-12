"""Matched three-model suite with shared extraction, true gated generation and batching."""
import argparse,fcntl,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_baselines import Worker,ROOT
from scripts.run_paper_baselines_resilient import process_with_failures,quota_error,INTERVAL
from scripts.paper_batch_inference import BatchedHFTask,MicroBatch,LockedCachedTask,run_rows
from scripts.paper_shared_extraction import SharedLunaExtraction
from scripts.paper_v4_pipeline import EvaluatedPipeline
from scripts.paper_cli_pool import PooledCLI
from src.paper_baselines import digest,write_json,sha


class V4Worker(Worker):
    def __init__(self,args):
        super().__init__(args)
        if args.model=='luna6':self.methods=[m for m in self.methods if m!='probe_gate']
        adapter={f:sha(ROOT/'scripts'/f) for f in ('run_paper_baselines_v4.py','paper_v4_pipeline.py','paper_batch_inference.py','paper_shared_extraction.py','paper_cli_pool.py')}
        self.identity={**self.identity,'v4_adapter_sha256':adapter}
        write_json(self.out/'execution_policies'/(digest(adapter)+'.json'),adapter,immutable=True)
        self.cli=PooledCLI(self.external['task_model'],args.out,self.out/'STOP') if args.model=='luna6' else None

    def task(self,messages):
        return self.cli(messages) if self.cli else super().task(messages)

    def external_call(self,role,messages):
        cfg=self.external[role+'_model'];key=digest({'config':cfg,'messages':messages,'role':role,'backend_sha':sha(ROOT/'src/fpqa_cli_backend.py')})
        base=self.args.out/'shared/optimization_calls';base.mkdir(parents=True,exist_ok=True)
        path=base/(key+'.json')
        with (base/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if path.exists():return json.loads(path.read_text())['text']
            while True:
                try:text,meta=PooledCLI(cfg,self.args.out,self.out/'STOP')(messages)
                except (RuntimeError,ValueError) as error:
                    if not quota_error(error):raise
                    write_json(self.out/'optimization_quota.json',{'state':'waiting_quota','next_retry_epoch':time.time()+INTERVAL})
                    for _ in range(INTERVAL):
                        if (self.out/'STOP').exists():raise RuntimeError('STOP requested')
                        time.sleep(1)
                    continue
                write_json(path,{'config':cfg,'messages':messages,'text':text,'metadata':meta},immutable=True)
                return text

    def optimize(self,split,method):
        selected={**split,'dev':split['gepa_dev']}
        return super().optimize(selected,method)

    def feature(self,question):
        base=self.out/'feature_locks';base.mkdir(parents=True,exist_ok=True)
        with (base/(digest(question)+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return super().feature(question)

    def process(self,row,split,method,candidate,gate,qa):
        self.pipeline.context.value=(row,split,qa)
        try:
            return process_with_failures(self,row,split,method,candidate,gate,qa,Worker.process)
        finally:self.pipeline.context.value=None

    def execute(self):
        self.args.generate_only=True
        opts=self.config['batching'][self.args.model];dispatch=None
        if self.cli:
            call=self.task;workers=opts['pipeline_workers']
        else:
            self.backend=BatchedHFTask(self.config,self.args.model)
            dispatch=MicroBatch(self.backend,opts['max_batch'],opts['token_budget'])
            call=dispatch;workers=opts['max_batch']*2
        cache=LockedCachedTask(call,self.out,self.identity,self.reg['transport_common'])
        self.pipeline=EvaluatedPipeline(self.reg,cache,self.bridge,SharedLunaExtraction(self),self)
        priority=['plain','balanced','direct_gate','cot_gate','cot','tfidf_gate','probe_gate','prewome','extract_verify','gepa_gate','gepa','gepa_both']
        methods=[m for m in priority if m in self.methods]
        if self.args.command=='smoke':
            splits=[{**self.bundle['splits'][0],'test':self.bundle['splits'][0]['smoke'][:2]}]
        else:splits=self.bundle['splits']
        state={'state':'running','protocol':self.config['protocol_id'],'model':self.args.model,'batching':opts,'methods':methods,'final_judge':'separate Haiku ledger'}
        status=self.out/'generation_progress.json';write_json(status,state)
        try:
            # Main FPQA tables first, including GEPA; native QA cannot delay them.
            for qa in (False,True):
                if qa and self.args.command=='smoke':continue
                for method in methods:
                    for split in splits:
                        # QA transfer uses one preselected Cancer-trained system.
                        if qa and split['dataset']=='cancer_myth' and split['name']!='fold0':continue
                        # Fixed shared QA needs no duplicate pass for CREPE.
                        if qa and split['dataset']=='crepe' and method in ('plain','balanced','cot','direct_gate','cot_gate'):continue
                        rows=self.bundle['qa'] if qa else [self.lookup[i] for i in split['test']]
                        state.update(stage='table3' if qa else 'table12',method=method,dataset=split['dataset'],split=split['name'],processed=0,total=len(rows),updated_epoch=time.time());write_json(status,state)
                        print(f'[v4] {self.args.model} {state["stage"]} {split["dataset"]}/{split["name"]} {method}',flush=True)
                        gate=self.classifier(split,method) if method in ('tfidf_gate','probe_gate') else None
                        candidate=self.optimize(split,method) if method.startswith('gepa') else None
                        def completed(n):
                            if n%10==0 or n==len(rows):
                                state.update(processed=n,updated_epoch=time.time());write_json(status,state)
                        run_rows(self,rows,split,method,candidate,gate,qa,workers,completed)
            state.update(state='complete',updated_epoch=time.time());write_json(status,state)
        except BaseException as error:
            state.update(state='failed',error=str(error),updated_epoch=time.time());write_json(status,state);raise
        finally:
            if dispatch:dispatch.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','smoke','init'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['qwen25','gemma3','luna6'],required=True);p.add_argument('--methods');p.add_argument('--phase',default='all');p.add_argument('--generate-only',action='store_true',default=True)
    a=p.parse_args();c=json.loads(a.config.read_text());os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,c['models'][a.model]['gpus']))
    w=V4Worker(a)
    if a.command!='init':w.execute()
