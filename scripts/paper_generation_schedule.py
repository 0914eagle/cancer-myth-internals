"""Generate all FPQA baselines before native QA; final judging is a separate service.
Fixed prompts traverse disjoint test-ID folders only for cache compatibility.
They are neither fitted nor selected by fold.
"""
from src.paper_baselines import FIXED, CachedTask, Pipeline, write_json, sha
from pathlib import Path
import time
import os


def jobs(methods, splits):
    local=[m for m in methods if not m.startswith('gepa')]
    # Fixed methods first, then cross-fit TF-IDF/probe. GEPA needs live training
    # feedback and therefore runs after the judge-independent jobs.
    priority=['plain','balanced','direct_gate','cot_gate','cot','tfidf_gate','probe_gate','prewome','extract_verify']
    local=sorted(local,key=lambda m:priority.index(m))
    for qa in (False, True):
        for method in local:
            for split in splits:
                yield method,split,qa
    for method in methods:
        if method.startswith('gepa'):
            for split in splits:
                for qa in (False, True):yield method,split,qa


def execute(worker):
    worker.args.generate_only=True
    batch_size=int(os.environ.get('PAPER_BATCH_SIZE_'+getattr(worker.args,'model','').upper(),os.environ.get('PAPER_BATCH_SIZE','1')))
    token_budget=int(os.environ.get('PAPER_BATCH_TOKEN_BUDGET_'+getattr(worker.args,'model','').upper(),os.environ.get('PAPER_BATCH_TOKEN_BUDGET','65536')))
    dispatcher=None
    cache_class=CachedTask
    if batch_size>1:
        from scripts.paper_batch_inference import BatchedHFTask,MicroBatch,LockedCachedTask
        worker.backend=BatchedHFTask(worker.config,worker.args.model)
        dispatcher=MicroBatch(worker.backend,max_batch=batch_size,
            token_budget=token_budget)
        # Keep worker.backend as the HF object for probe feature extraction.
        worker.task=dispatcher
        original_feature=worker.feature
        def locked_feature(question):
            import fcntl
            from src.paper_baselines import digest
            path=worker.out/'feature_locks'/(digest(question)+'.lock')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                return original_feature(question)
        worker.feature=locked_feature
        cache_class=LockedCachedTask
    worker.pipeline=Pipeline(worker.reg,cache_class(worker.task,worker.out,worker.identity,
                                                 worker.reg['transport_common']),worker.bridge)
    if os.environ.get('PAPER_SHARED_LUNA_EXTRACTION')=='1':
        from scripts.paper_shared_extraction import SharedLunaExtraction,SharedExtractionPipeline
        # Existing own-model extraction results must not be relabelled as shared.
        for method in ('prewome','extract_verify'):
            for table in ('table12','table3'):
                for path in (worker.out/table).glob('*/*/'+method+'/*.json'):
                    import json
                    if json.loads(path.read_text()).get('extraction_protocol')!='shared_premise_extraction_v1':
                        raise ValueError('Legacy own-model extraction requires a separate output namespace: '+str(path))
        worker.pipeline=SharedExtractionPipeline(worker.reg,worker.pipeline.call,worker.bridge,SharedLunaExtraction(worker))
    state_path=worker.out/'generation_progress.json'
    state={'state':'running','schedule_sha256':sha(Path(__file__)),
           'order':'FPQA: Plain/Balanced -> Direct/CoT gates -> CoT -> TF-IDF/Probe -> shared-Luna pipelines; then native QA -> GEPA',
           'final_judging':'separate queue; no final judge calls from this worker'}
    state['extraction_model']='gpt-6-luna' if os.environ.get('PAPER_SHARED_LUNA_EXTRACTION')=='1' else getattr(worker,'identity',{}).get('model',{}).get('model_id','unspecified')
    state.update(batch_cap=batch_size,pipeline_workers=batch_size*2 if dispatcher else 1,
                 batch_token_budget=token_budget,
                 batch_note='Greedy outputs may differ across batch shapes; actual batch saved per call.')
    if dispatcher:
        state['batch_code_sha256']=sha(Path(__file__).with_name('paper_batch_inference.py'))
        policy={k:state[k] for k in ('extraction_model','batch_cap','pipeline_workers','batch_token_budget','batch_note','batch_code_sha256')}
        from src.paper_baselines import digest
        write_json(worker.out/'execution_policies'/(digest(policy)+'.json'),policy,immutable=True)
    write_json(state_path,state)
    try:
        for method,split,qa in jobs(worker.methods,worker.bundle['splits']):
            rows=worker.bundle['qa'] if qa else [worker.lookup[i] for i in split['test']]
            state.update(method=method,dataset=split['dataset'],split=split['name'],
                         stage='table3' if qa else 'table12',processed=0,total=len(rows))
            write_json(state_path,state)
            print(f'[generate] {state["stage"]} {split["dataset"]}/{split["name"]} {method}',flush=True)
            gate=worker.classifier(split,method) if method in ('tfidf_gate','probe_gate') else None
            candidate=worker.optimize(split,method) if method.startswith('gepa') else None
            def completed(n):
                if n%10==0 or n==len(rows):
                    state.update(processed=n,updated_epoch=time.time());write_json(state_path,state)
            if dispatcher:
                from scripts.paper_batch_inference import run_rows
                run_rows(worker,rows,split,method,candidate,gate,qa,batch_size*2,completed)
            else:
                for i,row in enumerate(rows):
                    if (worker.out/'STOP').exists():raise RuntimeError('STOP requested')
                    worker.process(row,split,method,candidate,gate,qa)
                    completed(i+1)
        state.update(state='complete',updated_epoch=time.time())
        write_json(state_path,state)
    except BaseException as error:
        state.update(state='failed',error=str(error),updated_epoch=time.time())
        write_json(state_path,state)
        raise

    finally:
        if dispatcher:dispatcher.close()
