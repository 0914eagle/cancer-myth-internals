"""Generate all FPQA baselines before native QA; final judging is a separate service.
Fixed prompts traverse disjoint test-ID folders only for cache compatibility.
They are neither fitted nor selected by fold.
"""
from src.paper_baselines import FIXED, CachedTask, Pipeline, write_json, sha
from pathlib import Path
import time


def jobs(methods, splits):
    local=[m for m in methods if not m.startswith('gepa')]
    # Fixed methods first, then cross-fit TF-IDF/probe. GEPA needs live training
    # feedback and therefore runs after the judge-independent jobs.
    local=[m for m in local if m in FIXED]+[m for m in local if m not in FIXED]
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
    worker.pipeline=Pipeline(worker.reg,CachedTask(worker.task,worker.out,worker.identity,
                                                 worker.reg['transport_common']),worker.bridge)
    state_path=worker.out/'generation_progress.json'
    state={'state':'running','schedule_sha256':sha(Path(__file__)),
           'order':'fixed FPQA -> learned local FPQA -> native QA -> GEPA',
           'final_judging':'separate queue; no final judge calls from this worker'}
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
            for i,row in enumerate(rows):
                if (worker.out/'STOP').exists():raise RuntimeError('STOP requested')
                worker.process(row,split,method,candidate,gate,qa)
                if i%10==0 or i+1==len(rows):
                    state.update(processed=i+1,updated_epoch=time.time());write_json(state_path,state)
        state.update(state='complete',updated_epoch=time.time())
        write_json(state_path,state)
    except BaseException as error:
        state.update(state='failed',error=str(error),updated_epoch=time.time())
        write_json(state_path,state)
        raise
