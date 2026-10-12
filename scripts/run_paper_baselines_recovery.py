"""Transport/failure recovery without changing frozen scientific cache identities."""
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import digest,sha,write_json
from src.fpqa_prompting import parse_rating


def validated_optimization_call(worker,role,messages,original):
    if role!='judge':return original(worker,role,messages)
    cfg=worker.external['judge_model']
    key=digest({'config':cfg,'messages':messages,'role':role,'backend_sha':sha(ROOT/'src/fpqa_cli_backend.py')})
    base=worker.args.out/'shared/optimization_calls'
    for attempt in range(3):
        text=original(worker,role,messages)
        try:parse_rating(text)
        except ValueError:
            # The original call releases this same per-request lock before returning.
            # Do not remove a newer valid response published by another process.
            base.mkdir(parents=True,exist_ok=True)
            with (base/(key+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                path=base/(key+'.json')
                archive=base.parent/'invalid_optimization_calls'/f'{key}-{time.time_ns()}.json'
                archive.parent.mkdir(parents=True,exist_ok=True)
                if path.exists() and json.loads(path.read_text()).get('text')==text:
                    path.replace(archive)
                else:
                    write_json(archive,{'text':text,'config':cfg,'messages':messages,'role':role})
            print(f'[optimization judge] Invalid rating archived; attempt {attempt+1}/3, unchanged request',flush=True)
        else:return text
    raise ValueError('Three malformed optimization ratings; archived, no invented score')


def recoverable_item_error(error,original):
    if isinstance(error,ValueError) and str(error)=='Empty generation':return 'empty_generation'
    return original(error)


def install():
    from scripts import run_paper_baselines_v4 as runner
    from scripts import run_paper_baselines_resilient as resilient
    original=runner.V4Worker.external_call
    runner.V4Worker.external_call=lambda self,role,messages:validated_optimization_call(self,role,messages,original)
    old_item_error=resilient.item_error
    replacement=lambda error:recoverable_item_error(error,old_item_error)
    resilient.item_error=replacement
    runner.item_error=replacement


def main():
    install()
    model=sys.argv[sys.argv.index('--model')+1]
    out=Path(sys.argv[sys.argv.index('--out')+1])/model
    write_json(out/'execution_policies/recovery_20261013.json',{
        'wrapper_sha256':sha(Path(__file__)),
        'optimization_judge':'Strict original parser; archive malformed cache and retry same prompt up to 3 times',
        'empty_generation':'Record per-item failure, retain full denominator, continue other items; no invented rating',
        'model_prompt_split_decoding_and_valid_caches':'unchanged'})
    from scripts import run_paper_baselines_sol as entry
    if model=='gemma3':
        from scripts.run_paper_gemma_placement import PlacementWorker
        entry.PrefetchWorker=PlacementWorker
    entry.main()


if __name__=='__main__':main()
