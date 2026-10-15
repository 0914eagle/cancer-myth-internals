"""Frozen Plain/GEPA held-out evaluation through the original Well evaluator.

No optimization or prompt selection. Thread-local callers/evaluators avoid the
upstream balanced evaluator's mutable template leaking across question types.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from itertools import zip_longest
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import digest, freeze, load_rows, response_metrics, validate_splits
from src.fpqa_well_upstream import FatalTransportError, UncachedCaller, load_upstream, make_args, verify_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_jobs(dataset_rows, prompts):
    lanes = [[(ds, method, row) for row in rows] for ds, rows in dataset_rows.items()
             for method in prompts[ds]]
    return [job for group in zip_longest(*lanes) for job in group if job is not None]


def upstream_row(row):
    return {'id': row['id'], 'question': row['question'], 'answer': row['reference_answer'],
            'presuppositions': row['presuppositions'], 'few_shot_data': [], 'passages': [],
            '_gepa_question_type': 'fpq' if row['label'] else 'tpq'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimization-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--crepe-data', type=Path, default=ROOT/'results/fpqa_prompting/data')
    parser.add_argument('--cancer-data', type=Path, default=ROOT/'results/fpqa_prompting/cancer_data_test100_v2')
    parser.add_argument('--well-root', type=Path, default=ROOT/'external/fpqa_20261002/Well')
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 5:
        parser.error('Use 1..5 workers; each makes one CLI call at a time')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'STOP').exists():
        raise RuntimeError('STOP present; not restarting')
    if (out/'run.json').exists() and not args.preflight and not args.resume:
        raise RuntimeError('Run exists; inspect and explicitly --resume')
    source_hashes = verify_source(args.well_root)
    context = threading.local()
    stopped = threading.Event()
    def dispatch(role, messages):
        if stopped.is_set() or (out/'STOP').exists():
            raise FatalTransportError('Test evaluation stopped')
        return context.caller(role, messages)
    balanced, direct, _ = load_upstream(args.well_root, dispatch)
    rows_by_dataset, configs, prompts, original_args, provenance = {}, {}, {}, {}, {}
    for ds, data in [('cancer_myth',args.cancer_data), ('crepe',args.crepe_data)]:
        opt = args.optimization_root/ds
        if (opt/'FAILED.json').exists() or (opt/'INVALID_OPTIMIZATION.json').exists():
            raise RuntimeError(f'{ds}: invalid optimization')
        run = json.loads((opt/'run.json').read_text())
        summary = json.loads((opt/'gepa_result_summary.json').read_text())
        if run['source_sha256'] != source_hashes:
            raise RuntimeError('Well source changed since optimization')
        # Match the exact generation/scoring adapters used in the successful run.
        for path in ('src/fpqa_cli_backend.py','src/fpqa_well_upstream.py'):
            if run['adapter_sha256'][path] != sha(ROOT/path):
                raise RuntimeError(f'Adapter changed since optimization: {path}')
        manifest = json.loads((data/'manifest.json').read_text())
        for rel, checksum in manifest['files'].items():
            if rel.startswith(ds+'/') and sha(data/rel) != checksum:
                raise RuntimeError(f'Dataset changed: {rel}')
        splits = [load_rows(data/ds/f'{s}.jsonl') for s in ('train','dev','test')]
        validate_splits(*splits)
        rows = splits[-1]
        if len({r['id'] for r in rows}) != len(rows):
            raise RuntimeError('Duplicate test IDs')
        rows_by_dataset[ds] = rows
        configs[ds] = run['config']
        family = {'crepe':'CREPE','cancer_myth':'CancerMyth'}[ds]
        original_args[ds] = make_args(balanced,family,opt/'inputs',out/ds,configs[ds])
        prompts[ds] = {'plain':direct.load_seed_prompt(original_args[ds]), 'gepa':summary['best_system_prompt']}
        if summary['best_system_prompt'] != (opt/'best_system_prompt.txt').read_text():
            raise RuntimeError('Selected prompt mismatch')
        provenance[ds] = {'optimization_summary_sha256':sha(opt/'gepa_result_summary.json'),
                          'optimization_run_sha256':sha(opt/'run.json'), 'test_sha256':sha(data/ds/'test.jsonl'),
                          'test_ids':[r['id'] for r in rows], 'counts':dict(Counter('fpq' if r['label'] else 'nfp' for r in rows)),
                          'prompt_sha256':{m:hashlib.sha256(t.encode()).hexdigest() for m,t in prompts[ds].items()}}
        for method, prompt in prompts[ds].items():
            freeze(out/'prompts'/f'{ds}_{method}.json',{'system_prompt':prompt,'user_prompt':'Original question only'})
    jobs = test_jobs(rows_by_dataset,prompts)
    plan = {'protocol':'well_upstream_frozen_test_v1','source_sha256':source_hashes,
            'configs':configs,'provenance':provenance,'workers':args.workers,'n_answers':len(jobs),
            'script_sha256':sha(__file__),'metrics_code_sha256':sha(ROOT/'src/fpqa_prompting.py'),
            'remaining_transport_differences':run['remaining_transport_differences'],
            'scope':'Frozen Plain and selected GEPA only. Same held-out IDs and Well judge. No optimization.'}
    freeze(out/'preflight.json',plan)
    if args.preflight:
        print(json.dumps({'n_answers':len(jobs),'n_model_calls_without_retries':2*len(jobs),
                          'counts':{ds:v['counts'] for ds,v in provenance.items()},'workers':args.workers},indent=2))
        return
    freeze(out/'run.json',plan)
    records = defaultdict(list)
    def evaluate(job):
        ds,method,row = job
        if stopped.is_set() or (out/'STOP').exists():
            return None
        folder = out/ds/method/digest(row['id'])
        path = folder/'record.json'
        if path.exists():
            return json.loads(path.read_text())
        attempts = folder/'attempts'
        attempt = attempts/f'{len(list(attempts.glob("*")))+1:03d}'
        context.caller = UncachedCaller(configs[ds],attempt)
        try:
            evaluator = balanced.BalancedDirectQAGEPAEvaluator(original_args[ds],'gemini')
            score,feedback = evaluator({'system_prompt':prompts[ds][method]},upstream_row(row))
            record = {'dataset':ds,'method':method,'id':row['id'],'label':row['label'],
                      'score':feedback['raw_score'],'normalized_score':score,'feedback':feedback,
                      'attempt':str(attempt.relative_to(out))}
            freeze(path,record)
            return record
        except BaseException:
            stopped.set()
            raise
    def status(state):
        data = {'state':state,'updated_kst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
                'completed':sum(len(v) for v in records.values()),'total':len(jobs),'conditions':{}}
        for (ds,method),rr in records.items():
            data['conditions'][f'{ds}/{method}'] = {'n_completed':len(rr),'n_expected':len(rows_by_dataset[ds]),
                'partial':len(rr)!=len(rows_by_dataset[ds]),'metrics':response_metrics(rr)}
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(data,indent=2)+'\n')
        tmp.replace(out/'status.json')
        return data
    status('running')
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(evaluate,job) for job in jobs]
            for future in as_completed(pending):
                r = future.result()
                if r is not None:
                    records[r['dataset'],r['method']].append(r)
                    n = sum(len(v) for v in records.values())
                    if n%10==0 or n==len(jobs):
                        status('running')
                        print(f'Completed {n}/{len(jobs)}',flush=True)
        complete = sum(len(v) for v in records.values())==len(jobs)
        report = status('complete' if complete else 'stopped')
        if complete:
            # Completion time belongs to status, not immutable measurement data.
            report.pop('updated_kst')
            freeze(out/'metrics.json',json.loads(json.dumps(report,sort_keys=True)))
    except BaseException:
        status('failed')
        raise


if __name__=='__main__':
    main()
