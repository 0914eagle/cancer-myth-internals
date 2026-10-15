"""Paired Cancer-Myth thinking OFF/ON suite using one resident Transformers model."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gemma4_diagnostics as base
from scripts import run_frontier_claude as io
from scripts import qwen38_transformers_runtime as rt
from scripts.run_qwen38_vllm import BASE_PLAN
from src.pilot import output_lock

MODES = ('thinking_off', 'thinking_on')
STOP = False


def stop(*_):
    global STOP
    STOP = True


def prepare(out, mode, batch_size):
    rows = json.loads(base.SOURCE.read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 732
    refs = {r['id']: r for r in io.read_rows(base.ANNOTATIONS)}
    assert all(refs[r['id']]['question'] == r['question'] for r in rows if r['id'] in refs)
    plan = copy.deepcopy(json.loads(BASE_PLAN.read_text()))
    plan.update(model=rt.MODEL, revision=rt.REVISION, dtype='FP8 weights / BF16 activations',
                quantization='official fine-grained FP8, 128x128 blocks', engine='transformers',
                attention='sdpa', thinking_enabled=mode == 'thinking_on',
                reasoning_effort='medium' if mode == 'thinking_on' else None,
                do_sample=True, sampling=rt.SAMPLING, initial_batch_size=batch_size,
                seed_rule='SHA256 of ordered stage/item IDs per batch; saved in each record. Same rule across modes.',
                max_new_tokens=8192, context_limit=32768,
                truncation_policy='Save initial output, extend once to 16384 total completion tokens; otherwise invalid.',
                format_policy='Separate native thinking from final content, then use the frozen Gemma JSON parser. Invalid is missing, not negative.',
                limitations=['Native thinking differs from explicit CoT prompts; crossed as separate conditions.',
                             'FP8 sampling differs from greedy BF16 Gemma.',
                             'Reasoning consumes the same total completion budget in ON; only final content is judged.',
                             'Batch sampling is not an identical per-item random stream across modes.',
                             'Source annotations are privileged diagnosis; extraction is not expert quality evaluation.',
                             'Reference PyTorch attention/convolution fallbacks; no throughput optimization claim.',
                             'Triton 3.5.1 explicitly overrides torch 2.8 default pin; eager FP8 smoke validated.'],
                source_sha256={str(p): io.sha(p) for p in
                    (Path(__file__), Path(rt.__file__), Path(base.__file__), BASE_PLAN, base.SOURCE,
                     base.ANNOTATIONS, Path(base.gate.__file__), Path(base.premise.__file__), rt.OVERRIDE)})
    io.write(out / 'plan.json', plan, frozen=True)
    io.write(out / 'evaluation_questions.json', rows, frozen=True)
    base.jsonl(out / 'evaluation_questions.jsonl', rows)
    io.write(out / 'generation_inputs.json', [{'id': r['id'], 'question': r['question']} for r in rows], frozen=True)
    return plan, rows, refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 16
    root = args.out_dir.resolve()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def suite(state, **kw):
        io.write(root / 'status.json', dict(state=state, pid=os.getpid(), updated_unix=time.time(), **kw))
    try:
        with output_lock(root / 'suite'):
            plans, hashes, records = {}, {}, {}
            for mode in MODES:
                out = root / mode
                plans[mode], rows, refs = prepare(out, mode, args.batch_size)
                hashes[mode] = io.sha(out / 'plan.json')
                records[mode] = {}
                for path in (out / 'records').glob('*/*.json'):
                    r = json.loads(path.read_text())
                    assert r['plan_hash'] == hashes[mode]
                    assert r['status'] != 'started', 'Interrupted batch requires explicit review'
                    records[mode][r['stage'], r['id']] = r
                base.report(out, plans[mode], rows, records[mode], 'loading')
            suite('loading')
            runtime = rt.Runtime(root, args.batch_size)
            for stage in plans[MODES[0]]['stages']:
                pending = {m: [task for row in rows if (stage, row['id']) not in records[m]
                               if (task := base.prompt_for(stage, row, records[m], refs)) is not None]
                           for m in MODES}
                streak = {m: 0 for m in MODES}
                while any(pending.values()):
                    for mode in MODES:
                        out = root / mode
                        if STOP or (root / 'STOP').exists() or (out / 'STOP').exists():
                            suite('stopped', stage=stage, mode=mode)
                            return
                        tasks = pending[mode][:runtime.batch_size]
                        if not tasks:
                            continue
                        pending[mode] = pending[mode][len(tasks):]
                        suite('running', stage=stage, mode=mode, batch_size=len(tasks))
                        for task in tasks:
                            io.write(out / 'records' / stage / (task['id'] + '.json'),
                                     dict(**task, status='started', plan_hash=hashes[mode]))
                        results = runtime.safe_generate(tasks, mode == 'thinking_on', 8192)
                        for task, result in zip(tasks, results, strict=True):
                            r = dict(**task, plan_hash=hashes[mode], thinking_enabled=mode == 'thinking_on',
                                     attempts=[result], **result)
                            if result['truncated']:
                                result = runtime.safe_generate([task], mode == 'thinking_on', 16384)[0]
                                r['attempts'].append(result)
                                r.update(result)
                            try:
                                if result['truncated'] or result.get('format_error'):
                                    raise ValueError(result.get('format_error', 'Truncated generation'))
                                r['result'] = base.parse(stage, result['response'], task)
                                r['status'] = 'complete'
                            except ValueError as exc:
                                r.update(status='invalid', error=str(exc))
                            r['finished_unix'] = time.time()
                            records[mode][stage, r['id']] = r
                            io.write(out / 'records' / stage / (r['id'] + '.json'), r)
                            streak[mode] = streak[mode] + 1 if r['status'] == 'invalid' else 0
                        state = base.report(out, plans[mode], rows, records[mode], 'running', stage)
                        print(json.dumps({'mode': mode, 'stage': stage, 'counts': state['counts'][stage],
                                          'batch_seconds': results[0]['batch_wall_seconds']}), flush=True)
                        if streak[mode] >= 3:
                            raise RuntimeError('Three consecutive invalid outputs; inspect before continuing')
                if stage in base.ANSWER_STAGES:
                    for mode in MODES:
                        out = root / mode
                        answers = [{'id': row['id'], 'question': row['question'], 'method': stage,
                                    'response': records[mode][stage, row['id']]['response'], 'status': 'complete'}
                                   for row in rows if records[mode].get((stage, row['id']), {}).get('status') == 'complete']
                        path = out / 'answers' / (stage + '.jsonl')
                        base.jsonl(path, answers)
                        io.write(out / 'answers' / (stage + '.ready.json'),
                                 {'valid': len(answers), 'expected': len(rows), 'answer_sha256': io.sha(path)})
            for mode in MODES:
                base.report(root / mode, plans[mode], rows, records[mode], 'generation_finished')
            suite('generation_finished')
    except Exception as exc:
        suite('failed', error=f'{type(exc).__name__}: {exc}')
        io.write(root / 'error.json', {'error': str(exc), 'at': time.time()})
        for mode in MODES:
            io.write(root / mode / 'error.json', {'error': str(exc), 'at': time.time()})
        raise


if __name__ == '__main__':
    main()
