"""Watch completed Gemma answer files; use unchanged Well rubric and Sonnet judge."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import datetime as dt
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frontier_claude as tr
from scripts.run_well_parallel import call
from scripts.run_gemma4_diagnostics import ANSWER_STAGES, MODEL
from scripts.summarize_gemma4_diagnostics import summarize
from src import well_eval as we
from src.pilot import output_lock, file_digest

JUDGE = 'claude-sonnet-5'


def judge_method(run, method, workers):
    out = run / 'well' / method
    out.mkdir(parents=True, exist_ok=True)
    (out / 'isolated_cwd').mkdir(exist_ok=True)
    we.prepare(run / 'evaluation_questions.jsonl', [run / 'answers' / (method + '.jsonl')],
               out, backend='claude', model=JUDGE)
    tr.write(out / 'execution_manifest.json', {'generator': MODEL, 'judge': JUDGE,
             'workers': workers, 'same_model_generator_and_judge': False,
             'rubric': 'unchanged Well 0-5; FPQ original premise and correction; NFP original rubric',
             'source_sha256': {str(p): file_digest(p) for p in
                              (Path(__file__), Path(we.__file__), ROOT / 'scripts/run_well_parallel.py')}}, frozen=True)
    plan, plan_hash = we.load_plan(out)
    with output_lock(out / 'judge'):
        events, scores = we.read_ledger(out, plan, plan_hash)
        pending = iter(k for k in plan['order'] if k not in events)
        def update(state):
            events, scores = we.read_ledger(out, plan, plan_hash)
            value = {'state': state, 'valid': len(scores), 'target': len(plan['jobs']),
                     'attempted': len(events), 'generator': MODEL, 'judge': JUDGE,
                     'same_model_generator_and_judge': False, 'updated_unix': time.time()}
            tr.write(out / 'status.json', value)
            return value
        text, served, native = call('Reply with the single word OK.', JUDGE, out)
        tr.write(out / 'preflight.json', native)
        if text.strip() != 'OK' or not we.model_matches(served, JUDGE):
            raise ValueError('Sonnet preflight failed')
        invalid_streak = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            def fill():
                while not tr.STOP.is_set() and len(active) < workers:
                    key = next(pending, None)
                    if key is None:
                        break
                    we._append(out / 'attempts.jsonl', {'id': key, 'plan_hash': plan_hash,
                               'status': 'started', 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
                    active[pool.submit(call, plan['jobs'][key]['prompt'], JUDGE, out)] = key
            fill()
            while active:
                if (run / 'STOP').exists() or (out / 'STOP').exists():
                    tr.terminate_active()
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    event = {'id': key, 'plan_hash': plan_hash, 'status': 'finished'}
                    try:
                        raw, served, native = future.result()
                        tr.write(out / 'native' / (key + '.json'), native)
                        score = we.parse_rating(raw)
                        valid = score is not None and we.model_matches(served, JUDGE)
                        event.update(raw=raw, model=served, valid=valid, score=score if valid else None)
                    except Exception as exc:
                        event.update(valid=False, score=None, error=f'{type(exc).__name__}: {exc}')
                    we._append(out / 'attempts.jsonl', event)
                    invalid_streak = 0 if event['valid'] else invalid_streak + 1
                    if 'error' in event or not we.model_matches(event.get('model'), JUDGE) or invalid_streak >= 3:
                        tr.terminate_active()
                    current = update('stopping' if tr.STOP.is_set() else 'running')
                    print(json.dumps({'method': method, **current}), flush=True)
                fill()
        current = update('stopped' if tr.STOP.is_set() else 'finished')
        if current['valid'] == current['target']:
            current = update('complete')
    we.report(out)
    if current['state'] != 'complete':
        raise RuntimeError('Judge incomplete; inspect ledger before any retry')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=5)
    args = ap.parse_args()
    assert 1 <= args.workers <= 10
    run = args.run_dir.resolve()
    signal.signal(signal.SIGTERM, tr.terminate_active)
    signal.signal(signal.SIGINT, tr.terminate_active)
    try:
        for method in ANSWER_STAGES:
            ready = run / 'answers' / (method + '.ready.json')
            while not ready.exists():
                if (run / 'STOP').exists() or (run / 'error.json').exists() or tr.STOP.is_set():
                    raise RuntimeError('Generation stopped before answer export')
                tr.write(run / 'judge_status.json', {'state': 'waiting_for_answers', 'method': method,
                                                   'updated_unix': time.time()})
                time.sleep(15)
            metadata = json.loads(ready.read_text())
            assert metadata['answer_sha256'] == tr.sha(run / 'answers' / (method + '.jsonl'))
            tr.write(run / 'judge_status.json', {'state': 'judging', 'method': method, 'updated_unix': time.time()})
            judge_method(run, method, args.workers)
            summarize(run)
        tr.write(run / 'judge_status.json', {'state': 'complete', 'updated_unix': time.time()})
    except Exception as exc:
        tr.write(run / 'judge_status.json', {'state': 'stopped', 'error': str(exc), 'updated_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
