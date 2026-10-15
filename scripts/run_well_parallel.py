"""Five concurrent Sonnet calls using the existing frozen Well prompts and ledger.

Keeps src.well_eval's no-retry rule for started jobs and its unchanged score parser/report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import well_eval as we
from src.llm_backend import CLAUDE_SYSTEM_PROMPT
from src.pilot import output_lock, file_digest
from scripts import run_frontier_claude as transport


def judge_command(model):
    cmd = transport.command(model)
    cmd[cmd.index('--system-prompt') + 1] = CLAUDE_SYSTEM_PROMPT
    return cmd


def call(prompt, model, out):
    raw = transport.invoke(judge_command(model), prompt, out / 'isolated_cwd', 240)
    value, served = transport.parse_result(raw, model)
    return value['result'].strip(), served, value


def status(out, plan, plan_hash, state):
    events, scores = we.read_ledger(out, plan, plan_hash)
    value = {'state': state, 'target': len(plan['jobs']), 'attempted': len(events),
             'valid': len(scores), 'finished': sum(r['status'] == 'finished' for r in events.values()),
             'invalid_finished': sum(r['status'] == 'finished' and not r.get('valid') for r in events.values()),
             'unstarted': len(plan['jobs']) - len(events), 'workers': 5,
             'judge': plan['model'], 'generator': 'claude-sonnet-5',
             'same_model_generator_and_judge': True, 'updated_unix': time.time()}
    transport.write(out / 'status.json', value)
    return value


def score_parallel(out, max_calls, caller=None, preflight=True):
    """Coordinator alone writes ledger; workers only call the model."""
    out = Path(out)
    transport.STOP.clear()
    signal.signal(signal.SIGTERM, transport.terminate_active)
    signal.signal(signal.SIGINT, transport.terminate_active)
    with output_lock(out / 'judge'):
        plan, plan_hash = we.load_plan(out)
        if plan['sources']['implementation_sha256'] != file_digest(we.__file__):
            raise ValueError('Frozen Well evaluator implementation changed')
        if plan['backend'] != 'claude':
            raise ValueError('This runner is for the approved Claude judge')
        if (out / 'STOP').exists():
            raise ValueError('STOP marker exists; no new calls')
        (out / 'isolated_cwd').mkdir(exist_ok=True)
        previous, _ = we.read_ledger(out, plan, plan_hash)
        pending = [key for key in plan['order'] if key not in previous][:max_calls]
        model = plan['model']
        caller = caller or (lambda prompt: call(prompt, model, out))
        if pending and preflight:
            text, used, native = caller('Reply with the single word OK.')
            if not we.model_matches(used, model) or text.strip() != 'OK':
                raise ValueError('Judge authentication/model preflight failed; ledger untouched')
            transport.write(out / 'preflight.json', native)
        status(out, plan, plan_hash, 'running')
        cursor = iter(pending)
        invalid_streak = 0
        with ThreadPoolExecutor(max_workers=5) as pool:
            active = {}

            def fill():
                while not transport.STOP.is_set() and len(active) < 5:
                    key = next(cursor, None)
                    if key is None:
                        break
                    we._append(out / 'attempts.jsonl', {'id': key, 'plan_hash': plan_hash,
                               'status': 'started', 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
                    active[pool.submit(caller, plan['jobs'][key]['prompt'])] = key

            fill()
            while active:
                if (out / 'STOP').exists():
                    transport.terminate_active()
                completed, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    key = active.pop(future)
                    event = {'id': key, 'plan_hash': plan_hash, 'status': 'finished'}
                    try:
                        raw, used, native = future.result()
                        transport.write(out / 'native' / (key + '.json'), native)
                        rating = we.parse_rating(raw)
                        valid = rating is not None and we.model_matches(used, model)
                        event.update(raw=raw, model=used, valid=valid, score=rating if valid else None)
                    except Exception as exc:
                        event.update(valid=False, score=None, error=f'{type(exc).__name__}: {str(exc)[:800]}')
                    we._append(out / 'attempts.jsonl', event)
                    invalid_streak = 0 if event['valid'] else invalid_streak + 1
                    if 'error' in event or not we.model_matches(event.get('model'), model) or invalid_streak >= 3:
                        transport.terminate_active()
                    current = status(out, plan, plan_hash, 'stopping' if transport.STOP.is_set() else 'running')
                    print(json.dumps({'job': key[:12], 'score': event['score'], 'valid': current['valid'],
                                      'target': current['target'], 'error': event.get('error')}, ensure_ascii=False), flush=True)
                fill()
        current = status(out, plan, plan_hash, 'stopped' if transport.STOP.is_set() else 'finished_batch')
        if current['valid'] == current['target']:
            current = status(out, plan, plan_hash, 'complete')
    result = we.report(out)
    print(json.dumps({'status': current, 'methods': result['methods']}, ensure_ascii=False), flush=True)
    return current


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'score', 'report'])
    ap.add_argument('--questions', type=Path, required=True)
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--max-calls', type=int, default=732)
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.max_calls < 1:
        raise ValueError('Positive max-calls required')
    if args.stage == 'prepare':
        print(json.dumps(we.prepare(args.questions, [args.answers], args.out_dir,
                                   backend='claude', model='claude-sonnet-5')))
        manifest = {'workers': 5, 'timeout_seconds': 240,
                    'generator_model': 'claude-sonnet-5', 'judge_model': 'claude-sonnet-5',
                    'same_model_generator_and_judge': True,
                    'command': judge_command('claude-sonnet-5'),
                    'note': 'Unchanged Well 0-5 templates/parser; existing [SYSTEM]/[USER] serialization. CLI does not enforce API temperature/max_tokens settings.',
                    'sha256': {str(p.resolve()): file_digest(p) for p in
                               (Path(__file__), ROOT / 'scripts/run_frontier_claude.py', Path(we.__file__))}}
        transport.write(args.out_dir / 'parallel_manifest.json', manifest, frozen=True)
    elif args.stage == 'score':
        manifest = json.loads((args.out_dir / 'parallel_manifest.json').read_text())
        if any(file_digest(path) != h for path, h in manifest['sha256'].items()):
            raise ValueError('Frozen parallel runner changed')
        score_parallel(args.out_dir, args.max_calls)
    else:
        print(json.dumps(we.report(args.out_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
