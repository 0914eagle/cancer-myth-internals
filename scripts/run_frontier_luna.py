"""Luna correction answers with bounded concurrency, then a Sonnet Well judge.

No model delegation: each Codex invocation independently answers one benchmark
question, with tools disabled and tool events rejected. Does not recreate Mac Plain.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frontier_claude as tr
from scripts.run_frontier_corrections import PROMPTS
from scripts.run_well_parallel import call as judge_call
from scripts.run_well_parallel import judge_command
from src import well_eval as we
from src.llm_backend import parse_codex_banner_model
from src.pilot import output_lock

MODEL = 'gpt-5.6-luna'
JUDGE = 'claude-sonnet-5'


def parse_events(stdout, final):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    allowed = {'thread.started', 'turn.started', 'turn.completed', 'item.started', 'item.completed', 'item.updated'}
    for event in events:
        if event.get('type') not in allowed:
            raise ValueError(f"Unexpected Codex event: {event.get('type')}")
        if event.get('type', '').startswith('item.'):
            item = event.get('item', {})
            if item.get('type') not in {'agent_message', 'reasoning'}:
                raise ValueError(f"Tool/error event rejected: {item.get('type')}")
    if sum(e.get('type') == 'turn.completed' for e in events) != 1:
        raise ValueError('Expected exactly one completed turn')
    if sum(e.get('type') == 'turn.started' for e in events) != 1:
        raise ValueError('Expected exactly one started turn')
    messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                and e.get('item', {}).get('type') == 'agent_message']
    if not final.strip() or not messages or messages[-1].strip() != final.strip():
        raise ValueError('Missing or inconsistent final answer')
    return events


def setup(out, source):
    inputs = json.loads((source / 'generation_inputs.json').read_text())
    evaluation = json.loads((source / 'evaluation_questions.json').read_text())
    assert len(inputs) == 735 and len(evaluation) == 732
    assert all(set(r) == {'id', 'question'} for r in inputs)
    by_id = {r['id']: r['question'] for r in inputs}
    assert len(by_id) == 735 and all(by_id[r['id']] == r['question'] for r in evaluation)
    opts = json.loads((out / 'transport_options.json').read_text())
    assert opts[opts.index('--model') + 1] == MODEL
    assert 'model_reasoning_effort="medium"' in opts
    preflight = json.loads((out / 'model_preflight.json').read_text())
    assert preflight['returncode'] == 0 and preflight['stdout'].strip() == 'READY'
    resolved = parse_codex_banner_model(preflight['stdout'], preflight['stderr'])
    assert resolved == MODEL, resolved
    assert (out / 'system_prompt.txt').read_text() == tr.SYSTEM_PROMPT
    files = [Path(__file__), ROOT / 'scripts/run_frontier_claude.py',
             ROOT / 'scripts/run_frontier_corrections.py', ROOT / 'scripts/run_well_parallel.py',
             ROOT / 'src/llm_backend.py', Path(we.__file__),
             source / 'generation_inputs.json', source / 'evaluation_questions.json',
             out / 'system_prompt.txt', out / 'transport_options.json', out / 'model_preflight.json']
    manifest = {'model': MODEL, 'reasoning_effort': 'medium', 'judge_model': JUDGE,
                'system_prompt': tr.SYSTEM_PROMPT, 'prompts': PROMPTS,
                'generation_n_per_method': 735, 'evaluation_n_per_method': 732,
                'command_options': opts, 'timeout_seconds': 240,
                'model_identity_source': 'Explicit CLI model flag and preflight resolved-model banner; JSON events do not expose backend model ID.',
                'transport_limitations': 'Codex CLI may inject runtime context in addition to model_instructions_file. No API-equivalence claim. No token-cap override.',
                'mac_plain_available': False, 'well_templates': we.template_manifest(),
                'sha256': {str(p): tr.sha(p) for p in files},
                'cli_version': subprocess.run(['codex', '--version'], text=True, capture_output=True, check=True).stdout.strip()}
    tr.write(out / 'manifest.json', manifest, frozen=True)
    qpath = out / 'evaluation_questions.jsonl'
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in evaluation)
    if qpath.exists() and qpath.read_text() != payload:
        raise ValueError('Changed evaluation inventory')
    qpath.write_text(payload)
    specs = {}
    for method, prompt in PROMPTS.items():
        spec = {'method': method, 'model': MODEL, 'user_template': prompt,
                'workers': 'recorded per batch in execution_history.jsonl',
                'manifest_sha256': tr.sha(out / 'manifest.json')}
        tr.write(out / method / 'plan.json', spec, frozen=True)
        tr.write(out / method / 'evaluation_questions.json', evaluation, frozen=True)
        specs[method] = spec
    return inputs, specs, opts


def invoke(out, method, row, spec, opts):
    folder = out / method
    rawdir = folder / 'native' / row['id']
    rawdir.mkdir(parents=True, exist_ok=True)
    final_path = rawdir / 'answer.txt'
    cmd = ['codex', 'exec'] + opts + ['--json', '--output-last-message', str(final_path), '-']
    prompt = spec['user_template'].format(question=row['question'])
    record = {**row, 'method': method, 'signature': tr.digest([spec, row]),
              'user_prompt': prompt, 'model_requested': MODEL,
              'model_identity_source': 'CLI flag; preflight resolved model; backend model ID absent',
              'started_unix': time.time()}
    proc = None
    try:
        with tr.ACTIVE_LOCK:
            if tr.STOP.is_set():
                raise RuntimeError('Run stopped')
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, cwd=out / 'isolated_cwd', start_new_session=True)
            tr.ACTIVE[proc.pid] = proc
        stdout, stderr = proc.communicate(prompt, timeout=240)
        (rawdir / 'events.jsonl').write_text(stdout)
        (rawdir / 'stderr.txt').write_text(stderr)
        if proc.returncode:
            raise RuntimeError(f'Codex exit {proc.returncode}: {(stderr or stdout)[-700:]}')
        final = final_path.read_text()
        events = parse_events(stdout, final)
        record.update(status='complete', response=final.strip(), model_used=MODEL,
                      usage=next(e['usage'] for e in events if e['type'] == 'turn.completed'),
                      thread_id=next(e['thread_id'] for e in events if e['type'] == 'thread.started'))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        (rawdir / 'events.jsonl').write_text(stdout)
        (rawdir / 'stderr.txt').write_text(stderr)
        record.update(status='failed', error='Timeout after 240 seconds')
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {str(exc)[:900]}')
    finally:
        if proc is not None:
            with tr.ACTIVE_LOCK:
                tr.ACTIVE.pop(proc.pid, None)
    if record['status'] != 'complete':
        tr.terminate_active()
    record['wall_seconds'] = time.time() - record['started_unix']
    tr.write(folder / 'records' / (row['id'] + '.json'), record)
    return record


def schedule(out, pending, worker, consume, workers):
    cursor = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        def fill():
            while not tr.STOP.is_set() and len(active) < workers:
                item = next(cursor, None)
                if item is None:
                    break
                active[pool.submit(worker, item)] = item
        if (out / 'STOP').exists():
            raise RuntimeError('STOP marker exists')
        fill()
        while active:
            if (out / 'STOP').exists():
                tr.terminate_active()
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                item = active.pop(future)
                consume(item, future)
            fill()


def generation(out, inputs, specs, opts, workers, limit):
    pending = []
    # Interleave the two conditions and FPQ/NFP while retaining a fixed order.
    fpq = [r for r in inputs if r['id'].startswith('fpq_')]
    nfp = [r for r in inputs if r['id'].startswith('nfp_')]
    ordered = [r for i in range(max(len(fpq), len(nfp))) for group in (fpq, nfp) for r in group[i:i+1]]
    for row in ordered:
        for method, spec in specs.items():
            path = out / method / 'records' / (row['id'] + '.json')
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['signature'] != tr.digest([spec, row]) or saved['status'] != 'complete':
                    raise ValueError(f'Changed/failed record requires review: {path}')
            else:
                pending.append((method, row))
    if limit:
        pending = pending[:limit]
    start = time.time()
    completed = []
    def consume(item, future):
        record = future.result()
        completed.append(record)
        method = item[0]
        current = tr.summarize(out / method, specs[method], inputs, 'running')
        current['workers'] = workers
        tr.write(out / method / 'status.json', current)
        print(json.dumps({'stage': 'generation', 'method': method, 'id': record['id'],
                          'completed': current['counts'].get('complete', 0), 'error': record.get('error')}, ensure_ascii=False), flush=True)
    schedule(out, pending, lambda item: invoke(out, item[0], item[1], specs[item[0]], opts), consume, workers)
    elapsed = time.time() - start
    summary = {'stage': 'generation', 'workers': workers, 'calls': len(completed),
               'valid': sum(r['status'] == 'complete' for r in completed), 'seconds': elapsed,
               'calls_per_minute': 60 * len(completed) / elapsed if elapsed else 0,
               'at': dt.datetime.now(dt.timezone.utc).isoformat()}
    we._append(out / 'execution_history.jsonl', summary)
    print(json.dumps(summary), flush=True)
    for method, spec in specs.items():
        state = 'stopped' if tr.STOP.is_set() else 'finished_batch'
        status = tr.summarize(out / method, spec, inputs, state)
        if status['counts'].get('complete') == len(inputs):
            status = tr.summarize(out / method, spec, inputs, 'complete')
        status['workers'] = workers
        tr.write(out / method / 'status.json', status)
    if tr.STOP.is_set():
        raise RuntimeError('Generation stopped; inspect failure records')


def judging(out, workers, limit=0):
    judge = out / 'well_judge_sonnet_v1'
    paths = [out / method / 'answers_eval732.jsonl' for method in PROMPTS]
    assert all(len(tr.read_rows(p)) == 732 for p in paths)
    we.prepare(out / 'evaluation_questions.jsonl', paths, judge, backend='claude', model=JUDGE)
    tr.write(judge / 'parallel_manifest.json', {'generator': MODEL, 'judge': JUDGE,
             'same_model_generator_and_judge': False, 'manifest_sha256': tr.sha(out / 'manifest.json'),
             'command': judge_command(JUDGE)}, frozen=True)
    (judge / 'isolated_cwd').mkdir(exist_ok=True)
    with output_lock(judge / 'judge'):
        plan, plan_hash = we.load_plan(judge)
        previous, scores = we.read_ledger(judge, plan, plan_hash)
        if len(previous) != len(scores):
            raise ValueError('Invalid/interrupted judgment requires review; no automatic retry')
        pending = [key for key in plan['order'] if key not in previous]
        if limit:
            pending = pending[:limit]
        current = dict(scores)
        def worker(key):
            return judge_call(plan['jobs'][key]['prompt'], JUDGE, judge)
        def mark_started(key):
            we._append(judge / 'attempts.jsonl', {'id': key, 'plan_hash': plan_hash, 'status': 'started',
                       'at': dt.datetime.now(dt.timezone.utc).isoformat()})
            return key
        def consume(key, future):
            event = {'id': key, 'plan_hash': plan_hash, 'status': 'finished'}
            try:
                raw, used, native = future.result()
                tr.write(judge / 'native' / (key + '.json'), native)
                score = we.parse_rating(raw)
                valid = score is not None and we.model_matches(used, JUDGE)
                event.update(raw=raw, model=used, valid=valid, score=score if valid else None)
                if valid:
                    current[key] = score
            except Exception as exc:
                event.update(valid=False, score=None, error=str(exc)[:900])
            we._append(judge / 'attempts.jsonl', event)
            if not event['valid']:
                tr.terminate_active()
            tr.write(judge / 'status.json', {'state': 'running', 'valid': len(current),
                     'target': len(plan['jobs']), 'workers': workers, 'generator': MODEL, 'judge': JUDGE})
            print(json.dumps({'stage': 'well', 'valid': len(current), 'target': len(plan['jobs']),
                              'error': event.get('error')}, ensure_ascii=False), flush=True)
        # Generator is advanced only by the coordinator, before worker submission.
        start = time.time()
        schedule(out, (mark_started(k) for k in pending), worker, consume, workers)
        state = 'complete' if len(current) == len(plan['jobs']) else 'finished_batch'
        if tr.STOP.is_set():
            state = 'stopped'
        tr.write(judge / 'status.json', {'state': state, 'valid': len(current), 'target': len(plan['jobs']),
                 'workers': workers, 'generator': MODEL, 'judge': JUDGE})
        we._append(out / 'execution_history.jsonl', {'stage': 'well', 'workers': workers,
                   'planned_calls': len(pending), 'new_valid': len(current) - len(scores),
                   'seconds': time.time() - start})
    we.report(judge)
    if tr.STOP.is_set():
        raise RuntimeError('Judgment stopped; inspect ledger')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'generate', 'judge', 'run'])
    ap.add_argument('--source-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--judge-workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if not 1 <= args.workers <= 20 or not 1 <= args.judge_workers <= 20 or args.limit < 0:
        raise ValueError('Workers must be 1–20 and limit nonnegative')
    out = args.out_dir.resolve()
    tr.STOP.clear()
    signal.signal(signal.SIGTERM, tr.terminate_active)
    signal.signal(signal.SIGINT, tr.terminate_active)
    with output_lock(out / 'pipeline'):
        inputs, specs, opts = setup(out, args.source_dir.resolve())
        if args.stage == 'prepare':
            return
        try:
            if args.stage in ('generate', 'run'):
                tr.write(out / 'status.json', {'state': 'running', 'stage': 'generation', 'pid': os.getpid(), 'workers': args.workers})
                generation(out, inputs, specs, opts, args.workers, args.limit)
            if args.stage in ('judge', 'run'):
                tr.write(out / 'status.json', {'state': 'running', 'stage': 'well', 'pid': os.getpid(), 'workers': args.judge_workers})
                judging(out, args.judge_workers, args.limit)
            tr.write(out / 'status.json', {'state': 'finished_batch' if args.limit or args.stage == 'generate' else 'complete', 'pid': os.getpid()})
        except Exception as exc:
            tr.write(out / 'status.json', {'state': 'stopped', 'error': str(exc), 'pid': os.getpid()})
            raise


if __name__ == '__main__':
    main()
