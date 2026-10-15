"""Run a frozen Well plan through tool-disabled Codex calls; never retry jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
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
from src import well_eval as we
from src.pilot import output_lock, file_digest
from scripts import run_frontier_claude as tr
from scripts.run_frontier_luna import parse_events as strict_parse_events

def parse_events(stdout, final):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    filtered = [e for e in events if not (e.get("type") == "error" and e.get("message", "").startswith("Reconnecting...")) and not (e.get("type") == "item.completed" and e.get("item", {}).get("type") == "error" and e["item"].get("message", "").startswith("Falling back from WebSockets to HTTPS transport."))]
    strict_parse_events("\n".join(json.dumps(e) for e in filtered), final)
    return events



def invoke(out, opts, key, prompt):
    folder = out / 'native' / key
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'answer.txt'
    cmd = ['codex', 'exec'] + opts + ['--json', '--output-last-message', str(final), '-']
    proc = None
    try:
        with tr.ACTIVE_LOCK:
            if tr.STOP.is_set():
                raise RuntimeError('Run stopped')
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True,
                                    cwd=out / 'isolated_cwd', start_new_session=True)
            tr.ACTIVE[proc.pid] = proc
        stdout, stderr = proc.communicate(prompt, timeout=240)
        (folder / 'events.jsonl').write_text(stdout)
        (folder / 'stderr.txt').write_text(stderr)
        if proc.returncode:
            raise RuntimeError(f'Codex exit {proc.returncode}: {(stderr or stdout)[-900:]}')
        answer = final.read_text().strip()
        events = parse_events(stdout, answer)
        usage = next(e.get('usage') for e in events if e['type'] == 'turn.completed')
        return answer, usage
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        (folder / 'events.jsonl').write_text(stdout)
        (folder / 'stderr.txt').write_text(stderr)
        raise RuntimeError('Timeout after 240 seconds; no retry')
    finally:
        if proc is not None:
            with tr.ACTIVE_LOCK:
                tr.ACTIVE.pop(proc.pid, None)


def run(out, workers, limit):
    plan, plan_hash = we.load_plan(out)
    opts = json.loads((out / 'transport_options.json').read_text())
    pre = json.loads((out / 'preflight.json').read_text())
    assert plan['backend'] == 'codex'
    assert opts[opts.index('--model') + 1] == plan['model']
    assert pre['returncode'] == 0 and pre['model'] == plan['model']
    assert (out / 'preflight_answer.txt').read_text().strip() == 'OK'
    assert plan['sources']['implementation_sha256'] == file_digest(we.__file__)
    for source in [plan['sources']['questions'], *plan['sources']['answers']]:
        assert file_digest(source['path']) == source['sha256']
    manifest = {
        'model': plan['model'], 'workers': workers, 'timeout_seconds': 240,
        'reasoning_effort': 'medium', 'tools': 'disabled; tool events rejected',
        'model_identity': 'Explicit model flag and successful preflight banner; JSON events do not expose backend model ID',
        'temperature_and_max_tokens': 'CLI does not enforce API fields in Well plan',
        'generator': 'Codex directly authored drafts with labels/references visible',
        'same_GPT_family_author_and_judge': True,
        'rubric': 'Unchanged Well 0-5; FPQ reference premise/correction, NFP original no-false-premise rubric',
        'purpose': 'Training-target quality check, not held-out model performance',
        'sha256': {str(p): file_digest(p) for p in
                   [Path(__file__).resolve(), out / 'reconciliation.json', Path(we.__file__), ROOT / 'scripts/run_frontier_luna.py',
                    ROOT / 'scripts/run_frontier_claude.py', out / 'transport_options.json', out / 'system_prompt.txt']},
    }
    tr.write(out / 'execution_manifest.json', manifest, frozen=True)
    tr.STOP.clear()
    signal.signal(signal.SIGTERM, tr.terminate_active)
    signal.signal(signal.SIGINT, tr.terminate_active)
    with output_lock(out / 'judge'):
        if (out / 'STOP').exists():
            raise RuntimeError('STOP marker exists')
        previous, scores = we.read_ledger(out, plan, plan_hash)
        reconciliation = json.loads((out / 'reconciliation.json').read_text())
        if set(previous) - set(scores) != set(reconciliation['unrecoverable_keys']):
            raise RuntimeError('Unreviewed invalid attempt; stopping')
        pending = iter([key for key in plan['order'] if key not in previous][:limit])
        def update(state):
            events, scores = we.read_ledger(out, plan, plan_hash)
            value = {'state': state, 'pid': os.getpid(), 'target': len(plan['jobs']),
                     'valid': len(scores), 'attempted': len(events), 'judge': plan['model'],
                     'workers': workers, 'updated_unix': time.time()}
            tr.write(out / 'status.json', value)
            return value
        update('running')
        with ThreadPoolExecutor(max_workers=workers) as pool:
            active = {}
            def fill():
                while len(active) < workers and not tr.STOP.is_set():
                    key = next(pending, None)
                    if key is None:
                        break
                    we._append(out / 'attempts.jsonl', {'id': key, 'plan_hash': plan_hash,
                               'status': 'started', 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
                    active[pool.submit(invoke, out, opts, key, plan['jobs'][key]['prompt'])] = key
            fill()
            while active:
                if (out / 'STOP').exists():
                    tr.terminate_active()
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    event = {'id': key, 'plan_hash': plan_hash, 'status': 'finished'}
                    try:
                        raw, usage = future.result()
                        rating = we.parse_rating(raw)
                        event.update(raw=raw, model=plan['model'], usage=usage,
                                     valid=rating is not None, score=rating)
                    except Exception as exc:
                        event.update(error=f'{type(exc).__name__}: {exc}', valid=False, score=None)
                    we._append(out / 'attempts.jsonl', event)
                    if not event['valid']:
                        tr.terminate_active()
                    state = update('stopping' if tr.STOP.is_set() else 'running')
                    print(json.dumps({'job': key[:12], 'score': event['score'], **state,
                                      'error': event.get('error')}), flush=True)
                fill()
        state = update('stopped' if tr.STOP.is_set() else 'finished_batch')
        if state['valid'] == state['target']:
            state = update('complete')
    we.report(out)
    return state


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-calls', type=int, default=438)
    args = ap.parse_args()
    assert 1 <= args.workers <= 10 and args.max_calls > 0
    try:
        print(json.dumps(run(args.out_dir.resolve(), args.workers, args.max_calls)), flush=True)
    except Exception as exc:
        tr.write(args.out_dir / 'runner_error.json', {'error': str(exc), 'at': time.time()})
        raise
