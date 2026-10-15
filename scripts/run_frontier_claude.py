"""Independent Claude Plain answers, bounded concurrency, resumable per-item records.

No tools, reference annotations, answer scoring, or premise rewriting.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pilot import output_lock

SYSTEM_PROMPT = "You are a helpful assistant. Respond to the user's question in English. Use only your existing knowledge; do not use tools. Return only your answer."
STOP = threading.Event()
ACTIVE = {}
ACTIVE_LOCK = threading.Lock()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value, frozen=False):
    if frozen and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen configuration changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def command(model):
    return ['claude', '-p', '--tools', '', '--disable-slash-commands', '--no-session-persistence',
            '--system-prompt', SYSTEM_PROMPT, '--output-format', 'json', '--permission-mode', 'default',
            '--setting-sources', '', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
            '--settings', '{"disableAllHooks":true}', '--model', model]


def parse_result(stdout, model):
    value = json.loads(stdout)
    if value.get('type') != 'result' or value.get('is_error') or value.get('subtype') != 'success':
        raise ValueError(f"CLI result error: {str(value.get('result', value.get('subtype')))[:600]}")
    if value.get('stop_reason') != 'end_turn':
        raise ValueError(f"Incomplete answer: stop_reason={value.get('stop_reason')}")
    if not isinstance(value.get('result'), str) or not value['result'].strip():
        raise ValueError('Empty answer')
    models = value.get('modelUsage', {})
    served = [name for name in models if name == model or name.startswith(model + '-')]
    if len(served) != 1:
        raise ValueError(f'Model mismatch: expected {model}; got {list(models)}')
    if value.get('num_turns', 1) != 1 or value.get('permission_denials'):
        raise ValueError('Unexpected agent/tool turn')
    server_tools = value.get('usage', {}).get('server_tool_use', {})
    if any(server_tools.get(k, 0) for k in ('web_search_requests', 'web_fetch_requests')):
        raise ValueError('Unexpected server tool use')
    return value, served[0]


def terminate_active(*_):
    STOP.set()
    with ACTIVE_LOCK:
        for proc in list(ACTIVE.values()):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def invoke(cmd, prompt, cwd, timeout):
    with ACTIVE_LOCK:
        if STOP.is_set():
            raise RuntimeError('Run stopped')
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=cwd, start_new_session=True)
        ACTIVE[proc.pid] = proc
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f'CLI exit {proc.returncode}: {(stderr or stdout)[-600:]}')
        return stdout
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    finally:
        with ACTIVE_LOCK:
            ACTIVE.pop(proc.pid, None)


def prepare(args):
    source = [r for r in read_rows(args.questions) if r['set'] in ('fpq', 'nfp')]
    evaluation = read_rows(args.eval_questions)
    assert Counter(r['set'] for r in source) == {'fpq': 585, 'nfp': 150}
    assert len({r['id'] for r in source}) == len(source)
    by_id = {r['id']: r for r in source}
    assert len(evaluation) == 732 and len({r['id'] for r in evaluation}) == 732
    assert all(by_id[r['id']]['question'] == r['question'] for r in evaluation)
    assert len({' '.join(r['question'].split()) for r in evaluation}) == 732
    from src.well_eval import example_questions
    demos = {' '.join(q.split()) for q in example_questions()}
    assert not any(' '.join(r['question'].split()) in demos for r in evaluation)
    inputs = [{k: r[k] for k in ('id', 'question')} for r in sorted(source, key=lambda r: r['id'])]
    spec = {'version': 1, 'method': 'plain', 'model': args.model, 'system_prompt': SYSTEM_PROMPT,
            'user_prompt': 'original question, verbatim', 'timeout_seconds': 240,
            'output_token_limit': None, 'reasoning_effort_override': None, 'workers': 5,
            'command': command(args.model), 'generation_n': len(source), 'evaluation_n': len(evaluation),
            'excluded_evaluation_ids': sorted(set(by_id) - {r['id'] for r in evaluation}),
            'questions_sha256': sha(args.questions), 'eval_questions_sha256': sha(args.eval_questions),
            'implementation_sha256': sha(Path(__file__)),
            'cli_version': subprocess.run(['claude', '--version'], capture_output=True, text=True, check=True).stdout.strip()}
    write(args.out_dir / 'plan.json', spec, frozen=True)
    write(args.out_dir / 'generation_inputs.json', inputs, frozen=True)
    write(args.out_dir / 'evaluation_questions.json', evaluation, frozen=True)
    (args.out_dir / 'isolated_cwd').mkdir(exist_ok=True)
    return spec, inputs


def job(row, spec, out):
    signature = digest([spec, row])
    record_path = out / 'records' / (row['id'] + '.json')
    if record_path.exists():
        old = json.loads(record_path.read_text())
        assert old['signature'] == signature
        if old['status'] == 'complete':
            return old
    start = time.time()
    errors = []
    for attempt in range(1, 3):
        if STOP.is_set():
            break
        try:
            raw = invoke(spec['command'], row['question'], out / 'isolated_cwd', 240)
            value, served = parse_result(raw, spec['model'])
            write(out / 'raw' / (row['id'] + '.json'), value)
            record = {**row, 'method': 'plain', 'status': 'complete', 'signature': signature,
                      'response': value['result'].strip(), 'model_used': served,
                      'details': {k: value.get(k) for k in ('usage', 'modelUsage', 'stop_reason', 'duration_ms',
                                                           'duration_api_ms', 'session_id', 'num_turns', 'total_cost_usd')},
                      'attempts': attempt, 'prior_errors': errors, 'wall_seconds': time.time() - start}
            write(record_path, record)
            return record
        except Exception as exc:
            detail = f'{type(exc).__name__}: {str(exc)[:700]}'
            errors.append(detail)
            fatal = any(word in detail.lower() for word in ('oauth', 'not logged in', 'authentication',
                        'usage limit', 'rate_limit', 'rate limit', 'quota', 'token has expired', 'model mismatch'))
            if fatal:
                terminate_active()
                break
            if attempt < 2:
                STOP.wait(5)
    record = {**row, 'method': 'plain', 'status': 'stopped' if STOP.is_set() else 'failed',
              'signature': signature, 'errors': errors, 'wall_seconds': time.time() - start}
    write(record_path, record)
    return record


def summarize(out, spec, inputs, state):
    records = [json.loads(p.read_text()) for p in sorted((out / 'records').glob('*.json'))]
    by_id = {r['id']: r for r in inputs}
    assert len({r['id'] for r in records}) == len(records)
    assert all(r['id'] in by_id and r['signature'] == digest([spec, by_id[r['id']]]) for r in records)
    counts = Counter(r['status'] for r in records)
    completed = [r for r in records if r['status'] == 'complete']
    evaluation = json.loads((out / 'evaluation_questions.json').read_text())
    eval_ids = {r['id'] for r in evaluation}
    summary = {'state': state, 'target': len(inputs), 'counts': dict(counts),
               'pending': len(inputs) - len(completed), 'evaluation_completed': sum(r['id'] in eval_ids for r in completed),
               'models': dict(Counter(r['model_used'] for r in completed)),
               'failed_ids': [r['id'] for r in records if r['status'] != 'complete'],
               'updated_unix': time.time(), 'pid': os.getpid(), 'workers': spec['workers']}
    write(out / 'status.json', summary)
    for name, selected in [('answers_all.jsonl', completed), ('answers_eval732.jsonl', [r for r in completed if r['id'] in eval_ids])]:
        temp = out / (name + '.tmp')
        temp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
        temp.replace(out / name)
    return summary


def generate(args, spec, inputs):
    signal.signal(signal.SIGTERM, terminate_active)
    signal.signal(signal.SIGINT, terminate_active)
    out = args.out_dir
    with output_lock(out / 'generation'):
        if (out / 'STOP').exists():
            raise RuntimeError('STOP marker exists; refusing to start requests')
        pending = []
        for r in inputs:
            p = out / 'records' / (r['id'] + '.json')
            old = json.loads(p.read_text()) if p.exists() else None
            if old:
                assert old['signature'] == digest([spec, r])
            if not old or old['status'] != 'complete':
                pending.append(r)
        if args.limit:
            pending = pending[:args.limit]
        cursor = iter(pending)
        with ThreadPoolExecutor(max_workers=5) as pool:
            active = {}
            def fill():
                while not STOP.is_set() and len(active) < 5:
                    row = next(cursor, None)
                    if row is None:
                        break
                    active[pool.submit(job, row, spec, out)] = row['id']
            fill()
            while active:
                if (out / 'STOP').exists():
                    terminate_active()
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    key = active.pop(future)
                    record = future.result()
                    status = summarize(out, spec, inputs, 'stopping' if STOP.is_set() else 'running')
                    print(json.dumps({'id': key, 'status': record['status'], 'completed': status['counts'].get('complete', 0),
                                      'target': status['target'], 'errors': record.get('errors')}, ensure_ascii=False), flush=True)
                fill()
        status = summarize(out, spec, inputs, 'stopped' if STOP.is_set() else 'finished_batch')
        if status['counts'].get('complete') == len(inputs):
            status = summarize(out, spec, inputs, 'complete')
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return 2 if STOP.is_set() or status['failed_ids'] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'generate', 'status'])
    ap.add_argument('--questions', type=Path, required=True)
    ap.add_argument('--eval-questions', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--model', default='claude-sonnet-5')
    ap.add_argument('--limit', type=int, default=0, help='Optional bounded smoke batch; resumes the same run')
    args = ap.parse_args()
    args.out_dir = args.out_dir.resolve()
    spec, inputs = prepare(args)
    if args.stage == 'generate':
        return generate(args, spec, inputs)
    if args.stage == 'status' and (args.out_dir / 'status.json').exists():
        print((args.out_dir / 'status.json').read_text())
    else:
        print(json.dumps(summarize(args.out_dir, spec, inputs, 'prepared'), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
