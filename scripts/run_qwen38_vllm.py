"""Paired thinking OFF/ON Qwen3.8-FP8 experiments against a local vLLM server."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_gemma4_diagnostics as base
from scripts import run_frontier_claude as io
from src.pilot import output_lock

MODEL = 'Qwen/Qwen3.8-27B-FP8'
REVISION = '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
BASE_PLAN = ROOT / 'results/gemma4/full_20260926_v2/plan.json'
SAMPLING = {'temperature': 0.7, 'top_p': 0.95, 'top_k': 20,
            'presence_penalty': 0.0, 'repetition_penalty': 1.0}


def prepare(out, thinking, limit=0):
    plan = copy.deepcopy(json.loads(BASE_PLAN.read_text()))
    plan.update(model=MODEL, revision=REVISION, dtype='official FP8 weights / BF16 activations',
                quantization='official fine-grained FP8, 128x128 blocks', engine='vllm',
                attention='vLLM automatic selection; actual backend recorded in server log',
                thinking_enabled=thinking, reasoning_effort='medium' if thinking else None,
                do_sample=True, sampling=SAMPLING, max_new_tokens=8192,
                truncation_policy='One recorded extension from 8192 to 16384 total completion tokens; otherwise invalid',
                limitations=['Matched prompts, sampling and total completion caps across OFF/ON; ON reasoning effort medium.',
                             'Qwen sampling and FP8 differ from greedy BF16 Gemma; not a controlled model-size comparison.',
                             'Reasoning tokens count toward completion budget; only final content is judged.',
                             'Source annotations are privileged input, not a deployed gate.',
                             'Generated explanations do not establish an internal causal mechanism.'],
                source_sha256={str(p): io.sha(p) for p in
                               (Path(__file__), BASE_PLAN, Path(base.__file__), base.SOURCE,
                                base.ANNOTATIONS, Path(base.gate.__file__), Path(base.premise.__file__))})
    rows = json.loads(base.SOURCE.read_text())
    refs = {r['id']: r for r in io.read_rows(base.ANNOTATIONS)}
    assert len(rows) == len({r['id'] for r in rows}) == 732
    assert all(refs[r['id']]['question'] == r['question'] for r in rows if r['id'] in refs)
    if limit:
        groups = [[r for r in rows if r['set'] == k] for k in ('fpq', 'nfp')]
        rows = [r for pair in zip(*groups) for r in pair][:limit]
    plan['expected'] = {k: sum(r['set'] == k for r in rows) for k in ('fpq', 'nfp')}
    plan['missing_annotation_ids'] = [r['id'] for r in rows if r['id'] not in refs]
    io.write(out / 'plan.json', plan, frozen=True)
    io.write(out / 'evaluation_questions.json', rows, frozen=True)
    base.jsonl(out / 'evaluation_questions.jsonl', rows)
    io.write(out / 'generation_inputs.json', [{'id': r['id'], 'question': r['question']} for r in rows], frozen=True)
    io.write(out / 'annotation_inputs.json',
             [{'id': r['id'], 'question': r['question'], 'premises': refs[r['id']]['premises']}
              for r in rows if r['id'] in refs], frozen=True)
    return plan, rows, refs


def completion(url, task, thinking, cap):
    seed = int(hashlib.sha256((task['stage'] + '/' + task['id']).encode()).hexdigest()[:8], 16) % (2**31)
    body = {'model': MODEL, 'messages': [{'role': 'system', 'content': task['system']},
                                       {'role': 'user', 'content': task['prompt']}],
            **SAMPLING, 'max_tokens': cap, 'seed': seed, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': thinking, 'reasoning_effort': 'medium'}}
    req = urllib.request.Request(url.rstrip('/') + '/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.time()
    with urllib.request.urlopen(req, timeout=1800) as f:
        native = json.load(f)
    assert native['model'] == MODEL, 'Unexpected served model'
    assert len(native['choices']) == 1
    choice = native['choices'][0]
    msg = choice['message']
    assert not msg.get('tool_calls'), 'Unexpected tool call'
    content = msg.get('content') or ''
    reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
    assert isinstance(content, str) and isinstance(reasoning, str)
    if '<think>' in content or '</think>' in content:
        raise ValueError('Reasoning parser failed to separate final answer')
    if not thinking and reasoning.strip():
        raise ValueError('Thinking OFF unexpectedly returned reasoning')
    return {'request': body, 'native': native, 'response': content, 'reasoning': reasoning,
            'finish_reason': choice['finish_reason'], 'wall_seconds': time.time() - start,
            'usage': native.get('usage'), 'truncated': choice['finish_reason'] == 'length'}


def invoke(out, task, thinking, url, plan_hash):
    record = {**task, 'plan_hash': plan_hash, 'thinking_enabled': thinking, 'status': 'started'}
    path = out / 'records' / task['stage'] / (task['id'] + '.json')
    io.write(path, record)
    try:
        result = completion(url, task, thinking, 8192)
        attempts = [result]
        if result['truncated']:
            result = completion(url, task, thinking, 16384)
            attempts.append(result)
        record.update(result, attempts=attempts)
        if result['finish_reason'] != 'stop':
            raise ValueError('Incomplete generation: ' + str(result['finish_reason']))
        record['result'] = base.parse(task['stage'], result['response'], task)
        record['status'] = 'complete'
    except Exception as exc:
        record.update(status='invalid', error=f'{type(exc).__name__}: {exc}')
    record['finished_unix'] = time.time()
    io.write(path, record)
    return record


def run_mode(args, mode):
    out = args.out_dir / mode
    thinking = mode == 'thinking_on'
    plan, rows, refs = prepare(out, thinking, args.limit)
    h = io.sha(out / 'plan.json')
    records = {}
    for path in (out / 'records').glob('*/*.json'):
        record = json.loads(path.read_text())
        assert record['plan_hash'] == h
        if record['status'] == 'started':
            raise ValueError('Interrupted request requires review before resuming')
        records[record['stage'], record['id']] = record
    with output_lock(out / 'generation'):
        for stage in plan['stages']:
            if (args.out_dir / 'STOP').exists() or (out / 'STOP').exists():
                base.report(out, plan, rows, records, 'stopped', stage)
                return
            tasks = [base.prompt_for(stage, row, records, refs) for row in rows
                     if (stage, row['id']) not in records]
            tasks = [t for t in tasks if t is not None]
            base.report(out, plan, rows, records, 'running', stage)
            for offset in range(0, len(tasks), args.workers):
                if (args.out_dir / 'STOP').exists() or (out / 'STOP').exists():
                    base.report(out, plan, rows, records, 'stopped', stage)
                    return
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(invoke, out, t, thinking, args.url, h)
                               for t in tasks[offset:offset + args.workers]]
                    for future in as_completed(futures):
                        record = future.result()
                        records[stage, record['id']] = record
                        state = base.report(out, plan, rows, records, 'running', stage)
                        print(json.dumps({'mode': mode, 'stage': stage, 'counts': state['counts'][stage]}), flush=True)
                        if record.get('error', '').startswith(('URLError:', 'HTTPError:', 'TimeoutError:')):
                            raise RuntimeError('Local server request failed; inspect record before restart')
            if stage in base.ANSWER_STAGES:
                answers = [{'id': r['id'], 'question': r['question'], 'method': stage,
                            'response': records[stage, r['id']]['response'], 'status': 'complete'}
                           for r in rows if records.get((stage, r['id']), {}).get('status') == 'complete']
                base.jsonl(out / 'answers' / (stage + '.jsonl'), answers)
                io.write(out / 'answers' / (stage + '.ready.json'),
                         {'valid': len(answers), 'expected': len(rows),
                          'answer_sha256': io.sha(out / 'answers' / (stage + '.jsonl'))})
        base.report(out, plan, rows, records, 'generation_finished')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--url', default='http://127.0.0.1:8038/v1')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--mode', choices=['both', 'thinking_off', 'thinking_on'], default='both')
    args = ap.parse_args()
    assert 1 <= args.workers <= 32 and args.limit >= 0
    for mode in (('thinking_off', 'thinking_on') if args.mode == 'both' else (args.mode,)):
        try:
            run_mode(args, mode)
        except Exception as exc:
            io.write(args.out_dir / mode / 'error.json', {'error': str(exc), 'at': time.time()})
            raise


if __name__ == '__main__':
    main()
