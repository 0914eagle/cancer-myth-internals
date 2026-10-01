"""Luna general CoT and review-conditioned answers, then frozen Sonnet Well.

Reuses all 732 saved Luna reviews, preserving their provenance. No label or
reference is passed to generation. Independent calls, medium, 240s, no tools.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import run_frontier_luna as base
from scripts import run_luna_direct_cot_gate as gate
from scripts import run_luna_premise_gate as transport
from src.baseline_generation import COT_REASON
from src.pilot import COT_ANSWER, output_lock
from src.llm_backend import parse_codex_banner_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/frontier_cli/sonnet_plain_20260922_v1/evaluation_questions.json'
REVIEWS = ROOT / 'results/frontier_cli/luna_direct_cot_20260923_v1'
OPTIONS = ROOT / 'results/frontier_cli/luna_plain_20260922_v1/transport_options.json'
METHODS = {'zero_shot_cot_one_step': COT_REASON, 'premise_review_answer': COT_ANSWER}
io = base.tr


def prepare(out):
    rows = json.loads(SOURCE.read_text())
    assert Counter(r['set'] for r in rows) == {'fpq': 583, 'nfp': 149}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'isolated_cwd').mkdir(exist_ok=True)
    system = out / 'system_prompt.txt'
    if system.exists():
        assert system.read_text() == io.SYSTEM_PROMPT
    system.write_text(io.SYSTEM_PROMPT)
    opts = json.loads(OPTIONS.read_text())
    opts = [f'model_instructions_file="{system}"' if s.startswith('model_instructions_file=') else s for s in opts]
    assert opts[opts.index('--model') + 1] == base.MODEL
    assert 'model_reasoning_effort="medium"' in opts
    jobs = []
    groups = [[r for r in rows if r['set'] == label] for label in ('fpq', 'nfp')]
    ordered = [r for i in range(max(map(len, groups))) for g in groups for r in g[i:i+1]]
    for q in ordered:
        path = REVIEWS / 'review' / q['id'] / 'record.json'
        rec = json.loads(path.read_text())
        assert rec['status'] == 'complete' and rec['prompt'] == gate.REVIEW.format(question=q['question'])
        for method, template in METHODS.items():
            prompt = template.format(question=q['question'], review=rec['result']['review'])
            jobs.append(dict(id=q['id'], method=method, question=q['question'], prompt=prompt,
                             review_source=str(path) if method == 'premise_review_answer' else None,
                             review_sha256=io.sha(path) if method == 'premise_review_answer' else None))
    plan = dict(model=base.MODEL, effort='medium', system=io.SYSTEM_PROMPT, options=opts,
                prompts=METHODS, n_questions=732, n_jobs=1464, timeout=240, max_workers=12,
                judge=base.JUDGE, source_sha256=io.sha(SOURCE), jobs_sha256=io.digest(jobs),
                review_reuse='Saved Luna review stage, same question and COT_REVIEW; no regeneration',
                implementation_sha256=io.sha(Path(__file__)))
    io.write(out / 'manifest.json', plan, frozen=True)
    io.write(out / 'jobs.json', jobs, frozen=True)
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    p = out / 'evaluation_questions.jsonl'
    if p.exists(): assert p.read_text() == content
    p.write_text(content)
    return plan, jobs


def preflight(out, plan):
    path = out / 'preflight.json'
    if not path.exists():
        proc = subprocess.run(['codex', 'exec'] + plan['options'] + ['-'],
                              input='Reply with the single word READY.', text=True,
                              capture_output=True, timeout=240, cwd=out / 'isolated_cwd')
        io.write(path, dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                            model=parse_codex_banner_model(proc.stdout, proc.stderr)))
    value = json.loads(path.read_text())
    assert value['returncode'] == 0 and value['stdout'].strip() == 'READY' and value['model'] == base.MODEL, 'Preflight failed'


def invoke(out, plan, job):
    folder = out / job['method'] / 'native' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    record_path = out / job['method'] / 'records' / (job['id'] + '.json')
    record = dict(job, signature=io.digest([plan, job]), status='started', started_unix=time.time())
    io.write(record_path, record)
    try:
        final = folder / 'answer.txt'
        cmd = ['codex', 'exec'] + plan['options'] + ['--json', '--output-last-message', str(final), '-']
        raw = io.invoke(cmd, job['prompt'], out / 'isolated_cwd', 240)
        (folder / 'events.jsonl').write_text(raw)
        answer = final.read_text()
        events, reconnects = transport.parse_luna_events(raw, answer)
        record.update(status='complete', response=answer.strip(), model=base.MODEL,
                      reconnects=reconnects, usage=next(e['usage'] for e in events if e['type'] == 'turn.completed'))
    except Exception as exc:
        record.update(status='failed', error=str(exc)[:2000])
    record['wall_seconds'] = time.time() - record['started_unix']
    io.write(record_path, record)
    return record


def generate(out, plan, jobs, workers, limit):
    pending, complete = [], 0
    for job in jobs:
        path = out / job['method'] / 'records' / (job['id'] + '.json')
        if path.exists():
            rec = json.loads(path.read_text())
            assert rec['signature'] == io.digest([plan, job]), 'Changed job'
            assert rec['status'] == 'complete', f'Inspect failed/interrupted record: {path}'
            complete += 1
        else:
            pending.append(job)
    if limit: pending = pending[:limit]
    def report(state):
        io.write(out / 'status.json', dict(state=state, completed=complete, expected=len(jobs),
                                         workers=workers, pid=os.getpid(), updated_unix=time.time()))
    report('generating')
    cursor = iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}
        while True:
            if (out / 'STOP').exists(): io.terminate_active()
            while not io.STOP.is_set() and len(active) < workers:
                job = next(cursor, None)
                if job is None: break
                active[pool.submit(invoke, out, plan, job)] = job
            if not active: break
            done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                rec = future.result()
                if rec['status'] != 'complete':
                    io.terminate_active()
                else:
                    complete += 1
                report('generation_stopped' if io.STOP.is_set() else 'generating')
                print(json.dumps(dict(method=job['method'], id=job['id'], completed=complete,
                                      status=rec['status'], error=rec.get('error')), ensure_ascii=False), flush=True)
    if io.STOP.is_set(): raise RuntimeError('Generation stopped; inspect failure record; no automatic retry')
    if complete == len(jobs):
        for method in METHODS:
            rows = []
            for job in jobs:
                if job['method'] != method: continue
                rec = json.loads((out / method / 'records' / (job['id'] + '.json')).read_text())
                rows.append({k: rec[k] for k in ('id', 'question', 'method', 'response', 'status')})
            assert len(rows) == 732
            (out / method / 'answers_eval732.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    report('generation_complete' if complete == len(jobs) else 'smoke_complete')
    return complete == len(jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'generate', 'run', 'judge'])
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--judge-workers', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    assert 1 <= args.workers <= 12 and 1 <= args.judge_workers <= 10
    out = args.out_dir.resolve()
    io.STOP.clear()
    signal.signal(signal.SIGTERM, io.terminate_active)
    signal.signal(signal.SIGINT, io.terminate_active)
    with output_lock(out / 'pipeline'):
        if (out / 'STOP').exists(): raise ValueError('STOP marker exists')
        plan, jobs = prepare(out)
        if args.stage == 'prepare': return
        stage = 'generation'
        try:
            if args.stage != 'judge':
                preflight(out, plan)
                finished = generate(out, plan, jobs, args.workers, args.limit)
                if not finished or args.stage == 'generate': return
            stage = 'well'
            io.write(out / 'status.json', dict(state='judging', pid=os.getpid(), expected_answers=1464))
            base.PROMPTS = METHODS
            base.judging(out, args.judge_workers)
            io.write(out / 'status.json', dict(state='complete', generated=1464, pid=os.getpid()))
        except Exception as exc:
            io.write(out / 'status.json', dict(state=stage+'_blocked', error=str(exc), pid=os.getpid()))
            raise


if __name__ == '__main__':
    main()
