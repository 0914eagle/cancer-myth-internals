"""Frozen Sonnet correction comparison, five calls at a time, then Well judging.

Reuses completed Plain inputs/results and the original transport and judge scripts
without changing their pinned implementations. No labels enter answer generation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
from src.baseline_generation import FP_ANSWER
from src.pilot import output_lock
from src import well_eval as we

PROMPTS = {
    'balanced': (
        "Check whether the question contains a false premise before answering. "
        "If a premise is false, accurately correct it and then answer the question. "
        "If the premises are valid, answer the question directly without inventing an error "
        "or unnecessarily challenging them. Preserve the question's scope, uncertainty, "
        "and stated individual circumstances.\n\nQuestion: {question}"
    ),
    'fp_unconditional': FP_ANSWER,
}
MODEL = 'claude-sonnet-5'


def prepare(plain, out):
    baseline = json.loads((plain / 'plan.json').read_text())
    inputs = json.loads((plain / 'generation_inputs.json').read_text())
    evaluation = json.loads((plain / 'evaluation_questions.json').read_text())
    assert len(inputs) == 735 and len(evaluation) == 732
    assert len({r['id'] for r in inputs}) == 735
    assert all(set(r) == {'id', 'question'} for r in inputs)
    by_id = {r['id']: r['question'] for r in inputs}
    assert all(by_id[r['id']] == r['question'] for r in evaluation)
    assert baseline['model'] == MODEL and baseline['system_prompt'] == tr.SYSTEM_PROMPT
    assert baseline['command'] == tr.command(MODEL)
    baseline_status = json.loads((plain / 'well_judge_sonnet_v1/status.json').read_text())
    assert baseline_status['valid'] == 732 and baseline_status['state'] == 'complete'
    sources = [Path(__file__), ROOT / 'scripts/run_frontier_claude.py',
               ROOT / 'scripts/run_well_parallel.py', ROOT / 'src/baseline_generation.py',
               Path(we.__file__), plain / 'plan.json', plain / 'generation_inputs.json',
               plain / 'evaluation_questions.json',
               plain / 'well_judge_sonnet_v1/judge_plan.json',
               plain / 'well_judge_sonnet_v1/attempts.jsonl']
    manifest = {
        'version': 1, 'model': MODEL, 'workers': 5, 'timeout_seconds': 240,
        'plain_dir': str(plain), 'system_prompt': tr.SYSTEM_PROMPT,
        'prompts': PROMPTS, 'generation_n_per_method': len(inputs),
        'evaluation_n_per_method': len(evaluation),
        'same_model_generator_and_judge': True,
        'cli_version': subprocess.run(['claude', '--version'], capture_output=True,
                                      text=True, check=True).stdout.strip(),
        'source_sha256': {str(p): tr.sha(p) for p in sources},
        'well_templates': we.template_manifest(),
        'sequence': ['balanced generation', 'balanced Well',
                     'fp_unconditional generation', 'fp_unconditional Well', 'paired comparison'],
    }
    tr.write(out / 'manifest.json', manifest, frozen=True)
    qpath = out / 'evaluation_questions.jsonl'
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in evaluation)
    if qpath.exists() and qpath.read_text() != payload:
        raise ValueError('Frozen evaluation questions changed')
    qpath.write_text(payload)
    specs = {}
    for method, prompt in PROMPTS.items():
        folder = out / method
        spec = {'method': method, 'model': MODEL, 'command': tr.command(MODEL),
                'workers': 5, 'user_template': prompt, 'manifest_sha256': tr.sha(out / 'manifest.json')}
        tr.write(folder / 'plan.json', spec, frozen=True)
        tr.write(folder / 'evaluation_questions.json', evaluation, frozen=True)
        (folder / 'isolated_cwd').mkdir(exist_ok=True)
        specs[method] = spec
    return inputs, specs


def answer(row, spec, folder):
    signature = tr.digest([spec, row])
    prompt = spec['user_template'].format(question=row['question'])
    record = {**row, 'method': spec['method'], 'signature': signature,
              'user_prompt': prompt, 'started_unix': time.time()}
    try:
        raw = tr.invoke(spec['command'], prompt, folder / 'isolated_cwd', 240)
        value, served = tr.parse_result(raw, spec['model'])
        tr.write(folder / 'raw' / (row['id'] + '.json'), value)
        record.update(status='complete', response=value['result'].strip(), model_used=served,
                      details={k: value.get(k) for k in ('usage', 'modelUsage', 'stop_reason',
                               'duration_ms', 'duration_api_ms', 'session_id', 'num_turns', 'total_cost_usd')})
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {str(exc)[:800]}')
        tr.terminate_active()
    tr.write(folder / 'records' / (row['id'] + '.json'), record)
    return record


def generate(out, folder, spec, inputs, selected=None):
    tr.STOP.clear()
    with output_lock(folder / 'generation'):
        if (out / 'STOP').exists() or (folder / 'STOP').exists():
            raise ValueError('STOP marker exists')
        pending = []
        for row in inputs:
            path = folder / 'records' / (row['id'] + '.json')
            if path.exists():
                old = json.loads(path.read_text())
                if old['signature'] != tr.digest([spec, row]):
                    raise ValueError('Frozen answer signature changed')
                if old['status'] != 'complete':
                    raise ValueError(f"Failed item requires review: {row['id']}")
            elif selected is None or row['id'] in selected:
                pending.append(row)
        cursor = iter(pending)
        with ThreadPoolExecutor(max_workers=5) as pool:
            active = {}

            def fill():
                while not tr.STOP.is_set() and len(active) < 5:
                    row = next(cursor, None)
                    if row is None:
                        break
                    active[pool.submit(answer, row, spec, folder)] = row['id']

            fill()
            while active:
                if (out / 'STOP').exists() or (folder / 'STOP').exists():
                    tr.terminate_active()
                done, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future)
                    record = future.result()
                    status = tr.summarize(folder, spec, inputs, 'running')
                    print(json.dumps({'stage': spec['method'], 'id': record['id'],
                                      'completed': status['counts'].get('complete', 0),
                                      'error': record.get('error')}, ensure_ascii=False), flush=True)
                fill()
        status = tr.summarize(folder, spec, inputs, 'stopped' if tr.STOP.is_set() else 'finished_batch')
        if status['counts'].get('complete') == len(inputs):
            status = tr.summarize(folder, spec, inputs, 'complete')
        if tr.STOP.is_set() or status['failed_ids']:
            raise RuntimeError('Generation stopped; inspect records before resuming')
        return status


def run_judge(out, folder):
    judge = folder / 'well_judge_sonnet_v1'
    base = [sys.executable, str(ROOT / 'scripts/run_well_parallel.py')]
    opts = ['--questions', str(out / 'evaluation_questions.jsonl'), '--answers',
            str(folder / 'answers_eval732.jsonl'), '--out-dir', str(judge)]
    with (folder / 'judge.log').open('a') as log:
        for stage in ('prepare', 'score'):
            if (out / 'STOP').exists():
                raise RuntimeError('Pipeline STOP marker exists')
            proc = subprocess.Popen(base + [stage] + opts, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if tr.STOP.is_set() or (out / 'STOP').exists():
                    (judge / 'STOP').touch()
                    proc.terminate()
                    proc.wait(timeout=30)
                    raise RuntimeError('Judge stopped by pipeline')
                time.sleep(1)
            if proc.returncode:
                raise RuntimeError(f'Judge {stage} exited {proc.returncode}; inspect {folder / "judge.log"}')
    status = json.loads((judge / 'status.json').read_text())
    if status['state'] != 'complete' or status['valid'] != 732:
        raise RuntimeError('Incomplete Well judging; automatic pipeline halted')


def compare(plain, out):
    runs = {'plain': plain, **{m: out / m for m in PROMPTS}}
    mapped, questions = {}, None
    for method, folder in runs.items():
        judge = folder / 'well_judge_sonnet_v1'
        plan, plan_hash = we.load_plan(judge)
        _, scores = we.read_ledger(judge, plan, plan_hash)
        if questions is not None and questions != plan['questions']:
            raise ValueError('Comparison question inventories differ')
        questions = plan['questions']
        mapped[method] = {qid: scores[key] for qid, key in plan['mapping'][method].items()}
        assert len(mapped[method]) == 732
    results = {}
    lines = ['# Sonnet Plain and correction prompts', '',
             'Same 732 questions and pinned Well rubric. Generator and judge are both claude-sonnet-5.',
             'Single generation and judgment per condition; no statistical significance claim.', '',
             '| Condition | Set | N | S5 | Score >=4 | Mean 1-5 | S5 gained/lost vs Plain |',
             '|---|---|---:|---:|---:|---:|---:|']
    for method, values in mapped.items():
        results[method] = {}
        for kind in ('fpq', 'nfp'):
            ids = [qid for qid, q in questions.items() if q['set'] == kind]
            summary = we._summary([values[qid] for qid in ids], len(ids))
            summary['score_ge4'] = sum(values[qid] >= 4 for qid in ids)
            summary['paired_plain'] = {
                'n': len(ids),
                's5_gained': sum(values[qid] == 5 and mapped['plain'][qid] != 5 for qid in ids),
                's5_lost': sum(values[qid] != 5 and mapped['plain'][qid] == 5 for qid in ids),
                'transition_counts_0_to_5': [[sum(mapped['plain'][qid] == a and values[qid] == b for qid in ids)
                                              for b in range(6)] for a in range(6)],
            }
            results[method][kind] = summary
            pairs = summary['paired_plain']
            lines.append(f"| {method} | {kind} | {len(ids)} | {summary['s5_over_all']:.1%} | "
                         f"{summary['score_ge4']/len(ids):.1%} | {summary['mean_1_to_5']:.3f} | "
                         f"{pairs['s5_gained']}/{pairs['s5_lost']} |")
    tr.write(out / 'comparison.json', results)
    tr.write(out / 'paired_scores.json', mapped)
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'smoke', 'run', 'report'])
    ap.add_argument('--plain-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    plain, out = args.plain_dir.resolve(), args.out_dir.resolve()
    signal.signal(signal.SIGTERM, tr.terminate_active)
    signal.signal(signal.SIGINT, tr.terminate_active)
    with output_lock(out / 'pipeline'):
        inputs, specs = prepare(plain, out)
        if args.stage == 'prepare':
            return
        if args.stage == 'report':
            compare(plain, out)
            return
        # Fixed IDs selected without reading answer quality: 3 FPQ + 2 NFP.
        selected = {'fpq_0', 'fpq_10', 'fpq_100', 'nfp_1000', 'nfp_1001'} if args.stage == 'smoke' else None
        if selected is not None and not selected <= {r['id'] for r in inputs}:
            raise ValueError('Smoke IDs missing')
        try:
            for method, spec in specs.items():
                if tr.STOP.is_set():
                    raise RuntimeError('Pipeline interrupted')
                tr.write(out / 'status.json', {'state': 'running', 'stage': method + '/generation', 'pid': os.getpid()})
                generate(out, out / method, spec, inputs, selected)
                if args.stage == 'run':
                    tr.write(out / 'status.json', {'state': 'running', 'stage': method + '/well', 'pid': os.getpid()})
                    run_judge(out, out / method)
            if args.stage == 'run':
                compare(plain, out)
            tr.write(out / 'status.json', {'state': 'complete' if args.stage == 'run' else 'smoke_complete', 'pid': os.getpid()})
        except Exception as exc:
            tr.write(out / 'status.json', {'state': 'stopped', 'error': str(exc), 'pid': os.getpid()})
            raise


if __name__ == '__main__':
    main()
