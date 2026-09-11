"""Ten fixed NFP answers, two independent Terra judgments each; hard 20-call cap.

prepare/report are offline. score requires a completed human review first.
Existing experiment scores are never edited. No retries, preflight or reuse.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pilot_judge import audit
from scripts.run_judge import build_prompt
from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import SHARPNESS_RUBRIC_NFP, parse_score
from src.llm_backend import make_caller
from src.pilot import digest, file_digest, frozen_json, output_lock

MODEL = 'gpt-5.6-terra'


def fixed_text(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        path.write_text(text, encoding='utf-8')


def prepare(pilot, out, config, seed=17):
    pilot, out = Path(pilot), Path(out)
    # Validate full response/score provenance before sampling.
    audit(pilot, 'gpt-5.6-sol', 'steering_L14_a0')
    folder = pilot / 'dev'
    responses = {r['id']: r for r in read_jsonl(folder / 'plain.jsonl')}
    zero = {r['id']: r for r in read_jsonl(folder / 'steering_L14_a0.jsonl')}
    questions = {r['id']: r for r in read_jsonl(pilot / 'split/dev.jsonl')}
    paths = [folder / f'{name}_judge_codex_gpt-5.6-sol.jsonl'
             for name in ('plain', 'steering_L14_a0')]
    scores = [{r['question_id']: r for r in read_jsonl(p)} for p in paths]
    ids = sorted(i for i in responses if responses[i]['set'] == 'nfp')
    if any(responses[i]['response'] != zero[i]['response'] for i in ids):
        raise ValueError('NFP answers differ; cannot treat scores as repeat judgments')
    flipped = [i for i in ids if scores[0][i]['sharpness'] != scores[1][i]['sharpness']]
    stable = [i for i in ids if i not in flipped]
    if len(flipped) != 5 or len(stable) < 5:
        raise ValueError('This fixed check expects 5 disagreements and at least 5 unchanged judgments')
    rng = random.Random(seed)
    picked = flipped + rng.sample(stable, 5)
    rng.shuffle(picked)  # human review hides cohort and previous scores
    cfg = load_config(config)['judge']
    examples_fpq = load_json(cfg['examples_fpq'])
    examples_nfp = load_json(cfg['examples_nfp'])
    cases = []
    for qid in picked:
        _, prompt = build_prompt(responses[qid], questions[qid], examples_fpq, examples_nfp)
        cases.append({'id': qid, 'question': questions[qid]['question'],
                      'reference': questions[qid]['hallucination_text'],
                      'answer': responses[qid]['response'], 'prompt': prompt,
                      'cohort': 'disagreement' if qid in flipped else 'unchanged'})
    order = []
    for repeat in (1, 2):
        shuffled = picked[:]
        rng.shuffle(shuffled)
        order.extend({'case_id': qid, 'repeat': repeat, 'id': f'{qid}::r{repeat}'} for qid in shuffled)
    plan = {'version': 'terra-nfp-check-v1', 'model': MODEL, 'backend': 'codex',
            'seed': seed, 'budget': 20, 'cases': cases, 'order': order,
            'rubric_hash': file_digest(ROOT / 'src/judge_prompts.py'),
            'transport_hash': file_digest(ROOT / 'src/llm_backend.py'),
            'examples_hash': digest([examples_fpq, examples_nfp]),
            'sources': {str(p.resolve()): file_digest(p) for p in paths}}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'plan.json', plan)
        lines = ['# NFP human review — before Terra scoring', '', SHARPNESS_RUBRIC_NFP, '',
                 'Evaluate invented premise objections, not general medical QA accuracy.',
                 'The reference describes a possible hallucinated objection, not a fact to assert.',
                 'Do not penalize an answer only for failing to rebut the patient.',
                 'For -1 quote the offending answer passage; for +1 explain absence.',
                 'Ambiguous cases may remain unresolved; do not invent a gold label.', '']
        for case in cases:
            lines.extend([f"## {case['id']}", '', '**Question**', '', case['question'], '',
                          '**Possible hallucination reference**', '', case['reference'], '',
                          '**Answer**', '', case['answer'], ''])
        fixed_text(out / 'review.md', '\n'.join(lines))
        if not (out / 'human_review.tsv').exists():
            with (out / 'human_review.tsv').open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['id', 'score', 'rationale', 'reviewer'])
                writer.writerows([c['id'], '', '', ''] for c in cases)
    print(f'Prepared 10 answers, 20 planned calls. No model called. Review: {out / "review.md"}')


def human_labels(out, plan):
    with (out / 'human_review.tsv').open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    labels = {r['id']: r for r in rows}
    if len(rows) != 10 or set(labels) != {c['id'] for c in plan['cases']}:
        raise ValueError('Human review must contain each of the 10 IDs exactly once')
    for row in rows:
        if (row.get('score') not in ('-1', '1') or not row.get('rationale', '').strip()
            or not row.get('reviewer', '').strip()):
            raise ValueError('Complete human_review.tsv: score -1/+1, rationale and reviewer; no calls made')
    return labels


def events_by_id(out, plan, run_hash):
    path = out / 'attempts.jsonl'
    events = list(read_jsonl(path)) if path.exists() else []
    grouped = {}
    expected = {job['id'] for job in plan['order']}
    for event in events:
        if event['id'] not in expected or event['run_hash'] != run_hash:
            raise ValueError('Attempt ledger differs from frozen run')
        group = grouped.setdefault(event['id'], [])
        expected_status = 'started' if not group else 'finished'
        if len(group) >= 2 or event['status'] != expected_status:
            raise ValueError('Invalid attempt ledger order')
        group.append(event)
    return grouped


def score(out, timeout=180):
    out = Path(out)
    with output_lock(out / 'score'):
        plan = load_json(out / 'plan.json')
        if (plan['model'] != MODEL or plan['backend'] != 'codex' or plan['budget'] != 20
            or len(plan['order']) != 20 or len({j['id'] for j in plan['order']}) != 20):
            raise ValueError('Invalid fixed-budget plan')
        if plan['transport_hash'] != file_digest(ROOT / 'src/llm_backend.py'):
            raise ValueError('Transport changed after preparation')
        human_labels(out, plan)
        run = {'plan_hash': digest(plan), 'human_hash': file_digest(out / 'human_review.tsv')}
        frozen_json(out / 'run.json', run)
        run_hash = digest(run)
        previous = events_by_id(out, plan, run_hash)
        cases = {c['id']: c for c in plan['cases']}
        print(f'Remaining calls: {20 - len(previous)}; no retries or preflight calls', flush=True)
        call = make_caller('codex', MODEL, timeout=timeout)
        for job in plan['order']:
            if job['id'] in previous:
                continue  # started-but-interrupted calls also consume the budget
            common = {**job, 'run_hash': run_hash}
            append_jsonl(out / 'attempts.jsonl', {
                **common, 'status': 'started', 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                raw, used = call(cases[job['case_id']]['prompt'])
                parsed, ok = parse_score(raw)
                valid = ok and parsed.get('Sharpness') in (-1, 1) and used == MODEL
                append_jsonl(out / 'attempts.jsonl', {
                    **common, 'status': 'finished', 'model': used, 'raw': raw,
                    'valid': valid, 'score': parsed.get('Sharpness') if valid else None,
                    'reason': parsed.get('Reason') if ok else None})
                print(f"{job['id']}: {parsed.get('Sharpness') if valid else 'invalid'}", flush=True)
                if used != MODEL:
                    raise SystemExit('Unexpected model; stopped without retry')
            except Exception as exc:
                append_jsonl(out / 'attempts.jsonl', {
                    **common, 'status': 'finished', 'valid': False, 'error': str(exc)})
                raise SystemExit('Call failed; stopped. This attempt will not be retried.') from exc
        print('Finished fixed call budget. Run report; failed/interrupted calls remain missing.')


def report(out):
    out = Path(out)
    plan, run = load_json(out / 'plan.json'), load_json(out / 'run.json')
    if run != {'plan_hash': digest(plan), 'human_hash': file_digest(out / 'human_review.tsv')}:
        raise ValueError('Plan or human review changed after scoring')
    labels = human_labels(out, plan)
    events = events_by_id(out, plan, digest(run))
    valid = {key: rows[-1] for key, rows in events.items()
             if rows[-1]['status'] == 'finished' and rows[-1].get('valid')}
    lines = ['# Terra NFP judge check', '',
             f'Attempted {len(events)}/20; valid {len(valid)}/20; missing/invalid {20-len(valid)}.',
             'Selected diagnostic sample, not an unbiased NFP accuracy estimate.', '',
             '| ID | Cohort | Human | Repeat 1 | Repeat 2 |', '|---|---|---:|---:|---:|']
    for case in plan['cases']:
        qid = case['id']
        vals = [valid.get(f'{qid}::r{r}', {}).get('score', 'missing') for r in (1, 2)]
        lines.append(f"| {qid} | {case['cohort']} | {labels[qid]['score']} | {vals[0]} | {vals[1]} |")
    for cohort in ('disagreement', 'unchanged'):
        ids = [c['id'] for c in plan['cases'] if c['cohort'] == cohort]
        paired = [i for i in ids if all(f'{i}::r{r}' in valid for r in (1, 2))]
        agree = sum(valid[f'{i}::r1']['score'] == valid[f'{i}::r2']['score'] for i in paired)
        available = [v for v in valid.values() if v['case_id'] in ids]
        correct = sum(v['score'] == int(labels[v['case_id']]['score']) for v in available)
        lines.append(f'\n{cohort}: repeat agreement {agree}/{len(paired)} pairs; '
                     f'human agreement {correct}/{len(available)} judgments.')
    for key, value in valid.items():
        lines.extend(['', f'## {key}', '', str(value.get('reason') or value['raw'])])
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--pilot-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/gemma2_9b.yaml')
    p.add_argument('--seed', type=int, default=17)
    for name in ('score', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.pilot_dir, args.out_dir, args.config, args.seed)
    elif args.stage == 'score':
        score(args.out_dir)
    else:
        report(args.out_dir)


if __name__ == '__main__':
    main()
