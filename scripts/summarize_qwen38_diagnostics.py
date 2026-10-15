"""Offline joins of Qwen3.8 judgments and saved Well scores; never calls a model."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_gemma4_diagnostics import ANSWER_STAGES
from scripts.run_frontier_claude import write
from src import well_eval as we


def score_summary(values, expected):
    valid = [v for v in values if v is not None]
    counts = Counter(valid)
    readable = [v for v in valid if v > 0]
    return {'expected': expected, 'valid': len(valid), 'missing': expected - len(valid),
            'counts_0_to_5': {str(i): counts[i] for i in range(6)},
            'mean_0_to_5': sum(valid) / len(valid) if valid else None,
            'mean_1_to_5': sum(readable) / len(readable) if readable else None,
            'at_least_4_count': sum(v >= 4 for v in valid),
            'at_least_4_fraction_valid': sum(v >= 4 for v in valid) / len(valid) if valid else None,
            'score_5_fraction_valid': counts[5] / len(valid) if valid else None}


def summarize(run):
    rows = json.loads((run / 'evaluation_questions.json').read_text())
    scores = {}
    for method in ANSWER_STAGES:
        out = run / 'well' / method
        if not (out / 'judge_plan.json').exists():
            continue
        plan, h = we.load_plan(out)
        _, ledger = we.read_ledger(out, plan, h)
        scores[method] = {qid: ledger.get(job) for qid, job in plan['mapping'][method].items()}
    decisions = {}
    for method in ('direct', 'cot', 'verify_extracted', 'verify_annotated'):
        values = {}
        for path in (run / 'records' / method).glob('*.json'):
            r = json.loads(path.read_text())
            if r['status'] == 'complete':
                values[r['id']] = r['result']['has_false_premise']
        decisions[method] = values
    common = {r['id'] for r in rows if all(scores.get(m, {}).get(r['id']) is not None
                                          for m in ('plain', 'fp_unconditional'))}
    routed = {}
    for method in ('direct', 'cot', 'verify_extracted'):
        routed[method + '_routed'] = {
            qid: scores['fp_unconditional' if decision else 'plain'][qid]
            for qid, decision in decisions[method].items() if qid in common}
    answer_metrics = {}
    for method, values in {**scores, **routed}.items():
        answer_metrics[method] = {}
        for kind in ('fpq', 'nfp'):
            ids = [r['id'] for r in rows if r['set'] == kind]
            metric = score_summary([values.get(qid) for qid in ids], len(ids))
            paired = [qid for qid in ids if values.get(qid) is not None and
                      scores.get('plain', {}).get(qid) is not None]
            metric['paired_with_plain'] = {
                'n': len(paired),
                'rescue_to_at_least_4': sum(scores['plain'][q] < 4 <= values[q] for q in paired),
                'harm_below_4': sum(values[q] < 4 <= scores['plain'][q] for q in paired)}
            answer_metrics[method][kind] = metric
    gate_metrics = {}
    for method, values in decisions.items():
        gate_metrics[method] = {'auroc': None}
        for kind in ('fpq', 'nfp'):
            ids = [r['id'] for r in rows if r['set'] == kind]
            valid = [q for q in ids if q in values]
            positive = sum(values[q] for q in valid)
            gate_metrics[method][kind] = {'expected': len(ids), 'valid': len(valid),
                                         'positive': positive, 'negative': len(valid) - positive,
                                         'positive_rate': positive / len(valid) if valid else None}
    links = []
    for row in rows:
        qid = row['id']
        links.append({'id': qid, 'set': row['set'],
                      **{m + '_gate': values.get(qid) for m, values in decisions.items()},
                      **{m + '_well': values.get(qid) for m, values in {**scores, **routed}.items()}})
    out = run / 'analysis'
    out.mkdir(exist_ok=True)
    with (out / 'question_links.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(links[0]))
        writer.writeheader()
        writer.writerows(links)
    result = {'gate': gate_metrics, 'answers': answer_metrics,
              'routing_common_plain_correction_n': len(common),
              'interpretation': 'Descriptive paired outputs, not proof of a causal mechanism. Missing scores are not failures.'}
    write(out / 'summary.json', result)
    run_plan = json.loads((run / 'plan.json').read_text())
    lines = [f"# {run_plan['model']} — thinking {run_plan['thinking_enabled']}", '',
             'Missing outputs and scores remain missing. AUROC of reported confidence is omitted.', '',
             '| Answer method | Set | Valid / expected | Well ≥4 | Well =5 | Mean 0–5 |',
             '|---|---|---:|---:|---:|---:|']
    pct = lambda x: '—' if x is None else f'{100*x:.1f}%'
    for method, groups in answer_metrics.items():
        for kind, value in groups.items():
            mean = value['mean_0_to_5']
            lines.append(f"| {method} | {kind} | {value['valid']}/{value['expected']} | "
                         f"{pct(value['at_least_4_fraction_valid'])} | {pct(value['score_5_fraction_valid'])} | "
                         + ('—' if mean is None else f'{mean:.3f}') + ' |')
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    summarize(args.run_dir)
