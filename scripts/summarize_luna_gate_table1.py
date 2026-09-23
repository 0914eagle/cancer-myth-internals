"""Export frozen Luna gate scores and summarize with source-group bootstrap.

Standard-library only; no model calls. Recompute a published summary using
--scores docs/reviews/luna_premise_gate_2026-09-23/table1_scores.csv.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(rows):
    data = sorted((float(r['false_score']), r['set'] == 'fpq') for r in rows)
    npos = sum(y for _, y in data)
    nneg = len(data) - npos
    if not npos or not nneg:
        return None
    total, i = 0.0, 0
    while i < len(data):
        j = i + 1
        while j < len(data) and data[j][0] == data[i][0]:
            j += 1
        total += sum(y for _, y in data[i:j]) * (i + 1 + j) / 2
        i = j
    return (total - npos * (npos + 1) / 2) / (npos * nneg)


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] * (hi - x) + values[hi] * (x - lo) if lo != hi else values[lo]


def summarize(rows, seed, bootstraps):
    assert rows and len({r['id'] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    group_rows = list(groups.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstraps):
        sampled = [r for _ in group_rows for r in rng.choice(group_rows)]
        value = auc(sampled)
        if value is not None:
            estimates.append(value)
    result = {'n_valid': len(rows), 'source_groups': len(groups), 'auroc': auc(rows),
              'auroc_ci95': [quantile(estimates, .025), quantile(estimates, .975)],
              'bootstrap_valid_replicates': len(estimates)}
    for label in ('fpq', 'nfp'):
        subset = [r for r in rows if r['set'] == label]
        detected = sum(float(r['false_score']) >= .5 for r in subset)
        result[label] = {'n': len(subset), 'gate_on': detected, 'rate': detected / len(subset)}
    return result


def export(run, target):
    plan_path = run / 'plan.json'
    plan = json.loads(plan_path.read_text())
    questions_path = run / 'evaluation_questions.json'
    questions = json.loads(questions_path.read_text())
    rows, omitted = [], []
    for method in ('verify_extracted', 'verify_annotated'):
        for q in questions:
            p = run / method / q['id'] / 'record.json'
            if not p.exists():
                omitted.append({'method': method, 'id': q['id'], 'reason': 'no_record'})
                continue
            record = json.loads(p.read_text())
            if record['status'] != 'complete':
                omitted.append({'method': method, 'id': q['id'], 'reason': record['status'],
                                'error': record.get('error')})
                continue
            assert record['id'] == q['id'] and record['stage'] == method
            judgments = record['result']['judgments']
            assert all(type(j['index']) is int and j['index'] == i for i, j in enumerate(judgments))
            values = [j['false_probability'] for j in judgments]
            assert all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            score = max(values, default=0.0)
            rows.append({'method': method, 'id': q['id'], 'set': q['set'], 'group_id': q['group_id'],
                         'partition': q['partition'], 'false_score': score, 'gate_on': int(score >= .5),
                         'premise_count': len(values), 'record_sha256': sha(p)})
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'table1_scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'run_dir': str(run.resolve()), 'plan_sha256': sha(plan_path),
                'evaluation_questions_sha256': sha(questions_path), 'expected_questions': len(questions),
                'omitted': omitted, 'model': plan['model'], 'reasoning_effort': plan['reasoning_effort'],
                'extract_prompt': plan['extract_prompt'], 'verify_prompt': plan['verify_prompt'],
                'system_prompt': plan['system'], 'score_kind': plan['score_kind'],
                'verification_batching': plan['verification_batching'], 'gate_rule': plan['gate_rule'],
                'interpretation': 'Extracted row is automatic Luna gate. Annotated row is diagnostic only; references unavailable at deployment.',
                'transport_recovery': 'v1/v2 raw responses preserved and revalidated into v3 without prompt/threshold changes; per-record hashes identify scored records.'}
    (target / 'table1_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return target / 'table1_scores.csv'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--run-dir', type=Path)
    source.add_argument('--scores', type=Path)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--bootstraps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20260923)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = export(args.run_dir, args.out_dir) if args.run_dir else args.scores
    with path.open() as f:
        rows = list(csv.DictReader(f))
    methods = sorted({r['method'] for r in rows})
    common = set.intersection(*[{r['id'] for r in rows if r['method'] == method} for method in methods])
    summary = {'threshold': .5, 'score_type': 'model-reported confidence, not token probabilities',
               'auc_protocol': 'pooled AUROC, not fold-weighted crossfit AUROC',
               'ci_protocol': 'percentile source-group bootstrap; conditional on frozen responses; not API resampling',
               'bootstrap_seed': args.seed, 'bootstrap_replicates': args.bootstraps,
               'scores_sha256': sha(path), 'all_valid': {}, 'common_questions': {}}
    for method in methods:
        rr = [r for r in rows if r['method'] == method]
        summary['all_valid'][method] = summarize(rr, args.seed, args.bootstraps)
        summary['common_questions'][method] = summarize([r for r in rr if r['id'] in common], args.seed, args.bootstraps)
    (args.out_dir / 'table1_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
