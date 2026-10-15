#!/usr/bin/env python3
"""Audit saved out-of-fold text/hidden decisions; no training or model calls."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate-dir', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result_path = args.gate_dir / 'result.json'
    result = json.loads(result_path.read_text())
    questions = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    by_id = {q['id']: q for q in questions}
    assert len(by_id) == len(questions), 'Duplicate question IDs'
    predictions = {}
    for signal in ('text', 'hidden'):
        rows = [r for r in result['predictions'] if r['signal'] == signal]
        predictions[signal] = {r['id']: r for r in rows}
        assert len(rows) == len(predictions[signal]), 'Duplicate prediction IDs'
        assert set(predictions[signal]) == set(by_id), 'Incomplete predictions'
        for r in rows:
            assert math.isfinite(r['score']) and math.isfinite(r['threshold'])
            assert r['gate_on'] == (r['score'] > r['threshold'])
            assert (r['set'], r['group_id']) == (by_id[r['id']]['set'], by_id[r['id']]['group_id'])
    for qid in by_id:
        assert predictions['text'][qid]['fold'] == predictions['hidden'][qid]['fold']
    # Verify that each scored question/group was excluded from its fit and calibration data.
    group_folds = {}
    for qid, r in predictions['text'].items():
        assert group_folds.setdefault(r['group_id'], r['fold']) == r['fold']
    for s in result['selections']:
        if s['signal'] not in predictions:
            continue
        train, cal, held = (set(s[k]) for k in ('train_ids', 'calibration_ids', 'evaluation_ids'))
        assert not (train & cal or train & held or cal & held)
        groups = [{by_id[i]['group_id'] for i in ids} for ids in (train, cal, held)]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert held == {i for i, r in predictions[s['signal']].items() if r['fold'] == s['fold']}

    modes = [('calibrated', None), ('matched_5pct', .05), ('matched_8pct', .08), ('matched_10pct', .10)]
    output = []
    summaries = {}
    for mode, target in modes:
        flags, thresholds = {}, {}
        for signal, rows in predictions.items():
            thresholds[signal] = {}
            for fold in sorted({r['fold'] for r in rows.values()}):
                fold_rows = [r for r in rows.values() if r['fold'] == fold]
                if target is None:
                    values = {r['threshold'] for r in fold_rows}
                    assert len(values) == 1
                    threshold = values.pop()
                else:
                    negatives = sorted((r['score'] for r in fold_rows if r['set'] == 'nfp'), reverse=True)
                    threshold = negatives[math.floor(target * len(negatives))]
                thresholds[signal][fold] = threshold
            flags[signal] = {i: r['score'] > thresholds[signal][r['fold']] for i, r in rows.items()}
        counts = {s: Counter() for s in ('fpq', 'nfp')}
        for qid in sorted(by_id):
            q = by_id[qid]
            t, h = flags['text'][qid], flags['hidden'][qid]
            bucket = 'both' if t and h else 'text_only' if t else 'hidden_only' if h else 'neither'
            counts[q['set']][bucket] += 1
            output.append(dict(mode=mode, id=qid, set=q['set'], fold=predictions['text'][qid]['fold'],
                               group_id=q['group_id'], category=q.get('category', ''), bucket=bucket,
                               text_on=t, hidden_on=h, text_score=predictions['text'][qid]['score'],
                               hidden_score=predictions['hidden'][qid]['score'],
                               text_threshold=thresholds['text'][predictions['text'][qid]['fold']],
                               hidden_threshold=thresholds['hidden'][predictions['hidden'][qid]['fold']],
                               question=q['question'], premise_text=q.get('premise_text', ''),
                               correction=q.get('correction', '')))
        summaries[mode] = {s: {k: counts[s][k] for k in ('both', 'text_only', 'hidden_only', 'neither')}
                           for s in counts}
        for s, c in summaries[mode].items():
            assert sum(c.values()) == sum(q['set'] == s for q in questions)
        if target is not None:
            assert counts['nfp']['text_only'] == counts['nfp']['hidden_only'], 'Ties prevented equal FPR'

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'items.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    sources = {str(p.resolve()): digest(p) for p in (result_path, args.gate_dir / 'plan.json', args.questions)}
    manifest = dict(sources=sources, script_sha256=digest(Path(__file__)),
                    counts=dict(Counter(q['set'] for q in questions)),
                    checks='IDs, labels, group/fold alignment, finite scores, saved decisions, split/group disjointness, equal realized matched FPR',
                    modes=summaries)
    (args.out / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Text vs hidden gate overlap — 2026-09-22', '',
             'Saved Qwen2.5-7B crossfit_v2_style predictions; original FPQ/NFP only. No new training, generation or judging.',
             'Positive decision = flag a false premise. FPQ positives are detections; NFP positives are false positives.',
             'Calibrated uses the saved thresholds (calibration FPR target 5%). Matched thresholds use evaluation NFP labels per fold,',
             'with strict score > threshold and floor(target × fold negatives). They are retrospective ranking diagnostics, not deployable thresholds.',
             'Raw scores are never pooled across folds. Earlier dev inspection remains exploratory.', '',
             '| Mode | Set | Both flag | Text only | Hidden only | Neither flags |',
             '|---|---|---:|---:|---:|---:|']
    for mode, counts in summaries.items():
        for s, c in counts.items():
            lines.append(f"| {mode} | {s} | {c['both']} | {c['text_only']} | {c['hidden_only']} | {c['neither']} |")
    lines += ['', '| Mode | Rule | FPQ detected / total | TPR | NFP falsely flagged / total | FPR |',
              '|---|---|---:|---:|---:|---:|']
    for mode, counts in summaries.items():
        for rule, buckets in [('text', ('both', 'text_only')), ('hidden', ('both', 'hidden_only')),
                              ('OR (diagnostic)', ('both', 'text_only', 'hidden_only')), ('AND (diagnostic)', ('both',))]:
            tp, fp = [sum(counts[s][b] for b in buckets) for s in ('fpq', 'nfp')]
            pos, neg = [sum(counts[s].values()) for s in ('fpq', 'nfp')]
            lines.append(f'| {mode} | {rule} | {tp}/{pos} | {tp/pos:.1%} | {fp}/{neg} | {fp/neg:.1%} |')
    lines += ['', 'Hidden-only detections establish disagreement/complementarity at these thresholds, not a superior learned combination.',
              'OR adds false positives as well as true positives. No combined gate was fitted or independently evaluated.',
              'These are descriptive counts conditional on existing fitted models; no new significance or generalization claim.',
              'All question texts, original annotations, scores, thresholds and bucket assignments are in items.csv.',
              'Source hashes and integrity checks are in summary.json. No medical labels were changed.', '']
    (args.out / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
