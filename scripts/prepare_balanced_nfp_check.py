"""Prepare 5 normal + 5 overcorrection answers under the frozen v2 prompt.

Offline only. --accept-ai-review records the user's acceptance of the supplied
AI-assisted draft, not independent expert or blinded adjudication.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_terra_judge import PROTOCOL_V2, fixed_text, human_labels, render_v2, validate_protocol
from src.jsonl import load_json
from src.pilot import digest, file_digest, frozen_json, output_lock


def prepare(source, review_dir, labels_path, out, accept_ai_review=False, seed=17):
    source, review_dir, labels_path, out = map(Path, (source, review_dir, labels_path, out))
    if out.resolve() in (source.resolve(), review_dir.resolve()):
        raise ValueError('Use a new output directory')
    base = load_json(source / 'plan.json')
    validate_protocol(base)
    if base.get('protocol') != PROTOCOL_V2:
        raise ValueError('Freeze the existing v2 protocol before this check')
    human_labels(source, base)
    if load_json(source / 'run.json') != {
        'plan_hash': digest(base), 'human_hash': file_digest(source / 'human_review.tsv')
    }:
        raise ValueError('Source plan or labels changed after scoring')
    review = load_json(review_dir / 'review_cases.json')
    if review.get('version') != 'nfp-overcorrection-review-v1':
        raise ValueError('Unsupported review source')
    cases = {c['id']: c for c in review['cases']}
    if len(cases) != len(review['cases']) or set(cases) & {c['id'] for c in base['cases']}:
        raise ValueError('Duplicate review IDs or overlap with prompt-development cases')
    if any(c['answer_hash'] != digest(c['answer']) for c in cases.values()):
        raise ValueError('Review answer changed')
    with labels_path.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    labels = {r['id']: r for r in rows}
    if len(labels) != len(rows) or set(labels) != set(cases):
        raise ValueError('Review labels must cover every exported case exactly once')
    candidate_ids, held = [], []
    for row in rows:
        if row['status'] == 'hold':
            if row['score']:
                raise ValueError('Held cases must not have forced scores')
            held.append(row['id'])
            continue
        if (row['status'] != 'candidate' or row['score'] not in ('1', '-1')
            or not row['rationale'].strip()):
            raise ValueError('Invalid candidate label or missing rationale')
        if not row['reviewer'].strip() and not accept_ai_review:
            raise ValueError('Review the draft first; --accept-ai-review records explicit acceptance')
        if row['score'] == '-1':
            quote = row.get('answer_evidence', '')
            if not quote.strip() or quote not in cases[row['id']]['answer']:
                raise ValueError('Negative candidate lacks exact answer evidence')
        candidate_ids.append(row['id'])
    rng = random.Random(seed)
    selected = []
    for value in ('1', '-1'):
        pool = sorted(i for i in candidate_ids if labels[i]['score'] == value)
        if len(pool) < 5:
            raise ValueError('Need at least five reviewed candidates per class; do not force labels')
        selected.extend(rng.sample(pool, 5))
    rng.shuffle(selected)
    chosen = [{**cases[qid], 'cohort': 'normal' if labels[qid]['score'] == '1' else 'overcorrection',
               'prompt': render_v2(base['prompt_template'], cases[qid])} for qid in selected]
    order = []
    for repeat in (1, 2):
        shuffled = selected[:]
        rng.shuffle(shuffled)
        order.extend({'case_id': qid, 'repeat': repeat, 'id': f'{qid}::r{repeat}'} for qid in shuffled)
    plan = {**base, 'version': 'terra-nfp-balanced-v1', 'evaluation_scope': 'balanced selected unambiguous cases',
            'seed': seed, 'cases': chosen, 'order': order, 'budget': 20,
            'source_plan_hash': digest(base), 'source_dir': str(source.resolve()),
            'review_file_hash': file_digest(review_dir / 'review_cases.json'),
            'review_labels_hash': file_digest(labels_path),
            'review_acceptance': 'user-invoked --accept-ai-review' if accept_ai_review else 'named reviewers in source TSV',
            'held_ids': sorted(held), 'unselected_candidate_ids': sorted(set(candidate_ids) - set(selected))}
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(['id', 'score', 'rationale', 'reviewer'])
    for qid in selected:
        row = labels[qid]
        writer.writerow([qid, row['score'], row['rationale'], row['reviewer'].strip() or
                         'User-accepted AI-assisted draft via --accept-ai-review; not independent expert review'])
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'plan.json', plan)
        fixed_text(out / 'human_review.tsv', stream.getvalue())
        fixed_text(out / 'protocol.txt', base['prompt_template'])
        lines = ['# Balanced NFP judge check', '', 'No model calls during preparation.',
                 'Frozen v2 prompt; 5 normal + 5 overcorrection answers, 2 calls each.',
                 f'Excluded held IDs: {sorted(held)}',
                 f'Unselected candidates: {plan["unselected_candidate_ids"]}', '',
                 '| ID | Reference | Rationale |', '|---|---:|---|']
        lines.extend(f'| {qid} | {labels[qid]["score"]} | {labels[qid]["rationale"]} |' for qid in selected)
        fixed_text(out / 'selection.md', '\n'.join(lines))
    print(f'Prepared 5 + 5 cases; {len(held)} held out of scoring. GPT calls: 0. {out}')
    print('score will make at most 20 NEW Terra calls. Keep the v2 prompt fixed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', required=True)
    p.add_argument('--review-dir', required=True)
    p.add_argument('--labels', default='docs/reviews/nfp_overcorrection_20_ai_draft.tsv')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--accept-ai-review', action='store_true', help='I accept the supplied AI-assisted candidate labels; held labels stay unresolved')
    p.add_argument('--seed', type=int, default=17)
    a = p.parse_args()
    prepare(a.source_dir, a.review_dir, a.labels, a.out_dir, a.accept_ai_review, a.seed)


if __name__ == '__main__':
    main()
