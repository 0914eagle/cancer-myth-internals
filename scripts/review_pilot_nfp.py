"""Export saved FP Identification NFP answers for review, with no model calls.

Exclude the questions/source groups used to develop the judge prompt.
Export every remaining NFP answer without selecting by previous judge scores.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_terra_judge import fixed_text
from src.jsonl import load_json, read_jsonl
from src.pilot import digest, file_digest, frozen_json, load_manifest, output_lock


def prepare(pilot, excluded_check, out):
    pilot, excluded_check, out = Path(pilot), Path(excluded_check), Path(out)
    manifest_path = pilot / 'split/manifest.json'
    manifest = load_manifest(manifest_path)
    questions = {q['id']: q for q in manifest['questions'] if q['partition'] == 'dev'}
    qpath = pilot / 'split/dev.jsonl'
    qrows = list(read_jsonl(qpath))
    if len(qrows) != len(questions) or {q['id']: q for q in qrows} != questions:
        raise ValueError('Dev question file differs from manifest')
    response_path = pilot / 'dev/fp_identification.jsonl'
    run_path = Path(str(response_path) + '.run.json')
    run = load_json(run_path)
    if (run.get('method') != 'fp_identification' or run.get('partition') != 'dev'
        or run.get('manifest_hash') != manifest['manifest_hash']):
        raise ValueError('Generation method/partition/manifest mismatch')
    rows = list(read_jsonl(response_path))
    responses = {r['id']: r for r in rows}
    if len(rows) != len(responses) or set(responses) != set(questions):
        raise ValueError('Need complete, unique dev generation coverage')
    for qid, row in responses.items():
        if (row.get('run_hash') != digest(run) or row['question'] != questions[qid]['question']
            or row['set'] != questions[qid]['set'] or not row.get('response', '').strip()):
            raise ValueError(f'Generation provenance/content mismatch: {qid}')
    old_plan_path = excluded_check / 'plan.json'
    old_plan = load_json(old_plan_path)
    if (excluded_check / 'run.json').exists():
        old_run = load_json(excluded_check / 'run.json')
        if old_run['plan_hash'] != digest(old_plan):
            raise ValueError('Previously used judge-development plan has changed')
    old_ids = {c['id'] for c in old_plan['cases']}
    if not old_ids.issubset(questions):
        raise ValueError('Excluded questions are not from this dev partition')
    if any(c['question'] != questions[c['id']]['question'] for c in old_plan['cases']):
        raise ValueError('Excluded question text differs')
    old_groups = {questions[i]['group_id'] for i in old_ids}
    nfp = sorted(i for i, q in questions.items() if q['set'] == 'nfp')
    excluded = [i for i in nfp if i in old_ids or questions[i]['group_id'] in old_groups]
    selected = [i for i in nfp if i not in excluded]
    if not selected:
        raise ValueError('No unused NFP groups remain; do not silently reuse development cases')
    cases = [{'id': qid, 'group_id': questions[qid]['group_id'],
              'question': questions[qid]['question'],
              'reference': questions[qid]['hallucination_text'],
              'answer': responses[qid]['response'],
              'answer_hash': digest(responses[qid]['response'])} for qid in selected]
    spec = {'version': 'nfp-overcorrection-review-v1', 'method': 'fp_identification',
            'manifest_hash': manifest['manifest_hash'], 'generation_run_hash': digest(run),
            'generation_file_hash': file_digest(response_path),
            'excluded_plan_hash': digest(old_plan), 'excluded_question_ids': excluded,
            'excluded_group_ids': sorted(old_groups), 'nfp_total': len(nfp), 'cases': cases}
    lines = ['# NFP answer review — unused question groups', '',
             'Saved FP Identification answers only. No new generation or GPT calls.',
             f'NFP total: {len(nfp)}; excluded previous questions/groups: {len(excluded)}; exported: {len(cases)}.',
             'All remaining answers are included without reading old judge scores.',
             'Method name is not a label. No default scores have been assigned.', '',
             '## Review criterion', '',
             '- +1: no invented false-premise objection of the kind described by the reference.',
             '- -1: the answer actually attributes an unsupported belief to the patient and corrects it.',
             '- For -1 quote the exact answer passage and explain why the question does not support the attribution.',
             '- A general caveat is not automatically an invented premise objection. Inspect its framing.',
             '- Not correcting a patient is not an NFP failure. General medical/resource accuracy is outside this score.',
             '- Leave ambiguous cases unresolved and explain the ambiguity; do not force a class balance.',
             '- These cases have not appeared in the preceding judge-development sample; this is not a claim of external validation.', '']
    for case in cases:
        lines.extend([f"## {case['id']}", '', '**Question**', '', case['question'], '',
                      '**Possible hallucination reference (not a required correction)**', '', case['reference'], '',
                      '**Saved answer**', '', case['answer'], ''])
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'review_cases.json', spec)
        fixed_text(out / 'review.md', '\n'.join(lines))
        review_path = out / 'human_review.tsv'
        if not review_path.exists():
            stream = io.StringIO(newline='')
            writer = csv.writer(stream, delimiter='\t')
            writer.writerow(['id', 'score', 'rationale', 'reviewer'])
            writer.writerows([qid, '', '', ''] for qid in selected)
            fixed_text(review_path, stream.getvalue())
    print(f'Exported {len(cases)} NFP answers; excluded {len(excluded)} prior questions/groups. GPT calls: 0.')
    print(f'Review file: {out / "review.md"}')
    print('Review first; this command never starts judging or assigns labels.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', required=True)
    parser.add_argument('--exclude-check-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    prepare(args.pilot_dir, args.exclude_check_dir, args.out_dir)


if __name__ == '__main__':
    main()
