"""Read-only audit of repeat judgments on identical answers; no model calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_judge import load_reuse_scores
from src.jsonl import load_json, read_jsonl
from src.pilot import digest, file_digest


def audit(pilot, model, comparison):
    folder = Path(pilot) / 'dev'
    question_path = Path(pilot) / 'split' / 'dev.jsonl'
    questions = {r['id']: r for r in read_jsonl(question_path)}
    names = ['plain', comparison]
    answers, scores, signatures = [], [], []
    for name in names:
        response_path = folder / f'{name}.jsonl'
        score_path = folder / f'{name}_judge_codex_{model}.jsonl'
        signature = load_json(str(score_path) + '.run.json')
        load_reuse_scores(score_path, signature)  # validate signed score rows
        if signature['questions_hash'] != file_digest(question_path):
            raise ValueError('Question/reference file differs from judge run')
        if signature['responses_hash'] != file_digest(response_path):
            raise ValueError('Scored response file has changed')
        rows = list(read_jsonl(response_path))
        by_id = {r['id']: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError('Duplicate response IDs')
        scored = {}
        for row in read_jsonl(score_path):
            qid = row['question_id']
            if qid in scored or row['response_id'] != qid or qid not in by_id:
                raise ValueError('Audit requires one score per matching pilot question')
            if row['response_text_hash'] != digest(by_id[qid]['response']):
                raise ValueError('Score does not match answer')
            scored[qid] = row
        if scored.keys() != by_id.keys():
            raise ValueError('Incomplete scores; audit completed conditions only')
        answers.append(by_id)
        scores.append(scored)
        signatures.append(signature)
    ignored = {'responses_hash', 'response_run'}
    if ({k: v for k, v in signatures[0].items() if k not in ignored}
        != {k: v for k, v in signatures[1].items() if k not in ignored}):
        raise ValueError('Judge settings differ')
    if answers[0].keys() != answers[1].keys():
        raise ValueError('Question coverage differs')
    lines = [f'# Repeat-judge audit: Plain vs {comparison}', '',
             f'Judge: `{model}`. Saved data only; no new judge calls.', '',
             'This measures observed disagreement, not which judgment is correct.', '']
    details = []
    for kind in ('fpq', 'nfp'):
        ids = [i for i, r in answers[0].items() if r['set'] == kind]
        same, changed, flips = [], [], []
        for qid in ids:
            a, b = answers[0][qid], answers[1][qid]
            if a['question'] != b['question'] or a['set'] != b['set']:
                raise ValueError('Question text/set differs')
            if a['response'] != b['response']:
                changed.append(qid)
                continue
            same.append(qid)
            left, right = scores[0][qid], scores[1][qid]
            if left['sharpness'] != right['sharpness']:
                flips.append(qid)
                details.extend([f'## {kind}: {qid}', '', '**Question**', '', a['question'], '',
                                '**Rubric reference (not a gold answer)**', '',
                                str(questions[qid].get('correction') if kind == 'fpq'
                                    else questions[qid].get('hallucination_text')), '',
                                '**Identical answer**', '', a['response'], '',
                                f"**Plain score: {left['sharpness']}**", '',
                                str(left.get('reason') or left.get('judge_raw', '')), '',
                                f"**{comparison} score: {right['sharpness']}**", '',
                                str(right.get('reason') or right.get('judge_raw', '')), ''])
        lines.append(f'- {kind}: total={len(ids)}, identical answers={len(same)}, '
                     f'changed answers={len(changed)}, score disagreements on identical answers={len(flips)}')
        if changed:
            lines.append('  Changed answer IDs: ' + json.dumps(changed, ensure_ascii=False))
    return '\n'.join(lines + [''] + details) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', required=True)
    parser.add_argument('--model', default='gpt-5.6-sol')
    parser.add_argument('--comparison', default='steering_L14_a0')
    args = parser.parse_args()
    for value in (args.model, args.comparison):
        if '/' in value or '\\' in value or value in ('.', '..'):
            parser.error('Model and comparison must be filename components')
    print(audit(args.pilot_dir, args.model, args.comparison), end='')


if __name__ == '__main__':
    main()
