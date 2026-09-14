"""Offline review of <=18 saved FPQ answers, stratified by old Sol score.

Two answers per method/score where available. Old scores select cases, not gold
labels. Review text hides method and old score; selection.json retains provenance.
This selected dev sample is diagnostic, not an independent accuracy estimate.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pilot_judge import audit
from scripts.check_terra_judge import fixed_text
from scripts.reevaluate_pilot_nfp import METHODS, load_generation
from src.jsonl import read_jsonl
from src.pilot import file_digest, frozen_json, load_manifest, output_lock

MODEL = 'gpt-5.6-sol'


def select_cases(questions, responses, scores, seed=17):
    rng = random.Random(seed)
    selected, strata = [], []
    for method in METHODS:
        for value in (-1, 0, 1):
            pool = sorted(qid for qid, q in questions.items()
                          if q['set'] == 'fpq' and scores[method][qid]['sharpness'] == value)
            chosen = rng.sample(pool, min(2, len(pool)))
            strata.append({'method': method, 'old_score': value, 'available': len(pool), 'selected': len(chosen)})
            for qid in chosen:
                q = questions[qid]
                selected.append({'question_id': qid, 'group_id': q['group_id'], 'method': method,
                                 'old_score': value, 'old_reason': scores[method][qid].get('reason', ''),
                                 'question': q['question'], 'reference': q['correction'],
                                 'answer': responses[method][qid]['response']})
    rng.shuffle(selected)
    for i, case in enumerate(selected, 1):
        case['review_id'] = f'R{i:02}'
    return selected, strata


def prepare(pilot, out):
    pilot, out = Path(pilot), Path(out)
    manifest_path, qpath = pilot / 'split/manifest.json', pilot / 'split/dev.jsonl'
    manifest = load_manifest(manifest_path)
    questions = {q['id']: q for q in manifest['questions'] if q['partition'] == 'dev'}
    qrows = list(read_jsonl(qpath))
    if len(qrows) != len(questions) or {q['id']: q for q in qrows} != questions:
        raise ValueError('Dev questions differ from manifest')
    if not any(q['set'] == 'fpq' for q in questions.values()):
        raise ValueError('No FPQ questions')
    responses, scores = {}, {}
    sources = {str(p.resolve()): file_digest(p) for p in (manifest_path, qpath)}
    for method in METHODS:
        # Validates signed old scores, response hashes, complete coverage and common judge settings.
        audit(pilot, MODEL, method)
        responses[method], hashes = load_generation(pilot, method, manifest, questions)
        sources.update(hashes)
        spath = pilot / 'dev' / f'{method}_judge_codex_{MODEL}.jsonl'
        scores[method] = {r['question_id']: r for r in read_jsonl(spath)}
        sources.update({str(p.resolve()): file_digest(p) for p in (spath, Path(str(spath) + '.run.json'))})
    selected, strata = select_cases(questions, responses, scores)
    plan = {'version': 'fpq-review-v1', 'seed': 17, 'source_judge': MODEL,
            'sources': sources, 'manifest_hash': manifest['manifest_hash'], 'strata': strata, 'cases': selected}
    lines = ['# FPQ correction review — saved answers', '',
             f'{len(selected)} answer cases. No new model calls. No default review labels.',
             'Selected by old score strata; this is a diagnostic sample, not overall judge accuracy.',
             'Method names and previous scores/reasons are withheld here, but response style may reveal them.',
             'Dataset correction is a supplied reference, not independently verified clinical truth.', '',
             'For each answer: quote the target premise and any correction; assess recognition and factual correction separately.',
             'Use these review categories: accurate correction / recognized but incomplete or incorrect / not recognized / unresolved.',
             'Give a short evidence-based reason. If the reference is ambiguous or facts need verification, leave unresolved.',
             'A rebuttal alone is not an accurate correction. Do not rate general helpfulness as premise correction.',
             'Keep recognition, reference-supported accuracy, and material contradictions distinct. No score is changed by this export.', '']
    for c in selected:
        lines.extend([f'## {c["review_id"]}', '', '**Question**', '', c['question'], '',
                      '**Supplied correction reference**', '', c['reference'], '',
                      '**Saved final answer**', '', c['answer'], '',
                      '**Review category:**', '', '**Answer evidence:**', '', '**Reason / unresolved issue:**', ''])
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'selection.json', plan)
        # Never overwrite a user's filled review; immutable inputs stay in selection.json.
        if not (out / 'review.md').exists():
            fixed_text(out / 'review.md', '\n'.join(lines))
    print(f'Exported {len(selected)} saved FPQ answers (maximum 18). GPT calls: 0; Gemma generation: 0.')
    for s in strata:
        print(f"{s['method']} old score {s['old_score']}: {s['selected']} selected / {s['available']} available")
    print(f'Review: {out / "review.md"}')
    print('Old scores are not reference labels. Review first; no score runner is invoked.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir', required=True)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args()
    prepare(a.pilot_dir, a.out_dir)


if __name__ == '__main__':
    main()
