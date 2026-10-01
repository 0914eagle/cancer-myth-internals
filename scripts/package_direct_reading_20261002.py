"""Package the assistant's manually written reading notes; never classify answers.

No model/network calls. Unread source records remain explicitly pending.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/direct_reading_2026-10-02'
SOURCE = ROOT / 'results/analysis/direct_reading_20261002/source.json'


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    source = json.loads(SOURCE.read_text())
    notes = read_lines(OUT / 'reading_notes.jsonl')
    comparisons = read_lines(OUT / 'scoring_comparison.jsonl')
    by_index = {r['review_index']: r for r in notes}
    comparison_index = {r['review_index']: r for r in comparisons}
    assert len(by_index) == len(notes), 'Duplicate reading records'
    assert len(comparison_index) == len(comparisons), 'Duplicate scoring records'
    assert set(by_index) == set(comparison_index), 'Each read pair needs a score comparison'
    assert all(i == r['review_index'] for i, r in enumerate(source))
    for i, note in by_index.items():
        for condition in ('plain', 'alternative'):
            assert note[condition + '_quote'] in source[i][condition]['answer'], (i, condition)

    packed = []
    with (OUT / 'coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['review_index', 'cohort', 'model', 'id', 'method', 'status'], lineterminator='\n')
        writer.writeheader()
        for r in source:
            i = r['review_index']
            writer.writerow({**{k: r[k] for k in ('review_index', 'cohort', 'model', 'id', 'method')},
                             'status': 'both_full_answers_and_judges_read' if i in by_index else 'pending'})
            if i not in by_index:
                continue
            item = {k: r[k] for k in ('review_index', 'cohort', 'model', 'id', 'method')}
            item['question'] = r['plain']['question']
            item['reference'] = r['plain']['reference']
            for c in ('plain', 'alternative'):
                item[c] = {k: r[c].get(k) for k in ('method', 'score', 'answer', 'answer_sha256')}
            item['original_well_reasons'] = r['well_reasons']
            item['assistant_direct_reading'] = by_index[i]
            item['assistant_score_comparison'] = comparison_index[i]
            packed.append(item)
    (OUT / 'reviewed_evidence.json').write_text(json.dumps(packed, ensure_ascii=False, indent=2) + '\n')
    total = Counter((r['model'], r['cohort']) for r in source)
    read = Counter((r['model'], r['cohort']) for r in packed)
    summary = {
        'status': 'partial_direct_reading_not_full_cohort_result',
        'planned_pairs': len(source), 'read_pairs': len(packed),
        'read_complete_answers': 2 * len(packed), 'pending_pairs': len(source) - len(packed),
        'delegated_review_calls': 0, 'new_answer_generation_calls': 0, 'new_well_judging_calls': 0,
        'quote_validation': 'both evidence quotes match original answers for every read pair',
        'coverage': [{'model': m, 'cohort': c, 'read': read[(m, c)], 'total': total[(m, c)]}
                     for m, c in sorted(total)],
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'limitations': ['Selected score-transition cohorts, not all failures or population rates.',
                       'Direct AI textual review by the current assistant, not independent expert clinical adjudication.',
                       'Scores/codes omitted from current first-pass packets, but prior conversation and selection reveal outcomes: not a blinded study.',
                       'Different model-specific alternative prompts; cannot isolate model effects.',
                       'No medical facts or product claims validated solely by this reading.'],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
