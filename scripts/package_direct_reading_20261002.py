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
    taxonomy = json.loads((OUT / 'taxonomy.json').read_text())
    assignments = {r['review_index']: r for r in taxonomy['assignments']}
    assert len(assignments) == len(taxonomy['assignments']), 'Duplicate taxonomy records'
    assert set(assignments) == set(by_index), 'Taxonomy must cover exactly the read pairs'
    for i, note in by_index.items():
        category = assignments[i]['category']
        assert category in taxonomy['categories']
        assert category.startswith('F' if source[i]['cohort'] == 'fpq_disagreement' else 'N')
        for condition in ('plain', 'alternative'):
            assert note[condition + '_quote'] in source[i][condition]['answer'], (i, condition)
            answer = source[i][condition]
            assert hashlib.sha256(answer['answer'].encode()).hexdigest() == answer['answer_sha256']

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
            item['primary_descriptive_category'] = assignments[i]['category']
            packed.append(item)
    (OUT / 'reviewed_evidence.json').write_text(json.dumps(packed, ensure_ascii=False, indent=2) + '\n')
    total = Counter((r['model'], r['cohort']) for r in source)
    read = Counter((r['model'], r['cohort']) for r in packed)
    summary = {
        'status': ('complete_selected_cohorts_not_population_census' if len(packed) == len(source)
                   else 'partial_direct_reading_not_full_cohort_result'),
        'planned_pairs': len(source), 'read_pairs': len(packed),
        'read_complete_answers': 2 * len(packed), 'pending_pairs': len(source) - len(packed),
        'delegated_review_calls': 0, 'new_answer_generation_calls': 0, 'new_well_judging_calls': 0,
        'quote_validation': 'both evidence quotes match original answers for every read pair',
        'coverage': [{'model': m, 'cohort': c, 'read': read[(m, c)], 'total': total[(m, c)]}
                     for m, c in sorted(total)],
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'descriptive_counts': dict(sorted(Counter(r['category'] for r in assignments.values()).items())),
        'descriptive_counts_by_model': {
            m: dict(sorted(Counter(assignments[r['review_index']]['category']
                                   for r in packed if r['model'] == m).items()))
            for m in sorted({r['model'] for r in packed})},
        'limitations': ['Selected score-transition cohorts, not all failures or population rates.',
                       'Direct AI textual review by the current assistant, not independent expert clinical adjudication.',
                       'Mixed reading procedure: early packets omitted scores/codes; later packets included judge reasons. Selection and prior conversation also expose outcomes. Not blinded.',
                       'Post-hoc primary descriptive categories are not causal labels; secondary issues remain in the reading notes.',
                       'Different model-specific alternative prompts; cannot isolate model effects.',
                       'No medical facts or product claims validated solely by this reading.'],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
