"""Export paired answers and pre-answer explanations without automatic quality labels."""
import argparse
import collections
import csv
import datetime
import json
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.open():
        if line.endswith('\n'):
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    summaries = {}
    comparisons = []
    for model in ('qwen25', 'gemma3'):
        directory = args.root / model
        manifest = read_rows(directory / 'manifest.jsonl')
        targets = {row['id']: row for row in manifest}
        outputs = read_rows(directory / 'av_results.jsonl')
        if len({r['id'] for r in outputs}) != len(outputs):
            raise ValueError('Duplicate AV row')
        if any(r['id'] not in targets for r in outputs):
            raise ValueError('AV row outside frozen cohort')
        answers = {(r['dataset'], r['id'], r['condition']): r
                   for r in read_rows(directory / 'answers.jsonl')}
        scores = {r['id']: r for r in read_rows(directory / 'scored.jsonl')}
        by_question = collections.defaultdict(dict)
        for row in outputs:
            by_question[(row['source'], row['base_id'])][row['position_family']] = row
        for source, base_id in sorted({(r['source'], r['base_id']) for r in manifest}):
            plain = answers[source, base_id, 'plain']
            balanced = answers[source, base_id, 'balanced']
            entry = dict(model=model, dataset=source, id=base_id, label=plain['label'],
                         question=plain['question'], plain_answer=plain['answer'],
                         balanced_answer=balanced['answer'],
                         answers_identical=plain['answer'] == balanced['answer'],
                         behavior_quality='Not judged; output differences are not correction labels')
            for position, prefix in [('assistant_boundary_last_token', 'boundary'),
                                     ('prompt_keyword_1', 'keyword')]:
                row = by_question[source, base_id].get(position, {})
                score = scores.get(row.get('id'), {})
                entry.update({prefix + '_token': row.get('selected_token_text'),
                              prefix + '_explanation': row.get('nla_output'),
                              prefix + '_parsed': row.get('parsed_explanation_tag'),
                              prefix + '_finish_reason': row.get('finish_reason', 'legacy_unrecorded'),
                              prefix + '_origin': 'legacy_reused' if row.get('reused_legacy_av_source') else 'new_batch',
                              prefix + '_recon_mse': score.get('recon_mse'),
                              prefix + '_question_control_mse': score.get('question_text_control_mse')})
            comparisons.append(entry)
        summaries[model] = dict(
            expected_vectors=len(manifest), completed_vectors=len(outputs),
            reused_vectors=sum(bool(r.get('reused_legacy_av_source')) for r in outputs),
            new_vectors=sum(not bool(r.get('reused_legacy_av_source')) for r in outputs),
            parsed_explanations=sum(bool(r.get('parsed_explanation_tag')) for r in outputs),
            ar_scored_vectors=len(scores),
            strata=dict(collections.Counter(f"{r['source']}/{r['category']}/{r['position_family']}"
                                            for r in outputs)),
            ar_interpretation='Direction fidelity only; not explanation correctness',
        )
        if scores:
            summaries[model]['mean_recon_mse'] = sum(r['recon_mse'] for r in scores.values()) / len(scores)
            summaries[model]['mean_question_control_mse'] = sum(r['question_text_control_mse'] for r in scores.values()) / len(scores)
    result = dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='785 train questions/model; Plain pre-answer activations at two positions; AO deferred',
                  models=summaries, behavior_judge='Not run',
                  caveats=['Cancer uses fold0 train only; not complete three-fold Ours evaluation',
                           'Legacy and new batch explanations retain separate generation provenance',
                           'No semantic accuracy inferred from tag parsing, answer differences, or AR scores'])
    (args.root / 'analysis_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.root / 'question_answer_explanation_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (args.root / 'question_answer_explanation_comparison.jsonl').open('w') as stream:
        for row in comparisons:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
