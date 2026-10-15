"""CPU-only analysis of the completed, corrected-layer NLA cohort.

This exports evidence for semantic review; AR and text cues are not semantic
accuracy or answer-performance labels. It never alters generation artifacts.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

POSITIONS = ['question_last_token', 'prompt_keyword_1', 'assistant_boundary_last_token']
MODELS = ['qwen25', 'gemma3']
CONDITIONS = ['plain', 'balanced']


def read(path):
    return [json.loads(line) for line in path.open()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values))


def prepare(root):
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    config = json.loads((root / 'runtime_nla.json').read_text())
    gold = {(r['dataset'], r['id']): r for r in json.loads(Path(config['bundle']).read_text())['rows']}
    rng = random.Random(42)
    selected = []
    for dataset in ['cancer_myth', 'crepe']:
        for label in [1, 0]:
            pool = sorted([r for r in config['cohort'] if r['dataset'] == dataset and r['label'] == label], key=lambda r: r['id'])
            selected.extend(rng.sample(pool, 2))
    write(out / 'review_sample.json', dict(seed=42, selection='Two question IDs per dataset/label, sorted-ID pool; outputs not consulted',
        questions=selected, n_questions=8, expected_explanations=96,
        scope='Descriptive audit only; eight shared questions, not 96 independent questions'))
    selected_keys = {(r['dataset'], r['id']) for r in selected}
    all_rows = []
    paired = []
    review = []
    inputs = {}
    for model in MODELS:
        directory = root / model
        manifest = read(directory / 'manifest.jsonl')
        scores = read(directory / 'scored.jsonl')
        answers = {r['case_id']: r for r in read(directory / 'answers.jsonl')}
        assert len(manifest) == len(scores) == 4710
        assert len({r['id'] for r in scores}) == 4710
        assert {r['id'] for r in manifest} == {r['id'] for r in scores}
        assert all(r['layer'] == config['models'][model]['layer'] and r['hidden_state_index'] == r['layer'] + 1 for r in scores)
        by_question = defaultdict(dict)
        answer_by_question = {}
        for row in scores:
            assert row['partition'] == 'train'
            answer = answers[row['case_id']]
            enriched = dict(model=model, dataset=row['source'], question_id=row['base_id'], category=row['category'],
                condition=row['condition'], position=row['position_family'], token=row['selected_token_text'],
                question=row['prompt'], explanation=row['nla_output'], recon_mse=row['recon_mse'],
                recon_cos=row['recon_cos'], question_text_control_mse=row['question_text_control_mse'],
                parsed=row['parsed_explanation_tag'], words=len(row['nla_output'].split()), activation_id=row['id'],
                layer=row['layer'], hf_hidden_state_index=row['hidden_state_index'],
                correction_language=bool(re.search(r'\b(?:misconception|incorrect|mistaken|false premise|correct your|correct the|correcting|correction|rebuttal|clarif\w*)\b', row['nla_output'], re.I)))
            all_rows.append(enriched)
            by_question[row['source'], row['base_id']][row['condition'], row['position_family']] = enriched
            answer_by_question[row['source'], row['base_id'], row['condition']] = answer
            if (row['source'], row['base_id']) in selected_keys:
                target = gold[row['source'], row['base_id']]
                review.append({**enriched, 'reference_answer_for_reviewer_only': target.get('reference_answer'),
                    'presuppositions_for_reviewer_only': target.get('presuppositions'),
                    'native_answer_reference': answer['answer'],
                    'native_answer_provenance': 'Existing A6000 answer reference; activation capture was a separate prompt-only forward, not a generation-time trace',
                    'messages': answer['messages']})
        for (dataset, ident), variants in sorted(by_question.items()):
            assert set(variants) == {(c, p) for c in CONDITIONS for p in POSITIONS}
            first = variants['plain', POSITIONS[0]]
            wide = dict(model=model, dataset=dataset, question_id=ident, category=first['category'], question=first['question'])
            for condition in CONDITIONS:
                answer = answer_by_question[dataset, ident, condition]
                wide[condition + '__native_answer_reference'] = answer['answer']
                wide[condition + '__messages_sha256'] = hashlib.sha256(json.dumps(answer['messages'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            wide['answer_provenance'] = 'Existing A6000 reference, separate from fresh prompt-only activation capture; no generation-time trace'
            for (condition, position), row in variants.items():
                key = condition + '__' + position
                for field in ['token', 'explanation', 'recon_mse', 'correction_language', 'activation_id']:
                    wide[key + '__' + field] = row[field]
            paired.append(wide)
        for name in ['manifest.jsonl', 'scored.jsonl', 'answers.jsonl']:
            path = directory / name
            inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, rows in [('explanation_rows.jsonl', all_rows), ('review_explanations.jsonl', review)]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    with (out / 'paired_explanations.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    groups = defaultdict(list)
    for row in all_rows:
        groups[row['model'], row['condition'], row['position']].append(row)
    summary = []
    for (model, condition, position), rows in sorted(groups.items()):
        summary.append(dict(model=model, condition=condition, position=position,
            recon_mse=stats([r['recon_mse'] for r in rows]), question_text_control_mse=stats([r['question_text_control_mse'] for r in rows]),
            words=stats([r['words'] for r in rows]), parsed=sum(r['parsed'] for r in rows),
            correction_language=sum(r['correction_language'] for r in rows),
            token_counts=dict(Counter(repr(r['token']) for r in rows))))
    detailed = []
    groups = defaultdict(list)
    for row in all_rows:
        groups[row['model'], row['dataset'], row['category'], row['condition'], row['position']].append(row)
    for key, rows in sorted(groups.items()):
        detailed.append(dict(zip(['model', 'dataset', 'category', 'condition', 'position'], key),
            n=len(rows), mean_recon_mse=statistics.mean(r['recon_mse'] for r in rows),
            correction_language=sum(r['correction_language'] for r in rows)))
    write(out / 'aggregate_summary.json', dict(n_questions=785, n_models=2, n_vectors=9420,
        scope='Fixed training cohort only; corrected official layers; no final-test use',
        groups=summary, detailed_groups=detailed,
        caveats=['AR measures activation-direction fidelity, not semantic accuracy or answer correction',
                 'Correction-language counts are regex text cues; they do not identify the target false premise or an actual rebuttal',
                 'Question-text AR control is distinct from the mean-direction control',
                 'Model AR errors cannot be used directly to rank model explanation quality'],
        input_sha256=inputs))
    print('Prepared', len(all_rows), 'vectors,', len(paired), 'paired rows,', len(review), 'review explanations')


def direction_control(root):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    result = []
    details = []
    for model in MODELS:
        targets = read(root / model / 'manifest.jsonl')
        score = {r['id']: r for r in read(root / model / 'scored.jsonl')}
        groups = defaultdict(list)
        for row in targets:
            vector = torch.load(row['activation_path'], map_location='cpu', weights_only=True).numpy().astype(np.float64)
            norm = np.linalg.norm(vector)
            assert norm > 0 and np.isfinite(vector).all()
            groups[row['condition'], row['position_family']].append((row, vector / norm))
        for (condition, position), group in sorted(groups.items()):
            assert len(group) == 785 and len({(r['source'], r['base_id']) for r, _ in group}) == 785
            unit = np.stack([v for _, v in group])
            # Each mean contains the other 784 questions under the same condition/position.
            other = unit.sum(axis=0, keepdims=True) - unit
            other /= np.linalg.norm(other, axis=1, keepdims=True)
            mse = np.maximum(0, 2 * (1 - np.einsum('ij,ij->i', unit, other)))
            ar = np.array([score[r['id']]['recon_mse'] for r, _ in group])
            result.append(dict(model=model, condition=condition, position=position, n=785,
                mean_ar_mse=float(ar.mean()), mean_other_question_direction_mse=float(mse.mean()),
                ar_better_count=int((ar < mse).sum()), mean_ar_minus_direction=float((ar - mse).mean())))
            for (row, _), error in zip(group, mse):
                details.append(dict(model=model, activation_id=row['id'], other_question_direction_mse=float(error)))
        print('Direction control complete:', model, flush=True)
    write(root / 'analysis' / 'direction_control.json', dict(groups=result,
        method='Unit-normalize source vectors; mean other 784 question directions within model/condition/position; renormalize; 2*(1-cosine). No text, label, or self vector.',
        scope='Optional training-cohort geometry diagnostic; not held-out accuracy and not input to NLA'))
    (root / 'analysis' / 'direction_control_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in details))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--direction-control', action='store_true')
    args = parser.parse_args()
    direction_control(args.root) if args.direction_control else prepare(args.root)
