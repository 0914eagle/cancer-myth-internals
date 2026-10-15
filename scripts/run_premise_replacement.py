"""Frozen, fit-only premise replacement diagnostic; no answer generation or judge."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value, frozen=False):
    if frozen and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen file changed: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare(args):
    review = args.review_dir
    sample = json.loads((review / 'sample.json').read_text())
    edits = json.loads((review / 'replacements.json').read_text())
    annotations = {r['id']: r for r in json.loads((review / 'annotations.json').read_text())['items']}
    questions = {r['id']: r for r in map(json.loads, (args.suite_dir / 'questions.jsonl').read_text().splitlines())}
    answers = {r['id']: r for r in map(json.loads, (args.suite_dir / 'answers/extract_verify.jsonl').read_text().splitlines())}
    saved_identity = json.loads((args.suite_dir / 'model/identity.json').read_text())
    assert len(sample) == len(edits) == len({r['id'] for r in edits}) == 40
    assert [r['id'] for r in sample] == [r['id'] for r in edits]
    from src.baseline_generation import VERIFY
    assert VERIFY == saved_identity['prompts']['verify']
    items = []
    for row, edit in zip(sample, edits):
        key = row['id']
        q, a = questions[key], answers[key]
        assert q['partition'] == 'fit' and q['question'] == row['question'] == a['question']
        original = [p['premise'] for p in a['details']['premises']]
        assert original == row['premises']
        assert len(edit['keep']) == len(set(edit['keep']))
        assert all(1 <= i <= len(original) for i in edit['keep'])
        assert all(p['evidence'] and p['evidence'] in q['question'] for p in edit['add']), key
        kept = [original[i - 1] for i in edit['keep']]
        reviewed = kept + [p['premise'] for p in edit['add']]
        assert reviewed and len(reviewed) == len(set(reviewed))
        items.append({'id': key, 'set': q['set'], 'question': q['question'],
                      'reference_question_issue': annotations[key]['reference_question_issue'],
                      'arms': {'original': original, 'retained_only': kept, 'reviewed': reviewed},
                      'saved_true_scores': [p['true_score'] for p in a['details']['premises']],
                      'edit': edit})
    sources = [review / n for n in ('sample.json', 'annotations.json', 'replacements.json', 'replacement_protocol.md')]
    sources += [args.suite_dir / 'questions.jsonl', args.suite_dir / 'answers/extract_verify.jsonl',
                args.suite_dir / 'model/identity.json', Path(__file__), ROOT / 'src/baseline_generation.py', ROOT / 'src/pilot_model.py']
    plan = {'scope': 'AI-reviewed replacement, not independent gold; fit pilot', 'verify_prompt': VERIFY,
            'rule': 'score=1-min(true_scores, default=1); gate=any(true_score<0.5)',
            'saved_identity': saved_identity['identity'], 'items': items,
            'sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    write(args.out_dir / 'inputs.json', plan, frozen=True)
    counts = {arm: sum(len(r['arms'][arm]) for r in items) for arm in items[0]['arms']}
    print(json.dumps({'prepared_questions': len(items), 'premise_counts': counts, 'plan_sha256': digest(plan)}), flush=True)
    return plan


def run(args, plan):
    from src.baseline_generation import ModelRuntime
    from src.config import load_config
    from src.pilot import output_lock
    cfg = load_config(ROOT / 'configs/qwen25_7b.yaml')
    cfg['source_model']['revision'] = plan['saved_identity']['resolved_revision']
    with output_lock(args.out_dir / 'inference'):
        runtime = ModelRuntime(cfg)
        for key in ('resolved_revision', 'chat_template_hash', 'torch_version', 'transformers_version'):
            assert runtime.identity[key] == plan['saved_identity'][key], f'Runtime mismatch: {key}'
        write(args.out_dir / 'runtime.json', runtime.identity, frozen=True)
        jobs = {}
        for row in plan['items']:
            for premises in row['arms'].values():
                for premise in premises:
                    prompt = plan['verify_prompt'].format(question=row['question'], premise=premise)
                    jobs[digest(prompt)] = prompt
        print(f'{len(jobs)} unique verification prompts', flush=True)
        for i, (key, prompt) in enumerate(jobs.items(), 1):
            path = args.out_dir / 'cache' / (key + '.json')
            if not path.exists():
                value = runtime.binary(prompt, 'True', 'False')
                write(path, {'prompt_sha256': key, 'readout': value})
            if i % 20 == 0 or i == len(jobs):
                print(f'verified {i}/{len(jobs)}', flush=True)


def evaluate(args, plan):
    from sklearn.metrics import roc_auc_score
    items, drift, drift_gates = [], [], []
    for row in plan['items']:
        result = {k: row[k] for k in ('id', 'set', 'question', 'reference_question_issue')}
        result['arms'] = {}
        for arm, premises in row['arms'].items():
            checked = []
            for premise in premises:
                prompt = plan['verify_prompt'].format(question=row['question'], premise=premise)
                key = digest(prompt)
                value = json.loads((args.out_dir / 'cache' / (key + '.json')).read_text())
                assert value['prompt_sha256'] == key
                score = value['readout']['score']
                assert 0 <= score <= 1
                checked.append({'premise': premise, 'true_score': score})
            scores = [p['true_score'] for p in checked]
            result['arms'][arm] = {'checked': checked, 'score': 1 - min(scores, default=1),
                                  'gate': any(s < .5 for s in scores)}
        fresh = result['arms']['original']
        drift.extend(abs(p['true_score'] - s) for p, s in zip(fresh['checked'], row['saved_true_scores']))
        if fresh['gate'] != any(s < .5 for s in row['saved_true_scores']):
            drift_gates.append(row['id'])
        if row['arms']['original'] == row['arms']['reviewed']:
            assert result['arms']['original'] == result['arms']['reviewed']
        items.append(result)
    summary = {'n': len(items), 'scope': plan['scope'], 'groups': {},
               'saved_vs_fresh_original': {'max_abs_probability_delta': max(drift), 'changed_gate_ids': drift_gates}}
    for group, rows in [('all', items), ('reference_unflagged', [r for r in items if r['reference_question_issue'] is not True]),
                        ('reference_flagged_fpq', [r for r in items if r['reference_question_issue'] is True])]:
        summary['groups'][group] = {'counts': dict(Counter(r['set'] for r in rows)), 'arms': {}, 'transitions': {}}
        for arm in ('original', 'retained_only', 'reviewed'):
            y = [r['set'] == 'fpq' for r in rows]
            scores = [r['arms'][arm]['score'] for r in rows]
            summary['groups'][group]['arms'][arm] = {
                'fpq_detected': sum(r['arms'][arm]['gate'] for r in rows if r['set'] == 'fpq'),
                'nfp_false_positive': sum(r['arms'][arm]['gate'] for r in rows if r['set'] == 'nfp'),
                'auroc': float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
                'premises': sum(len(r['arms'][arm]['checked']) for r in rows)}
        for before, after in [('original', 'reviewed'), ('original', 'retained_only'), ('retained_only', 'reviewed')]:
            summary['groups'][group]['transitions'][before + '->' + after] = {
                label: {change: [r['id'] for r in rows if r['set'] == label
                                and r['arms'][before]['gate'] == a and r['arms'][after]['gate'] == b]
                        for change, a, b in [('off_to_on', False, True), ('on_to_off', True, False)]}
                for label in ('fpq', 'nfp')}
    write(args.out_dir / 'results.json', items)
    write(args.out_dir / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'report'])
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args)
    if args.stage == 'run':
        run(args, plan)
    if args.stage in ('run', 'report'):
        evaluate(args, plan)


if __name__ == '__main__':
    main()
