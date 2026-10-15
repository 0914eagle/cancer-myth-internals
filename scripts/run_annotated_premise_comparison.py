"""Export existing FPQ/Well TPQ annotations verbatim; optionally run the fixed Qwen verifier.

No generated premises, correction explanations, retrieval, answer generation, or external judge.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_premise_replacement import digest, run, write

TPQ_REVISION = 'f44ef11fc86805e2e09b5ac66536a37b8e1b410e'
TPQ_HASH = 'd7160733b1468443ffdce9796e85b65440458f2414523890e1ef510f0f9c55d0'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl(path, data):
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in data)
    if path.exists() and path.read_text() != payload:
        raise ValueError(f'Frozen export changed: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


def prepare(args):
    from src.baseline_generation import VERIFY
    question_path = args.suite_dir / 'questions.jsonl'
    answer_path = args.suite_dir / 'answers/extract_verify.jsonl'
    identity_path = args.suite_dir / 'model/identity.json'
    saved = json.loads(identity_path.read_text())
    assert saved['prompts']['verify'] == VERIFY
    assert hashlib.sha256(args.tpq_file.read_bytes()).hexdigest() == TPQ_HASH
    questions, answers, tpq = rows(question_path), rows(answer_path), rows(args.tpq_file)
    assert len(questions) == len({r['id'] for r in questions})
    assert len(answers) == len({r['id'] for r in answers})
    answer_by_id = {r['id']: r for r in answers}
    norm = lambda text: ' '.join(text.split())
    tpq_by_question = {norm(r['question']): r for r in tpq}
    assert len(tpq_by_question) == len(tpq) == 148
    exported, items, missing = [], [], []
    for q in questions:
        if q['set'] == 'fpq':
            premises = [q['premise_text']]
            source = {'dataset': 'Cancer-Myth', 'field': 'premise_text (source_myth)', 'suite_id': q['id']}
        elif q['set'] == 'nfp':
            ref = tpq_by_question.get(norm(q['question']))
            if ref is None:
                missing.append(q['id'])
                continue
            premises = ref['presuppositions']
            source = {'dataset': 'shenranw/CancerMyth-TPQ', 'revision': TPQ_REVISION,
                      'field': 'presuppositions', 'source_id': ref['id'],
                      'question_exact_match': q['question'] == ref['question']}
        else:
            raise ValueError(q['set'])
        assert premises and all(isinstance(p, str) and p.strip() for p in premises)
        flags = ['source_text_not_semantically_revalidated']
        if q['id'] == 'fpq_0':
            flags.append('known_source_fragment_From_physicians_preserved_verbatim')
        a = answer_by_id.get(q['id'])
        record = {k: q[k] for k in ('id', 'set', 'partition', 'group_id', 'question')}
        record.update(premises=premises, annotation_source=source, flags=flags,
                      baseline_available=a is not None)
        exported.append(record)
        arms = {'annotated': premises}
        if a is not None:
            assert a['status'] == 'complete' and a['question'] == q['question'] and a['set'] == q['set']
            arms = {'original': [p['premise'] for p in a['details']['premises']], **arms}
        items.append({**record, 'arms': arms,
                      'saved_true_scores': [p['true_score'] for p in a['details']['premises']] if a else None})
    source_paths = [question_path, answer_path, identity_path, args.tpq_file, Path(__file__),
                    ROOT / 'scripts/run_premise_replacement.py', ROOT / 'src/baseline_generation.py',
                    ROOT / 'src/pilot_model.py', args.protocol_file]
    plan = {'scope': 'Existing annotations supplied verbatim; descriptive diagnostic, not a deployable gate',
            'verify_prompt': VERIFY, 'saved_identity': saved['identity'], 'items': items,
            'rule': 'score=1-min(P(True),default=1); gate=any(P(True)<0.5)',
            'missing_annotation_ids': missing,
            'sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    write(args.out_dir / 'inputs.json', plan, frozen=True)
    jsonl(args.out_dir / 'source_annotations.jsonl', exported)
    # These are the only per-question fields used to build model inputs; no class/answer leakage.
    jsonl(args.out_dir / 'verification_inputs.jsonl', [{'id': r['id'], 'question': r['question'],
                                                     'premises': r['premises']} for r in exported])
    inventory = {'annotated_questions': len(exported),
                 'by_set': dict(Counter(r['set'] for r in exported)),
                 'premises_by_set': {s: sum(len(r['premises']) for r in exported if r['set'] == s) for s in ('fpq', 'nfp')},
                 'paired_by_set': dict(Counter(r['set'] for r in exported if r['baseline_available'])),
                 'missing_annotation_ids': missing,
                 'missing_baseline_ids': [r['id'] for r in exported if not r['baseline_available']],
                 'tpq_source_rows': len(tpq), 'tpq_source_premises': sum(len(r['presuppositions']) for r in tpq)}
    write(args.out_dir / 'inventory.json', inventory, frozen=True)
    print(json.dumps(inventory, ensure_ascii=False), flush=True)
    return plan


def report(args, plan):
    from sklearn.metrics import roc_auc_score
    result, deltas, changed_baseline = [], [], []
    for row in plan['items']:
        r = {k: row[k] for k in ('id', 'set', 'partition', 'baseline_available', 'flags')}
        r['arms'] = {}
        for arm, premises in row['arms'].items():
            checked = []
            for premise in premises:
                prompt = plan['verify_prompt'].format(question=row['question'], premise=premise)
                key = digest(prompt)
                value = json.loads((args.out_dir / 'cache' / (key + '.json')).read_text())
                assert key == value['prompt_sha256']
                score = value['readout']['score']
                assert 0 <= score <= 1
                checked.append({'premise': premise, 'true_score': score, 'predicted_true': score >= .5})
            r['arms'][arm] = {'checked': checked, 'score': 1 - min((v['true_score'] for v in checked), default=1),
                              'gate': any(not v['predicted_true'] for v in checked)}
        if row['baseline_available']:
            orig = r['arms']['original']
            assert len(orig['checked']) == len(row['saved_true_scores'])
            deltas.extend(abs(p['true_score'] - s) for p, s in zip(orig['checked'], row['saved_true_scores']))
            if orig['gate'] != any(s < .5 for s in row['saved_true_scores']):
                changed_baseline.append(row['id'])
        result.append(r)
    paired = [r for r in result if r['baseline_available']]
    summary = {'scope': plan['scope'], 'groups': {},
               'original_reproduction': {'premises': len(deltas), 'max_abs_delta': max(deltas),
                                         'changed_gate_ids': changed_baseline},
               'annotated_premise_agreement_with_dataset': {}}
    for label in ('fpq', 'nfp'):
        values = [c for r in result if r['set'] == label for c in r['arms']['annotated']['checked']]
        summary['annotated_premise_agreement_with_dataset'][label] = {
            'expected_true': label == 'nfp', 'total': len(values),
            'agree': sum(c['predicted_true'] == (label == 'nfp') for c in values)}
    for name, rr, arms in [('paired_all', paired, ['original', 'annotated']),
                            ('all_annotations', result, ['annotated']),
                            ('paired_fit', [r for r in paired if r['partition'] == 'fit'], ['original', 'annotated']),
                            ('paired_dev_test', [r for r in paired if r['partition'] != 'fit'], ['original', 'annotated'])]:
        counts = dict(Counter(r['set'] for r in rr))
        group = {'counts': counts, 'arms': {}}
        for arm in arms:
            y = [r['set'] == 'fpq' for r in rr]
            group['arms'][arm] = {'fpq_detected': sum(r['arms'][arm]['gate'] for r in rr if r['set'] == 'fpq'),
                                  'nfp_false_positive': sum(r['arms'][arm]['gate'] for r in rr if r['set'] == 'nfp'),
                                  'auroc': float(roc_auc_score(y, [r['arms'][arm]['score'] for r in rr])),
                                  'premises': sum(len(r['arms'][arm]['checked']) for r in rr)}
        if len(arms) == 2:
            group['transitions'] = {label: {
                change: [r['id'] for r in rr if r['set'] == label and r['arms']['original']['gate'] == a
                         and r['arms']['annotated']['gate'] == b]
                for change, a, b in [('off_to_on', False, True), ('on_to_off', True, False)]}
                for label in ('fpq', 'nfp')}
        summary['groups'][name] = group
    write(args.out_dir / 'results.json', result)
    write(args.out_dir / 'summary.json', summary)
    compact = {k: v for k, v in summary.items() if k != 'groups'}
    compact['groups'] = {k: {n: v for n, v in g.items() if n != 'transitions'} for k, g in summary['groups'].items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'run', 'report'])
    ap.add_argument('--suite-dir', type=Path, required=True)
    ap.add_argument('--tpq-file', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--protocol-file', type=Path, required=True)
    args = ap.parse_args()
    plan = prepare(args)
    if args.stage == 'run':
        run(args, plan)
    if args.stage in ('run', 'report'):
        report(args, plan)


if __name__ == '__main__':
    main()
