"""Freeze full paired-answer linkage and score-sensitivity controls, without LLM calls."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT = ROOT / 'results/audits/measurement_validity_20261001_v1/followups'


def write(path, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == text, f'Frozen file changed: {path}'
    else:
        path.write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in SOURCE.read_text().splitlines()]
    assert len(rows) == 3659
    assert len({(r['model'], r['id'], r['method']) for r in rows}) == len(rows)
    for row in rows:
        for side in ('plain', 'alternative'):
            rec = row[side]
            assert hashlib.sha256(rec['answer'].encode()).hexdigest() == rec['answer_sha256']
        assert row['plain']['question'] == row['alternative']['question']
    nfp = [r for r in rows if r['dataset'] == 'nfp']
    assert len(nfp) == 745
    losses = [r for r in nfp if r['plain']['score'] >= 4 and r['alternative']['score'] < 4]
    assert len(losses) == 121
    assert len({r['id'] for r in losses}) == 79
    rng = random.Random(20261001)
    stable = [r for r in nfp if r['plain']['score'] >= 4 and r['alternative']['score'] >= 4]
    rng.shuffle(stable)
    controls = []
    # No repeated answer within a setting. Same question across settings remains a cluster.
    for loss in sorted(losses, key=lambda r: (r['model'], r['id'])):
        candidates = [r for r in stable if r['model'] == loss['model'] and r['method'] == loss['method']]
        assert candidates
        candidates.sort(key=lambda r: abs(r['plain']['score'] - loss['plain']['score']))
        chosen = candidates[0]
        stable.remove(chosen)
        controls.append(chosen)
    selected = [('loss', r) for r in losses] + [('stable_high_control', r) for r in controls]
    rng.shuffle(selected)
    inputs, manifest = [], []
    for cohort, row in selected:
        key = hashlib.sha256(f"score-sensitivity:{row['model']}:{row['method']}:{row['id']}".encode()).hexdigest()[:24]
        answer = row['alternative']
        inputs.append({'key': key, 'question': answer['question'], 'answer': answer['answer']})
        manifest.append({'key': key, 'model': row['model'], 'method': row['method'],
                         'id': row['id'], 'cohort': cohort, 'plain_score': row['plain']['score'],
                         'alternative_score': answer['score'], 'answer_sha256': answer['answer_sha256']})
    assert len(inputs) == len({r['key'] for r in inputs}) == 242
    write(OUT / 'score_blind_inputs.json', inputs)
    write(OUT / 'score_private_manifest.json', manifest)
    fields = ['model', 'method', 'id', 'dataset', 'plain_score', 'alternative_score',
              'plain_answer_sha256', 'alternative_answer_sha256']
    matrix = [{**{k: row[k] for k in ('model', 'method', 'id', 'dataset')},
               **{f'{side}_score': row[side]['score'] for side in ('plain', 'alternative')},
               **{f'{side}_answer_sha256': row[side]['answer_sha256'] for side in ('plain', 'alternative')}} for row in rows]
    write(OUT / 'all_answer_pair_index.json', matrix)
    with (OUT / 'all_answer_pair_index.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(matrix)
    summary = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'pairs': len(rows), 'answers': 2 * len(rows), 'nfp_pairs': len(nfp),
               'losses': len(losses), 'unique_loss_questions': 79, 'controls': len(controls),
               'losses_by_model': dict(Counter(r['model'] for r in losses)),
               'controls_by_model': dict(Counter(r['model'] for r in controls)),
               'blind_fields': ['key', 'question', 'answer'], 'seed': 20261001,
               'control_rule': 'Same model/method, Plain and alternative >=4; nearest Plain score; seeded tie order; no replacement within setting.',
               'status': 'prepared_not_scored',
               'limitations': ['Selected contrasts do not estimate whole-population error rates.',
                              'Same question across models is not independent.',
                              'Model and intervention differ across settings.',
                              'Original benchmark labels and scores are preserved.']}
    write(OUT / 'validation.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
