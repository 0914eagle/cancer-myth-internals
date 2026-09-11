import csv
import json
from pathlib import Path

import pytest

from scripts.review_pilot_nfp import prepare
from src.jsonl import load_json, read_jsonl, write_jsonl
from src.pilot import digest, make_manifest


@pytest.fixture
def sample(tmp_path, monkeypatch):
    pilot, prior, out = tmp_path / 'pilot', tmp_path / 'prior', tmp_path / 'review'
    (pilot / 'dev').mkdir(parents=True)
    (pilot / 'split').mkdir()
    prior.mkdir()
    questions = [{'id': f'n{i}', 'set': 'nfp', 'partition': 'dev', 'question': f'Question{i}',
                  'hallucination_text': f'Reference{i}', 'group_id': f'g{max(i, 1)}'} for i in range(5)]
    questions.append({'id': 'f0', 'set': 'fpq', 'partition': 'dev', 'question': 'FPQ', 'group_id': 'gf'})
    manifest = make_manifest(questions, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash='ref')
    (pilot / 'split/manifest.json').write_text(json.dumps(manifest))
    write_jsonl(pilot / 'split/dev.jsonl', questions)
    run = {'method': 'fp_identification', 'partition': 'dev', 'manifest_hash': manifest['manifest_hash']}
    responses = [{'id': q['id'], 'set': q['set'], 'question': q['question'],
                  'response': f"Saved answer for {q['id']}", 'run_hash': digest(run)} for q in questions]
    path = pilot / 'dev/fp_identification.jsonl'
    write_jsonl(path, responses)
    Path(str(path) + '.run.json').write_text(json.dumps(run))
    plan = {'cases': [{'id': 'n0', 'question': 'Question0'}]}
    (prior / 'plan.json').write_text(json.dumps(plan))
    (prior / 'run.json').write_text(json.dumps({'plan_hash': digest(plan)}))
    def forbidden(*args, **kwargs):
        raise AssertionError('No external process/model calls allowed in offline preparation')
    monkeypatch.setattr('subprocess.run', forbidden)
    return pilot, prior, out


def test_exports_unused_groups_without_labels_or_existing_judges(sample):
    pilot, prior, out = sample
    before = (pilot / 'dev/fp_identification.jsonl').read_bytes()
    prepare(pilot, prior, out)
    spec = load_json(out / 'review_cases.json')
    assert spec['excluded_question_ids'] == ['n0', 'n1']
    assert [c['id'] for c in spec['cases']] == ['n2', 'n3', 'n4']
    with (out / 'human_review.tsv').open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert len(rows) == 3
    assert all(r['score'] == r['rationale'] == r['reviewer'] == '' for r in rows)
    assert 'Saved answer for n2' in (out / 'review.md').read_text()
    assert before == (pilot / 'dev/fp_identification.jsonl').read_bytes()
    # Human edits survive rerunning the same export.
    edited = (out / 'human_review.tsv').read_text().replace('n2\t\t\t', 'n2\t1\treviewed\tuser')
    (out / 'human_review.tsv').write_text(edited)
    prepare(pilot, prior, out)
    assert (out / 'human_review.tsv').read_text() == edited


def test_rejects_incomplete_generation(sample):
    pilot, prior, out = sample
    path = pilot / 'dev/fp_identification.jsonl'
    rows = list(read_jsonl(path))
    write_jsonl(path, rows[:-1])
    with pytest.raises(ValueError, match='complete'):
        prepare(pilot, prior, out)


def test_rejects_generation_provenance_mismatch(sample):
    pilot, prior, out = sample
    path = pilot / 'dev/fp_identification.jsonl'
    rows = list(read_jsonl(path))
    rows[0]['run_hash'] = 'changed'
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match='provenance'):
        prepare(pilot, prior, out)


def test_rejects_changed_previous_plan(sample):
    pilot, prior, out = sample
    plan = load_json(prior / 'plan.json')
    plan['cases'].append({'id': 'n2', 'question': 'Question2'})
    (prior / 'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='changed'):
        prepare(pilot, prior, out)


def test_changed_answers_require_new_review_directory(sample):
    pilot, prior, out = sample
    prepare(pilot, prior, out)
    path = pilot / 'dev/fp_identification.jsonl'
    rows = list(read_jsonl(path))
    rows[2]['response'] = 'Different answer'
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        prepare(pilot, prior, out)
