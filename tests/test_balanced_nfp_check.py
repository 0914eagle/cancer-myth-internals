import csv
import json

import pytest

from scripts import check_terra_judge as check
from scripts.prepare_balanced_nfp_check import prepare
from src.pilot import digest, file_digest
from src.jsonl import load_json


@pytest.fixture
def inputs(tmp_path):
    source, review, target = [tmp_path / n for n in ('v2', 'review', 'balanced')]
    source.mkdir()
    review.mkdir()
    template = '{{question}} / {{possible_hallucination}} / {{answer}}'
    cases = [{'id': f'old{i}', 'question': 'Q', 'reference': 'R', 'answer': 'A',
              'prompt': 'Q / R / A', 'cohort': 'unchanged'} for i in range(10)]
    base = {'model': check.MODEL, 'backend': 'codex', 'protocol': check.PROTOCOL_V2,
            'prompt_template': template, 'prompt_template_hash': digest(template),
            'cases': cases, 'transport_hash': file_digest(check.ROOT / 'src/llm_backend.py')}
    (source / 'plan.json').write_text(json.dumps(base))
    (source / 'human_review.tsv').write_text('id\tscore\trationale\treviewer\n'+
        ''.join(f'old{i}\t1\treason\treviewer\n' for i in range(10)))
    (source / 'run.json').write_text(json.dumps({
        'plan_hash': digest(base), 'human_hash': file_digest(source / 'human_review.tsv')}))
    candidate_cases = [{'id': f'n{i:02}', 'question': f'Q{i}', 'reference': 'R',
                        'answer': 'Quoted phrase', 'answer_hash': digest('Quoted phrase')} for i in range(20)]
    (review / 'review_cases.json').write_text(json.dumps({
        'version': 'nfp-overcorrection-review-v1', 'cases': candidate_cases}))
    labels = review / 'draft.tsv'
    with labels.open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['id','score','rationale','reviewer','status','answer_evidence'])
        for i in range(20):
            w.writerow([f'n{i:02}', '1' if i < 8 else '-1' if i < 14 else '',
                        'rationale', '', 'candidate' if i < 14 else 'hold',
                        'Quoted phrase' if 8 <= i < 14 else ''])
    return source, review, labels, target


def test_draft_acceptance_required(inputs):
    with pytest.raises(ValueError, match='accept-ai-review'):
        prepare(*inputs)
    assert not inputs[-1].exists()


def test_balanced_selection_preserves_protocol_and_exclusions(inputs):
    source, review, labels, target = inputs
    original = (source / 'plan.json').read_bytes()
    prepare(*inputs, accept_ai_review=True)
    plan = load_json(target / 'plan.json')
    assert len(plan['cases']) == 10 and len(plan['order']) == 20
    assert sum(c['cohort'] == 'normal' for c in plan['cases']) == 5
    assert sum(c['cohort'] == 'overcorrection' for c in plan['cases']) == 5
    assert plan['held_ids'] == [f'n{i:02}' for i in range(14,20)]
    assert not set(plan['held_ids']) & {c['id'] for c in plan['cases']}
    assert plan['prompt_template'] == load_json(source / 'plan.json')['prompt_template']
    snapshot = (target / 'plan.json').read_bytes()
    prepare(*inputs, accept_ai_review=True)
    assert snapshot == (target / 'plan.json').read_bytes()
    assert original == (source / 'plan.json').read_bytes()
    assert not (target / 'attempts.jsonl').exists()


def test_constant_positive_judge_fails_negative_group(inputs, monkeypatch, capsys):
    prepare(*inputs, accept_ai_review=True)
    calls = []
    def make(*args, **kwargs):
        def call(prompt):
            calls.append(prompt)
            return json.dumps({'Sharpness':1, 'Reason':'mock', 'AnswerEvidence':'', 'InventedPremise':''}), check.MODEL
        return call
    monkeypatch.setattr(check, 'make_caller', make)
    check.score(inputs[-1])
    check.score(inputs[-1])
    check.report(inputs[-1], reparse=True)
    text = capsys.readouterr().out
    assert len(calls) == 20
    assert 'normal: repeat agreement 5/5 pairs; human agreement 10/10 judgments' in text
    assert 'overcorrection: repeat agreement 5/5 pairs; human agreement 0/10 judgments' in text
    assert 'Held IDs (not scored)' in text
