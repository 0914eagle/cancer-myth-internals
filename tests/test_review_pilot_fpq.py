from scripts.review_pilot_fpq import METHODS, select_cases


def fixture():
    qs = {f'f{i}': {'set': 'fpq', 'group_id': f'g{i}', 'question': f'Q{i}', 'correction': f'C{i}'} for i in range(9)}
    qs['n0'] = {'set': 'nfp'}
    rs = {m: {q: {'response': f'{m} answer {q}'} for q in qs} for m in METHODS}
    ss = {m: {q: {'sharpness': i % 3 - 1, 'reason': 'old reason'} for i, q in enumerate(qs)} for m in METHODS}
    return qs, rs, ss


def test_stratified_selection_and_no_nfp():
    args = fixture()
    cases, strata = select_cases(*args)
    assert len(cases) == 18
    assert len({(c['method'], c['question_id']) for c in cases}) == 18
    assert all(c['question_id'].startswith('f') for c in cases)
    assert all(s['selected'] == 2 and s['available'] == 3 for s in strata)
    assert (cases, strata) == select_cases(*args)
    assert 'review_score' not in cases[0]


def test_does_not_force_missing_strata():
    qs, rs, ss = fixture()
    for values in ss.values():
        for v in values.values():
            v['sharpness'] = 1
    cases, strata = select_cases(qs, rs, ss)
    assert len(cases) == 6
    assert all(s['selected'] == 0 for s in strata if s['old_score'] != 1)


def test_offline_export_with_signed_inputs(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import pytest
    from scripts.review_pilot_fpq import MODEL, prepare
    from src.jsonl import read_jsonl, write_jsonl
    from src.pilot import digest, file_digest, make_manifest
    pilot, out = tmp_path / 'pilot', tmp_path / 'review'
    (pilot / 'split').mkdir(parents=True)
    (pilot / 'dev').mkdir()
    qs = [{'id': f'f{i}', 'set': 'fpq', 'question': f'Q{i}', 'correction': f'C{i}',
           'group_id': f'g{i}', 'partition': 'dev'} for i in range(9)]
    manifest = make_manifest(qs, seed=17, folds=5, test_fold=0, dev_fold=1, reference_hash='ref')
    (pilot / 'split/manifest.json').write_text(json.dumps(manifest))
    qpath = pilot / 'split/dev.jsonl'
    write_jsonl(qpath, qs)
    for m in METHODS:
        run = {'method': m, 'partition': 'dev', 'manifest_hash': manifest['manifest_hash']}
        path = pilot / 'dev' / f'{m}.jsonl'
        answers = [{'id': q['id'], 'set': 'fpq', 'question': q['question'], 'response': f'Answer {i}',
                    'run_hash': digest(run)} for i, q in enumerate(qs)]
        write_jsonl(path, answers)
        Path(str(path) + '.run.json').write_text(json.dumps(run))
        signature = {'requested_model': MODEL, 'response_run': run, 'responses_hash': file_digest(path),
                     'questions_hash': file_digest(qpath), 'backend': 'codex', 'temperature': None}
        spath = pilot / 'dev' / f'{m}_judge_codex_{MODEL}.jsonl'
        scored = [{'id': q['id'] + '::' + q['id'], 'question_id': q['id'], 'response_id': q['id'],
                   'set': 'fpq', 'sharpness': i % 3 - 1, 'judge_parsed': True,
                   'judge_run_hash': digest(signature), 'response_run_hash': digest(run),
                   'judge_model': MODEL, 'judge_backend': 'codex', 'judge_temperature': None,
                   'rubric': 'fpq', 'response_text_hash': digest(answers[i]['response'])}
                  for i, q in enumerate(qs)]
        write_jsonl(spath, scored)
        Path(str(spath) + '.run.json').write_text(json.dumps(signature))
    monkeypatch.setattr('subprocess.run', lambda *a, **k: pytest.fail('offline only'))
    prepare(pilot, out)
    text = (out / 'review.md').read_text()
    assert text.count('## R') == 18
    assert 'old reason' not in text and 'fp_identification' not in text
    (out / 'review.md').write_text(text + '\nUser annotation')
    prepare(pilot, out)
    assert (out / 'review.md').read_text().endswith('User annotation')
    rows = list(read_jsonl(pilot / 'dev/plain.jsonl'))
    rows[0]['response'] = 'Changed'
    write_jsonl(pilot / 'dev/plain.jsonl', rows)
    with pytest.raises(ValueError, match='changed'):
        prepare(pilot, out)
