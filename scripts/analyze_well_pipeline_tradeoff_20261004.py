#!/usr/bin/env python3
"""Reproduce stored-response diagnostics; no generation, judging, or automatic semantic coding.

Use --freeze once to copy the exact inputs into the review bundle. Thereafter the
bundle suffices; semantic notes are manually authored and separately validated.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/well_pipeline_tradeoff_diagnosis_2026-10-04'
SOURCE = ROOT / 'results/fpqa_prompting/well_pipelines_luna6_20261004'
REFERENCE = ROOT / 'results/fpqa_prompting/well_upstream_test_v1_20261003_run/cancer_myth'
TEST = ROOT / 'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl'
OLD = ROOT / 'docs/reviews/well_pipeline_nfp_direct_read_2026-10-04'
METHODS = ('plain', 'gepa', 'prewome', 'atomic')


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    if (OUT / 'evidence.jsonl').exists():
        raise SystemExit('Frozen evidence exists; refusing to overwrite.')
    run = read(SOURCE / 'run.json')
    ids = sorted(set(run['test_ids']) - set(run['fewshot_overlap_ids']))
    refs = {}
    manifest = {}

    def capture(p):
        manifest[str(p.relative_to(ROOT))] = sha(p)
        return read(p)

    for m in ('plain', 'gepa'):
        refs[m] = {}
        for p in sorted((REFERENCE / m).glob('*/record.json')):
            r = read(p)
            if r['id'] in ids:
                assert r['id'] not in refs[m]
                refs[m][r['id']] = capture(p)
        assert set(refs[m]) == set(ids)
    gold = {r['id']: r for r in lines(TEST)}
    old = {r['id']: r for r in lines(OLD / 'ledger.jsonl')}
    frozen = []
    for item in ids:
        row = {'id': item, 'gold': gold[item], 'methods': {}}
        for m in METHODS:
            r = refs[m][item] if m in refs else capture(SOURCE / 'items' / item / m / 'record.json')
            row['methods'][m] = r
        # Capture the real final input, rather than inferring its contents from code.
        q = capture(SOURCE / 'items' / item / 'stages/atomic/answer/request.json')
        row['atomic_final_user_input'] = [m['content'] for m in q['messages'] if m['role'] == 'user'][-1]
        row['previous_nfp_reading'] = old.get(item)
        frozen.append(row)
    for p in (SOURCE / 'run.json', TEST, OLD / 'ledger.jsonl', OLD / 'evidence.jsonl'):
        manifest[str(p.relative_to(ROOT))] = sha(p)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in frozen))
    dump(OUT / 'source_manifest.json', {'sha256': manifest, 'excluded_ids': run['fewshot_overlap_ids'],
        'config': run['config'], 'scope': run['scope'], 'well_revision': run['well_revision']})


def summarize():
    rows = lines(OUT / 'evidence.jsonl')
    assert len(rows) == len({r['id'] for r in rows}) == 199
    flat = []
    for r in rows:
        a, p = r['methods']['atomic'], r['methods']['prewome']
        assert a['claims'] == p['claims'] and a['question'] == p['question'] == r['gold']['question']
        assert not a['fewshot_overlap'] and not a['unparseable_check_indices']
        assert len(a['claims']) == len(a['factcheck_results']) > 0
        assert all(c.strip() for c in a['claims']) and set(a['factcheck_results']) <= {0, 1}
        assert all(d['label'] == r['gold']['label'] for d in r['methods'].values())
        for m in ('plain', 'gepa'):
            assert r['methods'][m]['feedback']['question'] == a['question']
        false = 0 in a['factcheck_results']
        user = r['atomic_final_user_input']
        if false:
            assert 'Feedback: The question contains false presuppositions that ' in user
            assert 'Action: Correct the false assumptions that ' in user
            assert all(c in user for c, v in zip(a['claims'], a['factcheck_results']) if v == 0)
        else:
            assert 'Feedback: The question is valid and does not contain false presuppositions.' in user
        f = {'id': r['id'], 'label': r['gold']['label'], 'n_extracted_entries': len(a['claims']),
             'any_false': false, **{m: d['score'] for m, d in r['methods'].items()}}
        if r['gold']['label'] == 0:
            l = r['previous_nfp_reading']
            assert l['evidence_p'] in p['raw_answer'] and l['evidence_a'] in a['raw_answer']
            f.update(nfp_extraction=l['extraction'], nfp_prewome_behavior=l['prewome'],
                     nfp_atomic_behavior=l['atomic'])
        flat.append(f)
    summary = {'n': len(rows), 'evidence_sha256': sha(OUT / 'evidence.jsonl'),
               'automatic_measures': 'Stored scores, flags, input structure; no new semantic judgments',
               'by_label': {}, 'nfp_previous_reading': {}}
    for label, name in ((1, 'FPQ'), (0, 'NFP')):
        sub = [r for r in flat if r['label'] == label]
        stats = {'n': len(sub), 'methods': {}, 'atomic_verdict_to_response': {}}
        for m in METHODS:
            stats['methods'][m] = {'ge4': sum(r[m] >= 4 for r in sub),
                'plain_lt4_to_ge4': [r['id'] for r in sub if r['plain'] < 4 <= r[m]],
                'plain_ge4_to_lt4': [r['id'] for r in sub if r[m] < 4 <= r['plain']],
                'score_distribution': dict(Counter(str(r[m]) for r in sub))}
        for value, category in ((True, 'any_false'), (False, 'all_true')):
            part = [r for r in sub if r['any_false'] == value]
            stats['atomic_verdict_to_response'][category] = {'n': len(part),
                'ge4_ids': [r['id'] for r in part if r['atomic'] >= 4],
                'lt4_ids': [r['id'] for r in part if r['atomic'] < 4]}
        stats['any_pipeline_lt4_ids'] = [r['id'] for r in sub if min(r['prewome'], r['atomic']) < 4]
        summary['by_label'][name] = stats
    nf = [r for r in flat if r['label'] == 0]
    for m in ('prewome', 'atomic'):
        newly = [r for r in nf if r['plain'] >= 4 and r[m] < 4]
        summary['nfp_previous_reading'][m] = {'new_lt4_n': len(newly),
            'extraction': dict(Counter(r['nfp_extraction'] for r in newly)),
            'behavior': dict(Counter(r[f'nfp_{m}_behavior'] for r in newly))}
    summary['nfp_previous_reading']['entry_count'] = {
        str(n): {'n': sum(r['n_extracted_entries'] == n for r in nf),
                 'atomic_lt4': sum(r['n_extracted_entries'] == n and r['atomic'] < 4 for r in nf)}
        for n in sorted({r['n_extracted_entries'] for r in nf})}
    notes = lines(OUT / 'manual_notes.jsonl') if (OUT / 'manual_notes.jsonl').exists() else []
    source = {r['id']: r for r in rows}
    for n in notes:
        assert n['id'] in source
        for q in n.get('quotes', []):
            rec = source[n['id']]['methods'][q['method']]
            txt = rec['feedback']['answer'] if q['method'] in ('plain', 'gepa') else rec['raw_answer']
            assert q['text'] in txt, (n['id'], q)
    for cohort in ('fpq_residual', 'fpq_all_true_ge4', 'fpq_recovery_examples'):
        selected = [n for n in notes if n['cohort'] == cohort]
        summary[cohort] = {'n': len(selected), 'ids': [n['id'] for n in selected],
                           'categories': dict(Counter(n['category'] for n in selected))}
    if notes:
        assert set(summary['fpq_residual']['ids']) == set(summary['by_label']['FPQ']['any_pipeline_lt4_ids'])
        assert set(summary['fpq_all_true_ge4']['ids']) == set(summary['by_label']['FPQ']['atomic_verdict_to_response']['all_true']['ge4_ids'])
        assert len({(n['cohort'], n['id']) for n in notes}) == len(notes)
        summary['fpq_residual_by_method'] = {}
        for m in ('prewome', 'atomic'):
            selected = [n for n in notes if n['cohort'] == 'fpq_residual' and m in n['per_method']]
            expected = {r['id'] for r in flat if r['label'] == 1 and r[m] < 4}
            assert {n['id'] for n in selected} == expected
            summary['fpq_residual_by_method'][m] = dict(Counter(n['per_method'][m] for n in selected))
    dump(OUT / 'summary.json', summary)
    fields = sorted(set().union(*(r.keys() for r in flat)))
    with (OUT / 'per_item.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(flat)
    # Whole texts are retained for audit; selection is explicitly bounded.
    book = ['# FPQ 逐語記録'.replace('逐語記録', '원문 대조 기록'), '',
            '선정 집단과 독해는 manual_notes.jsonl. 전체 199문항 원자료는 evidence.jsonl.',
            '이 파일은 판정을 새로 생성하지 않으며 기존 채점과 수동 독해를 구분한다.']
    for item in sorted({n['id'] for n in notes}):
        r = source[item]
        book += ['', f'## {item}', '', r['gold']['question'], '', '주석: ' + ' / '.join(r['gold']['presuppositions'])]
        for n in notes:
            if n['id'] == item:
                book += ['', n['cohort'] + ': ' + n['note']]
        for m in ('plain', 'prewome', 'atomic'):
            d = r['methods'][m]
            book += ['', f'### {m} 기존 점수 {d["score"]}', '']
            if m == 'plain':
                book += [d['feedback']['answer'], '', d['feedback']['score_explanation']]
            else:
                book += [json.dumps(d['claims'], ensure_ascii=False), '',
                         d.get('feedback_action', json.dumps(d.get('factcheck_results'))), '',
                         d['raw_answer'], '', d['score_explanation']]
    (OUT / 'casebook.md').write_text('\n'.join(book) + '\n')
    print(json.dumps({'n': len(flat), 'manual_records': len(notes), 'by_label': {
        k: {m: v['ge4'] for m, v in d['methods'].items()} for k, d in summary['by_label'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze', action='store_true')
    args = ap.parse_args()
    if args.freeze:
        freeze()
    summarize()
