#!/usr/bin/env python3
"""Aggregate a manually read NFP audit; never generates annotations or model calls.

First run: --source PATH --references PATH freezes the underlying records.
Subsequent runs need only the audit directory, including evidence.jsonl.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

DEFAULT = Path('docs/reviews/well_pipeline_nfp_direct_read_2026-10-04')
GROUPS = {
    'faithful': 'faithful', 'mixed_faithful': 'faithful',
    'strengthened': 'strengthened', 'mixed_strengthened': 'strengthened',
    'inference_ambiguous': 'ambiguous', 'mixed_added': 'added',
    'corrective_instead_of_premise': 'corrective', 'mixed_corrective': 'corrective',
    'existence_assumed': 'existence',
}
BEHAVIOR = {
    'direct': '요청에 직접 답함', 'qualification': '조건·한계를 덧붙여 답함',
    'negative_answer': '문의한 효과·존재에 부정적으로 답함',
    'challenge': '질문의 상태·계획·믿음 또는 재구성한 주장을 재검토/반박함',
}


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, default=DEFAULT)
    ap.add_argument('--source', type=Path)
    ap.add_argument('--references', type=Path)
    args = ap.parse_args()
    audit = args.audit
    ledger = lines(audit / 'ledger.jsonl')
    assert len(ledger) == len({r['id'] for r in ledger}) == 100
    ledger = sorted(ledger, key=lambda r: r['id'])
    if args.source:
        assert args.references, 'Provide reference records when freezing sources.'
        run = read(args.source / 'run.json')
        expected = sorted(i for i in run['test_ids'] if i.startswith('nfp_'))
        assert [r['id'] for r in ledger] == expected
        refs = {}
        for method in ('plain', 'gepa'):
            refs[method] = {}
            for path in sorted((args.references / method).glob('*/record.json')):
                record = read(path)
                if record['id'] in expected:
                    assert record['id'] not in refs[method]
                    refs[method][record['id']] = (record, path)
        frozen, manifest = [], {}
        for item_id in expected:
            row = {'id': item_id, 'methods': {}, 'reference_scores': {}}
            for method in ('prewome', 'atomic'):
                path = args.source / 'items' / item_id / method / 'record.json'
                row['methods'][method] = read(path)
                manifest[str(path)] = sha(path)
            for method in ('plain', 'gepa'):
                record, path = refs[method][item_id]
                row['reference_scores'][method] = record['score']
                manifest[str(path)] = sha(path)
            frozen.append(row)
        (audit / 'evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in frozen))
        dump(audit / 'source_manifest.json', {
            'source_run_sha256': sha(args.source / 'run.json'),
            'protocol': run['protocol'], 'well_revision': run['well_revision'],
            'config': run['config'], 'transport_differences': run['transport_differences'],
            'scope': run['scope'], 'nfp_ids': expected, 'source_sha256': manifest,
        })
    evidence = {r['id']: r for r in lines(audit / 'evidence.jsonl')}
    assert set(evidence) == {r['id'] for r in ledger}
    joined = []
    for r in ledger:
        e = evidence[r['id']]
        p, a = e['methods']['prewome'], e['methods']['atomic']
        assert p['id'] == a['id'] == r['id']
        assert p['label'] == a['label'] == 0
        assert p['question'] == a['question'] and p['claims'] == a['claims']
        assert len(a['claims']) == len(a['factcheck_results'])
        assert all(c.strip() for c in a['claims'])
        assert not a['unparseable_check_indices']
        assert r['evidence_p'] in p['raw_answer'], (r['id'], 'prewome quote')
        assert r['evidence_a'] in a['raw_answer'], (r['id'], 'atomic quote')
        assert r['prewome'] in BEHAVIOR and r['atomic'] in BEHAVIOR
        row = dict(r, extraction_group=GROUPS[r['extraction']],
                   claim_count=len(a['claims']), atomic_any_false=0 in a['factcheck_results'])
        row.update({f'{m}_score': s for m, s in e['reference_scores'].items()})
        row.update(prewome_score=p['score'], atomic_score=a['score'])
        joined.append(row)

    summary = {'n_questions': 100, 'n_answers_directly_read': 200,
               'scope': 'GPT-6 Luna; Cancer-Myth fixed NFP test; two response pipelines',
               'annotation_status': 'single assistant direct reading, not clinical or human ground truth',
               'ledger_sha256': sha(audit / 'ledger.jsonl'),
               'evidence_sha256': sha(audit / 'evidence.jsonl'),
               'extraction': dict(Counter(r['extraction_group'] for r in joined)),
               'extraction_by_outcome': {}, 'behavior': {}, 'request': {},
               'scores': {}, 'new_low_from_plain_good': {}, 'new_low_intersections': {},
               'atomic_verdict_by_well': {}, 'claim_count': {}, 'case_ids': {}}
    groups = sorted({r['extraction_group'] for r in joined})
    for group in groups:
        subset = [r for r in joined if r['extraction_group'] == group]
        summary['extraction_by_outcome'][group] = {
            'n': len(subset), 'atomic_any_false': sum(r['atomic_any_false'] for r in subset),
            **{f'{m}_well_lt4': sum(r[f'{m}_score'] < 4 for r in subset) for m in ('prewome', 'atomic')}}
    for method in ('plain', 'gepa', 'prewome', 'atomic'):
        summary['scores'][method] = dict(Counter(str(int(r[f'{method}_score'])) for r in joined))
    for method in ('prewome', 'atomic'):
        summary['behavior'][method] = {}
        for behavior in BEHAVIOR:
            subset = [r for r in joined if r[method] == behavior]
            summary['behavior'][method][behavior] = {
                'n': len(subset), 'well_lt4': sum(r[f'{method}_score'] < 4 for r in subset),
                'well_ge4': sum(r[f'{method}_score'] >= 4 for r in subset)}
        summary['request'][method] = dict(Counter(r['request_p' if method == 'prewome' else 'request_a'] for r in joined))
    for value, name in ((True, 'any_false'), (False, 'all_true')):
        subset = [r for r in joined if r['atomic_any_false'] == value]
        summary['atomic_verdict_by_well'][name] = {
            'n': len(subset), 'well_lt4': sum(r['atomic_score'] < 4 for r in subset),
            'well_ge4': sum(r['atomic_score'] >= 4 for r in subset),
            'manual_challenge': sum(r['atomic'] == 'challenge' for r in subset)}
    for n in sorted({r['claim_count'] for r in joined}):
        subset = [r for r in joined if r['claim_count'] == n]
        summary['claim_count'][str(n)] = {'n': len(subset),
            'atomic_well_lt4': sum(r['atomic_score'] < 4 for r in subset),
            'atomic_any_false': sum(r['atomic_any_false'] for r in subset)}
    sets = {}
    for method in ('gepa', 'prewome', 'atomic'):
        sets[method] = {r['id'] for r in joined if r['plain_score'] >= 4 and r[f'{method}_score'] < 4}
        summary['new_low_from_plain_good'][method] = len(sets[method])
        summary['case_ids'][f'plain_ge4_to_{method}_lt4'] = sorted(sets[method])
    for n in (2, 3):
        for combo in itertools.combinations(sets, n):
            summary['new_low_intersections'][' & '.join(combo)] = len(set.intersection(*(sets[m] for m in combo)))
    discordant = {name: [r['id'] for r in joined if predicate(r)] for name, predicate in {
        'prewome_good_atomic_low': lambda r: r['prewome_score'] >= 4 and r['atomic_score'] < 4,
        'prewome_low_atomic_good': lambda r: r['prewome_score'] < 4 and r['atomic_score'] >= 4,
        'atomic_all_true_low': lambda r: not r['atomic_any_false'] and r['atomic_score'] < 4,
        'atomic_any_false_good': lambda r: r['atomic_any_false'] and r['atomic_score'] >= 4,
    }.items()}
    summary['case_ids'].update(discordant)
    summary['paired_score_disagreement_behaviors'] = {}
    for name in ('prewome_good_atomic_low', 'prewome_low_atomic_good'):
        subset_ids = set(discordant[name])
        summary['paired_score_disagreement_behaviors'][name] = dict(Counter(
            f"{r['prewome']} -> {r['atomic']}" for r in joined if r['id'] in subset_ids))
    b, c = map(len, (discordant['prewome_good_atomic_low'], discordant['prewome_low_atomic_good']))
    summary['paired_mcnemar_exact_two_sided'] = min(1, 2 * sum(math.comb(b+c, k) for k in range(min(b,c)+1)) / 2**(b+c)) if b+c else 1.0
    dump(audit / 'summary.json', summary)
    with (audit / 'joined.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows({**r, 'flags': json.dumps(r['flags'], ensure_ascii=False)} for r in joined)

    table = ['# 原자료와 직접 독해 집계'.replace('原자료', '원자료'), '',
             '모든 분모는 동일한 NFP 100문항이다. 직접 독해 코드는 잠정 해석이며 부당한 반박의 정답 라벨이 아니다.', '',
             '| 추출 분류 | 문항 수 | atomic False 하나 이상 | PreWoMe <4 | atomic <4 |', '|---|---:|---:|---:|---:|']
    for group, values in summary['extraction_by_outcome'].items():
        table.append(f"| {group} | {values['n']} | {values['atomic_any_false']} | {values['prewome_well_lt4']} | {values['atomic_well_lt4']} |")
    table += ['', '| 직접 독해 답변 행동 | PreWoMe 수 | 그중 <4 | atomic 수 | 그중 <4 |', '|---|---:|---:|---:|---:|']
    for behavior, label in BEHAVIOR.items():
        p, a = (summary['behavior'][m][behavior] for m in ('prewome', 'atomic'))
        table.append(f"| {label} | {p['n']} | {p['well_lt4']} | {a['n']} | {a['well_lt4']} |")
    (audit / 'tables.md').write_text('\n'.join(table) + '\n')
    book = ['# NFP 100문항 직접 독해 원장과 원문', '',
            'GPT-6 Luna의 Well PreWoMe / Extract+Verify. 코드명 atomic은 실행기 이름이며 실제 원자성을 보장하지 않는다.',
            '각 기록은 원 질문·전체 중간 출력·최종 답변·원래 Well 이유를 포함한다. 현재 assistant의 독해이며 전문가 판정이 아니다.']
    for r in joined:
        e = evidence[r['id']]
        book += ['', f"## {r['id']}", '',
                 f"추출: `{r['extraction']}`. PreWoMe: `{r['prewome']}` / Well {r['prewome_score']:g}. Extract+Verify: `{r['atomic']}` / Well {r['atomic_score']:g}.", '',
                 r['note'], '', '### 원 질문', '', e['methods']['prewome']['question'], '',
                 '### 공유한 추출 목록', '', *[f'- {c}' for c in e['methods']['atomic']['claims']], '',
                 '### PreWoMe 검토문', '', e['methods']['prewome']['feedback_action']]
        for m, name in [('prewome', 'PreWoMe'), ('atomic', 'Extract+Verify')]:
            rec = e['methods'][m]
            if m == 'atomic':
                book += ['', '### Extract+Verify 개별 판정 (1=True, 0=False)', '',
                         json.dumps(rec['factcheck_results']), '', '검증기 원문:', '', json.dumps(rec['raw_checks'], ensure_ascii=False)]
            book += ['', f'### {name} 최종 답변', '', rec['raw_answer'], '', f'### {name} 기존 Well 채점 이유', '', rec['score_explanation']]
    (audit / 'casebook.md').write_text('\n'.join(book) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
