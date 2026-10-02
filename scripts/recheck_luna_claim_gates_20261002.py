"""Read saved Luna runs and print a reproducible report; no calls or file writes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = {}


def read(path):
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load_run(name, conditions):
    root = ROOT / 'results/frontier_cli' / name
    source = {q['id']: q for q in json.loads(read(root / 'evaluation_questions.json'))}
    records = {}
    for condition in conditions:
        records[condition] = {}
        for path in sorted((root / condition).glob('*/record.json')):
            record = json.loads(read(path))
            assert record['condition'] == condition
            if record['status'] == 'complete':
                records[condition][record['id']] = record['result']
    common = set.intersection(*(set(r) for r in records.values()))
    return source, records, common


def main():
    source, records, common = load_run('luna_explicit_claim_gate_20261001_v1',
                                       ['baseline', 'context_rule'])
    assert len(common) == 723
    structured = {}
    for label in ['fpq', 'nfp']:
        ids = sorted(i for i in common if source[i]['set'] == label)
        transitions = Counter(f"{int(records['baseline'][i]['has_false_premise'])}->"
                              f"{int(records['context_rule'][i]['has_false_premise'])}" for i in ids)
        conditions = {}
        for condition, rr in records.items():
            all_u = [i for i in ids if rr[i]['examined_premises'] and
                     all(p['assessment'] == 'undetermined' for p in rr[i]['examined_premises'])]
            conditions[condition] = {
                'yes': sum(rr[i]['has_false_premise'] for i in ids),
                'all_undetermined': len(all_u),
                'all_undetermined_yes': sum(rr[i]['has_false_premise'] for i in all_u),
                'no_with_false': sum(not rr[i]['has_false_premise'] and
                                     any(p['assessment'] == 'false' for p in rr[i]['examined_premises'])
                                     for i in ids),
            }
        structured[label] = dict(n=len(ids), transitions=dict(transitions), conditions=conditions)

    evidence = ROOT / 'docs/reviews'
    anchors = list(csv.DictReader(read(evidence / 'luna_explicit_claim_gate_2026-10-01' /
                                       'same_quote_claim_comparisons.csv').splitlines()))
    matched = [r for r in anchors if r['gate_transition'] == '1->0' and
               r['context_assessment'] == 'undetermined']
    quote_summary = {
        'claim_pair_rows': len(matched),
        'items_by_label': {label: len({r['id'] for r in matched if r['label'] == label})
                           for label in ['fpq', 'nfp']},
        'exact_claim_text_rows': [r['id'] for r in matched if r['baseline_claim'] == r['context_claim']],
        'warning': 'Exact quote/claim string matching is not semantic equivalence or target-error verification.'}
    cases = {i: {'source': source[i], 'outputs': {c: rr[i] for c, rr in records.items()}}
             for i in ['fpq_148', 'fpq_274', 'fpq_596', 'fpq_49', 'fpq_2',
                       'nfp_1141', 'nfp_1142', 'nfp_1089']}

    fixed_source, fixed, fixed_common = load_run('luna_fixed_claim_factorial_20261001_v1',
                                                ['control', 'content', 'criterion', 'both'])
    assert len(fixed_common) == 721
    fixed_summary = {}
    for label in ['fpq', 'nfp']:
        ids = [i for i in fixed_common if fixed_source[i]['set'] == label]
        fixed_summary[label] = dict(n=len(ids), conditions={})
        for condition, rr in fixed.items():
            no = [i for i in ids if not rr[i]['has_false_premise']]
            fixed_summary[label]['conditions'][condition] = dict(
                yes=len(ids)-len(no), no=len(no),
                no_with_false=sum(any(p['assessment'] == 'false' for p in rr[i]['claim_assessments'])
                                  for i in no))
    fixed_cases = {i: {'source': fixed_source[i], 'outputs': {c: rr[i] for c, rr in fixed.items()}}
                   for i in ['fpq_611', 'fpq_30', 'fpq_51']}

    links = list(csv.DictReader(read(evidence / 'boundary_followup_2026-10-01' /
                                     'all_732_claim_gate_links.csv').splitlines()))
    assert len(links) == 732
    strata = []
    for label in ['fpq', 'nfp']:
        for flag in ['True', 'False']:
            rows = [r for r in links if r['dataset'] == label and r['apparent_error_flag'] == flag]
            before = sum(float(r['baseline_mean']) for r in rows)
            after = sum(float(r['context_mean']) for r in rows)
            strata.append(dict(label=label, terra_apparent_error=flag == 'True', n=len(rows),
                               baseline_mean_yes_count=before, protected_mean_yes_count=after,
                               baseline_rate=before/len(rows), protected_rate=after/len(rows),
                               net_mean_yes_loss=before-after,
                               groups=dict(Counter(r['group'] for r in rows))))
    report = dict(scope='Saved results only. No model calls, new medical judgments or label changes.',
                  model='gpt-5.6-luna', reasoning='medium',
                  structured=structured, matched_quotes=quote_summary,
                  structured_examples=cases, fixed_claims=fixed_summary, fixed_examples=fixed_cases,
                  terra_direct_strata=strata)
    # A compact digest identifies the exact source snapshot, including raw records.
    report['source_snapshot'] = {
        'file_count': len(HASHES),
        'sha256_of_sorted_path_hashes': hashlib.sha256(json.dumps(HASHES, sort_keys=True).encode()).hexdigest()}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
