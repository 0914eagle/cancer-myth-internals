"""Reconcile existing cohorts for the meeting; no model calls or new labels."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/failure_candidates_2026-10-01'
FILES = {
    'pairs': 'docs/reviews/core_failures_2026-10-01/all_3659_pair_transitions.csv',
    'fpq_audit': 'docs/reviews/boundary_followup_2026-10-01/independent_audit_summary.json',
    'nfp_audit': 'docs/reviews/nfp_plain_pair_review_2026-10-01/paired_review_full.json',
    'claim_links': 'docs/reviews/boundary_followup_2026-10-01/all_732_claim_gate_links.csv',
}


def csv_rows(key):
    with (ROOT / FILES[key]).open() as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    pairs = csv_rows('pairs')
    fpq = json.loads((ROOT / FILES['fpq_audit']).read_text())['fpq']
    nfp = json.loads((ROOT / FILES['nfp_audit']).read_text())
    claims = csv_rows('claim_links')
    assert len(pairs) == 3659 and len(nfp) == 121 and len(claims) == 732
    assert len({(r['model'], r['dataset'], r['id']) for r in pairs}) == 3659
    assert len({(r['model'], r['id']) for r in nfp}) == 121
    models = ['qwen25', 'gemma', 'luna', 'qwen38_off', 'qwen38_on']
    table = []
    for model in models:
        pp = [r for r in pairs if r['model'] == model]
        methods = {r['method'] for r in pp}
        assert len(methods) == 1
        rr = [r for r in nfp if r['model'] == model]
        a = fpq[model]
        row = {'model': model, 'actual_method': methods.pop()}
        for dataset in ['fpq', 'nfp']:
            subset = [r for r in pp if r['dataset'] == dataset]
            row[dataset + '_n'] = len(subset)
            row[dataset + '_improved'] = sum(float(r['plain_score']) < 4 <= float(r['alternative_score']) for r in subset)
            row[dataset + '_worsened'] = sum(float(r['alternative_score']) < 4 <= float(r['plain_score']) for r in subset)
        assert row['nfp_worsened'] == len(rr)
        row.update(fpq_selected_pairs=a['n_pairs'],
                   fpq_selected_plain_direct=a['plain_relations'].get('direct_correction', 0),
                   fpq_selected_alt_direct=a['alternative_relations'].get('direct_correction', 0),
                   nfp_loss_new_challenge=sum(r['plain']['stance'] != 'challenge' and r['alternative']['stance'] == 'challenge' for r in rr),
                   nfp_loss_both_challenge=sum(r['plain']['stance'] == r['alternative']['stance'] == 'challenge' for r in rr),
                   nfp_loss_same_caveat=sum(r['same_substantive_caveat'] == 'yes' for r in rr))
        def flagged(r, side):
            return r[side]['stance'] == 'challenge' and r[side]['scope'] in {'strengthened', 'different_target'}
        row['nfp_loss_new_flagged_challenge'] = sum(flagged(r, 'alternative') and not flagged(r, 'plain') for r in rr)
        table.append(row)
    grouped = defaultdict(Counter)
    for r in claims:
        c = grouped[(r['dataset'], r['group'])]
        c['n'] += 1
        c['personal_code'] += r['has_personal'] == 'True'
        c['general_code'] += r['has_general'] == 'True'
    claim_table = [dict(dataset=k[0], transition=k[1], **v) for k, v in sorted(grouped.items())]
    assert sum(r['fpq_selected_pairs'] for r in table) == 178
    assert sum(r['nfp_loss_new_challenge'] for r in table) == 65
    assert sum(r['nfp_loss_new_flagged_challenge'] for r in table) == 41
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv('model_condition_evidence.csv', table)
    write_csv('luna_claim_codes_all_transitions.csv', claim_table)
    manifest = {
        'inputs': {k: {'path': v, 'sha256': hashlib.sha256((ROOT / v).read_bytes()).hexdigest()} for k, v in FILES.items()},
        'new_model_calls': 0,
        'scope': 'Existing 3659 answer pairs, selected FPQ 178 pairs, selected NFP 121 losses, Luna 732 question links.',
        'limitations': [
            'Selected cohorts are not population error rates; same questions recur across models.',
            'FPQ independent audit and NFP paired audit use different tasks and prompts.',
            'AI codes are not causal explanations or clinical truth labels.',
            'Gate results and answer outcomes are separate tasks, not interchangeable causes.',
            'No existing labels, Well scores, or coding judgments changed.',
        ],
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(table, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
