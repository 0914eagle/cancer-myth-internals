"""Read-only analysis of saved Luna structured gates; no model calls or repairs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/frontier_cli/luna_explicit_claim_gate_20261001_v1'
OUT = ROOT / 'docs/reviews/luna_explicit_claim_gate_2026-10-01'
EXAMPLES = ['nfp_1005', 'nfp_1133', 'fpq_2', 'fpq_56', 'fpq_30',
            'fpq_49', 'fpq_308', 'nfp_1048', 'fpq_29', 'nfp_1082']


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = {r['id']: r for r in json.loads((RUN / 'evaluation_questions.json').read_text())}
    data, failures, disagreements, hashes = {}, [], [], {}
    for condition in ['baseline', 'context_rule']:
        data[condition] = {}
        for path in sorted((RUN / condition).glob('*/record.json')):
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            record = json.loads(path.read_text())
            qid = record['id']
            assert record['condition'] == condition
            if record['status'] != 'complete':
                failures.append(dict(id=qid, condition=condition, status=record['status'], error=record.get('error')))
                continue
            data[condition][qid] = record
            result = record['result']
            counts = Counter(x['assessment'] for x in result['examined_premises'])
            if bool(counts['false']) != result['has_false_premise']:
                disagreements.append(dict(id=qid, condition=condition, label=source[qid]['set'],
                    gate=result['has_false_premise'], false_claims=counts['false'],
                    supported_claims=counts['supported'], undetermined_claims=counts['undetermined'],
                    question=source[qid]['question'], result=json.dumps(result, ensure_ascii=False)))
    paired = sorted(set(data['baseline']) & set(data['context_rule']))
    old = {r['id']: r for r in csv.DictReader((RUN / 'historical_repeat_items.csv').open())}
    pairs, changed, anchors = [], [], []
    summary = dict(model='gpt-5.6-luna', reasoning='medium', total_expected=1464,
                   complete=sum(len(d) for d in data.values()), failed=len(failures),
                   paired_questions=len(paired), paired_by_label={}, per_condition={},
                   historical_stable_yes_to_no={}, yes_to_no_diagnostics={})
    for qid in paired:
        a, b = [data[c][qid]['result'] for c in ['baseline', 'context_rule']]
        ay, by = int(a['has_false_premise']), int(b['has_false_premise'])
        row = dict(id=qid, label=source[qid]['set'], baseline=ay, context_rule=by,
                   transition=f'{ay}->{by}',
                   baseline_false_claims=sum(x['assessment']=='false' for x in a['examined_premises']),
                   context_false_claims=sum(x['assessment']=='false' for x in b['examined_premises']),
                   historical_stable_yes_to_no=int(old[qid]['stable_yes_to_no']))
        pairs.append(row)
        if ay != by:
            changed.append(dict(id=qid, label=source[qid]['set'], question=source[qid]['question'],
                                baseline=a, context_rule=b))
        # Exact same quote LIST only: an anchor match is not a semantic claim match.
        for i, x in enumerate(a['examined_premises']):
            if x['assessment'] != 'false':
                continue
            for j, y in enumerate(b['examined_premises']):
                if x['question_quotes'] and x['question_quotes'] == y['question_quotes']:
                    anchors.append(dict(id=qid, label=source[qid]['set'], gate_transition=f'{ay}->{by}',
                        a_index=i, b_index=j, quote=json.dumps(x['question_quotes'], ensure_ascii=False),
                        quote_exact=all(q in source[qid]['question'] for q in x['question_quotes']),
                        baseline_claim=x['claim'], baseline_assessment=x['assessment'], baseline_reason=x['reason'],
                        context_claim=y['claim'], context_assessment=y['assessment'], context_reason=y['reason']))
    for label in ['fpq', 'nfp']:
        sub = [r for r in pairs if r['label']==label]
        counts = Counter(r['transition'] for r in sub)
        summary['paired_by_label'][label] = dict(n=len(sub), transitions=dict(counts),
            baseline_yes=sum(r['baseline'] for r in sub), context_yes=sum(r['context_rule'] for r in sub))
        historical_ids = [i for i in old if old[i]['dataset']==label and old[i]['stable_yes_to_no']=='1']
        summary['historical_stable_yes_to_no'][label] = dict(expected=len(historical_ids),
            paired=sum(i in paired for i in historical_ids),
            current_transitions=dict(Counter(r['transition'] for r in sub if r['id'] in historical_ids)))
        loss = [r for r in sub if r['transition']=='1->0']
        ids = {r['id'] for r in loss}
        arows = [r for r in anchors if r['id'] in ids]
        summary['yes_to_no_diagnostics'][label] = dict(n=len(loss),
            no_false_claim_before=sum(r['baseline_false_claims']==0 for r in loss),
            any_false_claim_before=sum(r['baseline_false_claims']>0 for r in loss),
            any_false_claim_after=sum(r['context_false_claims']>0 for r in loss),
            all_claims_undetermined_after=sum(bool(data['context_rule'][i]['result']['examined_premises']) and
                all(x['assessment']=='undetermined' for x in data['context_rule'][i]['result']['examined_premises']) for i in ids),
            same_quote_false_to_undetermined_items=len({r['id'] for r in arows if r['context_assessment']=='undetermined'}),
            same_quote_false_to_supported_items=len({r['id'] for r in arows if r['context_assessment']=='supported'}))
    for c, records in data.items():
        counts = Counter(x['assessment'] for r in records.values() for x in r['result']['examined_premises'])
        summary['per_condition'][c] = dict(valid=len(records), claims=dict(counts),
            disagreements=sum(r['condition']==c for r in disagreements),
            warnings=dict(Counter(w['type'] for r in records.values() for w in r.get('warnings',[]))))
    # Keep complete saved results locally; a compact census and changed-pair transcript are published.
    with (RUN / 'paired_analysis_all_723.jsonl').open('w') as f:
        for row in pairs:
            i=row['id']
            obj=dict(row, question=source[i]['question'],
                     results={c:data[c][i]['result'] for c in data})
            f.write(json.dumps(obj, ensure_ascii=False)+'\n')
    write_csv('all_paired_decisions.csv', pairs)
    write_csv('failed_records.csv', failures)
    write_csv('claim_gate_disagreements.csv', disagreements)
    write_csv('same_quote_claim_comparisons.csv', anchors)
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (OUT/'changed_pairs.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in changed))
    examples=[]
    for qid in EXAMPLES:
        examples.append(dict(source=source[qid], results={c:data[c][qid]['result'] for c in data}))
    (OUT/'examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2)+'\n')
    manifest=dict(input_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='All completed records; complete-case paired comparison. No repairs, regeneration, medical relabeling, or new judge calls.',
        matching='Exact identical quote lists, not semantic equality of claims. Repeated matches are preserved; item counts use unique IDs.',
        interpretation='Generated justifications are prospective output evidence, not recovered internal/historical reasoning. Individual matched-anchor categories overlap.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    assert len(pairs)==723 and len(changed)==142
    assert summary['complete']==1455 and len(failures)==9 and len(disagreements)==22
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
