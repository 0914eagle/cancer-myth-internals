"""Export read-only evidence for the 2026-09-23 presentation; no model calls.

Routing is a retrospective selection of already generated/scored answers.
Run with the project Python from the repository root.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import well_eval as we


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', type=Path, required=True)
    ap.add_argument('--frontier', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source(p):
        p = Path(p)
        sources[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()

    def save(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    def csvfile(name, rows):
        with (out / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)

    def stats(values, expected):
        c = collections.Counter(values)
        return dict(valid=len(values), expected=expected, missing=expected-len(values),
                    counts_0_to_5={str(k): c[k] for k in range(6)},
                    ge4=c[4]+c[5], s5=c[5],
                    mean=sum(values)/len(values) if values else None)

    qpath = args.suite / 'questions.jsonl'
    source(qpath)
    questions = {r['id']: r for r in map(json.loads, qpath.read_text().splitlines())}
    assert len(questions) == 732
    jobs = [('qwen', args.suite / 'well_judge_claude'),
            ('qwen', args.suite / 'well_judge_claude_fpu'),
            ('sonnet', args.frontier / 'sonnet_plain_20260922_v1/well_judge_sonnet_v1'),
            ('sonnet', args.frontier / 'sonnet_corrections_20260922_v1/balanced/well_judge_sonnet_v1'),
            ('sonnet', args.frontier / 'sonnet_corrections_20260922_v1/fp_unconditional/well_judge_sonnet_v1'),
            ('luna', args.frontier / 'luna_corrections_20260922_v1/well_judge_sonnet_v1'),
            ('luna', args.frontier / 'luna_plain_20260922_v1/well_judge_sonnet_v1')]
    all_scores, flat, summaries = {}, [], {}
    template_hashes = None
    for model, folder in jobs:
        plan, ph = we.load_plan(folder)
        _, scores = we.read_ledger(folder, plan, ph)
        assert plan['model'] == 'claude-sonnet-5'
        current_templates = plan['sources']['templates']['derived_templates']
        if template_hashes is None:
            template_hashes = current_templates
        assert current_templates == template_hashes, 'Different evaluation rubrics'
        for name in ['judge_plan.json', 'attempts.jsonl']:
            source(folder / name)
        for qid, q in plan['questions'].items():
            assert qid in questions
            assert (q['question'], q['set']) == (questions[qid]['question'], questions[qid]['set'])
        for method, mapping in plan['mapping'].items():
            key = model + '/' + method
            values = {qid: scores[job] for qid, job in mapping.items() if job in scores}
            if key in all_scores:
                assert all_scores[key] == values, key
                continue
            all_scores[key] = values
            summaries[key] = {s: stats([v for q, v in values.items() if questions[q]['set'] == s],
                                      sum(q['set'] == s for q in questions.values()))
                              for s in ['fpq', 'nfp']}
            flat.extend(dict(model=model, method=method, id=q, set=questions[q]['set'], score=v)
                        for q, v in sorted(values.items()))
    csvfile('answer_scores.csv', flat)
    save('answer_summary.json', summaries)
    case_path = Path('docs/reviews/frontier_claude_2026-09-22/plain_nfp_audit/qwen_sonnet_paired_149.json')
    source(case_path)
    cases = [r for r in json.loads(case_path.read_text()) if r['id'] in ['nfp_1080', 'nfp_1061']]
    assert len(cases) == 2
    save('nfp_examples.json', cases)

    # Common support avoids treating Qwen's missing Plain judgment as a failure.
    plain, corrected = all_scores['qwen/plain'], all_scores['qwen/fp_unconditional']
    common = sorted(set(plain) & set(corrected))
    assert len(common) == 731
    gp = args.suite / 'gates/crossfit_v2_style/result.json'
    source(gp)
    gates = json.loads(gp.read_text())
    flags = {}
    for signal in ['text', 'hidden']:
        rows = [r for r in gates['predictions'] if r['signal'] == signal]
        assert len(rows) == len(questions)
        flags[signal] = {r['id']: r['gate_on'] for r in rows}
        assert set(flags[signal]) == set(questions)
        for r in rows:
            assert r['set'] == questions[r['id']]['set']
            assert r['gate_on'] == (r['score'] > r['threshold'])
        for selection in gates['selections']:
            if selection['signal'] != signal:
                continue
            groups = [{questions[q]['group_id'] for q in selection[k]}
                      for k in ['train_ids', 'calibration_ids', 'evaluation_ids']]
            assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    routed, route_summary = [], {}
    for method in ['plain', 'always_correct', 'text', 'hidden', 'label_oracle']:
        vals = {}
        for q in common:
            on = (method == 'always_correct' or
                  (method == 'label_oracle' and questions[q]['set'] == 'fpq') or
                  (method in flags and flags[method][q]))
            vals[q] = corrected[q] if on else plain[q]
            routed.append(dict(method=method, id=q, set=questions[q]['set'],
                               gate_on=int(on), plain=plain[q], corrected=corrected[q], score=vals[q]))
        route_summary[method] = {}
        for s in ['fpq', 'nfp']:
            ids = [q for q in common if questions[q]['set'] == s]
            info = stats([vals[q] for q in ids], len(ids))
            info.update(rescue_ge4=sum(plain[q] < 4 <= vals[q] for q in ids),
                        harm_ge4=sum(vals[q] < 4 <= plain[q] for q in ids))
            route_summary[method][s] = info
    # Exact expectation of a label-blind uniformly sampled subset of the same size.
    for signal in flags:
        k = sum(flags[signal][q] for q in common)
        rate = k / len(common)
        row = dict(interventions=k, n=len(common), probability=rate)
        for s in ['fpq', 'nfp']:
            ids = [q for q in common if questions[q]['set'] == s]
            row[s] = {name: sum((1-rate)*(plain[q] >= cutoff) + rate*(corrected[q] >= cutoff)
                               for q in ids) / len(ids)
                      for name, cutoff in [('ge4_fraction', 4), ('s5_fraction', 5)]}
        route_summary['random_expected_' + signal] = row
    csvfile('routing_scores.csv', routed)
    save('routing_summary.json', dict(common_n=len(common),
         excluded_ids=sorted(set(questions)-set(common)), results=route_summary))

    lora_summary, lora_scores = {}, []
    for run in ['twin_s4', 'twin_bal_s4', 'twin_dpo_s4', 'fpqonly_s4']:
        folder = args.suite / 'lora' / run / 'well_judge_claude'
        plan, ph = we.load_plan(folder)
        _, scores = we.read_ledger(folder, plan, ph)
        for name in ['judge_plan.json', 'attempts.jsonl']:
            source(folder / name)
        mapping = plan['mapping']['lora_' + run]
        groups = collections.defaultdict(list)
        for qid, job in mapping.items():
            assert job in scores
            q = plan['questions'][qid]
            group = q['set'] if qid in questions else 'modified_question'
            groups[group].append(scores[job])
            lora_scores.append(dict(run=run, id=qid, group=group, score=scores[job]))
        assert {k: len(v) for k, v in groups.items()} == {'fpq': 234, 'nfp': 149, 'modified_question': 132}
        lora_summary[run] = {k: stats(v, len(v)) for k, v in groups.items()}
    save('lora_summary.json', lora_summary)
    csvfile('lora_scores.csv', lora_scores)

    for name, rel in [('gate_summary.json', 'gates/crossfit_v2_style/summary_bootstrap500.json'),
                      ('transfer_summary.json', 'crepe/transfer/v1/summary_bootstrap500.json'),
                      ('style_summary.json', 'style_controls/v2/summary_bootstrap500.json'),
                      ('replacement_summary.json', 'controls/premise_replacement_full_v1/summary.json')]:
        p = args.suite / rel
        source(p)
        save(name, json.loads(p.read_text()))
    for name, rel in [('positions_tables.md', 'positions/pos_v1/report.md'),
                      ('signal_flow_tables.md', 'signal_flow/flow_v1/report.md'),
                      ('token_scan_tables.md', 'tokens/scan_v1/report.md')]:
        p = args.suite / rel
        source(p)
        # Preserve numerical tables, not superseded causal claims in old auto-reports.
        lines = [line for line in p.read_text().splitlines()
                 if line.startswith(('|', '#')) or not line.strip()]
        (out / name).write_text('> Numerical excerpt only; interpretation is in document 37.\n\n' + '\n'.join(lines).rstrip() + '\n')
    for name, path in [('annotated_summary.json', Path('docs/reviews/annotated_premises_2026-09-22/run_v1/summary.json')),
                       ('diagnostic_summary.json', Path('docs/reviews/existing_diagnostic_links_2026-09-22/summary.json'))]:
        source(path)
        save(name, json.loads(path.read_text()))
    status_files = list(args.frontier.glob('*/status.json')) + [
        args.frontier / 'luna_plain_20260922_v1/plain/status.json',
        args.frontier / 'luna_plain_20260922_v1/well_judge_sonnet_v1/status.json']
    save('run_status.json', {str(p): json.loads(p.read_text()) for p in status_files})
    for p in status_files:
        source(p)
    source(Path(__file__))
    save('provenance.json', dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         source_sha256=sources, model_calls=0,
         judge='claude-sonnet-5', rubric_hashes=template_hashes,
         note='Existing ledgers verified against frozen plans/raw ratings. Routing is retrospective; no thresholds refitted.'))
    print(json.dumps(route_summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
