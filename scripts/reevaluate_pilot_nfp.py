"""Frozen-v2 NFP-only dev evaluation; saved generations, <=90 calls, no retries.

Deduplicate by question ID + complete rendered prompt across three baselines.
Old judge-development repetitions are not pooled into this one-pass evaluation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_terra_judge import (
    MODEL, PROTOCOL_V2, events_by_id, evidence_issue, fixed_text,
    parse_nfp_json, render_v2, validate_protocol,
)
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.llm_backend import make_caller
from src.pilot import digest, file_digest, frozen_json, load_manifest, output_lock

METHODS = ('plain', 'fp_identification', 'premise_cot')
VERSION = 'nfp-baselines-v2-once-v1'


def load_generation(pilot, method, manifest, questions):
    path = pilot / 'dev' / f'{method}.jsonl'
    run_path = Path(str(path) + '.run.json')
    run = load_json(run_path)
    if (run.get('method') != method or run.get('partition') != 'dev'
        or run.get('manifest_hash') != manifest['manifest_hash']):
        raise ValueError(f'{method}: generation run mismatch')
    rows = list(read_jsonl(path))
    indexed = {r['id']: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(questions):
        raise ValueError(f'{method}: need complete unique dev generation coverage')
    for qid, row in indexed.items():
        q = questions[qid]
        if (row.get('run_hash') != digest(run) or row.get('question') != q['question']
            or row.get('set') != q['set'] or not row.get('response', '').strip()):
            raise ValueError(f'{method}: generation provenance/content mismatch: {qid}')
    return indexed, {str(p.resolve()): file_digest(p) for p in (path, run_path)}


def prepare(pilot, source, out):
    pilot, source, out = map(Path, (pilot, source, out))
    base = load_json(source / 'plan.json')
    validate_protocol(base)
    if base.get('protocol') != PROTOCOL_V2 or base.get('model') != MODEL or base.get('backend') != 'codex':
        raise ValueError('Use the completed Terra v2 diagnostic as protocol source')
    if load_json(source / 'run.json') != {
        'plan_hash': digest(base), 'human_hash': file_digest(source / 'human_review.tsv')
    }:
        raise ValueError('Protocol source changed after scoring')
    if base['transport_hash'] != file_digest(ROOT / 'src/llm_backend.py'):
        raise ValueError('Transport differs from diagnostic')
    manifest_path = pilot / 'split/manifest.json'
    manifest = load_manifest(manifest_path)
    questions = {q['id']: q for q in manifest['questions'] if q['partition'] == 'dev'}
    qpath = pilot / 'split/dev.jsonl'
    qrows = list(read_jsonl(qpath))
    if len(qrows) != len(questions) or {q['id']: q for q in qrows} != questions:
        raise ValueError('Dev questions differ from manifest')
    ids = sorted(i for i, q in questions.items() if q['set'] == 'nfp')
    if len(ids) != 30:
        raise ValueError('This pilot diagnostic expects 30 NFP dev questions')
    sources = {str(p.resolve()): file_digest(p) for p in (manifest_path, qpath)}
    cases, mapping = {}, {}
    for method in METHODS:
        rows, hashes = load_generation(pilot, method, manifest, questions)
        sources.update(hashes)
        mapping[method] = {}
        for qid in ids:
            case = {'question_id': qid, 'question': questions[qid]['question'],
                    'reference': questions[qid]['hallucination_text'], 'answer': rows[qid]['response']}
            case['prompt'] = render_v2(base['prompt_template'], case)
            key = digest([qid, case['prompt']])
            cases[key] = {**case, 'id': key}
            mapping[method][qid] = key
    order = [{'id': key, 'case_id': key} for key in sorted(cases)]
    random.Random(17).shuffle(order)
    plan = {'version': VERSION, 'model': MODEL, 'backend': 'codex', 'protocol': PROTOCOL_V2,
            'prompt_template': base['prompt_template'], 'prompt_template_hash': base['prompt_template_hash'],
            'transport_hash': base['transport_hash'], 'source_plan_hash': digest(base),
            'sources': sources, 'manifest_hash': manifest['manifest_hash'], 'seed': 17,
            'question_ids': ids, 'mapping': mapping, 'cases': list(cases.values()),
            'order': order, 'budget': len(order)}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'plan.json', plan)
        fixed_text(out / 'protocol.txt', base['prompt_template'])
    print(f'Prepared 3 methods x 30 NFP answers; {len(order)} unique NEW calls (maximum 90).', flush=True)
    print('Preparation: 0 calls. No generation, FPQ scoring, or diagnostic-score reuse.', flush=True)


def get_plan(out):
    plan = load_json(out / 'plan.json')
    validate_protocol(plan)
    ids, mapping = plan['question_ids'], plan['mapping']
    cases = {c['id']: c for c in plan['cases']}
    jobs = {j['id'] for j in plan['order']}
    if (plan['version'] != VERSION or plan['model'] != MODEL or plan['backend'] != 'codex'
        or plan['protocol'] != PROTOCOL_V2 or len(ids) != 30 or len(set(ids)) != 30
        or set(mapping) != set(METHODS) or not 30 <= plan['budget'] <= 90
        or len(plan['cases']) != len(cases) or plan['budget'] != len(plan['order'])
        or len(jobs) != plan['budget'] or jobs != set(cases)
        or any(j['id'] != j['case_id'] for j in plan['order'])):
        raise ValueError('Invalid fixed-budget baseline plan')
    referenced = set()
    for method in METHODS:
        if set(mapping[method]) != set(ids):
            raise ValueError('Incomplete method mapping')
        for qid, key in mapping[method].items():
            if key not in cases or cases[key]['question_id'] != qid:
                raise ValueError('Mapping differs from case')
            referenced.add(key)
    if referenced != set(cases) or any(
        c['id'] != digest([c['question_id'], c['prompt']]) for c in cases.values()
    ):
        raise ValueError('Deduplication key mismatch')
    return plan, cases


def score(out, timeout=180):
    out = Path(out)
    with output_lock(out / 'score'):
        plan, cases = get_plan(out)
        if plan['transport_hash'] != file_digest(ROOT / 'src/llm_backend.py'):
            raise ValueError('Transport changed after preparation')
        run = {'plan_hash': digest(plan)}
        frozen_json(out / 'run.json', run)
        previous = events_by_id(out, plan, digest(run))
        remaining = plan['budget'] - len(previous)
        print(f'Protocol: {PROTOCOL_V2}; model: {MODEL}; remaining NEW calls: {remaining}', flush=True)
        print('No retries/preflight. Started/interrupted calls consume budget.', flush=True)
        if not remaining:
            return
        call = make_caller('codex', MODEL, timeout=timeout)
        for job in plan['order']:
            if job['id'] in previous:
                continue
            case = cases[job['case_id']]
            common = {**job, 'run_hash': digest(run)}
            append_jsonl(out / 'attempts.jsonl', {**common, 'status': 'started',
                         'at': dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                raw, used = call(case['prompt'])
            except Exception as exc:
                append_jsonl(out / 'attempts.jsonl', {**common, 'status': 'finished',
                             'valid': False, 'error': str(exc)})
                raise SystemExit('Call failed; stopped. Attempt will not be retried.') from exc
            parsed, ok = parse_nfp_json(raw)
            valid = ok and used == MODEL
            append_jsonl(out / 'attempts.jsonl', {**common, 'status': 'finished', 'raw': raw,
                         'model': used, 'valid': valid, 'score': parsed.get('Sharpness') if valid else None,
                         'reason': parsed.get('Reason'),
                         'evidence_issue': evidence_issue(parsed, case['answer']) if ok else 'Invalid JSON'})
            print(f"{case['question_id']}: {parsed.get('Sharpness') if valid else 'invalid'}", flush=True)
            if used != MODEL:
                raise SystemExit('Unexpected model; stopped without retry')
        print('Finished. Run report (offline); missing/invalid answers are not counted as passes.')


def report(out):
    out = Path(out)
    plan, cases = get_plan(out)
    run = load_json(out / 'run.json')
    if run != {'plan_hash': digest(plan)}:
        raise ValueError('Plan changed after scoring')
    events = events_by_id(out, plan, digest(run))
    results, flags, details = {}, {}, {}
    for key, rows in events.items():
        event = rows[-1]
        parsed, ok = parse_nfp_json(event.get('raw'))
        if event['status'] == 'finished' and event.get('model') == MODEL and ok:
            results[key] = parsed['Sharpness']
            details[key] = parsed
            issue = evidence_issue(parsed, cases[key]['answer'])
            if issue:
                flags[key] = issue
    mapped = {m: {qid: results.get(key) for qid, key in plan['mapping'][m].items()} for m in METHODS}
    lines = ['# NFP baseline reevaluation — dev', '',
             f'Judge: {MODEL}; protocol: {PROTOCOL_V2}. NFP only; no FPQ reevaluation.',
             'Pass = +1: no reference-targeted invented premise objection. Not overall medical QA accuracy.',
             'Descriptive dev analysis includes judge-development/review cases; not independent test validation.',
             'One fresh judgment per unique question/prompt, shared across methods with identical answers.',
             f'Attempted {len(events)}/{plan["budget"]}; valid unique judgments {len(results)}; evidence flags {len(flags)}.',
             'Evidence flags retain scores and require review. Missing values are never imputed.', '',
             '| Method | Valid / 30 | +1 | NFP pass % | Paired n vs Plain | Rescue / harm | Paired delta (pp) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, scores in mapped.items():
        valid = [v for v in scores.values() if v is not None]
        passed = sum(v == 1 for v in valid)
        pct = f'{100*passed/30:.1f}' if len(valid) == 30 else 'incomplete'
        pairs = [i for i in plan['question_ids'] if scores[i] is not None and mapped['plain'][i] is not None]
        rescue = sum(scores[i] == 1 and mapped['plain'][i] == -1 for i in pairs)
        harm = sum(scores[i] == -1 and mapped['plain'][i] == 1 for i in pairs)
        delta = f'{100*(rescue-harm)/len(pairs):+.1f}' if pairs else 'NA'
        lines.append(f'| {method} | {len(valid)}/30 | {passed} | {pct} | {len(pairs)} | {rescue}/{harm} | {delta} |')
    lines.extend(['', 'Paired deltas use only displayed complete pairs; missing pairs can bias comparisons.', '',
                  '## All question scores', '', '| ID | Plain | FP Identification | Premise CoT |', '|---|---:|---:|---:|'])
    for qid in plan['question_ids']:
        lines.append('| ' + qid + ' | ' + ' | '.join(str(mapped[m][qid]) if mapped[m][qid] is not None else 'missing' for m in METHODS) + ' |')
    lines.extend(['', '## Negative, flagged, or missing judgments for review', '',
                  'All three baseline methods cover all 30 questions, including previously held/ambiguous cases. No labels are forced.'])
    for key, case in cases.items():
        if results.get(key) == 1 and key not in flags:
            continue
        methods = [m for m in METHODS if plan['mapping'][m][case['question_id']] == key]
        parsed = details.get(key, {})
        event = events.get(key, [{}])[-1]
        lines.extend(['', f'### {case["question_id"]} — {", ".join(methods)}', '',
                      f'Score: {results.get(key, "missing")}; evidence: {flags.get(key, "format OK; check meaning")}', '',
                      '**Question**', '', case['question'], '', '**Reference**', '', case['reference'], '',
                      '**Saved answer**', '', case['answer'], '', '**Judge reason**', '',
                      str(parsed.get('Reason') or event.get('error') or event.get('raw') or 'Not completed'), '',
                      f'Answer evidence: {parsed.get("AnswerEvidence", "")}',
                      f'Attributed premise: {parsed.get("InventedPremise", "")}'])
    text = '\n'.join(lines) + '\n'
    (out / 'report.md').write_text(text, encoding='utf-8')
    print(text)
    return mapped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--pilot-dir', required=True)
    prep.add_argument('--source-dir', required=True)
    prep.add_argument('--out-dir', required=True)
    for name in ('score', 'report'):
        sub.add_parser(name).add_argument('--out-dir', required=True)
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare(args.pilot_dir, args.source_dir, args.out_dir)
    elif args.stage == 'score':
        score(args.out_dir)
    else:
        report(args.out_dir)


if __name__ == '__main__':
    main()
