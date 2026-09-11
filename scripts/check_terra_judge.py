"""Ten fixed NFP answers, two independent Terra judgments each; hard 20-call cap.

prepare/report are offline. score requires a completed human review first.
Existing experiment scores are never edited. No retries, preflight or reuse.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pilot_judge import audit
from scripts.run_judge import build_prompt
from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import SHARPNESS_RUBRIC_NFP
from src.llm_backend import make_caller
from src.pilot import digest, file_digest, frozen_json, output_lock

MODEL = 'gpt-5.6-terra'


PARSER_VERSION = 'nfp-json-v2'


def parse_nfp_json(raw):
    """Read one unambiguous JSON object, independent of whitespace/fences.

    No coercion, default score, regex extraction of numbers, or choice among
    multiple JSON objects. Conservative failures remain available for audit.
    """
    if not isinstance(raw, str) or not raw.strip():
        return {}, False
    text = raw.strip()
    if text.startswith(('[', '"')):
        return {}, False

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    start = text.find('{')
    if start < 0:
        return {}, False
    try:
        value, end = json.JSONDecoder(object_pairs_hook=unique_keys).raw_decode(text, start)
    except (ValueError, RecursionError):
        return {}, False
    if '{' in text[end:] or '}' in text[end:]:
        return {}, False
    if (not isinstance(value, dict) or type(value.get('Sharpness')) is not int
        or value['Sharpness'] not in (-1, 1)
        or not isinstance(value.get('Reason'), str) or not value['Reason'].strip()):
        return {}, False
    return value, True


def fixed_text(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        path.write_text(text, encoding='utf-8')


def prepare(pilot, out, config, seed=17):
    pilot, out = Path(pilot), Path(out)
    # Validate full response/score provenance before sampling.
    audit(pilot, 'gpt-5.6-sol', 'steering_L14_a0')
    folder = pilot / 'dev'
    responses = {r['id']: r for r in read_jsonl(folder / 'plain.jsonl')}
    zero = {r['id']: r for r in read_jsonl(folder / 'steering_L14_a0.jsonl')}
    questions = {r['id']: r for r in read_jsonl(pilot / 'split/dev.jsonl')}
    paths = [folder / f'{name}_judge_codex_gpt-5.6-sol.jsonl'
             for name in ('plain', 'steering_L14_a0')]
    scores = [{r['question_id']: r for r in read_jsonl(p)} for p in paths]
    ids = sorted(i for i in responses if responses[i]['set'] == 'nfp')
    if any(responses[i]['response'] != zero[i]['response'] for i in ids):
        raise ValueError('NFP answers differ; cannot treat scores as repeat judgments')
    flipped = [i for i in ids if scores[0][i]['sharpness'] != scores[1][i]['sharpness']]
    stable = [i for i in ids if i not in flipped]
    if len(flipped) != 5 or len(stable) < 5:
        raise ValueError('This fixed check expects 5 disagreements and at least 5 unchanged judgments')
    rng = random.Random(seed)
    picked = flipped + rng.sample(stable, 5)
    rng.shuffle(picked)  # human review hides cohort and previous scores
    cfg = load_config(config)['judge']
    examples_fpq = load_json(cfg['examples_fpq'])
    examples_nfp = load_json(cfg['examples_nfp'])
    cases = []
    for qid in picked:
        _, prompt = build_prompt(responses[qid], questions[qid], examples_fpq, examples_nfp)
        cases.append({'id': qid, 'question': questions[qid]['question'],
                      'reference': questions[qid]['hallucination_text'],
                      'answer': responses[qid]['response'], 'prompt': prompt,
                      'cohort': 'disagreement' if qid in flipped else 'unchanged'})
    order = []
    for repeat in (1, 2):
        shuffled = picked[:]
        rng.shuffle(shuffled)
        order.extend({'case_id': qid, 'repeat': repeat, 'id': f'{qid}::r{repeat}'} for qid in shuffled)
    plan = {'version': 'terra-nfp-check-v1', 'model': MODEL, 'backend': 'codex',
            'seed': seed, 'budget': 20, 'cases': cases, 'order': order,
            'rubric_hash': file_digest(ROOT / 'src/judge_prompts.py'),
            'transport_hash': file_digest(ROOT / 'src/llm_backend.py'),
            'examples_hash': digest([examples_fpq, examples_nfp]),
            'sources': {str(p.resolve()): file_digest(p) for p in paths}}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'plan.json', plan)
        lines = ['# NFP human review — before Terra scoring', '', SHARPNESS_RUBRIC_NFP, '',
                 'Evaluate invented premise objections, not general medical QA accuracy.',
                 'The reference describes a possible hallucinated objection, not a fact to assert.',
                 'Do not penalize an answer only for failing to rebut the patient.',
                 'For -1 quote the offending answer passage; for +1 explain absence.',
                 'Ambiguous cases may remain unresolved; do not invent a gold label.', '']
        for case in cases:
            lines.extend([f"## {case['id']}", '', '**Question**', '', case['question'], '',
                          '**Possible hallucination reference**', '', case['reference'], '',
                          '**Answer**', '', case['answer'], ''])
        fixed_text(out / 'review.md', '\n'.join(lines))
        if not (out / 'human_review.tsv').exists():
            with (out / 'human_review.tsv').open('w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['id', 'score', 'rationale', 'reviewer'])
                writer.writerows([c['id'], '', '', ''] for c in cases)
    print(f'Prepared 10 answers, 20 planned calls. No model called. Review: {out / "review.md"}')


PROTOCOL_V2 = 'nfp-role-clarified-v2'


def render_v2(template, case):
    values = {'question': case['question'], 'possible_hallucination': case['reference'],
              'answer': case['answer']}
    for name in values:
        if template.count('{{' + name + '}}') != 1:
            raise ValueError(f'Expected exactly one template placeholder: {name}')
    # One pass: placeholder-like text inside an answer is never substituted.
    return re.sub(r'\{\{(question|possible_hallucination|answer)\}\}',
                  lambda match: values[match[1]], template)


def revise(source, out):
    """Freeze the same cases, order and approved labels under the new protocol."""
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError('Use a separate v2 directory; the original run is immutable')
    original = load_json(source / 'plan.json')
    human_labels(source, original)
    if (source / 'run.json').exists() and load_json(source / 'run.json') != {
        'plan_hash': digest(original), 'human_hash': file_digest(source / 'human_review.tsv')
    }:
        raise ValueError('Source plan or labels changed after scoring')
    template_path = ROOT / 'prompts/nfp_role_clarified_v2.txt'
    template = template_path.read_text(encoding='utf-8')
    plan = {**original, 'version': 'terra-nfp-check-v2', 'protocol': PROTOCOL_V2,
            'source_plan_hash': digest(original),
            'source_human_hash': file_digest(source / 'human_review.tsv'),
            'source_dir': str(source.resolve()),
            'prompt_template': template, 'prompt_template_hash': digest(template),
            'rubric_hash': file_digest(template_path),
            'examples_hash': digest(template),
            'cases': [{**case, 'prompt': render_v2(template, case)} for case in original['cases']]}
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / 'prepare'):
        frozen_json(out / 'plan.json', plan)
        fixed_text(out / 'human_review.tsv', (source / 'human_review.tsv').read_text())
        # Text-mode copies normalize CRLF. Pin the destination bytes used by score.
        fixed_text(out / 'protocol.txt', template)
    print(f'Prepared {PROTOCOL_V2}: same 10 answers and labels, 20 NEW planned calls. No model called. {out}')


def evidence_issue(parsed, answer):
    quote, premise = parsed.get('AnswerEvidence'), parsed.get('InventedPremise')
    if not isinstance(quote, str) or not isinstance(premise, str):
        return 'Missing/non-string AnswerEvidence or InventedPremise'
    if parsed['Sharpness'] == 1:
        return None if quote == premise == '' else '+1 requires empty evidence and premise fields'
    if not quote.strip() or not premise.strip():
        return '-1 requires an answer quote and an invented premise'
    if quote not in answer:
        return 'Quoted evidence is not an exact substring of the saved answer'
    return None


def validate_protocol(plan):
    protocol = plan.get('protocol', 'original')
    if protocol == 'original':
        return
    if protocol != PROTOCOL_V2:
        raise ValueError('Unknown judge protocol')
    template = plan['prompt_template']
    if digest(template) != plan['prompt_template_hash']:
        raise ValueError('Frozen prompt template changed')
    if any(case['prompt'] != render_v2(template, case) for case in plan['cases']):
        raise ValueError('Rendered prompt differs from frozen template/case')


def human_labels(out, plan):
    with (out / 'human_review.tsv').open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    labels = {r['id']: r for r in rows}
    if len(rows) != 10 or set(labels) != {c['id'] for c in plan['cases']}:
        raise ValueError('Human review must contain each of the 10 IDs exactly once')
    for row in rows:
        if (row.get('score') not in ('-1', '1') or not row.get('rationale', '').strip()
            or not row.get('reviewer', '').strip()):
            raise ValueError('Complete human_review.tsv: score -1/+1, rationale and reviewer; no calls made')
    return labels


def events_by_id(out, plan, run_hash):
    path = out / 'attempts.jsonl'
    events = list(read_jsonl(path)) if path.exists() else []
    grouped = {}
    expected = {job['id'] for job in plan['order']}
    for event in events:
        if event['id'] not in expected or event['run_hash'] != run_hash:
            raise ValueError('Attempt ledger differs from frozen run')
        group = grouped.setdefault(event['id'], [])
        expected_status = 'started' if not group else 'finished'
        if len(group) >= 2 or event['status'] != expected_status:
            raise ValueError('Invalid attempt ledger order')
        group.append(event)
    return grouped


def score(out, timeout=180):
    out = Path(out)
    with output_lock(out / 'score'):
        plan = load_json(out / 'plan.json')
        if (plan['model'] != MODEL or plan['backend'] != 'codex' or plan['budget'] != 20
            or len(plan['order']) != 20 or len({j['id'] for j in plan['order']}) != 20):
            raise ValueError('Invalid fixed-budget plan')
        validate_protocol(plan)
        if plan['transport_hash'] != file_digest(ROOT / 'src/llm_backend.py'):
            raise ValueError('Transport changed after preparation')
        human_labels(out, plan)
        run = {'plan_hash': digest(plan), 'human_hash': file_digest(out / 'human_review.tsv')}
        frozen_json(out / 'run.json', run)
        run_hash = digest(run)
        previous = events_by_id(out, plan, run_hash)
        cases = {c['id']: c for c in plan['cases']}
        print(f'Protocol: {plan.get("protocol", "original")} | model: {MODEL}', flush=True)
        print(f'Remaining calls: {20 - len(previous)}; no retries or preflight calls', flush=True)
        call = make_caller('codex', MODEL, timeout=timeout)
        for job in plan['order']:
            if job['id'] in previous:
                continue  # started-but-interrupted calls also consume the budget
            common = {**job, 'run_hash': run_hash}
            append_jsonl(out / 'attempts.jsonl', {
                **common, 'status': 'started', 'at': dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                raw, used = call(cases[job['case_id']]['prompt'])
                parsed, ok = parse_nfp_json(raw)
                valid = ok and parsed.get('Sharpness') in (-1, 1) and used == MODEL
                evidence = {}
                if ok and plan.get('protocol') == PROTOCOL_V2:
                    evidence = {
                        'answer_evidence': parsed.get('AnswerEvidence'),
                        'invented_premise': parsed.get('InventedPremise'),
                        'evidence_issue': evidence_issue(parsed, cases[job['case_id']]['answer'])}
                append_jsonl(out / 'attempts.jsonl', {
                    **common, **evidence, 'status': 'finished', 'model': used, 'raw': raw,
                    'valid': valid, 'parser_version': PARSER_VERSION,
                    'score': parsed.get('Sharpness') if valid else None,
                    'reason': parsed.get('Reason') if ok else None})
                print(f"{job['id']}: {parsed.get('Sharpness') if valid else 'invalid'}", flush=True)
                if used != MODEL:
                    raise SystemExit('Unexpected model; stopped without retry')
            except Exception as exc:
                append_jsonl(out / 'attempts.jsonl', {
                    **common, 'status': 'finished', 'valid': False, 'error': str(exc)})
                raise SystemExit('Call failed; stopped. This attempt will not be retried.') from exc
        print('Finished fixed call budget. Run report; failed/interrupted calls remain missing.')


def report(out, reparse=False):
    out = Path(out)
    plan, run = load_json(out / 'plan.json'), load_json(out / 'run.json')
    if run != {'plan_hash': digest(plan), 'human_hash': file_digest(out / 'human_review.tsv')}:
        raise ValueError('Plan or human review changed after scoring')
    validate_protocol(plan)
    labels = human_labels(out, plan)
    events = events_by_id(out, plan, digest(run))
    valid = {key: rows[-1] for key, rows in events.items()
             if rows[-1]['status'] == 'finished' and rows[-1].get('valid')}
    original_valid = len(valid)
    recovered, rejected, changed = [], [], []
    if reparse:
        valid = {}
        for key, rows in events.items():
            event = rows[-1]
            parsed, ok = parse_nfp_json(event.get('raw'))
            if event['status'] == 'finished' and event.get('model') == MODEL and ok:
                valid[key] = {**event, 'valid': True, 'score': parsed['Sharpness'],
                              'reason': parsed['Reason']}
                if not event.get('valid'):
                    recovered.append(key)
                elif event.get('score') != parsed['Sharpness']:
                    changed.append(key)
            elif event.get('valid'):
                rejected.append(key)
    evidence_flags = {}
    if plan.get('protocol') == PROTOCOL_V2:
        cases = {c['id']: c for c in plan['cases']}
        for key, value in valid.items():
            parsed, ok = parse_nfp_json(value.get('raw'))
            issue = evidence_issue(parsed, cases[value['case_id']]['answer']) if ok else 'Cannot parse evidence'
            value['answer_evidence'] = parsed.get('AnswerEvidence')
            value['invented_premise'] = parsed.get('InventedPremise')
            if issue:
                evidence_flags[key] = issue
    lines = ['# Terra NFP judge check', '',
             f'Protocol: {plan.get("protocol", "original")}.',
             'Current +1-only cases are development diagnostics; they do not test detection of actual overcorrection.',
             f'Evidence-format issues: {len(evidence_flags)}. Scores are retained, never flipped or silently dropped.',
             'Agreement below uses emitted scores; flagged judgments require manual evidence review.', '' ,
             f'Readout: {PARSER_VERSION if reparse else "stored results"}. No model calls.',
             f'Attempt ledger SHA-256: `{file_digest(out / "attempts.jsonl")}`.',
             f'Stored valid={original_valid}; recovered={len(recovered)}; '
             f'rejected={len(rejected)}; changed scores={len(changed)}.', '' ,
             f'Attempted {len(events)}/20; valid {len(valid)}/20; missing/invalid {20-len(valid)}.',
             'Selected diagnostic sample, not an unbiased NFP accuracy estimate.', '',
             '| ID | Cohort | Human | Repeat 1 | Repeat 2 |', '|---|---|---:|---:|---:|']
    for case in plan['cases']:
        qid = case['id']
        vals = [valid.get(f'{qid}::r{r}', {}).get('score', 'missing') for r in (1, 2)]
        lines.append(f"| {qid} | {case['cohort']} | {labels[qid]['score']} | {vals[0]} | {vals[1]} |")
    for cohort in ('disagreement', 'unchanged'):
        ids = [c['id'] for c in plan['cases'] if c['cohort'] == cohort]
        paired = [i for i in ids if all(f'{i}::r{r}' in valid for r in (1, 2))]
        agree = sum(valid[f'{i}::r1']['score'] == valid[f'{i}::r2']['score'] for i in paired)
        available = [v for v in valid.values() if v['case_id'] in ids]
        correct = sum(v['score'] == int(labels[v['case_id']]['score']) for v in available)
        lines.append(f'\n{cohort}: repeat agreement {agree}/{len(paired)} pairs; '
                     f'human agreement {correct}/{len(available)} judgments.')
    for key, value in valid.items():
        lines.extend(['', f'## {key}', '', str(value.get('reason') or value['raw'])])
        if plan.get('protocol') == PROTOCOL_V2:
            lines.extend(['', f"Answer evidence: {value.get('answer_evidence')}",
                          f"Attributed premise: {value.get('invented_premise')}",
                          f"Evidence check: {evidence_flags.get(key, 'format OK; semantic validity still requires review')}"])
    if reparse:
        lines.extend(['', '## Reparse audit', '',
                      f'Recovered IDs: {recovered}', f'Rejected IDs: {rejected}',
                      f'Changed score IDs: {changed}'])
        for job in plan['order']:
            if job['id'] in valid:
                continue
            event = events.get(job['id'], [{}])[-1]
            lines.extend(['', f"### Unresolved: {job['id']}", '',
                          f"Status: {event.get('status', 'not attempted')}; model: {event.get('model', 'unknown')}",
                          '', 'Raw saved response (or error):', '',
                          str(event.get('raw') or event.get('error') or 'No saved response')])
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--pilot-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--config', default='configs/gemma2_9b.yaml')
    p.add_argument('--seed', type=int, default=17)
    p = sub.add_parser('revise', help='Same fixed sample with detailed v2 prompt in a new directory; no calls')
    p.add_argument('--source-dir', required=True)
    p.add_argument('--out-dir', required=True)
    for name in ('score', 'report'):
        p = sub.add_parser(name)
        p.add_argument('--out-dir', required=True)
        if name == 'report':
            p.add_argument('--reparse', action='store_true', help='Reparse saved raw responses offline; ledger unchanged')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.pilot_dir, args.out_dir, args.config, args.seed)
    elif args.stage == 'revise':
        revise(args.source_dir, args.out_dir)
    elif args.stage == 'score':
        score(args.out_dir)
    else:
        report(args.out_dir, reparse=args.reparse)


if __name__ == '__main__':
    main()
