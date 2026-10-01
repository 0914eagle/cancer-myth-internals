"""Join frozen crossfit/LLM gate decisions to existing answer scores; no calls."""
import csv
import json
from collections import Counter
from pathlib import Path
from scripts import run_frontier_claude as io
from src import well_eval as we

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/table2_completion_2026-10-01'


def main():
    gate_file = ROOT / 'docs/reviews/default_gate_decisions_2026-09-23/decisions.csv'
    answer_file = ROOT / 'docs/reviews/advisor_presentation_2026-09-23/answer_scores.csv'
    gates = {}
    for r in csv.DictReader(gate_file.open()):
        gates.setdefault(r['method'], {})[r['id']] = int(r['predicted_fpq'])
    scores = {}
    for r in csv.DictReader(answer_file.open()):
        scores.setdefault(r['model'], {}).setdefault(r['method'], {})[r['id']] = int(r['score'])
    sources = {str(p.relative_to(ROOT)): io.sha(p) for p in (gate_file, answer_file)}
    runs = {'gemma': 'results/gemma4/full_20260926_v2',
            'qwen38_off': 'results/qwen38/full_transformers_20260926_v1/thinking_off',
            'qwen38_on': 'results/qwen38/full_transformers_20260926_v1/thinking_on'}
    inventory = json.loads((ROOT / 'results/frontier_cli/sonnet_plain_20260922_v1/evaluation_questions.json').read_text())
    questions = {r['id']: r['question'] for r in inventory}
    assert len(questions) == 732
    for model, folder in runs.items():
        for method in ('plain', 'fp_unconditional'):
            path = ROOT / folder / 'well' / method
            plan, h = we.load_plan(path)
            _, values = we.read_ledger(path, plan, h)
            assert all(questions[qid] == q['question'] for qid, q in plan['questions'].items())
            scores.setdefault(model, {})[method] = {qid: values[job] for qid, job in plan['mapping'][method].items()}
            sources[str(path / 'judge_plan.json')] = io.sha(path / 'judge_plan.json')
            sources[str(path / 'attempts.jsonl')] = io.sha(path / 'attempts.jsonl')
    OUT.mkdir(parents=True, exist_ok=True)
    result, details = {}, []
    for model in ('qwen', 'luna', 'sonnet', 'gemma', 'qwen38_off', 'qwen38_on'):
        for gate in (('text', 'hidden', 'direct', 'review') if model == 'qwen' else ('text',)):
            common = set(gates[gate]) & set(scores[model]['plain']) & set(scores[model]['fp_unconditional'])
            assert len(common) == (731 if model == 'qwen' else 732)
            counts = {label: Counter() for label in ('fpq', 'nfp')}
            for qid in sorted(common):
                flag = gates[gate][qid]
                method = 'fp_unconditional' if flag else 'plain'
                score = scores[model][method][qid]
                label = qid.split('_')[0]
                counts[label].update(n=1, ge4=int(score >= 4), s5=int(score == 5), gate_on=flag)
                details.append(dict(model=model, gate=gate, id=qid, set=label,
                                    gate_on=flag, selected_method=method, score=score))
            result[model + '/' + gate] = counts
    old = json.loads((ROOT / 'docs/reviews/default_gate_decisions_2026-09-23/routing_summary.json').read_text())
    for gate in ('text', 'hidden'):
        for label in ('fpq', 'nfp'):
            for key in ('n', 'ge4', 's5'):
                assert result['qwen/' + gate][label][key] == old['results'][gate][label][key]
    io.write(OUT / 'routing_summary.json', dict(results=result, sources=sources,
        note='Same frozen crossfit text gate across answer models. Hidden gate remains Qwen2.5-specific. No generation, retraining or new judging.'))
    with (OUT / 'routing_scores.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(details[0])); w.writeheader(); w.writerows(details)
    lines = ['# Table 2 missing routing rows', '',
             'All rows reuse saved decisions and scores. Text decisions are crossfit; no full-data training.', '',
             '| Answer model | Gate | FPQ ≥4 | NFP ≥4 | FPQ S5 | NFP S5 |',
             '|---|---|---:|---:|---:|---:|']
    for key, c in result.items():
        values = [f"{c[label][metric]}/{c[label]['n']} ({100*c[label][metric]/c[label]['n']:.1f}%)"
                  for metric in ('ge4', 's5') for label in ('fpq', 'nfp')]
        lines.append('| ' + ' | '.join(key.split('/') + values) + ' |')
    (OUT / 'routing_tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
