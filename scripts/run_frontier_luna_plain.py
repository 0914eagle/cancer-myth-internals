"""Server Luna Plain using the frozen correction-run transport, then Sonnet Well.

Original runners/results remain unchanged; this wrapper selects only the literal
question template and records its own implementation and source provenance.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frontier_luna as base
from scripts import resume_frontier_luna as recovery

PLAIN_PROMPTS = {'plain': '{question}'}


def assets(out, corrections):
    out.mkdir(parents=True, exist_ok=True)
    for name in ('transport_options.json', 'model_preflight.json'):
        base.tr.write(out / name, json.loads((corrections / name).read_text()), frozen=True)
    source_system = (corrections / 'system_prompt.txt').read_text()
    assert source_system == base.tr.SYSTEM_PROMPT
    dest = out / 'system_prompt.txt'
    if dest.exists() and dest.read_text() != source_system:
        raise ValueError('Changed system prompt')
    dest.write_text(source_system)
    (out / 'isolated_cwd').mkdir(exist_ok=True)
    files = [Path(__file__), Path(recovery.__file__), corrections / 'manifest.json',
             corrections / 'transport_options.json', corrections / 'system_prompt.txt',
             corrections / 'model_preflight.json']
    base.tr.write(out / 'plain_extension_manifest.json', {
        'method': 'plain', 'user_template': '{question}', 'generator': base.MODEL,
        'reasoning_effort': 'medium', 'judge': base.JUDGE,
        'generation_workers': 5, 'judge_workers': 10,
        'preflight_provenance': 'Copied original correction-run resolved-model banner; generation smoke separately verifies current access.',
        'correction_dir': str(corrections),
        'sha256': {str(p.resolve()): base.tr.sha(p) for p in files}}, frozen=True)


def compare(out, corrections):
    mapped, questions = {}, None
    for judge in (out / 'well_judge_sonnet_v1', corrections / 'well_judge_sonnet_v1'):
        plan, h = base.we.load_plan(judge)
        _, scores = base.we.read_ledger(judge, plan, h)
        if questions is not None and questions != plan['questions']:
            raise ValueError('Different evaluation questions')
        questions = plan['questions']
        for method, jobs in plan['mapping'].items():
            assert len(jobs) == 732 and all(k in scores for k in jobs.values())
            mapped[method] = {qid: scores[k] for qid, k in jobs.items()}
    summary = {}
    lines = ['# Luna Plain and correction prompts', '',
             'Generator: gpt-5.6-luna (medium). Judge: claude-sonnet-5. Same 732 questions and Well rubric.',
             'Server Plain generated separately; single generation/judgment, no repeat-stability claim.', '',
             '| Condition | Set | N | S5 | Score >=4 | Mean 1-5 | S5 gained/lost vs Plain |',
             '|---|---|---:|---:|---:|---:|---:|']
    for method, values in mapped.items():
        summary[method] = {}
        for kind in ('fpq', 'nfp'):
            ids = [qid for qid, row in questions.items() if row['set'] == kind]
            result = base.we._summary([values[qid] for qid in ids], len(ids))
            result['score_ge4'] = sum(values[qid] >= 4 for qid in ids)
            gain = sum(values[qid] == 5 and mapped['plain'][qid] != 5 for qid in ids)
            loss = sum(values[qid] != 5 and mapped['plain'][qid] == 5 for qid in ids)
            result['paired_plain'] = {'n': len(ids), 's5_gained': gain, 's5_lost': loss}
            summary[method][kind] = result
            lines.append(f"| {method} | {kind} | {len(ids)} | {result['s5_over_all']:.1%} | "
                         f"{result['score_ge4']/len(ids):.1%} | {result['mean_1_to_5']:.3f} | {gain}/{loss} |")
    base.tr.write(out / 'comparison.json', summary)
    base.tr.write(out / 'paired_scores.json', mapped)
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')


def main():
    out = Path(sys.argv[sys.argv.index('--out-dir') + 1]).resolve()
    index = sys.argv.index('--correction-dir')
    corrections = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
    assets(out, corrections)
    base.PROMPTS = PLAIN_PROMPTS
    base.parse_events = recovery.parse_events
    base.main()
    status = out / 'status.json'
    if status.exists() and json.loads(status.read_text())['state'] == 'complete':
        compare(out, corrections)


if __name__ == '__main__':
    main()
