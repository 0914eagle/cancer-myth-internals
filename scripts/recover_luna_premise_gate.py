"""Revalidate saved native responses into a NEW run; never rewrite source data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_luna_premise_gate as m


def recover(source, target):
    assert source.resolve() != target.resolve()
    plan = json.loads((target / 'plan.json').read_text())
    rows = json.loads((target / 'evaluation_questions.json').read_text())
    refs = json.loads((target / 'annotations.json').read_text())
    for name, expected in plan['sha256'].items():
        path = Path(name)
        assert m.tr.sha(path if path.is_absolute() else ROOT / path) == expected, name
    old_plan = json.loads((source / 'plan.json').read_text())
    for key in ('model', 'reasoning_effort', 'auditor', 'system', 'extract_prompt', 'verify_prompt', 'audit_prompt', 'gate_rule'):
        assert old_plan[key] == plan[key], key
    assert json.loads((source / 'evaluation_questions.json').read_text()) == rows
    byid = {r['id']: r for r in rows}
    recovered, skipped = [], []
    for stage in m.STAGES:
        for path in sorted((source / stage).glob('*/record.json')):
            old = json.loads(path.read_text())
            qid = old['id']
            dst = target / stage / qid
            if dst.exists():
                raise ValueError(f'Target already contains a record: {dst}')
            try:
                pp = []
                if stage in ('verify_extracted', 'audit'):
                    pp = json.loads((target / 'extract' / qid / 'record.json').read_text())['result']['premises']
                elif stage == 'verify_annotated':
                    pp = refs[qid]['premises']
                reference = refs.get(qid, {}).get('premises', [])
                prompt = m.make_prompt(stage, byid[qid]['question'], pp, reference)
                assert prompt == old['prompt']
                reconnects = []
                if old.get('no_call_empty_premises'):
                    assert not pp and stage.startswith('verify_')
                    value = {'judgments': []}
                elif stage == 'audit':
                    native, _ = m.tr.parse_result((path.parent / 'native.json').read_text(), m.AUDITOR)
                    value = m.decode(native['result'])
                else:
                    final = (path.parent / 'answer.txt').read_text()
                    _, reconnects = m.parse_luna_events((path.parent / 'events.jsonl').read_text(), final)
                    value = m.decode(final)
                m.validate(stage, value, pp, reference)
                shutil.copytree(path.parent, dst)
                shutil.copy2(path, dst / 'prior_record.json')
                updated = {**old, 'status': 'complete', 'result': value,
                           'signature': m.tr.digest([m.tr.sha(target / 'plan.json'), stage, qid, prompt]),
                           'recovery_source': str(path.resolve()), 'recovery_source_sha256': m.tr.sha(path),
                           'transport_reconnect_events': reconnects}
                updated.pop('error', None)
                m.tr.write(dst / 'record.json', updated)
                recovered.append({'stage': stage, 'id': qid, 'previous_status': old['status']})
            except (ValueError, AssertionError, OSError, KeyError) as exc:
                skipped.append({'stage': stage, 'id': qid, 'previous_status': old['status'], 'error': str(exc)[:500]})
    report = {'source': str(source.resolve()), 'target': str(target.resolve()), 'recovered': recovered,
              'not_recovered': skipped, 'policy': 'Only complete, schema-valid native answers reused. Other jobs may be explicitly rerun in new target; old attempts retained.'}
    m.tr.write(target / 'recovery_manifest.json', report, frozen=True)
    print(json.dumps({'recovered': dict(Counter(r['stage'] for r in recovered)), 'not_recovered': len(skipped)}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--target', type=Path, required=True)
    args = ap.parse_args()
    recover(args.source.resolve(), args.target.resolve())
