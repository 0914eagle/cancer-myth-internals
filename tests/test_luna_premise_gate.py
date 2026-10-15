import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts import run_luna_premise_gate as m


class PremiseGateTests(unittest.TestCase):
    def test_accept_reconnect_only_when_turn_completes_without_tools(self):
        start = [{'type': 'thread.started', 'thread_id': 't'}, {'type': 'turn.started'},
                 {'type': 'error', 'message': 'Reconnecting... 2/5 (transport error)'}]
        end = [{'type': 'item.completed', 'item': {'type': 'agent_message', 'text': 'OK'}},
               {'type': 'turn.completed', 'usage': {}}]
        encode = lambda events: '\n'.join(json.dumps(e) for e in events)
        parsed, reconnects = m.parse_luna_events(encode(start + end), 'OK')
        self.assertEqual(len(reconnects), 1)
        fallback = {'type': 'item.completed', 'item': {'type': 'error',
                    'message': 'Falling back from WebSockets to HTTPS transport. network failure'}}
        _, reconnects = m.parse_luna_events(encode(start + [fallback] + end), 'OK')
        self.assertEqual(len(reconnects), 2)
        for extra in ([{'type': 'turn.failed'}], [{'type': 'error', 'message': 'Usage limit reached'}],
                      [{'type': 'item.completed', 'item': {'type': 'command_execution'}}]):
            with self.assertRaises(ValueError):
                m.parse_luna_events(encode(start + extra + end), 'OK')
        with self.assertRaises(ValueError):
            m.parse_luna_events(encode(start + [fallback]), '')
        with self.assertRaises(ValueError):
            m.parse_luna_events(encode(start), '')

    def test_validate_requires_all_premises_and_valid_confidence(self):
        with self.assertRaises(ValueError):
            m.validate('verify_extracted', {'judgments': []}, ['A premise'])
        for invalid in (True, float('nan'), 1.1, -0.1):
            with self.assertRaises(ValueError):
                m.validate('verify_extracted', {'judgments': [
                    {'index': 0, 'false_probability': invalid, 'reason': 'x'}]}, ['A'])
        with self.assertRaises(ValueError):
            m.validate('extract', {'premises': ['A', 'A']})
        m.validate('extract', {'premises': []})
        m.validate('verify_extracted', {'judgments': []}, [])

    def test_reference_only_enters_audit(self):
        for stage in ('extract', 'verify_extracted'):
            prompt = m.make_prompt(stage, 'Question?', ['Extracted'], ['SECRET_REFERENCE'])
            self.assertNotIn('SECRET_REFERENCE', prompt)
        self.assertIn('SECRET_REFERENCE', m.make_prompt('audit', 'Question?', ['Extracted'], ['SECRET_REFERENCE']))

    def test_auc_ties_and_direction(self):
        self.assertEqual(m.auroc([False, True], [0, 1]), 1)
        self.assertEqual(m.auroc([False, True], [1, 0]), 0)
        self.assertEqual(m.auroc([False, True], [.5, .5]), .5)
        self.assertIsNone(m.auroc([True], [.5]))

    def test_dependencies_concurrency_and_resume(self):
        m.tr.STOP.clear()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'plan.json').write_text('{}')
            rows = [{'id': f'q{i}', 'set': 'fpq' if i % 2 == 0 else 'nfp', 'question': f'Q{i}'} for i in range(12)]
            refs = {r['id']: {'premises': ['Reference']} for r in rows[:-1]}
            calls, extracted = [], set()
            lock = threading.Lock()
            counts, peak = {'luna': 0, 'audit': 0}, {'luna': 0, 'audit': 0}

            def invoke(out, plan, stage, row, pp, references):
                kind = 'audit' if stage == 'audit' else 'luna'
                with lock:
                    if stage in ('verify_extracted', 'audit'):
                        self.assertIn(row['id'], extracted)
                        self.assertEqual(pp, ['Extracted'])
                    calls.append((stage, row['id']))
                    counts[kind] += 1
                    peak[kind] = max(peak[kind], counts[kind])
                time.sleep(.005)
                if stage == 'extract':
                    result = {'premises': ['Extracted']}
                    extracted.add(row['id'])
                elif stage == 'audit':
                    result = {'premise_reviews': [{'index': 0, 'grounding': 'supported', 'atomic': True}],
                              'missing_premises': [], 'reference_coverage': []}
                else:
                    result = {'judgments': [{'index': 0, 'false_probability': .8 if row['set'] == 'fpq' else .1, 'reason': 'test'}]}
                prompt = m.make_prompt(stage, row['question'], pp, references)
                record = {'id': row['id'], 'stage': stage, 'status': 'complete', 'result': result,
                          'prompt': prompt, 'signature': m.tr.digest([m.tr.sha(out / 'plan.json'), stage, row['id'], prompt])}
                m.tr.write(out / stage / row['id'] / 'record.json', record)
                with lock:
                    counts[kind] -= 1
                return record

            with patch.object(m, 'invoke', invoke), patch('builtins.print'):
                m.run(out, {}, rows, refs, 4, 2, True, 0)
                m.run(out, {}, rows, refs, 4, 2, True, 0)
            self.assertEqual(len(calls), 47)
            self.assertEqual(len(set(calls)), 47)
            self.assertLessEqual(peak['luna'], 4)
            self.assertLessEqual(peak['audit'], 2)
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['gate_results']['verify_extracted']['auroc_reported_confidence'], 1)

    def test_invalid_attempt_not_scored_as_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{'id': 'a', 'set': 'fpq'}, {'id': 'b', 'set': 'nfp'}]
            records = {('verify_extracted', 'a'): {'status': 'failed'}}
            summary = m.summarize(Path(tmp), rows, records, 'test', 1, 1)
            self.assertEqual(summary['gate_results']['verify_extracted']['n_valid'], 0)


if __name__ == '__main__':
    unittest.main()
