import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts import run_luna_direct_cot_gate as m


class DirectCoTTests(unittest.TestCase):
    def test_decision_schema_and_consistency(self):
        self.assertTrue(m.decision('{"has_false_premise":true,"false_probability":0.5}')['has_false_premise'])
        for value in ({'has_false_premise': False, 'false_probability': .9},
                      {'has_false_premise': 'false', 'false_probability': .1},
                      {'has_false_premise': True, 'false_probability': True},
                      {'has_false_premise': False, 'false_probability': float('nan')}):
            with self.assertRaises(ValueError):
                m.decision(json.dumps(value))

    def test_direct_never_receives_review(self):
        self.assertNotIn('REVIEW_MARKER', m.prompt_for('direct', 'Question?', 'REVIEW_MARKER'))
        self.assertIn('REVIEW_MARKER', m.prompt_for('cot', 'Question?', 'REVIEW_MARKER'))
        self.assertEqual(m.REVIEW, m.COT_REVIEW)

    def test_staged_schedule_and_resume(self):
        m.tr.STOP.clear()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / 'plan.json').write_text('{}')
            rows = [{'id': f'q{i}', 'set': 'fpq' if i % 2 == 0 else 'nfp',
                     'question': f'Question {i}?', 'group_id': f'g{i}'} for i in range(8)]
            seen, reviews = [], set()
            lock = threading.Lock()
            active, peak = 0, 0

            def invoke(out, plan, stage, row, review):
                nonlocal active, peak
                with lock:
                    if stage == 'cot':
                        self.assertIn(row['id'], reviews)
                        self.assertEqual(review, 'Saved review')
                    if stage == 'direct':
                        self.assertEqual(review, '')
                    active += 1
                    peak = max(peak, active)
                    seen.append((stage, row['id']))
                time.sleep(.003)
                if stage == 'review':
                    result = {'review': 'Saved review'}
                    reviews.add(row['id'])
                else:
                    result = {'has_false_premise': row['set'] == 'fpq',
                              'false_probability': .8 if row['set'] == 'fpq' else .1}
                prompt = m.prompt_for(stage, row['question'], review)
                rec = {'id': row['id'], 'stage': stage, 'status': 'complete', 'prompt': prompt,
                       'signature': m.tr.digest([m.tr.sha(out / 'plan.json'), stage, row['id'], prompt]), 'result': result}
                m.tr.write(out / stage / row['id'] / 'record.json', rec)
                with lock:
                    active -= 1
                return rec

            with patch.object(m, 'invoke', invoke), patch('builtins.print'):
                m.run(out, {}, rows, 4, 8)
                m.run(out, {}, rows, 4, 8)
            self.assertEqual(len(seen), 24)
            self.assertEqual(len(set(seen)), 24)
            self.assertLessEqual(peak, 4)
            status = json.loads((out / 'status.json').read_text())
            self.assertEqual(status['metrics']['direct']['auroc_reported_confidence'], 1)
            self.assertEqual(status['metrics']['cot']['auroc_reported_confidence'], 1)


if __name__ == '__main__':
    unittest.main()
