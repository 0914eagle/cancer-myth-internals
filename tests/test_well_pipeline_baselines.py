"""Protocol tests: exact source loading, data isolation, and per-claim routing."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_well_upstream import load_upstream, verify_source
from src.fpqa_well_pipelines import load_templates, extract, answer


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = ROOT/'external/fpqa_20261002/Well'
        verify_source(root)
        load_upstream(root, lambda *_: (_ for _ in ()).throw(AssertionError('Network forbidden')))
        cls.templates = load_templates(root)

    def test_extraction_is_shared_without_annotations(self):
        calls = []
        def call(stage, role, messages):
            calls.append((stage, role, messages))
            return 'First premise.\nSecond premise.'
        claims = extract('QUESTION_SENTINEL', [], self.templates, call)
        self.assertEqual(claims, ['first premise.', 'second premise.'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2][1]['content'], 'QUESTION_SENTINEL')
        self.assertIn('always be at least one', calls[0][2][0]['content'])

    def test_atomic_checks_only_claim_and_routes_false_claim(self):
        calls = []
        def call(stage, role, messages):
            calls.append((stage, messages))
            return {'atomic/check_000': 'true', 'atomic/check_001': 'false',
                    'atomic/answer': 'ANSWER TEXT'}[stage]
        result = answer('QUESTION_SENTINEL', [], ['CLAIM_A', 'CLAIM_B'],
                        'atomic', self.templates, call)
        self.assertEqual(result['factcheck_results'], [1, 0])
        self.assertEqual(result['answer'], 'answer text')  # Official lowercasing.
        self.assertNotIn('QUESTION_SENTINEL', str(calls[0][1]))
        final = calls[-1][1][1]['content']
        self.assertIn('false presuppositions that CLAIM_B', final)
        self.assertNotIn('CLAIM_A', final)

    def test_prewome_preserves_question_and_feedback(self):
        calls = []
        def call(stage, role, messages):
            calls.append((stage, messages))
            return 'Feedback: TARGET\nAction: CORRECT' if stage.endswith('review') else 'ANSWER'
        answer('QUESTION', [], ['CLAIM'], 'prewome', self.templates, call)
        self.assertIn('There is a clear and single answer', calls[0][1][1]['content'])
        self.assertEqual(calls[1][1][1]['content'], 'Question: QUESTION\nFeedback: TARGET\nAction: CORRECT\n')

    def test_malformed_verdict_is_reported_not_hidden(self):
        result = answer('Q', [], ['P'], 'atomic', self.templates,
                        lambda stage, *_: 'uncertain' if 'check_' in stage else 'Answer')
        self.assertEqual(result['factcheck_results'], [0])  # Upstream fallback.
        self.assertEqual(result['unparseable_check_indices'], [0])


if __name__ == '__main__':
    unittest.main()
