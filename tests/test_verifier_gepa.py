import json
import unittest

from src.fpqa_verifier_gepa import (verifier_messages, execute_pipeline, feedback,
                                    validate_splits, paired_metrics, demonstrations_for)


class VerifierGEPATests(unittest.TestCase):
    question = 'My doctor scheduled surgery. What should I pack?'
    claim = {'text': 'My doctor scheduled surgery.', 'kind': 'personal_report',
             'quotes': ['My doctor scheduled surgery.'], 'relation': 'explicit',
             'grounding': 'Explicit report.', 'source_candidate_indices': [0]}
    verdict = {'verdict': 'unknown', 'claim_relation': 'explicit', 'personal_report': True,
               'reason': 'An individual report is not independently confirmed.', 'source_ids': []}

    class Template:
        def generate(self):
            return [{'role': 'system', 'content': 'Fixed answer system'}, {'role': 'user', 'content': ''}]
        class ResponseClass:
            @staticmethod
            def model_validate_plain_text(raw):
                class Parsed:
                    def get(self): return raw
                return Parsed()

    def test_only_verifier_system_changes(self):
        a = verifier_messages('Seed', self.question, self.claim)
        b = verifier_messages('Winner', self.question, self.claim)
        self.assertEqual(a[1], b[1])
        self.assertEqual(b[0]['content'], 'Winner')
        self.assertNotIn('label', json.loads(b[1]['content']))
        self.assertNotIn('reference_answer', json.loads(b[1]['content']))

    def test_pipeline_uses_full_feedback_and_frozen_response_policy(self):
        calls = []
        def call(stage, role, messages, parser=None):
            calls.append((stage, messages))
            return parser(json.dumps(self.verdict)) if parser else 'Packing suggestions.'
        result = execute_pipeline('Verifier', self.question, [self.claim], self.Template(), call)
        self.assertEqual(result['answer'], 'Packing suggestions.')
        self.assertIn(self.verdict['reason'], calls[1][1][1]['content'])
        self.assertIn('Do not invent new false-premise findings', calls[1][1][1]['content'])
        self.assertEqual(calls[1][1][0]['content'], 'Fixed answer system')

    def test_no_claims_still_generates_answer(self):
        calls = []
        def call(stage, role, messages, parser=None):
            calls.append(stage); return 'An answer.'
        execute_pipeline('Verifier', self.question, [], self.Template(), call)
        self.assertEqual(calls, ['answer'])

    def test_feedback_is_observations_not_gold_or_blame(self):
        row = {'question': self.question, 'label': 0, 'reference_answer': 'SECRET GOLD'}
        generated = {'claims': [self.claim['text']], 'grounded_claims': [self.claim],
                     'verdicts': [self.verdict], 'answer': 'Answer'}
        fb = feedback(row, generated, 4, 'Reason')
        self.assertEqual(fb['normalized_score'], .8)
        self.assertNotIn('SECRET GOLD', json.dumps(fb))
        self.assertNotIn('responsibility', fb)

    def test_splits_disjoint_and_no_silent_truncation(self):
        train = [{'id': 'a', 'label': 1}, {'id': 'b', 'label': 0}]
        val = [{'id': 'c', 'label': 1}, {'id': 'd', 'label': 0}]
        test = [{'id': 'e', 'label': 0}]
        validate_splits(train, val, test)
        with self.assertRaises(ValueError): validate_splits(train, val, train)
        with self.assertRaises(ValueError): validate_splits(train[:1], val, test)

    def test_paired_changes_do_not_confuse_net_and_gross(self):
        seed = [{'id': str(i), 'label': 0, 'score': s} for i, s in enumerate([5, 2, 2])]
        best = [{'id': str(i), 'label': 0, 'score': s} for i, s in enumerate([2, 5, 5])]
        self.assertEqual(paired_metrics(seed, best)['nfp'], {'n': 3, 'gain': 2, 'loss': 1})

    def test_matching_demo_removed_without_changing_other_examples(self):
        examples = [{'question': '  SAME   question ', 'answer': 'gold'}, {'question': 'Other'}]
        self.assertEqual(demonstrations_for('same question', examples), [examples[1]])
        self.assertEqual(len(examples), 2)


if __name__ == '__main__':
    unittest.main()
