import json
import unittest
from copy import deepcopy

from src.fpqa_scope_factorial import (parse_grounded, ground_messages, grounded_verify_messages,
                                     response_messages, route)
from src.fpqa_uncertainty import verify_messages, final_messages


class ScopeFactorialTests(unittest.TestCase):
    question = 'My doctor scheduled surgery. What should I pack?'
    candidate = 'My doctor scheduled surgery.'

    def grounded(self):
        return {'text': self.candidate, 'kind': 'personal_report', 'quotes': [self.candidate],
                'relation': 'explicit', 'grounding': 'The patient reports this plan.',
                'source_candidate_indices': [0]}

    def test_quote_must_be_from_question_not_generated_claim(self):
        c = self.grounded()
        self.assertEqual(parse_grounded(json.dumps({'claims': [c]}), self.question, [self.candidate]), [c])
        c['quotes'] = ['Every patient needs surgery.']
        with self.assertRaises(ValueError):
            parse_grounded(json.dumps({'claims': [c]}), self.question, [self.candidate])

    def test_no_silent_truth_fields_or_invalid_candidate_indices(self):
        for change in [{'verdict': 'false'}, {'source_candidate_indices': [-1]},
                       {'source_candidate_indices': [True]}, {'source_candidate_indices': [1]}]:
            with self.assertRaises(ValueError):
                parse_grounded(json.dumps({'claims': [{**self.grounded(), **change}]}), self.question, [self.candidate])

    def test_empty_is_valid_not_invented_claim(self):
        self.assertEqual(parse_grounded('{"claims":[]}', self.question, [self.candidate]), [])

    def test_D_verifier_only_adds_grounding_data(self):
        a = verify_messages(self.question, self.candidate)
        b = grounded_verify_messages(self.question, self.grounded())
        self.assertEqual(a[0], b[0])
        payload = json.loads(b[1]['content'])
        self.assertEqual(payload.pop('candidate_grounding'), self.grounded())
        self.assertEqual(payload, json.loads(a[1]['content']))

    def test_E_changes_action_only_not_feedback_or_system(self):
        class Template:
            def generate(self):
                return [{'role': 'system', 'content': 'Original Well system and examples'},
                        {'role': 'user', 'content': 'Original user'}]
        verdicts = [{'verdict': 'unknown', 'claim_relation': 'explicit', 'personal_report': True,
                     'reason': 'The individual plan is unverified.', 'source_ids': []}]
        original = deepcopy(verdicts)
        b = response_messages(self.question, [self.candidate], verdicts, Template())
        e = response_messages(self.question, [self.candidate], verdicts, Template(), True)
        self.assertEqual(b, final_messages(self.question, [self.candidate], verdicts, Template()))
        self.assertEqual(b[0], e[0])
        self.assertEqual(b[1]['content'].split('\nAction:')[0], e[1]['content'].split('\nAction:')[0])
        self.assertNotEqual(b[1]['content'], e[1]['content'])
        self.assertEqual(verdicts, original)

    def test_paired_cells_reuse_exact_objects(self):
        b = {'claims': ['B claim'], 'verdicts': [{'verdict': 'unknown'}]}
        d = {'claims': ['D claim'], 'verdicts': [{'verdict': 'false'}]}
        for method in ['B', 'E']:
            claims, verdicts, enhanced = route(method, b, d)
            self.assertIs(claims, b['claims'])
            self.assertIs(verdicts, b['verdicts'])
            self.assertEqual(enhanced, method == 'E')
        for method in ['D', 'DE']:
            claims, verdicts, enhanced = route(method, b, d)
            self.assertIs(claims, d['claims'])
            self.assertIs(verdicts, d['verdicts'])
            self.assertEqual(enhanced, method == 'DE')

    def test_grounding_has_only_question_and_candidate_no_gold(self):
        messages = ground_messages(self.question, [self.candidate])
        self.assertEqual(set(json.loads(messages[1]['content'])), {'question', 'previous_candidates'})


if __name__ == '__main__':
    unittest.main()
