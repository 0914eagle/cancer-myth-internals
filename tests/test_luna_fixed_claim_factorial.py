"""Prevent label leakage, candidate drift, and deletion of diagnostic disagreements."""
import copy
import json
import unittest

from scripts import run_luna_fixed_claim_factorial_20261001 as run


class FixedClaimProtocolTests(unittest.TestCase):
    def setUp(self):
        self.job = dict(id='hidden_id', condition='control', question='Someone believes P. Is Q possible?',
                        candidates=[dict(claim_id='c0', question_quotes=['believes P'], claim='P'),
                                    dict(claim_id='c1', question_quotes=['Is Q possible?'], claim='Q')])
        self.value = dict(claim_assessments=[
            dict(claim_id='c0', assessment='undetermined', question_relation='relied_on', reason='Insufficient information.'),
            dict(claim_id='c1', assessment='undetermined', question_relation='queried_only', reason='This is a question.')],
            has_false_premise=False, false_probability=.2)

    def test_four_conditions_receive_identical_whitelisted_payload(self):
        plan = dict(common=run.COMMON, protection='Protect stated circumstances.',
                    additions=dict(control='', content=run.CONTENT, criterion=run.CRITERION,
                                   both=run.CONTENT+'\n\n'+run.CRITERION))
        expected = {k:self.job[k] for k in ('question','candidates')}
        for condition in run.CONDITIONS:
            job = dict(self.job, condition=condition, label='fpq', correction='SECRET_GOLD',
                       previous_reason='SECRET_REASON', previous_gate=True)
            prompt = run.prompt_for(plan, job)
            self.assertEqual(json.loads(prompt.split('Input data:\n',1)[1]), expected)
            for secret in ('hidden_id','SECRET_GOLD','SECRET_REASON'):
                self.assertNotIn(secret, prompt)

    def test_missing_and_reordered_candidate_ids_are_rejected(self):
        for assessments in [self.value['claim_assessments'][:1], self.value['claim_assessments'][::-1]]:
            value = dict(self.value, claim_assessments=assessments)
            with self.assertRaises(AssertionError):
                run.validate(json.dumps(value), self.job)

    def test_diagnostic_disagreement_is_preserved(self):
        value = dict(self.value, has_false_premise=True, false_probability=.8)
        result,warnings = run.validate(json.dumps(value), self.job)
        self.assertEqual(result, value)
        self.assertIn('listed_relied_on_false_gate_disagreement', warnings)

    def test_false_unasserted_candidate_does_not_force_gate(self):
        value = copy.deepcopy(self.value)
        value['claim_assessments'][1]['assessment']='false'
        result,warnings = run.validate(json.dumps(value), self.job)
        self.assertFalse(result['has_false_premise'])
        self.assertEqual(warnings, [])


if __name__ == '__main__':
    unittest.main()
