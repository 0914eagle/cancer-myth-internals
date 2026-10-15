import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('diagnostic_builder',ROOT/'scripts/prepare_well_diagnostic_pilot.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class DiagnosticIsolationTests(unittest.TestCase):
    def setUp(self):
        self.claims=['reported plan','embedded inference']
        self.verdicts=[{'verdict':'supported','claim_relation':'explicit','personal_report':True,
            'reason':'accepted as context','source_ids':[]},
            {'verdict':'false','claim_relation':'implied','personal_report':False,'reason':'reference conflict','source_ids':[]}]
        self.messages=[{'role':'system','content':'frozen system'}, {'role':'user',
            'content':'Question: frozen question\nFeedback: '+json.dumps([{'claim':c,**v} for c,v in zip(self.claims,self.verdicts)])+'\nAction: frozen action\n'}]

    def test_correction_is_only_added_to_reviewed_target(self):
        a=mod.make_messages(self.messages,self.claims,self.verdicts)
        b=mod.make_messages(self.messages,self.claims,self.verdicts,'CORRECTION-CANARY',[1])
        self.assertNotIn('CORRECTION-CANARY',json.dumps(a))
        feedback=json.loads(b[1]['content'].split('\nFeedback: ')[1].split('\nAction: ')[0])
        self.assertNotIn('reference_correction',feedback[0])
        self.assertEqual(feedback[1]['reference_correction'],'CORRECTION-CANARY')
        self.assertEqual(a[0],self.messages[0]);self.assertEqual(b[0],self.messages[0])
        self.assertEqual(b[1]['content'].split('\nAction: ')[1],'frozen action\n')
        self.assertNotIn('CORRECTION-CANARY',json.dumps(self.messages))

    def test_different_claims_rejected(self):
        with self.assertRaises(ValueError):mod.make_messages(self.messages,['other','claim'],self.verdicts)

    def test_frozen_manifest_has_no_duplicate_jobs_and_full_repeats(self):
        out=ROOT/'results/fpqa_prompting/well_reliability_20261005_v1'
        if not (out/'run.json').exists():self.skipTest('local result fixture absent')
        p=json.loads((out/'run.json').read_text());jobs=p['jobs']
        self.assertEqual(len({j['key'] for j in jobs}),1235)
        self.assertEqual(sum(j['phase']=='repeat' for j in jobs),995)
        self.assertEqual(sum(j['phase']=='rejudge' for j in jobs),240)
        self.assertEqual(len(p['flipped_ids']),36)
        self.assertFalse(set(p['flipped_ids'])&set(p['control_ids']))
        for row in p['rows']:
            fresh=[j for j in jobs if j['phase']=='repeat' and j['id']==row['id']]
            self.assertEqual(sorted((j['condition'],j['repeat']) for j in fresh),[('B',3),('DE',2),('DE',3),('Plain',2),('Plain',3)])

    def test_manual_quotes_and_coverage(self):
        out=ROOT/'results/fpqa_prompting/well_reliability_20261005_v1'
        if not (out/'audit_inputs.json').exists():self.skipTest('local result fixture absent')
        audit={r['id']:r for r in json.loads((out/'audit_inputs.json').read_text())}
        review=json.loads((ROOT/'docs/reviews/well_reliability_2026-10-05/manual_flip_review.json').read_text())
        self.assertEqual({r['id'] for r in review['cases']},{i for i,r in audit.items() if r['selection']=='flipped'})
        for r in review['cases']:
            for s in ['B1','B2']:self.assertIn(r['evidence_'+s],audit[r['id']]['answers'][s]['answer'])
        diagnostic=json.loads((out/'diagnostic_jobs.json').read_text())
        self.assertEqual(set(diagnostic['sources'])|set(diagnostic['excluded']),set(audit))
        self.assertFalse(set(diagnostic['sources'])&set(diagnostic['excluded']))

if __name__=='__main__':unittest.main()
