import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_answer_behavior_full as a


def fixture():
    inp = dict(key='test', question='Can you help me plan the trip?',
               answer='Yes. Here are travel planning steps.',
               reference=dict(kind='normal_question_context', premises=['A trip is being considered.']))
    code = dict(key='test', stance='none', target_ko='없음', stance_quote='', question_quote='',
                target_grounding='none', grounding_note_ko='문의에 답함', response='substantive',
                response_quote='Here are travel planning steps.', flags=[],
                reference_alignment=dict(status='not_applicable',reference_quality='not_applicable',
                    reference_question_link='not_applicable',reference_quote='',answer_quote='',note_ko='해당 없음'))
    return inp, code


class AuditContractTests(unittest.TestCase):
    def test_valid_individual_answer(self):
        inp, code = fixture()
        valid, repairs = a.validate([code], [inp])
        self.assertEqual(valid,[code]); self.assertEqual(repairs,[])

    def test_fabricated_quote_rejected(self):
        inp, code = fixture(); code['response_quote']='An invented sentence.'
        with self.assertRaises(AssertionError): a.validate([code],[inp])

    def test_missing_or_duplicate_answers_rejected(self):
        inp, code = fixture()
        with self.assertRaises(AssertionError): a.validate([], [inp])
        other=copy.deepcopy(inp);other['key']='other'
        with self.assertRaises(AssertionError):a.validate([code,code],[inp,other])

    def test_nfp_not_treated_as_annotated_error(self):
        inp, code = fixture();code['reference_alignment']['status']='corrects_target'
        with self.assertRaises(AssertionError):a.validate([code],[inp])

    def test_fpq_requires_reference_grounding(self):
        inp, code=fixture()
        inp['reference']=dict(kind='target_error',premise='Travel requires a car.',correction='Other travel modes are possible.')
        code['reference_alignment']=dict(status='corrects_target',reference_quality='usable',
            reference_question_link='implied',reference_quote='Invented reference.',
            answer_quote='Yes.',note_ko='test')
        with self.assertRaises(AssertionError):a.validate([code],[inp])

    def test_quota_returns_after_one_call_and_keeps_raw(self):
        inp,_=fixture()
        native=dict(is_error=True,subtype='error',result="You've hit your limit. Resets later.")
        proc=subprocess.CompletedProcess([],0,json.dumps(native),'')
        with tempfile.TemporaryDirectory() as temp, patch.object(a,'OUT',Path(temp)), patch.object(a.subprocess,'run',return_value=proc) as call:
            result=a.job(0,[inp])
            self.assertEqual(result['state'],'quota');self.assertEqual(call.call_count,1)
            self.assertEqual(len(list(Path(temp).glob('batches/0000/attempts/*/stdout.json'))),1)

    def test_success_cached_without_second_external_request(self):
        inp,code=fixture()
        native=dict(is_error=False,subtype='success',stop_reason='end_turn',modelUsage={a.MODEL:{}},
                    usage={},result=json.dumps(dict(items=[code])))
        proc=subprocess.CompletedProcess([],0,json.dumps(native),'')
        with tempfile.TemporaryDirectory() as temp, patch.object(a,'OUT',Path(temp)), patch.object(a.subprocess,'run',return_value=proc) as call:
            self.assertEqual(a.job(0,[inp])['state'],'complete')
            self.assertEqual(a.job(0,[inp])['state'],'cached')
            self.assertEqual(call.call_count,1)

    def test_rate_limit_waits_twenty_minutes_then_resumes_without_repeating_success(self):
        inputs=[dict(key=str(i)) for i in range(6)]
        clock=[10000.0]; calls=[]
        def fake_job(index,batch):
            calls.append(index)
            if len(calls)==1:return dict(index=index,state='quota')
            return dict(index=index,state='complete',items=batch)
        def fake_report(inputs,manifest,results,state,**extra):
            return dict(state='complete' if len(results)==len(inputs) else state,
                        valid=len(results),expected=len(inputs),remaining=len(inputs)-len(results))
        with tempfile.TemporaryDirectory() as temp, patch.object(a,'OUT',Path(temp)), \
             patch.object(a,'cached',return_value={}), patch.object(a,'job',side_effect=fake_job), \
             patch.object(a,'report',side_effect=fake_report), patch.object(a,'event') as events, \
             patch.object(a.time,'time',side_effect=lambda:clock[0]), \
             patch.object(a.time,'sleep',side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)):
            a.run(inputs,[],workers=1)
        self.assertEqual(clock[0],11200.0)
        self.assertEqual(calls,[0,0,1])
        self.assertIn('resumed',[call.args[0] for call in events.call_args_list])
        self.assertEqual(events.call_args.kwargs['state'],'complete')


if __name__=='__main__':unittest.main()
