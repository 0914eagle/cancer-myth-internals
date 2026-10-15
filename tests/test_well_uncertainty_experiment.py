import json
import unittest
from src.fpqa_uncertainty import parse_verdict, verify_messages, Retriever, final_messages

class UncertaintyTests(unittest.TestCase):
    def test_unknown_is_not_false_and_bad_output_not_mapped(self):
        x={'verdict':'unknown','claim_relation':'explicit','personal_report':True,
           'reason':'Individual event cannot be established.','source_ids':[]}
        self.assertEqual(parse_verdict(json.dumps(x))['verdict'],'unknown')
        for bad in ['false','uncertain',json.dumps({**x,'verdict':'maybe'})]:
            with self.assertRaises((ValueError,TypeError)): parse_verdict(bad)

    def test_citation_must_exist(self):
        x={'verdict':'false','claim_relation':'explicit','personal_report':False,
           'reason':'Contradiction.','source_ids':['invented']}
        with self.assertRaises(ValueError): parse_verdict(json.dumps(x),[{'id':'actual'}])

    def test_recheck_differs_only_in_evidence(self):
        previous={'verdict':'unknown'}
        a=verify_messages('Q','C',previous,[])
        b=verify_messages('Q','C',previous,[{'id':'p','text':'Evidence'}])
        self.assertEqual(a[0],b[0])
        x,y=json.loads(a[1]['content']),json.loads(b[1]['content'])
        x.pop('source_passages');y.pop('source_passages')
        self.assertEqual(x,y)
        self.assertEqual(set(x),{'question','candidate_claim','previous_verdict'})

    def test_retrieval_is_deterministic_and_source_limited(self):
        chunks=[{'id':str(i),'url':str(i//3),'title':'Lymphoma',
                 'text':'lymphoma treatment chemotherapy'} for i in range(9)]
        r=Retriever(chunks)
        result=r.retrieve('Lymphoma treatment?', 'Chemotherapy')
        self.assertEqual(result,r.retrieve('Lymphoma treatment?', 'Chemotherapy'))
        self.assertEqual(len(result),4)
        for url in {x['url'] for x in result}:
            self.assertLessEqual(sum(x['url']==url for x in result),2)

    def test_final_preserves_system_and_unknown(self):
        class T:
            def generate(self):return [{'role':'system','content':'OFFICIAL'}, {'role':'user','content':'old'}]
        m=final_messages('Q',['C'],[{'verdict':'unknown'}],T())
        self.assertEqual(m[0]['content'],'OFFICIAL')
        self.assertIn('unknown',m[1]['content'])
        self.assertNotIn('The question contains false presuppositions',m[1]['content'])

if __name__=='__main__': unittest.main()
