import json
from pathlib import Path
import tempfile
import unittest

from scripts import premise_sft as s


class TemplateTokenizer:
    """Token IDs encode roles, allowing loss-mask tests without model downloads."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for m in messages:
            ids += [1 if m['role'] == 'user' else 2]
            ids += [10 + ord(c) for c in m['content']]
            ids += [3]
        return ids + [2] if add_generation_prompt else ids


class PremiseSFTTests(unittest.TestCase):
    def test_bundle_integrity_and_split_isolation(self):
        rows = s.cancer_rows()
        counts = s.audit_splits(rows)
        self.assertEqual(len(rows), 732)
        self.assertEqual(counts['test_fpq'], 117)
        self.assertEqual(counts['test_nfp'], 30)
        self.assertEqual([r['id'] for r in rows if not r['target_usable']], ['fpq_0'])

    def test_answer_controls_use_identical_training_ids(self):
        rows = s.cancer_rows()
        ids = []
        for objective in ['binary', 'premise', 'answer', 'joint']:
            selected = s.select_train(rows, objective, 'answer_matched')
            ids.append({r['id'] for r in selected})
            self.assertTrue(all(r['partition'] == 'fit' for r in selected))
        self.assertTrue(all(x == ids[0] for x in ids))
        self.assertEqual(len(ids[0]), 272)

    def test_normal_targets_do_not_leak_normal_premise_annotations(self):
        row = next(r for r in s.cancer_rows() if not r['label'])
        row['tpq_premises'] = ['This must never be called false.']
        target = json.loads(s.target_text(row, 'premise'))
        self.assertEqual(target, {'has_false_premise': False, 'false_premises': []})

    def test_train_does_not_take_dev_or_test(self):
        selected = s.select_train(s.cancer_rows(), 'premise', 'all')
        self.assertEqual(len(selected), 438)
        self.assertEqual(sum(not r['label'] for r in selected), 89)
        with self.assertRaises(ValueError):
            s.select_train(s.cancer_rows(), 'joint', 'all')

    def test_group_leakage_rejected(self):
        rows = s.cancer_rows()
        rows[0]['group_id'] = rows[-1]['group_id']
        rows[0]['partition'] = 'dev' if rows[-1]['partition'] != 'dev' else 'test'
        with self.assertRaises(ValueError):
            s.audit_splits(rows)

    def test_user_tokens_masked_and_answer_end_supervised(self):
        tok = TemplateTokenizer()
        encoded = s.encode(tok, 'Question', 'Target', 100)
        prompt = tok.apply_chat_template([{'role': 'user', 'content': 'Question'}], add_generation_prompt=True)
        self.assertEqual(encoded['labels'][:len(prompt)], [-100]*len(prompt))
        self.assertEqual(encoded['labels'][len(prompt):], [10+ord(c) for c in 'Target']+[3])
        with self.assertRaises(ValueError):
            s.encode(tok, 'Question', 'Target', 5)

    def test_padding_not_supervised(self):
        batch = s.collate([{'input_ids':[1,2,3], 'labels':[-100,2,3]},
                           {'input_ids':[1,4], 'labels':[-100,4]}], 0)
        self.assertEqual(batch['labels'][1].tolist(), [-100,4]+[-100]*6)
        self.assertEqual(batch['attention_mask'][1].tolist(), [1,1]+[0]*6)

    def test_crepe_conflicts_and_duplicate_split_filter(self):
        def row(i, label, question, premises):
            return {'id':i, 'labels':label, 'question':question, 'presuppositions':premises}
        with tempfile.TemporaryDirectory() as tmp:
            s.write_rows(Path(tmp)/'test.jsonl', [row('a',['normal'],'same question',[])])
            s.write_rows(Path(tmp)/'validation.jsonl', [])
            s.write_rows(Path(tmp)/'train.jsonl', [row('b',['normal'],'same question',[]),
                row('c',['normal','false presupposition'],'ambiguous',['x']),
                row('d',['false_presupposition'],'false question',['x'])])
            rows, excluded = s.crepe_rows(tmp)
            self.assertEqual({r['id'] for r in rows}, {'crepe_a','crepe_d'})
            self.assertEqual(len(excluded), 2)

    def test_invalid_predictions_not_silently_normal(self):
        for text in ['No false premise.', '{"has_false_premise":"false"}',
                     '{"has_false_premise":false,"false_premises":["x"]}',
                     '{"has_false_premise":true,"false_premises":[]}',
                     '{"has_false_premise":false} garbage']:
            with self.assertRaises(ValueError):
                s.parse_prediction(text)
        self.assertFalse(s.parse_prediction('{"has_false_premise":false}\n\nAnswer:\nHello')['has_false_premise'])

    def test_answer_task_keeps_question_and_has_no_correction_instruction(self):
        q = 'Question text unchanged.'
        self.assertEqual(s.evaluation_messages(q, 'raw'), [{'role':'user','content':q}])
        messages = s.evaluation_messages(q, 'answer')
        self.assertEqual(messages[-1], {'role':'user','content':q})
        self.assertNotIn('premise', messages[0]['content'])

    def test_score_keeps_invalid_and_missing_separate(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs = [{'id':'a','label':True,'question':'q','false_premises':['x']},
                    {'id':'b','label':False,'question':'r','false_premises':[]}]
            s.write_rows(root/'refs.jsonl', refs)
            s.write_rows(root/'pred.jsonl', [{'id':'a','response':'invalid'}])
            s.score(SimpleNamespace(questions=root/'refs.jsonl', answers=root/'pred.jsonl', out=root/'score.json'))
            result=json.loads((root/'score.json').read_text())
            self.assertEqual(result['counts'], {'fpq_invalid':1, 'nfp_missing':1})
            self.assertFalse(result['complete_valid'])


if __name__ == '__main__':
    unittest.main()
