"""K1 in context: give the original question and the gold premise, ask whether the claim is true or false.

Same system message as the Plain native run. Claim = gold premise or its templated negation; two answer orders.
First-token logit True - False. knows_false = premise False in both orders AND negation True in both orders.
Compared against the out-of-context K1 in k1_fpq25.jsonl.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parents[1] / 'results/jlens_crepe50_20261014'
BASE = '/data3/heejae/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
NATIVE = Path('/home/eagle0914/cancer-myth/external/paper_baselines_b81ace2/results/qwen3_8b_ao_crepe_20261014/native')
BUNDLE = '/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/protocol/full_bundle.json'
TEMPLATES = {
    'tf': ('A user asked the following question:\n"{q}"\n\nIn the context of this question, is the following claim true or false?\n'
           'Claim: "{s}"\n\nAnswer "True" or "False" first, then explain in one sentence.'),
    'ft': ('A user asked the following question:\n"{q}"\n\nIn the context of this question, is the following claim true or false?\n'
           'Claim: "{s}"\n\nAnswer "False" or "True" first, then explain in one sentence.'),
}


def negate(s):
    s = s.strip().rstrip('.')
    return 'It is not the case that ' + s[0].lower() + s[1:] + '.'


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] not in ('6', '7'), 'GPU 6 and 7 are reserved by the user'
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16,
                                                 device_map={'': 0}, attn_implementation='sdpa').eval()
    bundle = {r['id']: r for r in json.loads(Path(BUNDLE).read_text())['rows']}
    recs = [json.loads((NATIVE / 'results' / (i + '.json')).read_text()) for i in json.loads((NATIVE / 'cohort_ids.json').read_text())]
    system = (recs[0]['messages'] if isinstance(recs[0]['messages'], list) else eval(recs[0]['messages']))[0]['content']
    t_true, t_false = tok.encode('True', add_special_tokens=False)[0], tok.encode('False', add_special_tokens=False)[0]
    rows = []
    for r in [x for x in recs if str(x['label']) == '1']:
        prem = bundle[r['row_id']]['presuppositions'][0]
        row = dict(question_id=r['row_id'], question=r['question'], premise=prem)
        for which, stmt in [('premise', prem), ('negation', negate(prem))]:
            for order, tmpl in TEMPLATES.items():
                msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': tmpl.format(q=r['question'], s=stmt)}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                x = torch.tensor([ids], device='cuda:0')
                with torch.inference_mode():
                    logits = model(input_ids=x).logits[0, -1].float()
                row[f'{which}_{order}_true_minus_false'] = float(logits[t_true] - logits[t_false])
                if order == 'tf':
                    with torch.inference_mode():
                        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=70, do_sample=False,
                                           top_p=None, top_k=None, temperature=None, pad_token_id=tok.pad_token_id)
                    row[f'{which}_tf_text'] = tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()
        p_false = all(row[f'premise_{o}_true_minus_false'] < 0 for o in TEMPLATES)
        p_true = all(row[f'premise_{o}_true_minus_false'] > 0 for o in TEMPLATES)
        n_true = all(row[f'negation_{o}_true_minus_false'] > 0 for o in TEMPLATES)
        row['k1_in_context'] = 'knows_false' if (p_false and n_true) else ('believes_true' if p_true else 'uncertain')
        rows.append(row)
    with (OUT / 'k1_in_context_fpq25.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print({k: sum(r['k1_in_context'] == k for r in rows) for k in ['knows_false', 'believes_true', 'uncertain']}, flush=True)


if __name__ == '__main__':
    main()
