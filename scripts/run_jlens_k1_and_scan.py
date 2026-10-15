"""(1) K1 knowledge check on the 25 CREPE FPQ gold premises; (2) J-lens scan over every question token.

K1: same system message as the Plain native run. Each premise and a templated negation, each asked with two
answer orders. Read first-token logits for 'True' vs 'False'. knows_false = premise judged False in both orders
AND negation judged True in both orders. believes_true = premise judged True in both orders.
Scan: Plain prefills, layers 15-34, every user-question token plus the last prompt token. A 'hit' is a strong
correction token in the top-12 J-lens readout (same readout as run_jlens_crepe.py).
"""
import json
import os
import re
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parents[1] / 'results/jlens_crepe50_20261014'
BASE = '/data3/heejae/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
LENS = ('/data3/heejae/hf_cache/hub/models--neuronpedia--jacobian-lens/snapshots/'
        '9fa267f841d62aa468dab5b9b38c3dc45a82d863/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt')
NATIVE = Path('/home/eagle0914/cancer-myth/external/paper_baselines_b81ace2/results/qwen3_8b_ao_crepe_20261014/native')
BUNDLE = '/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/protocol/full_bundle.json'
SCAN_LAYERS = list(range(15, 35))
TOPK = 12
STRONG = re.compile(r'false|actual|incorrect|untrue|inaccurate|mistaken|misconception|myth|wrong|实际|不可能|错误|误解|并非|其实', re.I)
K1_ORDERS = {
    'tf': 'Is the following statement accurate? Answer "True" or "False" first, then explain in one sentence.\nStatement: {s}',
    'ft': 'Is the following statement accurate? Answer "False" or "True" first, then explain in one sentence.\nStatement: {s}',
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
    t_true, t_false = tok.encode('True', add_special_tokens=False), tok.encode('False', add_special_tokens=False)
    assert len(t_true) == 1 and len(t_false) == 1

    # ---- K1 ----
    k1 = []
    for r in [x for x in recs if str(x['label']) == '1']:
        prem = bundle[r['row_id']]['presuppositions'][0]
        row = dict(question_id=r['row_id'], question=r['question'], premise=prem, negation=negate(prem))
        for which, stmt in [('premise', prem), ('negation', negate(prem))]:
            for order, tmpl in K1_ORDERS.items():
                msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': tmpl.format(s=stmt)}]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([ids], device='cuda:0')).logits[0, -1].float()
                d = float(logits[t_true[0]] - logits[t_false[0]])
                row[f'{which}_{order}_true_minus_false'] = d
                if which == 'premise' and order == 'tf':
                    with torch.inference_mode():
                        g = model.generate(input_ids=torch.tensor([ids], device='cuda:0'), max_new_tokens=60, do_sample=False,
                                           top_p=None, top_k=None, temperature=None, pad_token_id=tok.pad_token_id)
                    row['premise_tf_text'] = tok.decode(g[0, len(ids):], skip_special_tokens=True).strip()
        p_false = all(row[f'premise_{o}_true_minus_false'] < 0 for o in K1_ORDERS)
        p_true = all(row[f'premise_{o}_true_minus_false'] > 0 for o in K1_ORDERS)
        n_true = all(row[f'negation_{o}_true_minus_false'] > 0 for o in K1_ORDERS)
        row['k1'] = 'knows_false' if (p_false and n_true) else ('believes_true' if p_true else 'uncertain')
        k1.append(row)
    with (OUT / 'k1_fpq25.jsonl').open('w') as f:
        for row in k1:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('K1 done', {k: sum(r['k1'] == k for r in k1) for k in ['knows_false', 'believes_true', 'uncertain']}, flush=True)

    # ---- J-lens scan over every question token ----
    lens = torch.load(LENS, map_location='cpu', weights_only=True)
    J = {l: lens['J'][l].to('cuda:0', torch.float32) for l in SCAN_LAYERS}
    norm, head = model.model.norm, model.lm_head
    scan = []
    for r in recs:
        ids = r['input_ids'] if isinstance(r['input_ids'], list) else json.loads(r['input_ids'])
        starts = [i for i, t in enumerate(ids) if t == 151644]
        u0 = starts[-2] + 3
        u1 = next(i for i in range(u0, len(ids)) if ids[i] == 151645)
        qtext = tok.decode(ids[u0:u1])
        assert r['question'].strip()[:20] in qtext, (r['row_id'], qtext[:60])
        positions = list(range(u0, u1)) + [len(ids) - 1]
        with torch.inference_mode():
            hs = model(input_ids=torch.tensor([ids], device='cuda:0'), output_hidden_states=True, use_cache=False).hidden_states
            found = []
            for l in SCAN_LAYERS:
                h = hs[l + 1][0, positions].float() @ J[l].T
                top = head(norm(h.to(torch.bfloat16))).float().topk(TOPK, dim=-1).indices.cpu()
                for pi, row in enumerate(top):
                    words = [tok.decode([int(t)]) for t in row]
                    strong = [w.strip() for w in words if STRONG.search(w)]
                    if strong:
                        p = positions[pi]
                        found.append(dict(layer=l, pos=p, is_final=(p == len(ids) - 1),
                                          at_token=tok.decode([ids[p]]), words=strong))
        scan.append(dict(question_id=r['row_id'], label='FPQ' if str(r['label']) == '1' else 'NFP', question=r['question'],
                         n_question_tokens=u1 - u0, hits=found))
    with (OUT / 'scan_all_positions.jsonl').open('w') as f:
        for row in scan:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print('scan done', flush=True)


if __name__ == '__main__':
    main()
