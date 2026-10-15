"""Bauer et al. improved AO (ceselder/qwen3-8b-ao-v3-best) on the same CREPE50 Qwen3-8B prefills.

Stages:
  sanity  - compare prefix conventions (depth percent vs layer index) on known-answer prompts
  capture - re-capture layers 18,21-25 from stored native input_ids; verify L18 against stored tensors
  ao      - four generic queries x 50 questions x {plain, gepa27} + targeted gold-premise Yes/No logits (FPQ)
Injection follows the official multi-layer demo (activation_oracles experiments/multilayer_activation_oracle_demo.ipynb):
  prefix = for each layer: "Layer: {p}\\n" + " ?" * K + " \\n"; vectors concatenated layer-major;
  at decoder layer 1 output: h <- h + normalize(v) * ||h|| * coef, prompt pass only.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'results/bauer_ao_four_questions_crepe_20261014'
ACTS = Path('/data3/heejae/cancer-myth/bauer_ao_crepe_20261014/acts')
BASE = '/data3/heejae/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'
BAUER = '/data3/heejae/hf_cache/hub/models--ceselder--qwen3-8b-ao-v3-best/snapshots/a7344ac8f21ed848c9f0593893e94137c4cb38cb'
BUNDLE = '/data3/heejae/cancer-myth/paper_baselines_b81ace2_a6000_gpu01/protocol/full_bundle.json'
PLAIN_NATIVE = Path('/home/eagle0914/cancer-myth/external/paper_baselines_b81ace2/results/qwen3_8b_ao_crepe_20261014/native')
GEPA_NATIVE = Path('/home/eagle0914/cancer-myth/results/ao_four_questions_gepa27_text_prompt_crepe_20261014/source_runtime/native')
PREV_QUERIES = Path('/home/eagle0914/cancer-myth/results/ao_four_questions_crepe_20261014/protocol.json')
ACT_LAYERS = [21, 22, 23, 24, 25]
PREFIX = {'percent': [59, 62, 64, 67, 70], 'index': ACT_LAYERS}
HOOK_LAYER = 1
COEF = 1.0
YESNO = ('Answer with "Yes" or "No" only. Do these activations indicate that the model doubts the following '
         'claim from the user\'s question: "{claim}"?')


def load_records(native):
    ids = json.loads((native / 'cohort_ids.json').read_text())
    recs = []
    for i in ids:
        r = json.loads((native / 'results' / (i + '.json')).read_text())
        recs.append(r)
    assert len(recs) == 50
    return recs


def setup(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16,
                                                 device_map={'': 0}, attn_implementation='sdpa').eval()
    return torch, tok, model


def capture_states(torch, model, ids, positions, layers):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([ids], device='cuda:0'), output_hidden_states=True, use_cache=False)
    return {l: out.hidden_states[l + 1][0, positions].float().cpu() for l in layers}


class Oracle:
    def __init__(self, torch, tok, model):
        from peft import PeftModel
        self.torch, self.tok = torch, tok
        self.model = PeftModel.from_pretrained(model, BAUER, adapter_name='bauer', is_trainable=False).eval()
        from safetensors import safe_open
        with safe_open(str(Path(BAUER) / 'adapter_model.safetensors'), framework='pt') as f:
            keys = list(f.keys())
        loaded = self.model.state_dict()
        mapped = [k.replace('.lora_A.weight', '.lora_A.bauer.weight').replace('.lora_B.weight', '.lora_B.bauer.weight') for k in keys]
        missing = [m for m in mapped if m not in loaded]
        assert keys and not missing, f'LoRA keys not loaded: {len(missing)} of {len(keys)}'
        print('LoRA tensors loaded:', len(keys), flush=True)
        cfg = json.loads((Path(BAUER) / 'ao_config.json').read_text())
        assert cfg['act_layer_combinations'] == [ACT_LAYERS] and cfg['hook_onto_layer'] == HOOK_LAYER
        assert cfg['special_token'] == ' ?' and cfg['steering_coefficient'] == COEF
        self.q_id = tok.encode(' ?', add_special_tokens=False)
        assert len(self.q_id) == 1
        self.q_id = self.q_id[0]
        self.module = self.model.get_base_model().model.layers[HOOK_LAYER]
        self.yes = tok.encode('Yes', add_special_tokens=False)
        self.no = tok.encode('No', add_special_tokens=False)
        assert len(self.yes) == 1 and len(self.no) == 1

    def build(self, query, k, prefix_layers):
        prefix = ''.join(f'Layer: {p}\n' + ' ?' * k + ' \n' for p in prefix_layers)
        ids = self.tok.apply_chat_template([{'role': 'user', 'content': prefix + query}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False)
        pos = [i for i, t in enumerate(ids) if t == self.q_id][:k * len(prefix_layers)]
        assert len(pos) == k * len(prefix_layers)
        for b in range(len(prefix_layers)):
            blk = pos[b * k:(b + 1) * k]
            assert blk[-1] - blk[0] == k - 1
        return ids, pos

    def _hook(self, vecs, poss, coef):
        torch = self.torch
        normed = [torch.nn.functional.normalize(v, dim=-1) for v in vecs]

        def fn(_m, _i, output):
            resid = output[0] if isinstance(output, tuple) else output
            if resid.shape[1] <= 1:
                return output
            for b in range(resid.shape[0]):
                p = torch.tensor(poss[b], device=resid.device)
                orig = resid[b, p, :]
                resid[b, p, :] = (normed[b].to(resid.device) * orig.norm(dim=-1, keepdim=True) * coef).to(resid.dtype) + orig
            return output
        return fn

    def _batch(self, items):
        torch = self.torch
        width = max(len(x['ids']) for x in items)
        pad = self.tok.pad_token_id
        input_ids, attn, poss, vecs = [], [], [], []
        for x in items:
            n = width - len(x['ids'])
            input_ids.append([pad] * n + x['ids'])
            attn.append([0] * n + [1] * len(x['ids']))
            poss.append([p + n for p in x['pos']])
            vecs.append(x['vec'])
        return (torch.tensor(input_ids, device='cuda:0'), torch.tensor(attn, device='cuda:0'), poss, vecs)

    def generate(self, items, max_new_tokens=256, coef=COEF, bs=16):
        torch = self.torch
        outs = []
        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            ids, attn, poss, vecs = self._batch(chunk)
            h = self.module.register_forward_hook(self._hook(vecs, poss, coef))
            try:
                with torch.inference_mode():
                    gen = self.model.generate(input_ids=ids, attention_mask=attn, do_sample=False,
                                              max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                              top_p=None, top_k=None, temperature=None)
            finally:
                h.remove()
            for row in gen[:, ids.shape[1]:]:
                outs.append(dict(text=self.tok.decode(row, skip_special_tokens=True).strip(),
                                 eos=bool((row == self.tok.eos_token_id).any() or (row == self.tok.pad_token_id).any())))
        return outs

    def yesno(self, items, coef=COEF, bs=16):
        torch = self.torch
        outs = []
        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            ids, attn, poss, vecs = self._batch(chunk)
            h = self.module.register_forward_hook(self._hook(vecs, poss, coef))
            try:
                with torch.inference_mode():
                    pos_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)
                    logits = self.model(input_ids=ids, attention_mask=attn, position_ids=pos_ids).logits[:, -1, :].float()
            finally:
                h.remove()
            lp = torch.log_softmax(logits, dim=-1)
            for b in range(len(chunk)):
                outs.append(dict(logit_yes=float(logits[b, self.yes[0]]), logit_no=float(logits[b, self.no[0]]),
                                 diff=float(logits[b, self.yes[0]] - logits[b, self.no[0]]),
                                 p_yes=float(lp[b, self.yes[0]].exp()), p_no=float(lp[b, self.no[0]].exp()),
                                 top=self.tok.decode([int(logits[b].argmax())])))
        return outs


def stage_sanity(gpu):
    torch, tok, model = setup(gpu)
    probes = [
        dict(name='eiffel', user='The Eiffel Tower is located in the city of',
             query='What city is the model thinking about? Answer with one word.', expect='Paris'),
        dict(name='grief', user='My dog died this morning and I keep crying.',
             query='Answer with a single word. What emotion is being felt here?', expect='sadness / grief'),
        dict(name='hemlock', user='The philosopher who drank hemlock taught a student who founded an academy. That student\'s most famous pupil was',
             query='Can you name which person the model is thinking about?', expect='Aristotle (or Plato)'),
    ]
    captured = []
    for p in probes:
        text = tok.apply_chat_template([{'role': 'user', 'content': p['user']}], tokenize=False,
                                       add_generation_prompt=False, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)['input_ids']
        pos = list(range(len(ids)))
        st = capture_states(torch, model, ids, pos, ACT_LAYERS)
        captured.append((p, torch.cat([st[l] for l in ACT_LAYERS], 0), len(pos)))
    oracle = Oracle(torch, tok, model)
    results = []
    for mode, coef in [('percent', COEF), ('index', COEF), ('percent', 0.0)]:
        items = []
        for p, vec, k in captured:
            ids, pos = oracle.build(p['query'], k, PREFIX[mode])
            items.append(dict(ids=ids, pos=pos, vec=vec))
        outs = oracle.generate(items, max_new_tokens=40, coef=coef, bs=4)
        for (p, _, _), o in zip(captured, outs):
            results.append(dict(probe=p['name'], expect=p['expect'], prefix=mode, coef=coef, output=o['text']))
            print(f"[{mode:7s} coef={coef}] {p['name']:8s} expect={p['expect']:<22s} -> {o['text'][:120]!r}", flush=True)
    (OUT / 'sanity_prefix_convention.json').write_text(json.dumps(results, indent=1, ensure_ascii=False))


def stage_capture(gpu):
    torch, tok, model = setup(gpu)
    ACTS.mkdir(parents=True, exist_ok=True)
    verify = []
    for cond, native in [('plain', PLAIN_NATIVE), ('gepa27', GEPA_NATIVE)]:
        for r in load_records(native):
            ids = r['input_ids'] if isinstance(r['input_ids'], list) else json.loads(r['input_ids'])
            pos = r['source_positions'] if isinstance(r['source_positions'], list) else json.loads(r['source_positions'])
            st = capture_states(torch, model, ids, pos, [18] + ACT_LAYERS)
            paths = r['activation_paths'] if isinstance(r['activation_paths'], dict) else eval(r['activation_paths'])
            old = torch.load(paths['18'], map_location='cpu', weights_only=True).float()
            diff = (st[18] - old).abs().max().item()
            cos = torch.nn.functional.cosine_similarity(st[18], old, dim=-1).min().item()
            rel = ((st[18] - old).norm(dim=-1) / old.norm(dim=-1)).max().item()
            verify.append(dict(condition=cond, native_id=r['id'], row_id=r['row_id'], k=len(pos),
                               l18_max_abs_diff=diff, l18_min_cos=cos, l18_max_rel_err=rel))
            torch.save({l: st[l].to(torch.bfloat16) for l in ACT_LAYERS}, ACTS / f'{cond}_{r["id"]}.pt')
    (OUT / 'capture_verification.json').write_text(json.dumps(dict(
        summary=dict(n=len(verify), worst_min_cos=min(v['l18_min_cos'] for v in verify),
                     worst_rel_err=max(v['l18_max_rel_err'] for v in verify),
                     exact=sum(v['l18_max_abs_diff'] == 0 for v in verify)),
        rows=verify), indent=1))
    print('capture done', len(verify), 'worst cos', min(v['l18_min_cos'] for v in verify),
          'worst rel', max(v['l18_max_rel_err'] for v in verify), flush=True)


def stage_ao(gpu, prefix_mode):
    torch, tok, model = setup(gpu)
    oracle = Oracle(torch, tok, model)
    queries = json.loads(PREV_QUERIES.read_text())['queries']
    bundle = {r['id']: r for r in json.loads(Path(BUNDLE).read_text())['rows']}
    rows = []
    for cond, native in [('plain', PLAIN_NATIVE), ('gepa27', GEPA_NATIVE)]:
        recs = load_records(native)
        acts = {r['id']: torch.load(ACTS / f'{cond}_{r["id"]}.pt', weights_only=True) for r in recs}
        vec = {i: torch.cat([a[l].float() for l in ACT_LAYERS], 0) for i, a in acts.items()}
        k = {i: a[ACT_LAYERS[0]].shape[0] for i, a in acts.items()}
        out = {r['id']: dict(condition=cond, native_id=r['id'], question_id=r['row_id'],
                             label='FPQ' if str(r['label']) == '1' else 'NFP', question=r['question'],
                             answer=r['answer']) for r in recs}
        for qname, qtext in queries.items():
            items = []
            for r in recs:
                ids, pos = oracle.build(qtext, k[r['id']], PREFIX[prefix_mode])
                items.append(dict(ids=ids, pos=pos, vec=vec[r['id']]))
            t0 = time.time()
            res = oracle.generate(items)
            for r, o in zip(recs, res):
                out[r['id']][qname] = o['text']
                out[r['id']][qname + '_eos'] = o['eos']
            print(cond, qname, 'done', round(time.time() - t0, 1), 's', flush=True)
        fpq = sorted([r for r in recs if str(r['label']) == '1'], key=lambda r: r['row_id'])
        donor = {fpq[j]['id']: fpq[(j + 1) % len(fpq)]['id'] for j in range(len(fpq))}
        items_real, items_donor = [], []
        for r in fpq:
            claim = bundle[r['row_id']]['presuppositions'][0]
            q = YESNO.format(claim=claim)
            ids, pos = oracle.build(q, k[r['id']], PREFIX[prefix_mode])
            items_real.append(dict(ids=ids, pos=pos, vec=vec[r['id']]))
            d = donor[r['id']]
            ids, pos = oracle.build(q, k[d], PREFIX[prefix_mode])
            items_donor.append(dict(ids=ids, pos=pos, vec=vec[d]))
            out[r['id']]['yesno_claim'] = claim
            out[r['id']]['yesno_donor_native_id'] = d
        for r, a, b in zip(fpq, oracle.yesno(items_real), oracle.yesno(items_donor)):
            out[r['id']].update({'yesno_real_' + kk: v for kk, v in a.items()})
            out[r['id']].update({'yesno_donor_' + kk: v for kk, v in b.items()})
        rows.extend(out.values())
    with (OUT / 'bauer_outputs.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    keys = list(rows[0].keys()) + [k for k in rows[-1].keys() if k not in rows[0]]
    keys = list(dict.fromkeys([k for row in rows for k in row.keys()]))
    with (OUT / 'bauer_outputs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (OUT / 'protocol.json').write_text(json.dumps(dict(
        reader='Bauer et al. improved AO', checkpoint=dict(repo_id='ceselder/qwen3-8b-ao-v3-best',
        revision='a7344ac8f21ed848c9f0593893e94137c4cb38cb'), source='Qwen/Qwen3-8B b968826d', act_layers=ACT_LAYERS,
        prefix_mode=prefix_mode, prefix_layers=PREFIX[prefix_mode], hook_onto_layer=HOOK_LAYER, steering_coefficient=COEF,
        packing='layer-major: all K positions of layer 21, then 22, ...; one " ?" block per layer',
        positions='Same stored native question-end <=20 token positions as the Karvonen runs',
        conditions=dict(plain=str(PLAIN_NATIVE), gepa27=str(GEPA_NATIVE)), queries=queries, yesno_query=YESNO,
        yesno_scope='FPQ only (gold premise required); donor = next FPQ by question id, same query and claim',
        generation=dict(do_sample=False, max_new_tokens=256), external_model_calls=0), indent=1, ensure_ascii=False))
    print('ao done', len(rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['sanity', 'capture', 'ao'], required=True)
    ap.add_argument('--gpu', type=int, default=2)
    ap.add_argument('--prefix-mode', choices=['percent', 'index'], default='percent')
    a = ap.parse_args()
    assert a.gpu not in (6, 7), 'GPU 6 and 7 are reserved by the user'
    {'sanity': lambda: stage_sanity(a.gpu), 'capture': lambda: stage_capture(a.gpu),
     'ao': lambda: stage_ao(a.gpu, a.prefix_mode)}[a.stage]()
