"""Exploratory Jacobian-lens readout on the same Qwen3-8B CREPE50 prefills (discovery only).

Lens: neuronpedia/jacobian-lens qwen3-8b/jlens/Salesforce-wikitext (Anthropic jlens library format).
Readout (jlens.lens.JacobianLens.apply): residual = output of decoder block l (HF hidden_states[l+1]);
logits = lm_head(final_norm(h @ J_l.T)). Baseline 'logit lens' skips J. 'model' = actual final-layer logits.
Positions (pre-answer only): qend = last question token, final = last prompt token (next token is the answer's first).
Score: log of total probability on a correction lexicon fixed before looking at results (single-token entries only).
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
LENS = ('/data3/heejae/hf_cache/hub/models--neuronpedia--jacobian-lens/snapshots/'
        '9fa267f841d62aa468dab5b9b38c3dc45a82d863/qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt')
NATIVE = {'plain': Path('/home/eagle0914/cancer-myth/external/paper_baselines_b81ace2/results/qwen3_8b_ao_crepe_20261014/native'),
          'gepa27': Path('/home/eagle0914/cancer-myth/results/ao_four_questions_gepa27_text_prompt_crepe_20261014/source_runtime/native')}
LEXICON = ['not', 'Not', 'no', 'No', 'actually', 'Actually', 'myth', 'myths', 'misconception', 'misconceptions',
           'false', 'False', 'incorrect', 'wrong', 'mistaken', 'However', 'however', 'contrary', 'untrue',
           'inaccurate', 'misunderstanding', 'premise', 'assumption', 'debunk', 'clarify', 'Clarify', 'myth',
           "isn't", "doesn't", "don't", 'cannot', 'misleading', 'common']
LAYERS = list(range(0, 35))
TOPK = 12


def lexicon_ids(tok):
    ids, kept = set(), []
    for w in dict.fromkeys(LEXICON):
        for v in (w, ' ' + w):
            t = tok.encode(v, add_special_tokens=False)
            if len(t) == 1:
                ids.add(t[0])
                kept.append(v)
    return sorted(ids), kept


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] not in ('6', '7'), 'GPU 6 and 7 are reserved by the user'
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, torch_dtype=torch.bfloat16,
                                                 device_map={'': 0}, attn_implementation='sdpa').eval()
    lens = torch.load(LENS, map_location='cpu', weights_only=True)
    J = {l: lens['J'][l].to('cuda:0', torch.float32) for l in LAYERS}
    norm, head = model.model.norm, model.lm_head
    lex, kept = lexicon_ids(tok)
    lex_t = torch.tensor(lex, device='cuda:0')

    def readout(h):
        logits = head(norm(h.to(torch.bfloat16))).float()
        lp = torch.log_softmax(logits, dim=-1)
        score = torch.logsumexp(lp[:, lex_t], dim=-1)
        top = lp.topk(TOPK, dim=-1).indices
        return score.cpu().tolist(), [[tok.decode([int(i)]) for i in row] for row in top.cpu()]

    rows = []
    for cond, native in NATIVE.items():
        ids_list = json.loads((native / 'cohort_ids.json').read_text())
        for nid in ids_list:
            r = json.loads((native / 'results' / (nid + '.json')).read_text())
            ids = r['input_ids'] if isinstance(r['input_ids'], list) else json.loads(r['input_ids'])
            qpos = r['source_positions'] if isinstance(r['source_positions'], list) else json.loads(r['source_positions'])
            pos = {'qend': qpos[-1], 'final': len(ids) - 1}
            with torch.inference_mode():
                out = model(input_ids=torch.tensor([ids], device='cuda:0'), output_hidden_states=True, use_cache=False)
                hs = out.hidden_states
                rec = dict(condition=cond, native_id=nid, question_id=r['row_id'],
                           label='FPQ' if str(r['label']) == '1' else 'NFP', question=r['question'],
                           answer_first_words=r['answer'][:80], qend_token=tok.decode([ids[pos['qend']]]),
                           final_token=tok.decode([ids[pos['final']]]))
                p_idx = [pos['qend'], pos['final']]
                model_s, model_top = readout(hs[36][0, p_idx].float())
                rec['model'] = dict(qend=model_s[0], final=model_s[1], top_final=model_top[1])
                for kind in ('jlens', 'logitlens'):
                    rec[kind] = {}
                    for l in LAYERS:
                        h = hs[l + 1][0, p_idx].float()
                        if kind == 'jlens':
                            h = h @ J[l].T
                        s, top = readout(h)
                        rec[kind][str(l)] = dict(qend=s[0], final=s[1], top_qend=top[0], top_final=top[1])
            rows.append(rec)
        print(cond, 'done', flush=True)
    with (OUT / 'jlens_readout.jsonl').open('w') as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    (OUT / 'protocol.json').write_text(json.dumps(dict(
        lens=dict(repo='neuronpedia/jacobian-lens', revision='9fa267f841d62aa468dab5b9b38c3dc45a82d863',
                  file='qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt', fit='WikiText-103, 461 prompts'),
        readout='lm_head(final_norm(h @ J_l.T)); h = output of block l (HF hidden_states[l+1]); jlens library apply()',
        baselines=['logit lens (no J)', 'model final-layer logits'], positions=dict(
            qend='last question token (stored source_positions[-1])', final='last prompt token; next token is the answer start'),
        lexicon_kept=kept, score='log total probability on lexicon tokens', layers=LAYERS,
        conditions={k: str(v) for k, v in NATIVE.items()}, status='discovery only; 50 previously examined train questions'),
        indent=1, ensure_ascii=False))
    print('kept lexicon tokens:', kept, flush=True)


if __name__ == '__main__':
    main()
