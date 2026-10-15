"""Matched Qwen2.5-style Yes/No gates; separate from the frozen generation suite.

No system message, native thinking disabled, greedy review with 1024-token cap.
Final answers/Well scores are reused only for retrospective routing.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_frontier_claude import write, sha
from src.baseline_generation import DIRECT, REVIEW_DETECT, binary_probability
from src.pilot import COT_REVIEW, output_lock


def decision(yes, no):
    return {'positive_logit': yes, 'negative_logit': no,
            'score': binary_probability(yes, no),
            'logit_difference': yes - no, 'has_false_premise': yes > no}


def verbalizers(tokenizer):
    ids = [tokenizer(s, add_special_tokens=False)['input_ids'] for s in ('Yes', 'No')]
    if any(len(v) != 1 for v in ids) or ids[0] == ids[1]:
        raise ValueError(f'Expected distinct single-token Yes/No: {ids}')
    return [v[0] for v in ids]


def render(tokenizer, prompt):
    # Match src.pilot_model.render: user-only, model-native chat template.
    return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def summarize(out, source, rows):
    from sklearn.metrics import roc_auc_score
    from scripts.summarize_gemma4_diagnostics import score_summary
    from src import well_eval as we
    scores = {}
    for method in ('plain', 'fp_unconditional'):
        plan, h = we.load_plan(source / 'well' / method)
        _, ledger = we.read_ledger(source / 'well' / method, plan, h)
        scores[method] = {qid: ledger.get(job) for qid, job in plan['mapping'][method].items()}
    result = {'gates': {}, 'routed_answers': {},
              'note': 'Native thinking OFF. Missing outputs remain missing. Saved answer scores reused; no new judge calls.'}
    for stage in ('direct', 'review'):
        records = {p.stem: json.loads(p.read_text()) for p in (out / stage).glob('*.json')}
        valid = [r for r in rows if records.get(r['id'], {}).get('status') == 'complete']
        metric = {}
        metric['auroc'] = float(roc_auc_score([r['set'] == 'fpq' for r in valid],
            [records[r['id']]['logit_difference'] for r in valid])) if len({r['set'] for r in valid}) == 2 else None
        routed = {}
        for kind in ('fpq', 'nfp'):
            subset = [r for r in valid if r['set'] == kind]
            positives = sum(records[r['id']]['has_false_premise'] for r in subset)
            metric[kind] = {'valid': len(subset), 'positive': positives,
                            'positive_rate': positives / len(subset) if subset else None}
            values = [scores['fp_unconditional' if records[r['id']]['has_false_premise'] else 'plain'].get(r['id'])
                      for r in subset]
            expected = sum(r['set'] == kind for r in rows)
            routed[kind] = score_summary(values, expected)
        result['gates'][stage] = metric
        result['routed_answers'][stage] = routed
    write(out / 'summary.json', result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['gemma4', 'qwen38'], required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    args = ap.parse_args()
    out, source = args.out.resolve(), args.source.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def state(value, **extra):
        info = dict(state=value, pid=os.getpid(), updated_unix=time.time(), **extra)
        write(out / 'status.json', info)
        print(json.dumps(info), flush=True)
    try:
        with output_lock(out / 'run'):
            from scripts.smoke_qwen38_transformers import gpu_inventory
            gpus = gpu_inventory()
            assert all(gpus[g]['used_mib'] < 1000 and gpus[g]['free_mib'] > 22000 for g in (0, 1)), 'GPUs not free'
            assert not (out / 'STOP').exists(), 'STOP requested'
            rows = json.loads((source / 'evaluation_questions.json').read_text())
            assert len(rows) == len({r['id'] for r in rows}) == 732
            source_plan = json.loads((source / 'plan.json').read_text())
            plan = dict(model=source_plan['model'], revision=source_plan['revision'], thinking_enabled=False,
                messages='user only; no added system message', direct=DIRECT, review=COT_REVIEW,
                review_detect=REVIEW_DETECT, review_do_sample=False, review_max_new_tokens=1024,
                decision='Yes logit > No logit; ties are NFP', auroc_score='Yes logit minus No logit',
                source=str(source), source_questions_sha256=sha(source / 'evaluation_questions.json'),
                source_plan_sha256=sha(source / 'plan.json'), script_sha256=sha(Path(__file__)),
                limitations=['Model-native chat templates and weights differ.',
                    'Review generation matched to old Qwen greedy 1024-token cap; truncated review is missing.',
                    'Existing answer generation settings differ across models; routing is retrospective.'])
            write(out / 'plan.json', plan, frozen=True)
            plan_hash = sha(out / 'plan.json')
            state('loading')
            if args.model == 'qwen38':
                from scripts.qwen38_transformers_runtime import Runtime
                runtime = Runtime(out, args.batch_size)
            else:
                from scripts.run_gemma4_diagnostics import Runtime
                runtime = Runtime(out)
            tok, model, torch = runtime.tokenizer, runtime.model, runtime.torch
            token_ids = verbalizers(tok)
            write(out / 'verbalizers.json', dict(Yes=token_ids[0], No=token_ids[1]), frozen=True)
            def inputs(prompts):
                texts = [render(tok, p) for p in prompts]
                batch = tok(texts, padding=True, add_special_tokens=False, return_tensors='pt').to(runtime.device)
                assert batch['input_ids'].shape[1] + 1024 <= 32768, 'No silent truncation allowed'
                return batch
            def score(prompts):
                batch = inputs(prompts)
                with torch.inference_mode():
                    logits = model(**batch, use_cache=False, logits_to_keep=1, return_dict=True).logits
                    pairs = logits[:, -1, token_ids].float().cpu().tolist()
                return [decision(*pair) for pair in pairs]
            for stage in ('direct', 'review'):
                pending = []
                for row in rows:
                    path = out / stage / (row['id'] + '.json')
                    if path.exists():
                        saved = json.loads(path.read_text())
                        assert saved['plan_hash'] == plan_hash
                    else:
                        pending.append(row)
                for start in range(0, len(pending), args.batch_size):
                    if (out / 'STOP').exists():
                        state('stopped'); return
                    batch_rows = pending[start:start + args.batch_size]
                    reviews = {}
                    if stage == 'review':
                        batch = inputs([COT_REVIEW.format(question=r['question']) for r in batch_rows])
                        with torch.inference_mode():
                            generated = model.generate(**batch, do_sample=False, max_new_tokens=1024,
                                use_cache=True, logits_to_keep=1, pad_token_id=tok.pad_token_id)
                        for r, sequence in zip(batch_rows, generated[:, batch['input_ids'].shape[1]:].tolist()):
                            stop = next((j for j, v in enumerate(sequence) if v in runtime.eos), None)
                            used = sequence if stop is None else sequence[:stop + 1]
                            reviews[r['id']] = dict(response=tok.decode(used, skip_special_tokens=True).strip(),
                                truncated=stop is None, generated_token_ids=used)
                        del generated, batch
                    good = [r for r in batch_rows if stage == 'direct' or
                            (not reviews[r['id']]['truncated'] and reviews[r['id']]['response'])]
                    prompts = [(DIRECT if stage == 'direct' else REVIEW_DETECT).format(
                        question=r['question'], review=reviews.get(r['id'], {}).get('response', '')) for r in good]
                    outputs = dict(zip([r['id'] for r in good], score(prompts))) if good else {}
                    for r in batch_rows:
                        value = dict(id=r['id'], plan_hash=plan_hash, status='complete' if r['id'] in outputs else 'invalid',
                            **outputs.get(r['id'], {}))
                        if stage == 'review': value['review'] = reviews[r['id']]
                        write(out / stage / (r['id'] + '.json'), value, frozen=True)
                    state('running', stage=stage, recorded=len(list((out / stage).glob('*.json'))), expected=732)
            summarize(out, source, rows)
            state('complete')
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
