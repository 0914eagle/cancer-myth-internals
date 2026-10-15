"""Generate native Gemma27 answers and save their own generation-prefill states.

The reader explanations are never fed back to the source model. This small
training-only audit does not replace any baseline/test run.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    locks = []
    for gpu in [0, 1]:
        f = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(f)
    import torch
    from src.paper_baselines import HFTask, write_json, digest
    from scripts.nla_generation_capture import decoder_layer
    out = root / 'ao_analysis' / 'native_answers'
    out.mkdir(parents=True, exist_ok=True)
    sample = json.loads((root / 'analysis/review_sample.json').read_text())['questions']
    selected = {(r['dataset'], r['id']) for r in sample}
    config = json.loads((root / 'runtime_source_with_gemma27.json').read_text())
    registry = json.loads((ROOT / config['registry']).read_text())
    jobs = []
    for path in (root / 'gemma27/cases').glob('*.json'):
        case = json.loads(path.read_text())
        ref = case['reference']
        if (ref['dataset'], ref['id']) in selected:
            # Hidden-only Gemma27 references did not store messages. Rebuild
            # exactly as capture did and verify the sealed case identity.
            messages = [{'role': 'system', 'content': registry['transport_common'] + '\n\n' + registry['prompts'][ref['condition']]},
                        {'role': 'user', 'content': ref['question']}]
            assert path.stem == digest(['corrected_nla_ao_capture_v1', 'gemma27', 41, messages])
            ref['messages'] = messages
            jobs.append(case)
    assert len(jobs) == 16
    jobs.sort(key=lambda r: (r['reference']['dataset'], r['reference']['id'], r['reference']['condition']))
    config['models']['gemma27']['gpus'] = [0, 1]
    write_json(out / 'policy.json', dict(model=config['models']['gemma27'], gpus=[0, 1], batch_size=1,
        max_new_tokens=config['max_new_tokens'], do_sample=False, dtype='bfloat16',
        selection='Existing fixed eight training questions, both Plain/Balanced; no output-based selection',
        intervention='None: native source model only; NLA/AO descriptions never inserted',
        trace='Capture source decoder31/41 states on the prefill of this actual answer generation',
        settings_source=str(root / 'runtime_source_with_gemma27.json')), immutable=True)
    output = out / 'answers.jsonl'
    done = {json.loads(line)['case_id'] for line in output.open()} if output.exists() else set()
    pending = [c for c in jobs if c['reference']['case_id'] not in done]
    if not pending:
        print('All native review answers already complete'); return
    backend = HFTask(config, 'gemma27')
    with output.open('a') as stream:
        for case in pending:
            ref = case['reference']; batch = backend.inputs(ref['messages'])
            n = batch['input_ids'].shape[1]
            nla = {r['position_family']: r for r in case['nla']}
            assert int(nla['assistant_boundary_last_token']['position']) == n - 1
            ao = next(r for r in case['ao'] if r['layer'] == 31)
            observed = {}; handles = []
            for layer in [31, 41]:
                module, _ = decoder_layer(backend.model, layer + 1, 62)
                def hook(_module, _inputs, result, layer=layer):
                    hidden = result[0] if isinstance(result, tuple) else result
                    if hidden.shape[1] != n:
                        return  # Autoregressive single-token decode; prefill only.
                    assert layer not in observed
                    if layer == 41:
                        observed[layer] = {name: hidden[0, int(row['position'])].detach().float().cpu() for name, row in nla.items()}
                    else:
                        observed[layer] = {'ao_window': hidden[0, ao['source_positions']].detach().float().cpu()}
                handles.append(module.register_forward_hook(hook))
            try:
                answer, metadata = backend(ref['messages'])
            finally:
                for handle in handles: handle.remove()
            assert set(observed) == {31, 41}
            comparisons = []
            for layer, values in observed.items():
                for name, value in values.items():
                    original = ao if layer == 31 else nla[name]
                    prior = torch.load(original['activation_path'], map_location='cpu', weights_only=True)
                    assert value.shape == prior.shape
                    x = value.reshape(-1).double(); y = prior.reshape(-1).double()
                    cosine = torch.dot(x, y) / (x.norm() * y.norm())
                    saved = out / f"{ref['case_id']}_L{layer}_{name}.pt"
                    torch.save(value, saved)
                    comparisons.append(dict(layer=layer, position=name, bitwise_equal=torch.equal(value, prior),
                        cosine=float(cosine), max_abs_difference=float((value-prior).abs().max()),
                        new_activation_path=str(saved), original_activation_id=original['id'],
                        original_activation_path=original['activation_path']))
            record = {**ref, 'answer': answer, 'completed': True, 'metadata': metadata,
                'source_model_revision': config['models']['gemma27']['revision'],
                'messages_sha256': hashlib.sha256(json.dumps(ref['messages'], sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                'activation_comparisons': comparisons, 'provenance': 'Native answer generation with its own captured prefill; reader outputs never supplied',
                'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            record.pop('reference_status', None)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n'); stream.flush(); done.add(ref['case_id'])
            write_json(out / 'status.json', dict(state='running', complete=len(done), expected=16))
            print('Native answer',len(done),'/16',ref['id'],ref['condition'],'tokens',metadata['output_tokens'],flush=True)
    write_json(out / 'status.json', dict(state='complete', complete=len(done), expected=16))


if __name__ == '__main__':
    main()
