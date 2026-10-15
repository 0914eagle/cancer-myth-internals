"""Native answer generation and prefill capture under source-prompt interventions."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = config['models'][args.model]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, spec['gpus']))
    import torch
    import yaml
    from src.paper_baselines import HFTask, digest, write_json
    from scripts.nla_generation_capture import decoder_layer
    from scripts.nla_connection import snapshot
    baseline = json.loads(Path(config['baseline_config']).read_text())
    registry = json.loads((ROOT / baseline['registry']).read_text())
    out = Path(config['artifact_root']) / args.model
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    (out / 'activations').mkdir(exist_ok=True)
    jobs = []
    for row in config['cohort']:
        for condition, addition in config['source_prompt_additions'].items():
            messages = [
                {'role': 'system', 'content': registry['transport_common'] + '\n\n' + registry['prompts']['plain'] + '\n\n' + addition},
                {'role': 'user', 'content': row['question']},
            ]
            case_id = digest(['source_false_premise_v1', args.model, spec['layer'], messages])
            jobs.append((row, condition, addition, messages, case_id))
    pending = [j for j in jobs if not (out / 'cases' / (j[-1] + '.json')).exists()]
    if pending:
        backend = HFTask(baseline, args.model)
        layers = sorted(set([spec['layer']] + spec.get('ao_layers', [])))
        modules = {k: decoder_layer(backend.model, k + 1, baseline['models'][args.model]['n_layers'])[0] for k in layers}
        for index, (row, condition, addition, messages, case_id) in enumerate(pending):
            text = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = backend.inputs(messages)
            ids = batch['input_ids'][0].tolist()
            mapping = backend.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            assert mapping['input_ids'] == ids
            start = text.rfind(row['question'])
            assert start >= 0
            end = start + len(row['question'])
            def final_token(a, b):
                return [i for i, (x, y) in enumerate(mapping['offset_mapping']) if x < b and y > a and x != y][-1]
            keyword = row['keywords'][0]
            a, b = keyword['char_span']
            assert row['question'][a:b] == keyword['text']
            positions = {
                'question_last_token': final_token(start, end),
                'prompt_keyword_1': final_token(start + a, start + b),
                'assistant_boundary_last_token': len(ids) - 1,
            }
            qstart = next(i for i, (x, y) in enumerate(mapping['offset_mapping']) if x < end and y > start and x != y)
            qend = positions['question_last_token']
            window_start = max(qstart, qend - 19)
            observed = {}
            handles = []
            for layer, module in modules.items():
                def hook(_module, _inputs, result, layer=layer):
                    hidden = result[0] if isinstance(result, tuple) else result
                    if hidden.shape[1] != len(ids):
                        return
                    assert layer not in observed
                    observed[layer] = {name: hidden[0, pos].detach().float().cpu() for name, pos in positions.items()}
                    if layer in spec.get('ao_layers', []):
                        observed[layer]['ao_window'] = hidden[0, window_start:qend + 1].detach().float().cpu()
                handles.append(module.register_forward_hook(hook))
            try:
                answer, metadata = backend(messages)
            finally:
                for handle in handles:
                    handle.remove()
            assert set(observed) == set(layers) and metadata['finish_reason'] == 'eos'
            verification = 'Actual native generation prefill; no reader text supplied'
            if index == 0:
                with torch.inference_mode():
                    result = backend.model(**batch, use_cache=False, output_hidden_states=True)
                for layer in layers:
                    for name, pos in positions.items():
                        direct = result.hidden_states[layer + 1][0, pos].detach().float().cpu()
                        assert torch.equal(observed[layer][name].view(torch.int32), direct.view(torch.int32))
                    if layer in spec.get('ao_layers', []):
                        direct = result.hidden_states[layer + 1][0, window_start:qend + 1].detach().float().cpu()
                        assert torch.equal(observed[layer]['ao_window'].view(torch.int32), direct.view(torch.int32))
                del result
                verification += '; first case bytewise matches explicit HF hidden_states[K+1]'
            common = dict(base_id=row['id'], case_id=case_id, source=row['dataset'],
                category='FPQ' if row['label'] else 'NFP', condition=condition, prompt=row['question'],
                partition='train', source_model_revision=baseline['models'][args.model]['revision'],
                source_prompt_addition=addition, activation_mode='actual_native_answer_generation_prefill',
                answer_completed=True, capture_verification=verification)
            nla_records = []
            for name, pos in positions.items():
                vector = observed[spec['layer']][name]
                assert vector.shape == (spec['d_model'],) and torch.isfinite(vector).all()
                ident = digest([case_id, spec['layer'], name])
                path = out / 'activations' / (ident + '.pt')
                torch.save(vector, path)
                nla_records.append(dict(common, id=ident, activation_path=str(path), layer=spec['layer'],
                    hidden_state_index=spec['layer'] + 1, position_family=name, position=str(pos),
                    selected_token_id=ids[pos], selected_token_text=backend.tokenizer.decode([ids[pos]]),
                    shape=list(vector.shape), split=row['split']))
            ao_records = []
            for layer in spec.get('ao_layers', []):
                vectors = observed[layer]['ao_window']
                assert vectors.ndim == 2 and vectors.shape[1] == spec['d_model'] and torch.isfinite(vectors).all()
                ident = digest([case_id, layer, 'ao_question_window20'])
                path = out / 'activations' / (ident + '.pt')
                torch.save(vectors, path)
                ao_records.append(dict(common, id=ident, activation_path=str(path), layer=layer,
                    hidden_state_index=layer + 1, shape=list(vectors.shape),
                    source_positions=list(range(window_start, qend + 1)), source_token_ids=ids[window_start:qend + 1]))
            ref = dict(case_id=case_id, model=args.model, dataset=row['dataset'], id=row['id'],
                condition=condition, category=common['category'], question=row['question'], messages=messages,
                source_prompt_addition=addition, answer=answer, completed=True, metadata=metadata,
                source_model_revision=common['source_model_revision'], reader_output_intervention=False,
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(out / 'cases' / (case_id + '.json'), dict(nla=nla_records, ao=ao_records, reference=ref), immutable=True)
            write_json(out / 'capture_status.json', dict(state='running', saved_cases=len(jobs) - len(pending) + index + 1, expected_cases=len(jobs)))
            print(f"Saved {index + 1}/{len(pending)}: {row['id']} {condition}, {metadata['output_tokens']} answer tokens", flush=True)
    cases = [json.loads((out / 'cases' / (j[-1] + '.json')).read_text()) for j in jobs]
    manifests = {'manifest.jsonl': [r for c in cases for r in c['nla']],
                 'ao_manifest.jsonl': [r for c in cases for r in c['ao']],
                 'answers.jsonl': [c['reference'] for c in cases]}
    for name, rows in manifests.items():
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    cfg = dict(seed=42, paths={'cache_dir': config['cache_root']}, nla_model=dict(
        model_id=spec['av_local_path'] if 'av_local_path' in spec else snapshot(config['cache_root'], spec['av']),
        dtype='bfloat16', device_map='auto' if len(spec['gpus']) > 1 else 'cuda', trust_remote_code=False,
        expected_d_model=spec['d_model'], sidecar_filename='nla_meta.yaml'),
        generation={'do_sample': False, 'max_new_tokens': 256})
    (out / 'av_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / 'capture_status.json', dict(state='complete', questions=len(config['cohort']),
        conditions=list(config['source_prompt_additions']), answers=len(cases), vectors=len(manifests['manifest.jsonl']),
        ao_windows=len(manifests['ao_manifest.jsonl']), layer=spec['layer'], hf_hidden_state_index=spec['layer'] + 1))


if __name__ == '__main__':
    main()
