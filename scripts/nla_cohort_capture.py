"""Resumable matched pre-answer extraction with official zero-based layer indices."""
import argparse
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
    out = Path(config['artifact_root']) / args.model
    (out / 'cases').mkdir(parents=True, exist_ok=True)
    (out / 'activations').mkdir(exist_ok=True)
    baseline = json.loads(Path(config['baseline_config']).read_text())
    cohort = config['cohort']
    references = {}
    if args.model in ('qwen25', 'gemma3'):
        source = Path(config['reference_root']) / args.model / 'answers.jsonl'
        references = {(r['dataset'], r['id'], r['condition']): r
                      for r in (json.loads(line) for line in source.open())}
    registry = json.loads((ROOT / baseline['registry']).read_text())
    jobs = []
    for row in cohort:
        for condition in config['conditions']:
            if references:
                ref = references[row['dataset'], row['id'], condition]
                messages = ref['messages']
            else:
                messages = [{'role': 'system', 'content': registry['transport_common'] + '\n\n' + registry['prompts'][condition]},
                            {'role': 'user', 'content': row['question']}]
                ref = dict(case_id=digest([row['dataset'], row['id'], condition]), model=args.model,
                           dataset=row['dataset'], id=row['id'], condition=condition,
                           question=row['question'], answer='', completed=False,
                           reference_status='No target answer generated; hidden-only extraction')
            case_key = digest(['corrected_nla_ao_capture_v1', args.model, spec['layer'], messages])
            jobs.append((row, condition, messages, ref, case_key))
    pending = [job for job in jobs if not (out / 'cases' / (job[4] + '.json')).exists()]
    if pending:
        backend = HFTask(baseline, args.model)
        layers = sorted(set([spec['layer']] + spec.get('ao_layers', [])))
        modules = {k: decoder_layer(backend.model, k + 1, baseline['models'][args.model]['n_layers'])[0]
                   for k in layers}
        for index, (row, condition, messages, ref, case_key) in enumerate(pending):
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
            positions = {'question_last_token': final_token(start, end),
                         'prompt_keyword_1': final_token(start + keyword['char_span'][0], start + keyword['char_span'][1]),
                         'assistant_boundary_last_token': len(ids) - 1}
            captured = {}
            handles = []
            for k, module in modules.items():
                def hook(_module, _inputs, output, k=k):
                    hidden = output[0] if isinstance(output, tuple) else output
                    assert hidden.shape[:2] == batch['input_ids'].shape
                    captured[k] = {name: hidden[0, pos].detach().float().cpu() for name, pos in positions.items()}
                    if k in spec.get('ao_layers', []):
                        # Window ends inside the question, so no assistant/answer tokens enter AO.
                        qend = positions['question_last_token']
                        qstart = [i for i, (x, y) in enumerate(mapping['offset_mapping']) if x < end and y > start and x != y][0]
                        window_start = max(qstart, qend - 19)
                        captured[k]['ao_window'] = hidden[0, window_start:qend + 1].detach().float().cpu()
                handles.append(module.register_forward_hook(hook))
            try:
                with torch.inference_mode():
                    result = backend.model(**batch, use_cache=False, output_hidden_states=(index == 0))
                if index == 0:
                    for k in layers:
                        for name, pos in positions.items():
                            assert torch.equal(captured[k][name], result.hidden_states[k + 1][0, pos].detach().float().cpu())
                del result
            finally:
                for handle in handles:
                    handle.remove()
            records = []
            for name, pos in positions.items():
                activation = captured[spec['layer']][name]
                assert activation.shape == (spec['d_model'],) and torch.isfinite(activation).all()
                ident = digest([case_key, spec['layer'], name])
                path = out / 'activations' / (ident + '.pt')
                torch.save(activation, path)
                records.append(dict(id=ident, base_id=row['id'], case_id=ref['case_id'], source=row['dataset'],
                    category='FPQ' if row['label'] else 'NFP', condition=condition, prompt=row['question'],
                    activation_path=str(path), layer=spec['layer'], hidden_state_index=spec['layer'] + 1,
                    layer_index_convention='official zero-based decoder block output', position=str(pos),
                    position_family=name, selected_token_id=ids[pos], selected_token_text=backend.tokenizer.decode([ids[pos]]),
                    activation_mode='new_prompt_forward_before_generation', shape=list(activation.shape),
                    split=row['split'], partition='train', answer_completed=ref['completed'],
                    source_model_revision=baseline['models'][args.model]['revision']))
            ao_records = []
            for k in spec.get('ao_layers', []):
                window = captured[k]['ao_window']
                assert window.ndim == 2 and window.shape[1] == spec['d_model'] and torch.isfinite(window).all()
                ident = digest([case_key, k, 'ao_question_window20'])
                path = out / 'activations' / (ident + '.pt')
                torch.save(window, path)
                qend = positions['question_last_token']
                ao_records.append(dict(id=ident, base_id=row['id'], case_id=ref['case_id'], source=row['dataset'],
                    condition=condition, category='FPQ' if row['label'] else 'NFP', prompt=row['question'],
                    activation_path=str(path), layer=k, hidden_state_index=k + 1, shape=list(window.shape),
                    source_positions=list(range(qend - len(window) + 1, qend + 1)),
                    source_token_ids=ids[qend - len(window) + 1:qend + 1], partition='train',
                    source_model_revision=baseline['models'][args.model]['revision']))
            write_json(out / 'cases' / (case_key + '.json'), dict(nla=records, ao=ao_records, reference=ref), immutable=True)
            if (index + 1) % 50 == 0:
                write_json(out / 'capture_status.json', dict(state='running', new_cases=index + 1,
                    retained_cases=len(jobs) - len(pending), total_cases=len(jobs)))
                print('Captured', index + 1, 'new cases /', len(pending), flush=True)
    manifest = []
    ao_manifest = []
    answers = []
    for *_, case_key in jobs:
        case = json.loads((out / 'cases' / (case_key + '.json')).read_text())
        manifest.extend(case['nla']);ao_manifest.extend(case['ao']);answers.append(case['reference'])
    for name, rows in [('manifest.jsonl', manifest), ('ao_manifest.jsonl', ao_manifest), ('answers.jsonl', answers)]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    cfg = dict(seed=42, paths={'cache_dir': config['cache_root']}, nla_model=dict(
        model_id=spec['av_local_path'] if 'av_local_path' in spec else snapshot(config['cache_root'], spec['av']),
        dtype='bfloat16', device_map='auto' if len(spec['gpus']) > 1 else 'cuda', trust_remote_code=False,
        expected_d_model=spec['d_model'], sidecar_filename='nla_meta.yaml'),
        generation={'do_sample': False, 'max_new_tokens': 256})
    (out / 'av_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / 'capture_status.json', dict(state='complete', questions=len(cohort), conditions=config['conditions'],
        vectors=len(manifest), ao_windows=len(ao_manifest), layer=spec['layer'], hf_hidden_state_index=spec['layer'] + 1))
    print('Complete:', len(manifest), 'NLA vectors;', len(ao_manifest), 'AO windows', flush=True)


if __name__ == '__main__':
    main()
