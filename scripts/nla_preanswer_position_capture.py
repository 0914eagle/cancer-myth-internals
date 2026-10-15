"""Capture three adjacent-context positions in one prompt-only forward pass."""
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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(spec['gpu'])
    import torch
    import yaml
    from src.paper_baselines import HFTask, digest, sha, write_json
    from scripts.nla_generation_capture import decoder_layer
    from scripts.nla_connection import snapshot
    baseline = json.loads(Path(config['baseline_config']).read_text())
    backend = HFTask(baseline, args.model)
    out = Path(config['artifact_root']) / args.model
    out.mkdir(parents=True, exist_ok=True)
    source = Path(config['reference_root']) / args.model
    answers = [json.loads(line) for line in (source / 'answers.jsonl').open()]
    wanted = {(r['dataset'], r['id']) for r in config['cohort']}
    selected = [r for r in answers if (r['dataset'], r['id']) in wanted]
    assert len(selected) == len(wanted) * 2
    rows = [r for r in selected if r['condition'] == 'plain']
    (out / 'answers.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    records = []
    # Public extraction_layer_index K is zero-based decoder block K.
    # The legacy helper takes a one-based block number, so pass K+1.
    layer, layer_path = decoder_layer(backend.model, spec['layer'] + 1, baseline['models'][args.model]['n_layers'])
    for index, row in enumerate(rows):
        text = backend.tokenizer.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
        batch = backend.inputs(row['messages'])
        mapped = backend.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = batch['input_ids'][0].tolist()
        assert mapped['input_ids'] == ids
        assert backend.tokenizer.decode([ids[-1]], skip_special_tokens=False) == '\n'
        expected_role = 'assistant' if args.model == 'qwen25' else 'model'
        assert backend.tokenizer.decode([ids[-2]], skip_special_tokens=False) == expected_role
        start = text.rfind(row['question'])
        assert start >= 0
        end = start + len(row['question'])
        question_tokens = [i for i, (a, b) in enumerate(mapped['offset_mapping'])
                           if a < end and b > start and a != b]
        positions = {'question_last_token': question_tokens[-1],
                     'assistant_role_token': len(ids) - 2,
                     'assistant_boundary_last_token': len(ids) - 1}
        captured = {}
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            assert hidden.shape[:2] == batch['input_ids'].shape
            for name, pos in positions.items():
                captured[name] = hidden[0, pos].detach().float().cpu()
        handle = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                # Prefill only: no answer token generation or saved-answer replay.
                result = backend.model(**batch, use_cache=True, output_hidden_states=(index == 0))
            if index == 0:
                for name, pos in positions.items():
                    expected = result.hidden_states[spec['layer'] + 1][0, pos].detach().float().cpu()
                    assert torch.equal(captured[name], expected), 'Decoder/HF hidden-state index mismatch'
            del result
        finally:
            handle.remove()
        assert set(captured) == set(positions)
        for name, pos in positions.items():
            activation = captured[name]
            assert activation.shape == (spec['d_model'],) and torch.isfinite(activation).all()
            ident = digest(['three_position_prompt_forward_v2_official_zero_based_layer', args.model, row['case_id'], name])
            path = out / 'activations' / (ident + '.pt')
            path.parent.mkdir(exist_ok=True)
            torch.save(activation, path)
            records.append(dict(id=ident, base_id=row['id'], case_id=row['case_id'],
                source=row['dataset'], category='FPQ' if row['label'] else 'NFP', condition='plain',
                prompt=row['question'], activation_path=str(path.resolve()), layer=spec['layer'],
                hidden_state_index=spec['layer'] + 1, decoder_module=layer_path + f".{spec['layer']}",
                layer_index_convention='official zero-based decoder block index; output_hidden_states index=K+1',
                position=str(pos), position_family=name, position_mode='token_index',
                selected_token_id=ids[pos], selected_token_text=backend.tokenizer.decode([ids[pos]], skip_special_tokens=False),
                activation_mode='new_prompt_forward_before_generation', shape=list(activation.shape),
                dtype=str(activation.dtype), split=row['split'], partition='train',
                source_model_revision=baseline['models'][args.model]['revision'],
                answer_completed=row['completed'],
                interpretation='Paired positions from one new prefill; reference answers are previous generation, not regenerated'))
        if (index + 1) % 16 == 0:
            print('Captured', index + 1, 'questions', flush=True)
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    cfg = dict(seed=baseline['seed'], paths={'cache_dir': config['cache_root']},
        nla_model=dict(model_id=snapshot(config['cache_root'], spec['av']), dtype='bfloat16',
                       device_map='cuda', trust_remote_code=False, expected_d_model=spec['d_model'],
                       sidecar_filename='nla_meta.yaml'),
        generation={'do_sample': False, 'max_new_tokens': 256})
    (out / 'av_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / 'capture_status.json', dict(state='complete', questions=len(rows), vectors=len(records),
        source_model=baseline['models'][args.model], runner_sha256=sha(__file__),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_pass='new prompt forward; no answer generated'))


if __name__ == '__main__':
    main()
