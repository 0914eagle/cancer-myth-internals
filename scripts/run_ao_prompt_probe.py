"""Controlled investigator-query audit using the public AO helpers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/activation_oracles_official'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', default='gemma27')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    spec = config['models'][args.model]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, spec['gpus']))
    # Match the official inference example: dynamic steering hooks and
    # Accelerate model parallelism must not enter automatic torch.compile.
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    import fcntl
    locks = []
    for gpu in spec['gpus']:
        handle = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    import torch
    from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel
    from safetensors import safe_open
    from nl_probes.utils.dataset_utils import create_training_datapoint
    from nl_probes.utils.eval import run_evaluation
    from scripts.nla_generation_capture import decoder_layer
    from src.paper_baselines import digest, write_json
    out = Path(config['artifact_root']) / args.model
    downloads = json.loads(Path(config['download_specs']).read_text())
    base = downloads['gemma27']['local_path']
    adapter = downloads['ao']['local_path']
    ao_config = json.loads((Path(adapter) / 'ao_config.json').read_text())
    assert ao_config['model_name'] == 'google/gemma-3-27b-it'
    layer = spec['ao_primary_layer']
    assert [layer] in ao_config['act_layer_combinations']
    output = out / 'ao_results.jsonl'
    done = {json.loads(line)['id'] for line in output.open()} if output.exists() else set()
    rows = [json.loads(line) for line in Path(config['source_manifest']).open()]
    rows = [r for r in rows if r['case_id'] in set(config['selected_case_ids']) and r['layer'] == layer]
    assert len(rows) == 16 and len({r['case_id'] for r in rows}) == 16
    jobs = []
    for question in config['queries']:
        for row in rows:
            for width in config['ao_window_sizes']:
                ident = digest([row['id'], width, question['text'], 'ao_query_audit_v1'])
                if ident not in done:
                    jobs.append((ident, row, width, question['id'], question['text']))
    if args.limit:
        jobs = jobs[:args.limit]
    if not jobs:
        print('No pending AO rows', flush=True)
        return
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = Gemma3ForConditionalGeneration.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map='auto', max_memory={i: '40GiB' for i in range(len(spec['gpus']))},
        attn_implementation='sdpa').eval()
    assert all(str(v) not in ('cpu', 'disk') for v in model.hf_device_map.values())
    model = PeftModel.from_pretrained(model, adapter, adapter_name='ao', is_trainable=False).eval()
    # All checkpoint LoRA tensors must map to the loaded model; warnings alone are insufficient.
    with safe_open(str(Path(adapter) / 'adapter_model.safetensors'), framework='pt') as file:
        expected_keys = list(file.keys())
    loaded = model.state_dict()
    mapped = []
    for key in expected_keys:
        actual = key.replace('.lora_A.weight', '.lora_A.ao.weight').replace('.lora_B.weight', '.lora_B.ao.weight')
        assert actual in loaded, f'Unmapped AO adapter tensor: {key}'
        mapped.append(actual)
    assert mapped and model.active_adapter == 'ao'
    hook_module, hook_path = decoder_layer(model.get_base_model(), ao_config['hook_onto_layer'] + 1, 62)
    device = model.get_input_embeddings().weight.device
    assert next(hook_module.parameters()).device == device
    generation = dict(do_sample=False, max_new_tokens=256, disable_compile=True,
                      pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id)
    policy = dict(ao_checkpoint=downloads['ao'], source_checkpoint=downloads['gemma27'],
        source_layer=layer, source_hf_hidden_index=layer + 1, injection_block=ao_config['hook_onto_layer'],
        injection_module=hook_path, steering_coefficient=ao_config['steering_coefficient'],
        official_helpers='create_training_datapoint + run_evaluation + get_hf_activation_steering_hook',
        source_windows='Question-only consecutive tokens ending at last question token',
        queries=config['queries'], max_new_tokens=256, batch_size=config['ao_batch_size'],
        original_config_generation=ao_config['generation_kwargs'],
        layer_difference='Public AO primary layer31; public NLA layer41. Same source model/questions, different trained layers.')
    write_json(out / 'ao_execution_policy.json', policy, immutable=True)
    saved = 0
    with output.open('a') as stream:
        for offset in range(0, len(jobs), config['ao_batch_size']):
            chunk = jobs[offset:offset + config['ao_batch_size']]
            points = []
            for ident, row, width, variant, query in chunk:
                window = torch.load(row['activation_path'], map_location='cpu', weights_only=True)
                assert window.ndim == 2 and window.shape[1] == spec['d_model']
                vectors = window[-width:]
                actual_width = len(vectors)
                assert actual_width > 0
                point = create_training_datapoint(datapoint_type='FPQA_preanswer', prompt=query,
                    target_response='N/A', layer=layer, num_positions=actual_width, tokenizer=tokenizer,
                    acts_BD=vectors, feature_idx=-1, context_input_ids=None, context_positions=None,
                    ds_label=None, meta_info={'id': ident})
                points.append(point)
            results = run_evaluation(eval_data=points, model=model, tokenizer=tokenizer,
                submodule=hook_module, device=device, dtype=torch.bfloat16, global_step=-1,
                lora_path=None, eval_batch_size=config['ao_batch_size'],
                steering_coefficient=ao_config['steering_coefficient'], generation_kwargs=generation)
            assert len(results) == len(chunk)
            for (ident, row, width, variant, query), result in zip(chunk, results):
                actual_width = min(width, row['shape'][0])
                record = {**row, 'id': ident, 'prompt_variant': variant, 'oracle_question': query, 'requested_window_size': width, 'window_size': actual_width,
                          'short_question_window': actual_width < width, 'ao_output': result.api_response,
                          'ao_prompt_rendered': result.prompt, 'ao_model_revision': downloads['ao']['revision'],
                          'interpretation': 'Natural-language interpretation, not correctness label',
                          'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                saved += 1
            stream.flush()
            write_json(out / 'ao_status.json', dict(state='running', retained=len(done), new_rows=saved,
                                                   pending_at_start=len(jobs)))
            print('AO saved', saved, '/', len(jobs), flush=True)
    write_json(out / 'ao_status.json', dict(state='complete', retained=len(done), new_rows=saved,
        total_rows=len(done) + saved, limited_run=bool(args.limit)))


if __name__ == '__main__':
    main()
