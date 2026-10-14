"""Persistent, fixed-protocol NLA or AO reader; source text is never supplied."""
import argparse
import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/activation_oracles_official'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--kind', choices=['nla', 'ao'], required=True)
    p.add_argument('--gpus', default='2,3')
    a = p.parse_args(); c = json.loads(a.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    locks = []
    Path('/tmp/paper-baselines-gpu-locks').mkdir(parents=True, exist_ok=True)
    for gpu in a.gpus.split(','):
        handle = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    import torch
    import yaml
    out = Path(c['artifact_root']) / 'readers' / a.kind
    requests = out / 'requests'; results = out / 'results'
    for path in [requests, results]: path.mkdir(parents=True, exist_ok=True)
    def save(path, data):
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)
    save(out / 'status.json', dict(state='loading', kind=a.kind, gpus=a.gpus))
    torch.manual_seed(42)
    if a.kind == 'nla':
        from scripts.nla_connection import medical_package
        package = medical_package(c['medical_nla_root'])
        nla = importlib.import_module(package + '.nla')
        modeling = importlib.import_module(package + '.modeling')
        path = c['models']['gemma27']['av_local_path']
        original_download = nla.hf_hub_download
        def sidecar_download(repo_id, filename, **kwargs):
            return str(Path(repo_id) / filename) if Path(repo_id).is_dir() else original_download(repo_id, filename=filename, **kwargs)
        nla.hf_hub_download = sidecar_download
        tokenizer = modeling.load_tokenizer(path, cache_dir=c['cache_root'], trust_remote_code=False)
        sidecar = nla.load_nla_sidecar(path, tokenizer=tokenizer, cache_dir=c['cache_root'], expected_d_model=5376)
        assert yaml.safe_load((Path(path) / 'nla_meta.yaml').read_text())['extraction_layer_index'] == 41
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map='auto', max_memory={i: '40GiB' for i in range(torch.cuda.device_count())}).eval()
        assert all(str(v) not in ('cpu', 'disk') for v in model.hf_device_map.values())
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        def read_batch(jobs):
            built = []
            for job in jobs:
                value = torch.load(job['activation_path'], map_location='cpu', weights_only=True)
                assert value.shape == (5376,) and torch.isfinite(value).all()
                built.append(nla.build_nla_inputs_embeds(tokenizer=tokenizer, embed_layer=model.get_input_embeddings(),
                    sidecar=sidecar, activation=value, device=model.get_input_embeddings().weight.device))
            assert all(torch.equal(x.input_ids, built[0].input_ids) for x in built)
            with torch.inference_mode():
                tokens = model.generate(inputs_embeds=torch.cat([x.inputs_embeds for x in built]),
                    attention_mask=torch.cat([x.attention_mask for x in built]), do_sample=False,
                    max_new_tokens=256, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
            answer = []
            for value, built_input in zip(tokens.detach().cpu().tolist(), built):
                ended = False
                for i, token in enumerate(value):
                    if token in eos: value = value[:i + 1]; ended = True; break
                raw = tokenizer.decode(value, skip_special_tokens=False)
                explanation, parsed = nla.extract_explanation(raw)
                answer.append(dict(explanation=explanation, raw_output=raw, parsed_explanation_tag=parsed,
                    finish_reason='eos' if ended else 'length', query=built_input.prompt_text,
                    injection_scale=sidecar.injection_scale, output_tokens=len(value)))
            return answer
    else:
        from transformers import Gemma3ForConditionalGeneration, AutoTokenizer
        from peft import PeftModel
        from safetensors import safe_open
        from nl_probes.utils.dataset_utils import create_training_datapoint
        from nl_probes.utils.eval import run_evaluation
        from scripts.nla_generation_capture import decoder_layer
        downloads = json.loads(Path(c['download_specs']).read_text())
        base = downloads['gemma27']['local_path']; adapter = downloads['ao']['local_path']
        ao_config = json.loads((Path(adapter) / 'ao_config.json').read_text())
        assert ao_config['model_name'] == 'google/gemma-3-27b-it' and [31] in ao_config['act_layer_combinations']
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True); tokenizer.padding_side = 'left'
        model = Gemma3ForConditionalGeneration.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map='auto', max_memory={i: '40GiB' for i in range(torch.cuda.device_count())}, attn_implementation='sdpa').eval()
        assert all(str(v) not in ('cpu', 'disk') for v in model.hf_device_map.values())
        model = PeftModel.from_pretrained(model, adapter, adapter_name='ao', is_trainable=False).eval()
        with safe_open(str(Path(adapter) / 'adapter_model.safetensors'), framework='pt') as checkpoint:
            keys = list(checkpoint.keys())
        loaded = model.state_dict()
        assert keys and all(key.replace('.lora_A.weight', '.lora_A.ao.weight').replace('.lora_B.weight', '.lora_B.ao.weight') in loaded for key in keys)
        assert model.active_adapter == 'ao'
        module, _ = decoder_layer(model.get_base_model(), ao_config['hook_onto_layer'] + 1, 62)
        device = model.get_input_embeddings().weight.device
        assert next(module.parameters()).device == device
        def read_batch(jobs):
            points = []
            for job in jobs:
                value = torch.load(job['activation_path'], map_location='cpu', weights_only=True)
                assert value.ndim == 2 and value.shape[1] == 5376 and 0 < len(value) <= 20 and torch.isfinite(value).all()
                points.append(create_training_datapoint(datapoint_type='FPQA_preanswer', prompt=job.get('query', c['ao_prompt']),
                    target_response='N/A', layer=31, num_positions=len(value), tokenizer=tokenizer,
                    acts_BD=value, feature_idx=-1, context_input_ids=None, context_positions=None,
                    ds_label=None, meta_info={'id': job['id']}))
            returned = run_evaluation(eval_data=points, model=model, tokenizer=tokenizer, submodule=module,
                device=device, dtype=torch.bfloat16, global_step=-1, lora_path=None, eval_batch_size=8,
                steering_coefficient=ao_config['steering_coefficient'], generation_kwargs=dict(do_sample=False,
                    max_new_tokens=256, disable_compile=True, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id))
            return [dict(explanation=x.api_response, query=job.get('query', c['ao_prompt']), rendered_prompt=x.prompt,
                         finish_reason='not_recorded_by_official_helper') for x, job in zip(returned, jobs)]
    save(out / 'status.json', dict(state='ready', kind=a.kind, gpus=a.gpus))
    while not (out / 'STOP').exists():
        pending = [p for p in requests.glob('*.json') if not (results / p.name).exists()]
        pending.sort(key=lambda p: p.stat().st_mtime_ns)
        if not pending:
            time.sleep(1); continue
        chunk = pending[:16 if a.kind == 'nla' else 8]
        jobs = [json.loads(path.read_text()) for path in chunk]
        save(out / 'status.json', dict(state='running', kind=a.kind, pending=len(pending)))
        records = read_batch(jobs)
        assert len(records) == len(jobs)
        for path, job, record in zip(chunk, jobs, records):
            save(results / path.name, dict(record, id=job['id'], source_native_id=job['native_id'],
                source_activation_path=job['activation_path'], source_text_provided=False,
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        save(out / 'status.json', dict(state='ready', kind=a.kind, processed_results=len(list(results.glob('*.json')))))
    save(out / 'status.json', dict(state='stopped', kind=a.kind))


if __name__ == '__main__':
    main()
