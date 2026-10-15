"""Wait for the pinned Qwen download, then test FP8 loading and both thinking modes."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_frontier_claude import write
from src.pilot import output_lock

DOWNLOAD = ROOT / 'results/model_downloads/qwen38_27b_fp8'
MODEL = 'Qwen/Qwen3.8-27B-FP8'
REVISION = '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'


def gpu_inventory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(row[0]): {'used_mib': int(row[1]), 'free_mib': int(row[2])}
            for line in output.splitlines() if (row := line.split(','))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def state(name, **extra):
        value = {'state': name, 'pid': os.getpid(), 'updated_unix': time.time(), **extra}
        write(out / 'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        with output_lock(out / 'load_test'):
            start = time.time()
            while True:
                if (out / 'STOP').exists():
                    raise RuntimeError('User STOP marker exists')
                status = json.loads((DOWNLOAD / 'status.json').read_text())
                runtime_file = out / 'runtime_ready.json'
                runtime = json.loads(runtime_file.read_text()) if runtime_file.exists() else {}
                if runtime.get('state') == 'failed':
                    raise RuntimeError('Runtime setup failed: ' + runtime.get('error', ''))
                if status['state'] == 'complete' and runtime.get('state') == 'ready':
                    break
                if status['state'] == 'failed':
                    raise RuntimeError('Model download failed: ' + status.get('error', ''))
                if time.time() - start > 7200:
                    raise RuntimeError('Download was not complete within two hours')
                state('waiting_for_download_and_runtime', download_state=status['state'],
                      runtime_state=runtime.get('state', 'installing'))
                time.sleep(30)
            assert status['model'] == MODEL and status['revision'] == REVISION
            snapshot = Path(status['snapshot'])
            manifest = json.loads((DOWNLOAD / 'manifest.json').read_text())
            assert all((snapshot / f['name']).stat().st_size == f['bytes'] for f in manifest['files'])
            assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
            inventory = gpu_inventory()
            assert all(inventory[g]['used_mib'] < 1000 and inventory[g]['free_mib'] > 22000
                       for g in (0, 1)), 'GPUs 0/1 are no longer free; no model was loaded'
            import torch
            import transformers
            from transformers import AutoModelForMultimodalLM, AutoTokenizer, AutoConfig
            from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear
            state('testing_fp8_kernel', torch=torch.__version__, transformers=transformers.__version__)
            x = torch.randn(2, 512, device='cuda', dtype=torch.bfloat16)
            w = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            y = finegrained_fp8_linear(x, w, torch.ones(4, 4, device='cuda'), [128, 128])
            ref = x @ w.to(torch.bfloat16).T
            error = ((y - ref).float().norm() / ref.float().norm()).item()
            assert torch.isfinite(y).all() and error < .1, f'FP8 kernel mismatch: {error}'
            del x, w, y, ref
            torch.cuda.empty_cache()
            state('loading_model')
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            # The checkpoint includes MoE-router exclusions in this dense model.
            # Transformers interprets them as prefix regexes, so `.mlp.gate`
            # incorrectly also excludes the FP8 `.mlp.gate_proj` weights.
            # Remove only nonexistent router names, leaving the checkpoint intact.
            config = json.loads((snapshot / 'config.json').read_text())['quantization_config']
            dropped = [s for s in config['modules_to_not_convert'] if s.endswith('.mlp.gate')]
            config['modules_to_not_convert'] = [s for s in config['modules_to_not_convert'] if s not in dropped]
            from transformers.quantizers.quantizers_utils import should_convert_module
            expected_fp8 = 0
            unused_mtp_fp8 = 0
            for shard in snapshot.glob('*.safetensors'):
                with shard.open('rb') as handle:
                    size = int.from_bytes(handle.read(8), 'little')
                    header = json.loads(handle.read(size))
                for name, tensor in header.items():
                    if name == '__metadata__':
                        continue
                    if tensor['dtype'] == 'F8_E4M3':
                        # The standard Transformers Qwen model deliberately ignores
                        # the checkpoint's optional multi-token prediction branch.
                        if name.startswith('mtp.'):
                            unused_mtp_fp8 += math.prod(tensor['shape'])
                            continue
                        module = name.removesuffix('.weight')
                        assert should_convert_module(module, config['modules_to_not_convert']), name
                        expected_fp8 += math.prod(tensor['shape'])
            write(out / 'loader_override.json', {'removed_nonexistent_router_exclusions': dropped,
                  'expected_fp8_elements': expected_fp8, 'unused_mtp_fp8_elements': unused_mtp_fp8,
                  'quantization_config': config})
            model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
            model_config.quantization_config = config
            model, loading_info = AutoModelForMultimodalLM.from_pretrained(
                snapshot, config=model_config, local_files_only=True, dtype=torch.bfloat16,
                device_map='balanced', max_memory={0: '20GiB', 1: '20GiB'},
                attn_implementation='sdpa',
                output_loading_info=True)
            model.eval()
            write(out / 'loading_info.json', json.loads(json.dumps(loading_info,
                  default=lambda value: sorted(value) if isinstance(value, set) else str(value))))
            assert not any(loading_info.get(k) for k in ('unexpected_keys', 'missing_keys', 'mismatched_keys', 'error_msgs')), loading_info
            assert all(str(d) not in ('cpu', 'disk', 'meta') for d in model.hf_device_map.values()), model.hf_device_map
            fp8_elements = sum(p.numel() for p in model.parameters() if p.dtype == torch.float8_e4m3fn)
            assert fp8_elements == expected_fp8, f'FP8 weight count mismatch: {fp8_elements} != {expected_fp8}'
            write(out / 'runtime.json', {'model': MODEL, 'revision': REVISION,
                  'torch': torch.__version__, 'transformers': transformers.__version__,
                  'device_map': model.hf_device_map, 'fp8_elements': fp8_elements,
                  'kernel_relative_error': error, 'gpu_inventory_before': inventory,
                  'gpu_inventory_loaded': gpu_inventory(), 'smoke_only': True})
            device = model.get_input_embeddings().weight.device
            eos = model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            for thinking in (False, True):
                mode = 'thinking_on' if thinking else 'thinking_off'
                state('generating', mode=mode)
                messages = [{'role': 'system', 'content': 'You are a helpful assistant. Answer in English.'},
                            {'role': 'user', 'content': 'What is 17 times 23? Give the final answer in one sentence.'}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=thinking, reasoning_effort='medium')
                inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
                torch.manual_seed(42)
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=True, temperature=.7, top_p=.95,
                        top_k=20, max_new_tokens=4096, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                ids = output[0, inputs['input_ids'].shape[1]:].tolist()
                raw = tokenizer.decode(ids, skip_special_tokens=False)
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                if thinking:
                    assert '</think>' in decoded, 'ON did not finish a reasoning block'
                    reasoning, final = decoded.split('</think>', 1)
                    reasoning = reasoning.removeprefix('<think>').strip()
                    assert reasoning, 'ON produced no reasoning'
                else:
                    reasoning, final = '', decoded
                    assert '<think>' not in final and '</think>' not in final
                record = {'thinking_enabled': thinking, 'reasoning_effort': 'medium' if thinking else None,
                          'prompt': prompt, 'raw': raw, 'reasoning': reasoning, 'final': final.strip(),
                          'generated_tokens': len(ids), 'ended_with_eos': bool(eos.intersection(ids)),
                          'smoke_only': True}
                write(out / (mode + '.json'), record)
                assert record['ended_with_eos'] and final.strip(), 'Generation was empty or truncated'
            state('complete', note='Both modes generated successfully. Smoke process exits and releases GPU memory.')
    except Exception as exc:
        state('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
