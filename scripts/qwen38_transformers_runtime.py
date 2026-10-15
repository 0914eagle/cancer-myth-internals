"""Validated FP8 loader and batched generation; no benchmark labels enter this module."""
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.run_frontier_claude import write
from scripts.smoke_qwen38_transformers import gpu_inventory, MODEL, REVISION, DOWNLOAD

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'results/qwen38/transformers_smoke_20260926_v1'
OVERRIDE = SMOKE / 'loader_override.json'
SAMPLING = {'temperature': .7, 'top_p': .95, 'top_k': 20, 'repetition_penalty': 1.0}


def split_response(decoded, thinking):
    if thinking:
        if decoded.count('</think>') != 1:
            raise ValueError('Thinking ON did not finish exactly one reasoning block')
        reasoning, final = decoded.split('</think>', 1)
        reasoning = reasoning.removeprefix('<think>').strip()
        if not reasoning:
            raise ValueError('Thinking ON returned empty reasoning')
    else:
        reasoning, final = '', decoded
    if '<think>' in final or '</think>' in final:
        raise ValueError('Reasoning leaked into final answer')
    if not final.strip():
        raise ValueError('Final answer is empty')
    return reasoning, final.strip()


class Runtime:
    def __init__(self, out, batch_size):
        import torch
        import transformers
        import triton
        from transformers import AutoConfig, AutoTokenizer, AutoModelForMultimodalLM
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
        inventory = gpu_inventory()
        assert all(inventory[g]['used_mib'] < 1000 and inventory[g]['free_mib'] > 22000
                   for g in (0, 1)), 'GPUs 0/1 must be free before loading'
        download = json.loads((DOWNLOAD / 'status.json').read_text())
        assert download['state'] == 'complete' and download['revision'] == REVISION
        assert json.loads((SMOKE / 'status.json').read_text())['state'] == 'complete'
        override = json.loads(OVERRIDE.read_text())
        config = AutoConfig.from_pretrained(download['snapshot'], local_files_only=True)
        config.quantization_config = override['quantization_config']
        self.tokenizer = AutoTokenizer.from_pretrained(download['snapshot'], local_files_only=True,
                                                       padding_side='left')
        self.model, info = AutoModelForMultimodalLM.from_pretrained(
            download['snapshot'], config=config, local_files_only=True, dtype=torch.bfloat16,
            device_map='balanced', max_memory={0: '20GiB', 1: '20GiB'},
            attn_implementation='sdpa', output_loading_info=True)
        self.model.eval()
        write(out / 'loading_info.json', json.loads(json.dumps(info,
              default=lambda v: sorted(v) if isinstance(v, set) else str(v))))
        assert not any(info.get(k) for k in ('unexpected_keys', 'missing_keys', 'mismatched_keys', 'error_msgs')), info
        assert all(str(d) not in ('cpu', 'disk', 'meta') for d in self.model.hf_device_map.values())
        fp8 = sum(p.numel() for p in self.model.parameters() if p.dtype == torch.float8_e4m3fn)
        assert fp8 == override['expected_fp8_elements'], (fp8, override['expected_fp8_elements'])
        self.device = self.model.get_input_embeddings().weight.device
        self.torch = torch
        self.batch_size = batch_size
        self.out = out
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        write(out / 'runtime.json', {'model': MODEL, 'revision': REVISION,
              'torch': torch.__version__, 'transformers': transformers.__version__, 'triton': triton.__version__,
              'device_map': self.model.hf_device_map, 'fp8_elements': fp8,
              'initial_batch_size': batch_size, 'gpu_inventory_loaded': gpu_inventory(),
              'loader_override': override, 'generation_config': self.model.generation_config.to_dict()})

    def generate(self, tasks, thinking, cap):
        texts = [self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': t['system']}, {'role': 'user', 'content': t['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=thinking, reasoning_effort='medium')
            for t in tasks]
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(self.device)
        width = inputs['input_ids'].shape[1]
        assert width + cap <= 32768, 'Input plus output cap exceeds experiment context limit'
        seed = int(hashlib.sha256('\n'.join(t['stage'] + '/' + t['id'] for t in tasks).encode()).hexdigest()[:8], 16)
        self.torch.manual_seed(seed)
        start = time.time()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=True, **SAMPLING,
                max_new_tokens=cap, logits_to_keep=1,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        results = []
        for i, sequence in enumerate(generated[:, width:].tolist()):
            end = next((j + 1 for j, token in enumerate(sequence) if token in self.eos), len(sequence))
            ids = sequence[:end]
            decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
            result = {'raw_response': self.tokenizer.decode(ids, skip_special_tokens=False),
                      'decoded_response': decoded, 'generated_token_ids': ids,
                      'input_tokens': int(inputs['attention_mask'][i].sum()), 'output_tokens': len(ids),
                      'truncated': not bool(self.eos.intersection(ids)), 'max_new_tokens': cap,
                      'batch_size': len(tasks), 'batch_ids': [t['id'] for t in tasks], 'batch_seed': seed,
                      'batch_wall_seconds': time.time() - start}
            try:
                result['reasoning'], result['response'] = split_response(decoded, thinking)
            except ValueError as exc:
                result.update(response='', reasoning='', format_error=str(exc))
            results.append(result)
        return results

    def safe_generate(self, tasks, thinking, cap):
        try:
            return self.generate(tasks, thinking, cap)
        except self.torch.OutOfMemoryError:
            if len(tasks) == 1:
                raise
            self.batch_size = max(1, len(tasks) // 2)
            write(self.out / 'oom_retry.json', {'at': time.time(), 'cap': cap,
                  'batch_ids': [t['id'] for t in tasks], 'new_batch_size': self.batch_size})
        # Leave the exception handler before retrying so the traceback cannot
        # retain the failed forward pass's GPU tensors.
        self.torch.cuda.empty_cache()
        size = self.batch_size
        return (self.safe_generate(tasks[:size], thinking, cap) +
                self.safe_generate(tasks[size:], thinking, cap))
