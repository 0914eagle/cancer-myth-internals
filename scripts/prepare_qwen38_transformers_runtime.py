"""Install a compatible torch runtime in the isolated Qwen environment and validate FP8."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/qwen38/transformers_smoke_20260926_v1'
PYTHON = '/data1/heejae/uv/cancer_myth_qwen38_transformers/bin/python'
PROBE = r'''
import torch, transformers, kernels, triton, json
from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear
x=torch.randn(2,512,device='cuda',dtype=torch.bfloat16)
w=torch.randn(512,512,device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
y=finegrained_fp8_linear(x,w,torch.ones(4,4,device='cuda'),[128,128])
reference=x@w.to(torch.bfloat16).T
error=((y-reference).float().norm()/reference.float().norm()).item()
assert torch.isfinite(y).all() and error<.1,error
from transformers import AutoModelForMultimodalLM, AutoConfig
config=AutoConfig.from_pretrained('/data1/heejae/hf_cache/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',local_files_only=True)
# Import the actual model implementation without allocating its weights.
model_class=AutoModelForMultimodalLM._model_mapping[type(config)]
print(json.dumps(dict(torch=torch.__version__, transformers=transformers.__version__,
                     kernels=kernels.__version__, triton=triton.__version__,
                     fp8_relative_error=error, model_class=model_class.__name__)))
'''


def state(name, **extra):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'runtime_ready.json'
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps({'state': name, 'updated_unix': time.time(), **extra}, indent=2) + '\n')
    temp.replace(target)


try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    state('installing')
    env = dict(os.environ, UV_HTTP_TIMEOUT='600', UV_CONCURRENT_DOWNLOADS='4',
               UV_CACHE_DIR='/tmp/cancer_myth_vllm_uv_cache', UV_LINK_MODE='copy')
    if not args.verify_only:
        subprocess.run(['/home/eagle0914/.local/bin/uv', 'pip', 'install', '--python', PYTHON,
                        'torch==2.8.0', 'torchvision==0.23.0', 'torchaudio==2.8.0',
                        '--torch-backend=cu126'], env=env, check=True)
        # The official FP8 kernel uses Triton 3.5 autotuner APIs. This overrides
        # torch 2.8's standard 3.4 pin; eager inference is validated below and in smoke.
        subprocess.run(['/home/eagle0914/.local/bin/uv', 'pip', 'install', '--python', PYTHON,
                        '--no-deps', 'triton==3.5.1'], env=env, check=True)
    state('testing')
    subprocess.run([PYTHON, '-u', '-c', PROBE], env=env, check=True)
    state('ready')
except Exception as exc:
    state('failed', error=f'{type(exc).__name__}: {exc}')
    traceback.print_exc()
    raise
