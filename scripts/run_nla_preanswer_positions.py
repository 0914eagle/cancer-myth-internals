"""Run prompt-only capture, AV descriptions and AR for a fixed position comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
out = Path(config['artifact_root']) / args.model
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(config['models'][args.model]['gpu']),
       'HF_HOME': '/data3/heejae/hf_cache', 'PYTHONUNBUFFERED': '1'}
common = ['--config', str(args.config.resolve()), '--model', args.model]
stages = [
    ('capture', ['/data3/heejae/uv/paper_baselines_b81ace2/bin/python', '-u',
                 'scripts/nla_preanswer_position_capture.py', *common]),
    ('av', ['/data3/heejae/uv/medical_nla/bin/python', '-u', 'scripts/nla_av_batched.py',
            *common, '--gpu', str(config['models'][args.model]['gpu']), '--batch-size', '32',
            '--manifest', str(out / 'manifest.jsonl'), '--output', str(out / 'av_results.jsonl')]),
    ('ar', ['/data3/heejae/uv/medical_nla/bin/python', '-u',
            'scripts/nla_preanswer_reconstruction.py', *common]),
]
try:
    for stage, command in stages:
        (out / 'pipeline_status.json').write_text(json.dumps(dict(stage=stage, state='running', pid=os.getpid())) + '\n')
        print(args.model, stage, flush=True)
        with (out / (stage + '.log')).open('a') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    (out / 'pipeline_status.json').write_text(json.dumps(dict(stage='complete', state='complete', pid=os.getpid())) + '\n')
    print(args.model, 'complete', flush=True)
except Exception as error:
    (out / 'pipeline_status.json').write_text(json.dumps(dict(stage=stage, state='failed', error=str(error))) + '\n')
    raise
