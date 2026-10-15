"""Locked source generation -> canonical NLA -> optional fixed-query AO."""
import argparse
import fcntl
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
spec = config['models'][args.model]
out = Path(config['artifact_root']) / args.model
out.mkdir(parents=True, exist_ok=True)
locks = []
for gpu in spec['gpus']:
    handle = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(handle)
env = {**os.environ, 'HF_HOME': '/data3/heejae/hf_cache', 'PYTHONUNBUFFERED': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
common = ['--config', str(args.config.resolve()), '--model', args.model]
commands = [('native_capture', ['/data3/heejae/uv/paper_baselines_b81ace2/bin/python', '-u',
                               'scripts/source_false_premise_capture.py', *common])]
av_script = 'scripts/nla_av_multigpu.py' if len(spec['gpus']) > 1 else 'scripts/nla_av_batched.py'
commands.append(('av', ['/data3/heejae/uv/medical_nla/bin/python', '-u', av_script, *common,
    '--gpu', ','.join(map(str, spec['gpus'])), '--batch-size', '16' if len(spec['gpus']) > 1 else '32',
    '--manifest', str(out / 'manifest.jsonl'), '--output', str(out / 'av_results.jsonl')]))
status = out / 'pipeline_status.json'
try:
    for stage, command in commands:
        status.write_text(json.dumps(dict(state='running', stage=stage, pid=os.getpid())) + '\n')
        with (out / (stage + '.log')).open('a') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
    if args.model == 'gemma27':
        manifest = [json.loads(s) for s in (out / 'ao_manifest.jsonl').read_text().splitlines()]
        ao_config = dict(config, selected_case_ids=[x['case_id'] for x in manifest], source_manifest=str(out / 'ao_manifest.jsonl'))
        path = out / 'ao_runtime.json'
        path.write_text(json.dumps(ao_config, ensure_ascii=False, indent=2) + '\n')
        # AO runner owns its own locks. Release only this wrapper's locks first.
        for handle in locks:
            handle.close()
        locks.clear()
        status.write_text(json.dumps(dict(state='running', stage='ao', pid=os.getpid())) + '\n')
        with (out / 'ao.log').open('a') as log:
            subprocess.run(['/data3/heejae/uv/fpqa_ao_20261012/bin/python', '-u', 'scripts/run_ao_prompt_probe.py',
                            '--config', str(path), '--model', args.model], cwd=ROOT, env=env,
                           stdout=log, stderr=log, check=True)
    status.write_text(json.dumps(dict(state='complete', stage='all', pid=os.getpid())) + '\n')
except Exception as error:
    status.write_text(json.dumps(dict(state='failed', error=str(error))) + '\n')
    raise
finally:
    for handle in locks:
        handle.close()
