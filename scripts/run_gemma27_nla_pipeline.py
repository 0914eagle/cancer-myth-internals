"""Run the common-model NLA comparison independently of the AO GPU pair."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--config', type=Path, required=True)
a = p.parse_args()
c = json.loads(a.config.read_text())
out = Path(c['artifact_root']) / 'gemma27'
state_path = out / 'nla_pipeline_status.json'
converted = Path(c['models']['gemma27']['av_local_path']) / 'conversion_status.json'
capture = out / 'capture_status.json'
try:
    state_path.write_text(json.dumps(dict(state='waiting', stage='capture_and_nla_weights', pid=os.getpid())) + '\n')
    deadline = time.monotonic() + 14400
    while True:
        if capture.exists() and converted.exists():
            # Writers publish complete only after all corresponding files are sealed.
            if '"state": "complete"' in capture.read_text() and '"state": "complete"' in converted.read_text():
                break
        if time.monotonic() > deadline:
            raise TimeoutError('Common-model capture/NLA weights did not complete within4hours')
        time.sleep(5)
    locks = []
    for gpu in [4, 5]:
        lock = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    state_path.write_text(json.dumps(dict(state='running', stage='av', pid=os.getpid(), gpus=[4, 5])) + '\n')
    with (out / 'nla_av.log').open('a') as log:
        subprocess.run(['/data3/heejae/uv/medical_nla/bin/python', '-u', 'scripts/nla_av_multigpu.py',
                        '--config', str(a.config.resolve()), '--model', 'gemma27', '--gpu', '4,5',
                        '--batch-size', '16', '--manifest', str(out / 'manifest.jsonl'),
                        '--output', str(out / 'av_results.jsonl')], cwd=ROOT,
                       env={**os.environ, 'HF_HOME': '/data3/heejae/hf_cache', 'PYTHONUNBUFFERED': '1'},
                       stdout=log, stderr=log, check=True)
    state_path.write_text(json.dumps(dict(state='complete', stage='av')) + '\n')
except Exception as error:
    state_path.write_text(json.dumps(dict(state='failed', error=str(error))) + '\n')
    raise
