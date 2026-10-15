"""Wait for pinned AO weights, capture the full fixed cohort, then run AO."""
import argparse
import datetime
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
status_file = out / 'ao_pipeline_status.json'


def status(stage, state='running', **details):
    temp = status_file.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage, state=state, pid=os.getpid(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details), indent=2) + '\n')
    temp.replace(status_file)
    print(stage, state, flush=True)


try:
    status('waiting_for_model_download')
    ready = Path(c['artifact_root']) / 'download_status.json'
    deadline = time.monotonic() + 7200
    while '"stage": "ao_models_ready"' not in ready.read_text():
        if time.monotonic() > deadline:
            raise TimeoutError('AO model download did not finish within2hours')
        time.sleep(5)
    locks = []
    for gpu in c['models']['gemma27']['gpus']:
        lock = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    env = {**os.environ, 'HF_HOME': '/data3/heejae/hf_cache', 'PYTHONUNBUFFERED': '1'}
    for stage, interpreter, script in [
        ('capture', '/data3/heejae/uv/paper_baselines_b81ace2/bin/python', 'scripts/nla_cohort_capture.py'),
        ('ao', '/data3/heejae/uv/fpqa_ao_20261012/bin/python', 'scripts/run_fpqa_ao.py'),
    ]:
        status(stage)
        with (out / (stage + '.log')).open('a') as log:
            subprocess.run([interpreter, '-u', script, '--config', str(a.config.resolve()),
                            '--model', 'gemma27'], cwd=ROOT, env=env,
                           stdout=log, stderr=log, check=True)
    status('complete', state='complete')
except Exception as error:
    status('failed', state='failed', error=str(error))
    raise
