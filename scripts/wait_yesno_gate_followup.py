"""One-shot sequential nohup queue; no service manager or automatic restarts."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / 'results/qwen38/full_transformers_20260926_v1'
OUT = ROOT / 'results/yesno_gate_followup_20260928_v1'
METHODS = ('plain', 'zero_shot_cot_one_step', 'premise_review_answer', 'fp_unconditional')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ready(current):
    if read(current / 'status.json').get('state') != 'generation_finished':
        return False
    return all(read(current / mode / 'well' / method / 'status.json').get('state') == 'complete'
               for mode in ('thinking_off', 'thinking_on') for method in METHODS)


def gpu_free():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = {int(a): (int(b), int(c)) for a, b, c in (s.split(',') for s in text.splitlines())}
    return all(rows[i][0] < 1000 and rows[i][1] > 22000 for i in (0, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # flock releases on exit; never revive/retry a failed child automatically.
    import fcntl
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (OUT / 'queue.pid').write_text(str(os.getpid()) + '\n')
        def state(value, **extra):
            obj = dict(state=value, pid=os.getpid(), at=time.time(), **extra)
            temp = OUT / 'status.tmp'
            temp.write_text(json.dumps(obj, indent=2) + '\n')
            temp.replace(OUT / 'status.json')
            print(json.dumps(obj), flush=True)
        try:
            state('waiting_for_current_generation_and_well')
            while True:
                if (OUT / 'STOP').exists() or any((CURRENT / p).exists() for p in ('STOP', 'thinking_off/STOP', 'thinking_on/STOP')):
                    state('stopped'); return
                status = read(CURRENT / 'status.json').get('state')
                if status in ('failed', 'stopped') or (CURRENT / 'error.json').exists():
                    raise RuntimeError('Current experiment failed/stopped; follow-up not started')
                if ready(CURRENT) and not (CURRENT / 'suite.lock').exists() and gpu_free():
                    break
                time.sleep(30)
            for model, env_name, source in (
                ('gemma4', 'cancer_myth_gemma4', ROOT / 'results/gemma4/full_20260926_v2'),
                ('qwen38', 'cancer_myth_qwen38_transformers', CURRENT / 'thinking_off')):
                target = OUT / model
                if (OUT / 'STOP').exists():
                    state('stopped'); return
                if not gpu_free():
                    raise RuntimeError('GPU 0/1 no longer free; no process will be displaced')
                if read(target / 'status.json').get('state') == 'complete':
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1', HF_HOME='/data1/heejae/hf_cache',
                           HF_HUB_DISABLE_IMPLICIT_TOKEN='1', TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR='1')
                if model == 'qwen38':
                    # Use exactly the publisher-verified kernel files already approved for this run.
                    pin = read(CURRENT / 'kernel_pin.json')
                    assert pin['publisher_trusted'] and len(pin['files']) >= 10
                    for item in pin['files']:
                        path = Path(pin['cache']) / item['path']
                        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
                    env['LOCAL_KERNELS'] = pin['repo'] + '=' + pin['cache']
                state('running', model=model)
                subprocess.run([f'/data1/heejae/uv/{env_name}/bin/python', '-u',
                    str(ROOT / 'scripts/run_yesno_gate_followup.py'), '--model', model,
                    '--source', str(source), '--out', str(target)], cwd=ROOT, env=env, check=True)
            state('complete')
        except Exception as exc:
            state('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
