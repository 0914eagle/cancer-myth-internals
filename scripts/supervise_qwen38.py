"""OS-service child supervisor; does not alter experiment settings or retry jobs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_frontier_claude import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=['generation', 'judge'], required=True)
parser.add_argument('--run-dir', type=Path, required=True)
args = parser.parse_args()
run = args.run_dir.resolve()
child = None
requested_stop = False


def terminate(*_):
    global requested_stop
    requested_stop = True
    if child is not None and child.poll() is None:
        child.terminate()


signal.signal(signal.SIGTERM, terminate)
signal.signal(signal.SIGINT, terminate)
env = dict(os.environ)
env['PATH'] = '/home/eagle0914/.local/bin:/usr/local/bin:/usr/bin:/bin'
env['HF_HOME'] = '/data1/heejae/hf_cache'
env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
env['TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR'] = '1'
env['CUDA_VISIBLE_DEVICES'] = '0,1' if args.kind == 'generation' else ''
if args.kind == 'generation':
    command = [sys.executable, '-u', str(ROOT / 'scripts/run_qwen38_transformers.py'),
               '--out-dir', str(run), '--batch-size', '8']
else:
    approval = json.loads((run / 'judge_transfer_approval.json').read_text())
    assert approval['approved'] is True
    command = [sys.executable, '-u', str(ROOT / 'scripts/judge_qwen38_answers.py'),
               '--run-dir', str(run), '--paired', '--workers', '5']

start = time.time()
with (run / (args.kind + '.log')).open('a') as log:
    log.write(f'\nSupervisor start {start}; kind={args.kind}\n')
    log.flush()
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    while True:
        code = child.poll()
        state = 'running' if code is None else ('complete' if code == 0 else 'failed')
        if code is not None and requested_stop:
            state = 'stopped'
        write(run / (args.kind + '_process.json'), {
            'state': state, 'supervisor_pid': os.getpid(), 'child_pid': child.pid,
            'started_unix': start, 'updated_unix': time.time(), 'exit_code': code,
            'requested_stop': requested_stop})
        if code is not None:
            if code != 0 and args.kind == 'generation':
                error = {'error': f'Generator process exited with code {code}', 'at': time.time()}
                write(run / 'error.json', error)
                for mode in ('thinking_off', 'thinking_on'):
                    write(run / mode / 'error.json', error)
            break
        time.sleep(15)
sys.exit(code)
