"""Switch each live serial worker after its current GEPA selection is saved.

Do not interrupt an optimizer. Ask the normal STOP mechanism to drain test
generation, then resume the same suite with hash-pinned cache compatibility.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.paper_gepa_resume import load_policy, prepare_policy
from src.paper_baselines import write_json

BASE = ROOT / 'results/paper_baselines_v4'
ROLES = {'qwen25': 'qwen', 'gemma3': 'gemma', 'luna6': 'luna'}
PYTHON = '/data1/heejae/uv/cancer_myth_internals/bin/python'
MARKER = 'GEPA parallel scheduling handoff: current selection already saved\n'


def selected_path(root, progress):
    return root / 'gepa' / progress['dataset'] / progress['split'] / progress['method'] / 'selected.json'


def safe_generation_boundary(root, progress, margin):
    """Avoid the end of test generation, when another optimizer could start."""
    return (progress.get('state') == 'running' and progress.get('stage') == 'table12'
            and progress.get('method', '').startswith('gepa')
            and selected_path(root, progress).exists()
            and progress.get('total', 0) - progress.get('processed', 0) > margin)


def unit(model):
    return 'paper-v4-' + ROLES[model] + '-sol' + ('-gpu123' if model == 'gemma3' else '') + '-20261012.service'


def pid(model):
    return subprocess.check_output(['systemctl', '--user', 'show', unit(model),
                                    '-p', 'MainPID', '--value'], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    directory = BASE / 'parallel_handoff'
    directory.mkdir(exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = directory / 'status.json'
        state = json.loads(status.read_text()) if status.exists() else {}
        for model in ROLES:
            root = BASE / 'main' / model
            prepare_policy(root)
            # The already queued self-extraction stage has frozen identities too.
            other = BASE / 'self_extraction' / model
            if (other / 'identity.json').exists():
                prepare_policy(other)
            progress = json.loads((root / 'generation_progress.json').read_text())
            state.setdefault(model, {'state': 'waiting_current_selection',
                                     'initial_progress': progress, 'registered_epoch': time.time()})
        write_json(status, state)
        while True:
            for model, entry in state.items():
                root = BASE / 'main' / model
                load_policy(root)  # Never silently adopt edits made after registration.
                stop = root / 'STOP'
                if entry['state'] == 'waiting_current_selection':
                    progress = json.loads((root / 'generation_progress.json').read_text())
                    entry['observed_progress'] = progress
                    # An unrelated crash is not permission to take over/restart.
                    if not pid(model) or pid(model) == '0':
                        entry['waiting_reason'] = 'Worker absent; preserving existing failure for review'
                        continue
                    margin = {'qwen25': 64, 'gemma3': 128, 'luna6': 16}[model]
                    if not safe_generation_boundary(root, progress, margin):
                        continue
                    try:
                        with stop.open('x') as f:
                            f.write(MARKER)
                    except FileExistsError:
                        entry['waiting_reason'] = 'Another STOP request exists; not taking ownership'
                        continue
                    entry.update(state='draining_generation', boundary=progress,
                                 selected=str(selected_path(root, progress)), requested_epoch=time.time())
                elif entry['state'] == 'draining_generation':
                    if pid(model) not in ('', '0'):
                        continue
                    if not stop.exists() or stop.read_text() != MARKER:
                        raise RuntimeError('Handoff STOP ownership changed: ' + model)
                    # Prevent the old service restart timer, then use the existing
                    # launcher with the same output paths, GPU placement and slots.
                    subprocess.run(['systemctl', '--user', 'stop', unit(model)], check=True)
                    stop.replace(directory / (model + '.drained_stop'))
                    subprocess.run([PYTHON, 'scripts/launch_paper_v4_sol.py', ROLES[model]],
                                   cwd=ROOT, check=True)
                    entry.update(state='resumed_parallel', resumed_epoch=time.time())
                elif entry['state'] == 'resumed_parallel':
                    records = list((root / 'gepa').glob('**/execution_policies/*.json'))
                    policies = [(p, json.loads(p.read_text())) for p in records]
                    if any(v.get('evaluation_workers') == 4 and v.get('pair_workers') == 2 for _, v in policies):
                        entry.update(state='parallel_confirmed', confirmed_epoch=time.time())
            write_json(status, state)
            if args.once or all(s['state'] == 'parallel_confirmed' for s in state.values()):
                return
            time.sleep(5)


if __name__ == '__main__':
    main()
