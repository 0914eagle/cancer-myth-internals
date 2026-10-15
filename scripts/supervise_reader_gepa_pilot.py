"""Keep the approved pilot alive in tmux; adopt an already-running driver."""
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
from scripts.reader_gepa_io import save


def retry_decision(error, consecutive_failures, made_progress=False):
    """Retry capacity/quota indefinitely; bound consecutive opaque failures."""
    message = error.lower()
    capacity_or_quota = any(word in message for word in (
        'rate limit', 'usage limit', 'quota', '429', 'temporarily unavailable',
        'at capacity', '503', 'service unavailable', '529', 'overloaded_error',
        'rate_limit_error', 'hit your limit', 'reached your limit', 'limit reached'))
    provider_failure = any(word in message for word in (
        'codex turn failed', 'claude failed', 'cli exited', 'timeoutexpired'))
    failures = 0 if made_progress else consecutive_failures
    if capacity_or_quota:
        return True, failures, 'capacity_or_quota'
    if provider_failure and failures < 3:
        return True, failures + 1, 'opaque_provider_failure'
    return False, failures, 'attention_required'


def provider_progress(out):
    directory = out / 'external_calls'
    return max((entry.stat().st_mtime_ns for entry in directory.glob('*.json')), default=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--adopt-driver-pid', type=int)
    p.add_argument('--adopt-native-pid', type=int)
    p.add_argument('--adopt-monitor-pid', type=int)
    a = p.parse_args(); config = json.loads(a.config.read_text())
    out = Path(config['artifact_root']); state_path = out / 'supervisor_status.json'
    assert json.loads((out / 'export_consent.json').read_text())['state'] == 'approved'
    native = None; driver = None; monitor = None
    handles = []
    def alive(pid, expected):
        if not pid: return False
        try:
            cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
            return expected in cmd and 'gepa_reader_pilot/runtime.json' in cmd
        except (FileNotFoundError, ProcessLookupError): return False
    class Adopted:
        def __init__(self, pid, script): self.pid = pid; self.script = script; self.returncode = None
        def poll(self):
            self.returncode = None if alive(self.pid, self.script) else 1
            return self.returncode
        def wait(self, timeout):
            deadline = time.time() + timeout
            while self.poll() is None:
                if time.time() > deadline: raise subprocess.TimeoutExpired(self.script, timeout)
                time.sleep(1)
            return self.returncode
    if a.adopt_driver_pid and alive(a.adopt_driver_pid, 'run_reader_gepa_pilot.py'):
        driver = Adopted(a.adopt_driver_pid, 'run_reader_gepa_pilot.py')
    if a.adopt_native_pid and alive(a.adopt_native_pid, 'reader_gepa_native_worker.py'):
        native = Adopted(a.adopt_native_pid, 'reader_gepa_native_worker.py')
    if a.adopt_monitor_pid and alive(a.adopt_monitor_pid, 'export_reader_gepa_cohort.py'):
        monitor = Adopted(a.adopt_monitor_pid, 'export_reader_gepa_cohort.py')
    def start(script, python, args, log_name):
        handle = (out / log_name).open('a'); handles.append(handle)
        return subprocess.Popen([python, '-u', 'scripts/' + script, '--config', str(a.config.resolve()), *args],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=handle, start_new_session=True,
            env={**os.environ, 'HF_HOME': '/data3/heejae/hf_cache'})
    python = '/data3/heejae/uv/paper_baselines_b81ace2/bin/python'
    next_driver_retry = 0
    transient_restarts = 0
    last_provider_progress = provider_progress(out)
    while not (out / 'SUPERVISOR_STOP').exists():
        pipeline_path = out / 'pipeline_status.json'
        pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
        if pipeline.get('state') == 'complete':
            save(out / 'native/STOP', {'reason': 'Pilot complete'})
            if native is not None: native.wait(timeout=120)
            save(state_path, dict(state='complete', pid=os.getpid(), updated_utc=time.time()))
            return
        if native is not None and native.poll() is not None:
            save(state_path, dict(state='attention_required', component='native_worker', exit_code=native.returncode,
                detail='Native worker exited unexpectedly; cached answers retained.', updated_utc=time.time()))
            return
        if native is None:
            (out / 'native/STOP').unlink(missing_ok=True)
            native = start('reader_gepa_native_worker.py', python,
                ['--gpus', config.get('native_gpus', '0,1')], 'native_worker.log')
        local_summary = out / 'local_cohort/summary.json'
        local_done = local_summary.exists() and json.loads(local_summary.read_text()).get('state') == 'complete'
        if not local_done and (monitor is None or monitor.poll() is not None):
            monitor = start('export_reader_gepa_cohort.py', 'python3', ['--watch'], 'local_cohort_watch.log')
        if driver is not None and driver.poll() is not None:
            pipeline = json.loads(pipeline_path.read_text()) if pipeline_path.exists() else {}
            if pipeline.get('state') == 'complete': continue
            error = str(pipeline.get('error', ''))
            progress = provider_progress(out)
            should_retry, transient_restarts, failure_kind = retry_decision(
                error, transient_restarts, progress > last_provider_progress)
            last_provider_progress = progress
            if should_retry:
                next_driver_retry = time.time() + 600
                save(state_path, dict(state='waiting_provider', retry_interval_seconds=600, next_retry_utc=next_driver_retry,
                    error=error, native_pid=native.pid, transient_restarts=transient_restarts,
                    failure_kind=failure_kind, updated_utc=time.time()))
                driver = None
            else:
                save(state_path, dict(state='attention_required', component='optimization_driver', error=error,
                    exit_code=driver.returncode, native_pid=native.pid, updated_utc=time.time()))
                # Local generation may continue; do not endlessly repeat a code/format failure.
                while native.poll() is None and not (out / 'SUPERVISOR_STOP').exists(): time.sleep(10)
                return
        if driver is None and time.time() >= next_driver_retry:
            driver = start('run_reader_gepa_pilot.py', python, [], 'pipeline.log')
        if time.time() >= next_driver_retry:
            save(state_path, dict(state='running', pid=os.getpid(), native_pid=native.pid,
                driver_pid=driver.pid if driver else None,
                monitor_pid=monitor.pid if monitor and monitor.poll() is None else None,
                launched_in='tmux detached session', updated_utc=time.time()))
        time.sleep(5)
    # Explicit stop marker affects only this supervisor's owned processes.
    save(out / 'native/STOP', {'reason': 'Explicit supervisor stop marker'})
    if driver is not None and driver.poll() is None: os.kill(driver.pid, signal.SIGINT)
    save(state_path, dict(state='stopping', updated_utc=time.time()))


if __name__ == '__main__': main()
