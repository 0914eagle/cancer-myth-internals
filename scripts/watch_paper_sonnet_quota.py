"""Adopt live baseline suites; probe Sonnet every 600s after quota failures.

Evaluation implementations/configuration stay immutable. Successful cached outputs
are reused by the original launcher. Other failures and STOP never auto-restart.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def utc(timestamp):
    return dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def quota_error(text):
    text = text.lower().replace('’', "'")
    return bool(re.search(
        r"(?:you(?:'ve| have)? (?:hit|reached|exceeded) (?:your |the )?(?:usage )?limit|usage[_ -]?limit|"
        r"rate[_ -]?limit|quota.{0,40}(?:exceed|exhaust|limit)|"
        r"too many requests|too_many_requests|\b429\b|"
        r"claude failed:\s*limit reached)", text))


def fingerprint(pid):
    try:
        # comm may contain spaces or parentheses; fields after the last ')' start at state.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def supervisor_state(path):
    result = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                result[key] = value
    return result


class Supervisor:
    def __init__(self, config, bundle, out, python, interval=600, clock=time.time):
        self.config, self.bundle, self.out, self.python = config, bundle, out, python
        self.interval = interval
        self.clock = clock
        self.models, self.children = {}, {}
        self.next_probe = None
        self.last_probe = None
        self.directory = out / 'quota_watchdog'
        self.directory.mkdir(parents=True, exist_ok=True)
        external = json.loads((ROOT / json.loads(config.read_text())['external_config']).read_text())
        self.judge = external['judge_model']
        if self.judge['backend'] != 'claude':
            raise ValueError('This supervisor is specifically for the Claude Sonnet judge')

    def event(self, kind, **details):
        record = {'time_utc': utc(self.clock()), 'event': kind, **details}
        with (self.directory / 'events.jsonl').open('a') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def adopt(self, model, pid):
        state = supervisor_state(self.out / 'logs' / f'{model}-supervisor.state')
        if state.get('pid') != str(pid):
            raise ValueError(f'{model}: PID does not match original supervisor state')
        if model not in json.loads(self.config.read_text())['models']:
            raise ValueError('Unknown model')
        self.models[model] = dict(pid=pid, fingerprint=fingerprint(pid), adopted=True,
            phase=state.get('phase', 'fixed'), state='running', started_at=0,
            attempts=0, returncode=None)
        self.event('adopted', model=model, pid=pid, phase=self.models[model]['phase'])

    def alive(self, model):
        item = self.models[model]
        if model in self.children:
            return self.children[model].poll() is None
        return item['fingerprint'] is not None and fingerprint(item['pid']) == item['fingerprint']

    def failure(self, model):
        item = self.models[model]
        paths = [self.out / 'logs' / f"{model}-run-{item['phase']}.log"]
        if item.get('attempt_log'):
            paths.append(Path(item['attempt_log']))
        errors = []
        for path in paths:
            if path.exists() and path.stat().st_mtime >= item['started_at']:
                text = path.read_text()[-12000:]
                index = text.rfind('Traceback (most recent call last)')
                errors.append(text[index:] if index >= 0 else text)
        return '\n'.join(errors)[-15000:] or 'Process exited without a recorded quota error'

    def launch(self, model, now):
        item = self.models[model]
        item['attempts'] += 1
        attempt = self.directory / f"{model}-{item['phase']}-attempt{item['attempts']}.log"
        env = dict(os.environ, PAPER_PYTHON=str(self.python), PAPER_CONFIG=str(self.config),
            PAPER_BUNDLE=str(self.bundle), PAPER_OUT=str(self.out), PAPER_MODELS=model,
            PYTHONUNBUFFERED='1')
        with attempt.open('w') as output:
            child = subprocess.Popen(['bash', 'scripts/run_paper_baselines_4090.sh',
                'run', item['phase']], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[model] = child
        item.update(pid=child.pid, fingerprint=fingerprint(child.pid), adopted=False,
            state='running', started_at=now, attempt_log=str(attempt), returncode=None)
        self.event('resumed', model=model, pid=child.pid, phase=item['phase'])

    def probe(self):
        from src.fpqa_cli_backend import CLIBackend
        try:
            answer, metadata = CLIBackend(self.judge)([
                {'role': 'system', 'content': 'Availability check only. Reply exactly READY.'},
                {'role': 'user', 'content': 'READY'}])
            if answer.strip() != 'READY':
                raise ValueError('Availability check did not return READY')
            return {'ok': True, 'requested_model': self.judge['model'],
                'served_models': list(metadata['raw']['modelUsage']),
                'cli_version': metadata['cli_version']}
        except Exception as error:
            return {'ok': False, 'quota_error': quota_error(str(error)), 'error': str(error)}

    def step(self, now):
        for model, item in self.models.items():
            if item['state'] != 'running':
                continue
            original = supervisor_state(self.out / 'logs' / f'{model}-supervisor.state')
            if item['adopted']:
                item['phase'] = original.get('phase', item['phase'])
            if self.alive(model):
                continue
            stopped = (self.out / model / 'STOP').exists()
            if stopped:
                item['state'] = 'stopped'
                self.event('stopped', model=model)
                continue
            if item['adopted']:
                succeeded = original.get('exit_code') == '0' and original.get('state') == 'complete'
                item['returncode'] = original.get('exit_code')
            else:
                item['returncode'] = self.children[model].poll()
                succeeded = item['returncode'] == 0
            if succeeded:
                if not item['adopted'] and item['phase'] == 'fixed':
                    item['phase'] = 'trained'
                    self.launch(model, now)
                else:
                    item['state'] = 'complete'
                    self.event('complete', model=model)
                continue
            error = self.failure(model)
            item['error'] = error
            if quota_error(error):
                item['state'] = 'waiting_for_quota'
                if self.next_probe is None:
                    self.next_probe = now + self.interval
                self.event('quota_wait', model=model, phase=item['phase'],
                    next_probe_utc=utc(self.next_probe), error=error[-2000:])
            else:
                item['state'] = 'failed_other'
                self.event('non_quota_failure', model=model, error=error[-2000:])
        for model, item in self.models.items():
            if item['state'] == 'waiting_for_quota' and (self.out / model / 'STOP').exists():
                item['state'] = 'stopped'
                self.event('stopped', model=model)
        waiting = [name for name, item in self.models.items() if item['state'] == 'waiting_for_quota']
        if waiting and self.next_probe is not None and now >= self.next_probe:
            result = self.probe()
            self.last_probe = {'checked_utc': utc(now), **result}
            self.event('availability_probe', **result)
            if result['ok']:
                self.next_probe = None
                for name in waiting:
                    # STOP may be requested during the provider call.
                    if (self.out / name / 'STOP').exists():
                        self.models[name]['state'] = 'stopped'
                    else:
                        self.launch(name, self.clock())
            else:
                self.next_probe = now + self.interval
        if not any(item['state'] == 'waiting_for_quota' for item in self.models.values()):
            self.next_probe = None
        save(self.directory / 'status.json', {
            'pid': os.getpid(), 'checked_utc': utc(self.clock()), 'interval_seconds': self.interval,
            'next_probe_utc': utc(self.next_probe) if self.next_probe else None,
            'last_probe': self.last_probe, 'models': self.models})
        return any(item['state'] in ('running', 'waiting_for_quota') for item in self.models.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--adopt', action='append', required=True, metavar='MODEL=PID')
    args = parser.parse_args()
    supervisor = Supervisor(args.config, args.bundle, args.out, args.python)
    with (supervisor.directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for entry in args.adopt:
            model, pid = entry.split('=', 1)
            supervisor.adopt(model, int(pid))
        while supervisor.step(time.time()):
            time.sleep(15)


if __name__ == '__main__':
    main()
