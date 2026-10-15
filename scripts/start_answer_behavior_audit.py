"""Launch the existing audit with nohup in an independent OS session.

The launcher exits; logs and the runner's existing 20-minute quota recovery
continue without a live exec terminal. Never start a second locked runner.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/audits/answer_behavior_full_20260929_v2'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'launch.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (OUT/'run.lock').open('a') as run_lock:
            try:
                fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps(dict(state='already_running')))
                return
        if (OUT/'STOP').exists():
            raise SystemExit('STOP is present; no automatic removal or restart.')
        command=['nohup',sys.executable,'-u','-m','scripts.audit_answer_behavior_full','run','--workers','8']
        with (OUT/'run.log').open('a') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True,close_fds=True)
        record=dict(at=time.time(),pid=process.pid,session_id=os.getsid(process.pid),
                    launcher_pid=os.getpid(),launcher_session=os.getsid(0),
                    interpreter=sys.executable,state='launched_detached',command=command)
        with (OUT/'launches.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        print(json.dumps(record))


if __name__=='__main__':
    main()
