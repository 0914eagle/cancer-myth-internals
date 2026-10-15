"""Retry invalid audit batches after the initial runner releases its lock.

Keep every original response and use the same frozen inputs/prompt/validator.
Never overlap the initial five workers, and never retry valid results.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import time

from scripts.audit_ungated_failure_census import OUT, collect, job, prepare, write


def main():
    inputs, manifest = prepare()
    with (OUT / 'retry.lock').open('w') as retry_lock:
        fcntl.flock(retry_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(OUT / 'retry_status.json', dict(state='waiting_for_initial_runner', at=time.time()))
        with (OUT / 'run.lock').open('a') as main_lock:
            fcntl.flock(main_lock, fcntl.LOCK_EX)
            for round_index in [1, 2]:
                failed = []
                for i in range(0, len(inputs), 3):
                    folder = OUT / 'batches' / f'{i//3:03d}'
                    if (folder / 'result.json').exists():
                        continue
                    if folder.exists():
                        # No active worker holds the main lock. Preserve failed or
                        # interrupted attempts in full before any fresh request.
                        archive = OUT / 'retry_archive' / f'{i//3:03d}' / f'{time.time_ns()}'
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        folder.rename(archive)
                    failed.append(i)
                if not failed:
                    break
                write(OUT / 'retry_status.json', dict(state='running', round=round_index,
                      batches=[i//3 for i in failed], at=time.time()))
                print('retry round', round_index, 'batches', [i//3 for i in failed], flush=True)
                with ThreadPoolExecutor(max_workers=5) as pool:
                    pending = [pool.submit(job, i//3, inputs[i:i+3]) for i in failed]
                    for future in as_completed(pending):
                        print(future.result(), flush=True)
                        collect(inputs, manifest)
            summary = collect(inputs, manifest)
            state = 'complete' if summary['complete'] else 'incomplete_requires_inspection'
            write(OUT / 'retry_status.json', dict(state=state, valid=summary['valid'],
                  expected=summary['expected'], at=time.time()))
            write(OUT / 'execution_status.json', dict(state=state, valid=summary['valid'],
                  expected=summary['expected'], at=time.time(),
                  authorization='User explicitly approved external Claude transfer of these 234 pairs.',
                  original_responses='Preserved under retry_archive for all retried batches.'))
            print(state, summary['valid'], '/', summary['expected'], flush=True)


if __name__ == '__main__':
    main()
