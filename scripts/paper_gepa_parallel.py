"""Concurrency within a GEPA candidate; no changes to sampling or scoring."""
import fcntl
from pathlib import Path

from src.paper_baselines import write_json


def check_stop(worker):
    if (worker.out / 'STOP').exists():
        raise RuntimeError('STOP requested')


def evaluate_pair(worker, method, dataset, candidate, rows, pool=None):
    """Return input-order feedback even when the normal question finishes first.

    The pool belongs to one optimize() call, separately from GEPA's evaluator
    pool. Never submit nested work back to the GEPA pool (which can deadlock).
    """
    if len(rows) != 2 or [r['label'] for r in rows] != [1, 0]:
        raise ValueError('Expected one FPQ followed by one NFP')

    def run(row):
        check_stop(worker)
        record = worker.pipeline.run(method, row['question'], dataset, candidate=candidate)
        check_stop(worker)
        rating, reason = worker.judge(row, record['answer'])
        return {'id': row['id'], 'label': row['label'], 'question': row['question'],
                'answer': record['answer'], 'rating': rating, 'reason': reason,
                'detector_raw': record.get('detector_raw')}

    check_stop(worker)
    if pool is None:
        feedback = [run(row) for row in rows]
    else:
        futures = [pool.submit(run, row) for row in rows]
        try:
            feedback = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return sum(r['rating'] / 5 for r in feedback) / 2, {'examples': feedback}


def write_quota_status(worker, request_id, status):
    """Serialize shared status writes; keep a separate record per request too."""
    root = Path(worker.out)
    directory = root / 'optimization_quota_requests'
    directory.mkdir(parents=True, exist_ok=True)
    value = {**status, 'request_id': request_id}
    with (directory / '.status.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(directory / (request_id + '.json'), value)
        # Compatibility: this aggregate is the latest event, not queue totals.
        write_json(root / 'optimization_quota.json', value)
