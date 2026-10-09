"""Transport-only quota recovery; preserve the frozen experiment/cache identity."""
from pathlib import Path
import json
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_baselines as runner
from src.paper_baselines import FIXED, digest, sha, write_json

INTERVAL = 600


def item_error(error):
    """Only known input/output length failures; never hide infrastructure errors."""
    if isinstance(error, RuntimeError) and str(error) == 'Generation truncated; retained task has no completed answer':
        return 'output_length_limit'
    if isinstance(error, ValueError) and str(error) == 'Input exceeds pinned limit; no silent truncation':
        return 'input_length_limit'
    return None


def item_failure_path(worker, row, split, method, qa):
    source, name = split['dataset'], split['name']
    if qa and method in FIXED:
        name = 'fixed'
        source = source if method in ('prewome', 'extract_verify') else 'shared'
    table = 'smoke' if worker.args.command == 'smoke' else 'table3' if qa else 'table12'
    return worker.out/'failures'/table/source/name/method/(digest([row['dataset'], row['id']])+'.json')


def process_with_failures(worker, row, split, method, candidate, gate, qa, process):
    path = item_failure_path(worker, row, split, method, qa)
    if path.exists():
        return
    try:
        return process(worker, row, split, method, candidate, gate, qa)
    except (RuntimeError, ValueError) as error:
        kind = item_error(error)
        if not kind:
            raise
        write_json(path, {'id': row['id'], 'dataset': row['dataset'], 'method': method,
                         'label': None if qa else row['label'], 'state': 'generation_failed',
                         'failure_kind': kind, 'error': str(error), 'recorded_epoch': time.time(),
                         'policy': 'Continue; keep in full denominator; never invent a judge rating.'},
                   immutable=True)
        print(f'[skip] {row["dataset"]}/{row["id"]} {method}: {kind}', flush=True)


def quota_error(error):
    if not isinstance(error, (RuntimeError, ValueError)):
        return False
    return bool(re.search(
        r"you[’']ve hit your limit|usage limit (?:reached|exceeded)|"
        r"rate[_ -]limit(?:ed|_error| exceeded| reached)|"
        r"too many requests|(?:http|status(?: code)?)\s*[:=]?\s*429\b",
        str(error), re.I))


def with_quota_retry(worker, role, messages, call, *, now=time.time, sleep=time.sleep):
    # No separate ping: retry the actual pending request and cache its successful
    # answer. Authentication, parsing, OOM and unrelated errors still fail fast.
    if worker.external[role + '_model']['backend'] != 'claude':
        return call(worker, role, messages)
    state_path = worker.out / 'quota_retry.json'
    request_id = digest({'role': role, 'messages': messages})
    attempts = 0
    deadline = 0
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if previous.get('state') == 'waiting':
            deadline = previous['next_retry_epoch']
            attempts = previous['quota_failures']
    while True:
        if (worker.out / 'STOP').exists():
            raise RuntimeError('STOP requested')
        remaining = deadline - now()
        if remaining > 0:
            sleep(min(1, remaining))
            continue
        try:
            result = call(worker, role, messages)
        except (RuntimeError, ValueError) as error:
            if not quota_error(error):
                if attempts:
                    write_json(state_path, {'state': 'failed_non_quota', 'role': role,
                                           'request_id': request_id, 'error_type': type(error).__name__})
                raise
            attempts += 1
            deadline = now() + INTERVAL
            write_json(state_path, {'state': 'waiting', 'role': role, 'request_id': request_id,
                                   'quota_failures': attempts, 'interval_seconds': INTERVAL,
                                   'next_retry_epoch': deadline})
            print(f'[{role}] Sonnet quota limit; retrying pending request in {INTERVAL}s '
                  f'(quota failure {attempts}).', flush=True)
        else:
            if attempts:
                write_json(state_path, {'state': 'recovered', 'role': role, 'request_id': request_id,
                                       'quota_failures': attempts, 'recovered_epoch': now()})
                print(f'[{role}] Sonnet available; continuing saved experiment.', flush=True)
            return result


def main():
    original = runner.Worker.external_call
    original_process = runner.Worker.process
    original_task = runner.Worker.task
    original_init = runner.Worker.__init__
    def initialize(worker, args):
        original_init(worker, args)
        policy = {'entrypoint_sha256': sha(Path(__file__)), 'quota_interval_seconds': INTERVAL,
                  'item_errors': ['output_length_limit', 'input_length_limit'],
                  'failure_denominator': 'retain', 'successful_cache_identity': 'unchanged'}
        write_json(worker.out/'execution_policies'/(digest(policy)+'.json'), policy, immutable=True)
    def resilient_task(worker, messages):
        # The same failed task can be reached by several gate methods/folds.
        # Remember it so deterministic generation is not repeated each time.
        identity = {'identity': worker.identity, 'messages': messages}
        path = worker.out/'failed_calls'/(digest(identity)+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            cls = ValueError if saved['failure_kind'] == 'input_length_limit' else RuntimeError
            raise cls(saved['error'])
        try:
            return original_task(worker, messages)
        except (RuntimeError, ValueError) as error:
            kind = item_error(error)
            if kind:
                write_json(path, {'request': identity, 'failure_kind': kind, 'error': str(error)}, immutable=True)
            raise
    def resilient_process(worker, row, split, method, candidate, gate, qa):
        return process_with_failures(worker, row, split, method, candidate, gate, qa, original_process)
    def resilient_call(worker, role, messages):
        write_json(worker.out / 'transport_retry_policy.json', {
            'interval_seconds': INTERVAL, 'entrypoint_sha256': sha(Path(__file__)),
            'only': 'Claude usage/rate limit errors',
            'experiment_identity': 'unchanged; retry same request, reuse successful cache'})
        return with_quota_retry(worker, role, messages, original)
    runner.Worker.external_call = resilient_call
    runner.Worker.__init__ = initialize
    runner.Worker.task = resilient_task
    runner.Worker.process = resilient_process
    runner.main()


if __name__ == '__main__':
    main()
