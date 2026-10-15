"""Also retain recoverable MMLU item errors as pending and evaluate the next row.

Quota/STOP, storage, provenance, authentication, and fatal GPU errors still stop
the worker. No failed answer is scored or removed from an eligible denominator.
"""
from __future__ import annotations
import gc
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_baselines as original
from scripts.run_paper_baselines_resilient import PendingWorker, record_paths, record_failure
from scripts.watch_paper_sonnet_quota import quota_error
from src.paper_baselines import sha, write_json


def recoverable_item_error(error):
    if isinstance(error, subprocess.CalledProcessError):
        text = error.stderr or ''
        return 'model_validate_plain_text' in text and any(word in text for word in ('ValidationError', 'ValueError', 'ParseError'))
    if not isinstance(error, (ValueError, RuntimeError)) or isinstance(error, json.JSONDecodeError):
        return False
    text = str(error).lower()
    if quota_error(text):
        return False
    fatal = ('stop requested', 'frozen artifact', 'cache mismatch', 'bundle hash',
        'pinned local checkpoint', 'cpu/disk offload', 'unauthorized', 'authentication',
        'invalid api key', 'permission denied', 'unexpected served model',
        'device-side assert', 'illegal memory access', 'driver shutting down',
        'cuda initialization', 'no cuda gpus', 'disk full', 'no space left')
    return not any(word in text for word in fatal)


class MMLUWorker(PendingWorker):
    def __init__(self, args):
        super().__init__(args)
        self.needs_cuda_cleanup = False
        write_json(self.out / 'execution_policy_mmlu.json', {
            'policy': 'mmlu_item_errors_pending_continue_v1',
            'runner': str(Path(__file__).resolve()), 'runner_sha256': sha(__file__),
            'evaluation_identity_unchanged': True, 'generation_settings_unchanged': True,
            'scope': 'MMLU evaluation rows, not training/optimization or provider quota',
            'failures': 'Pending with original denominator; no fabricated score',
        }, immutable=True)

    def process(self, row, split, method, candidate, gate, qa):
        if self.needs_cuda_cleanup:
            gc.collect()
            if self.backend is not None:
                self.backend.torch.cuda.empty_cache()
            self.needs_cuda_cleanup = False
        try:
            return super().process(row, split, method, candidate, gate, qa)
        except Exception as error:
            if self.args.command != 'run' or not qa or row['dataset'] != 'mmlu' or not recoverable_item_error(error):
                raise
            _, path = record_paths(self.out, row, split, method, qa)
            # Reuse the same separate evidence schema but retain the actual error.
            record_failure(self.out, self.config, row, split, method, qa)
            record = json.loads(path.read_text())
            details = str(error)
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                details += '\n' + error.stderr[-2000:]
            record.update(error=details, error_type=type(error).__name__,
                reason='recoverable_mmlu_item_error')
            write_json(path, record)
            self.note_pending(row, split, method)
            self.needs_cuda_cleanup = 'out of memory' in str(error).lower()
            print(f'Skipped MMLU item {row["id"]}: {type(error).__name__}: {error}; evidence: {path}', flush=True)


if __name__ == '__main__':
    original.Worker = MMLUWorker
    original.main()
