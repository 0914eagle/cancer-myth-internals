"""Keep bounded generation failures pending and continue remaining evaluation rows.

Original model/decoding/prompt/cache identities remain unchanged. This execution
policy is recorded separately; failures never become answers or fabricated scores.
"""
from __future__ import annotations
from collections import defaultdict
import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_baselines as original
from src.paper_baselines import FIXED, digest, qa_question, sha, write_json

TRUNCATED = 'Generation truncated; retained task has no completed answer'


def record_paths(out, row, split, method, qa):
    source, name = split['dataset'], split['name']
    if qa and method in FIXED:
        name = 'fixed'
        if method not in ('prewome', 'extract_verify'):
            source = 'shared'
    kind = 'table3' if qa else 'table12'
    suffix = Path(kind) / source / name / method / (digest([row['dataset'], row['id']]) + '.json')
    return out / suffix, out / 'generation_failures' / suffix


def record_failure(out, config, row, split, method, qa, provenance=None):
    _, path = record_paths(out, row, split, method, qa)
    record = dict(id=row['id'], dataset=row['dataset'], source=split['dataset'],
        split=split['name'], method=method, qa=qa, state='pending_generation_failure',
        question=qa_question(row) if qa else row['question'], error=TRUNCATED,
        max_new_tokens=config['max_new_tokens'], output_text_available=False,
        policy='Retain original eligible denominator; no answer or score imputation',
        observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if provenance:
        record['provenance'] = provenance
    # Existing failures are immutable evidence, not a reason to regenerate indefinitely.
    if not path.exists():
        write_json(path, record, immutable=True)
    return path


class PendingWorker(original.Worker):
    def __init__(self, args):
        super().__init__(args)
        self.pending = defaultdict(set)
        write_json(self.out / 'execution_policy.json', {
            'policy': 'generation_length_failure_pending_continue_v1',
            'runner': str(Path(__file__).resolve()), 'runner_sha256': sha(__file__),
            'evaluation_identity_unchanged': True,
            'generation_settings_unchanged': True,
            'failures': 'Separate records; retain full eligible denominator; no score imputation',
        }, immutable=True)

    def sync_status(self):
        for key, ids in self.pending.items():
            item = self.status.setdefault(key, {'state': 'running_with_pending_failures'})
            item['pending_generation_failures'] = len(ids)
            if item['state'] == 'complete':
                item['state'] = 'partial'
        if self.pending and self.status.get('table3', {}).get('state') == 'generated':
            self.status['table3']['state'] = 'partial'
        if self.pending:
            write_json(self.out / 'status.json', self.status)

    def note_pending(self, row, split, method):
        self.pending[f"{split['dataset']}/{split['name']}/{method}"].add((row['dataset'], row['id']))
        self.sync_status()

    def process(self, row, split, method, candidate, gate, qa):
        if self.args.command != 'run':
            return super().process(row, split, method, candidate, gate, qa)
        self.sync_status()
        output, failure = record_paths(self.out, row, split, method, qa)
        if failure.exists() and not output.exists():
            self.note_pending(row, split, method)
            return
        try:
            return super().process(row, split, method, candidate, gate, qa)
        except RuntimeError as error:
            if str(error) != TRUNCATED:
                raise
            path = record_failure(self.out, self.config, row, split, method, qa)
            self.note_pending(row, split, method)
            print(f'Pending generation failure: {row["id"]} ({method}); evidence: {path}', flush=True)

    def execute(self):
        try:
            return super().execute()
        finally:
            self.sync_status()


if __name__ == '__main__':
    original.Worker = PendingWorker
    original.main()
