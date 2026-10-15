"""All Cancer-Myth/CREPE conditions before any general QA; reuse frozen caches."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_paper_baselines as original
from scripts.run_paper_baselines_mmlu_resilient import MMLUWorker
from src.paper_baselines import CachedTask, Pipeline, sha, write_json


def schedule(splits, methods):
    for qa in (False, True):
        for method in methods:
            for split in splits:
                yield qa, split, method


class FPQAFirstWorker(MMLUWorker):
    def __init__(self, args):
        super().__init__(args)
        write_json(self.out / 'execution_policy_fpqa_first.json', {
            'policy': 'all_fpqa_methods_before_general_qa_v1',
            'runner': str(Path(__file__).resolve()), 'runner_sha256': sha(__file__),
            'evaluation_identity_unchanged': True, 'generation_settings_unchanged': True,
            'order': 'All methods across Cancer-Myth folds and CREPE, then all QA',
            'failure_policy': 'Inherited length and recoverable MMLU pending handling',
        }, immutable=True)

    def execute(self):
        if self.args.command != 'run':
            return super().execute()
        calls = CachedTask(self.task, self.out, self.identity, self.reg['transport_common'])
        self.pipeline = Pipeline(self.reg, calls, self.bridge)
        for split in self.bundle['splits']:
            row = self.lookup[split['smoke'][0]]
            self.bridge(op='render', family='prewome', template='PresuppositionExtractionTemplate',
                kwargs={'question': row['question'], 'passages': [],
                        'few_shot_data': self.reg['fixed_examples'][split['dataset']]['examples']})
            self.bridge(op='judge', row=row, answer='Preflight placeholder')
        try:
            for qa, split, method in schedule(self.bundle['splits'], self.methods):
                key = f"{split['dataset']}/{split['name']}/{method}"
                print(f"Stage {'QA' if qa else 'FPQA'}: {key}", flush=True)
                try:
                    if (self.out / 'STOP').exists():
                        raise RuntimeError('STOP requested')
                    gate = self.classifier(split, method) if method in ('tfidf_gate', 'probe_gate') else None
                    candidate = self.optimize(split, method) if method.startswith('gepa') else None
                    rows = self.bundle['qa'] if qa else [self.lookup[i] for i in split['test']]
                    for row in rows:
                        self.process(row, split, method, candidate, gate, qa=qa)
                    self.status[key] = {'state': 'complete' if qa else 'fpqa_complete',
                        'n_fpqa': len(split['test']), 'n_qa': len(rows) if qa else 0,
                        'fpqa_state': 'partial' if self.pending.get(key) else 'complete'}
                    write_json(self.out / 'status.json', self.status)
                    self.sync_status()
                except Exception as error:
                    self.status[key] = {**self.status.get(key, {}), 'state': 'failed',
                        'failed_stage': 'qa' if qa else 'fpqa', 'error': str(error)}
                    write_json(self.out / 'status.json', self.status)
                    raise
            self.status['table3'] = {'state': 'generated' if self.bundle['qa'] else 'pending',
                'reason': self.bundle['audit']['qa_status']}
            write_json(self.out / 'status.json', self.status)
        finally:
            self.sync_status()


if __name__ == '__main__':
    original.Worker = FPQAFirstWorker
    original.main()
