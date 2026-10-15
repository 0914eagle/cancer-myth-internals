"""Keep independent FPQA generation moving while Sonnet scoring is unavailable."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_paper_baselines as original
from scripts import run_paper_baselines_fpqa_first as first
from src.paper_baselines import sha,write_json


def generation_schedule(splits,methods):
    for method in methods:
        if method.startswith('gepa'):
            raise ValueError('GEPA requires the unavailable reflection provider')
        for split in splits:yield False,split,method


class GenerationOnlyWorker(first.FPQAFirstWorker):
    def __init__(self,args):
        if args.command!='run' or not args.generate_only:
            raise ValueError('This queue requires run --generate-only')
        super().__init__(args)
        if any(m.startswith('gepa') for m in self.methods):
            raise ValueError('Select only provider-independent methods')
        write_json(self.out/'execution_policy_generation_only.json',dict(
            policy='fpqa_generation_before_sonnet_scoring_v1',runner=str(Path(__file__).resolve()),
            runner_sha256=sha(__file__),evaluation_identity_unchanged=True,
            generation_settings_unchanged=True,existing_scores_retained=True,
            methods=self.methods,qa='Continue through original queue after provider recovery'),immutable=True)

    def sync_status(self):
        for key,item in self.status.items():
            if item.get('state')=='fpqa_complete':
                item.update(state='answers_generated',well_scoring='pending; existing scores retained')
        if self.status.get('table3',{}).get('state')=='generated':
            self.status['table3']={'state':'pending','reason':'This queue generates FPQA only; cached QA retained'}
        super().sync_status()
        if self.status:write_json(self.out/'status.json',self.status)

    def execute(self):
        try:return super().execute()
        finally:self.sync_status()


if __name__=='__main__':
    first.schedule=generation_schedule
    original.Worker=GenerationOnlyWorker
    original.main()
