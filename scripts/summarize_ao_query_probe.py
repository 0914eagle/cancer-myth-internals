"""Summarize fixed-activation AO query comparisons; exploratory GPT labels only."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import OUT
from scripts.reader_gepa_io import save

p=OUT/'ao_query_probe'
records=[json.loads(f.read_text()) for f in (p/'trajectory_audits').glob('*.json')]
assert len(records)==100 and all(r['valid'] for r in records)
def summarize(selected):
    results=[]
    for label in [1,0]:
      for key,name in [('seed','neutral_query'),('current','source_stance_query')]:
        views=[r['audit'][key] for r in selected if r['label']==label]
        judged=[v for v in views if v['stance_matches_answer'] is not None]
        results.append(dict(question_type='FPQ' if label else 'NFP',query=name,n=len(views),
            attributed_stances=dict(Counter(v['source_stance'] for v in views)),
            specific_target_error_identified=sum(v['reader_specific_target_error_identified'] for v in views),
            correction_intent=sum(v['reader_explicit_correction_intent'] for v in views),
            spurious_nfp_correction=sum(v['reader_spurious_correction_on_nfp'] for v in views),
            retrospective_stance_agreement_count=sum(v['stance_matches_answer'] is True for v in judged),
            retrospective_stance_agreement_denominator=len(judged),
            added_information_beyond_answer=sum(v['new_information_beyond_answer'] for v in views)))
    return results
results=summarize(records)
save(p/'comparison_summary.json',dict(results=results,
    by_dataset={dataset:summarize([r for r in records if r['dataset']==dataset])
                for dataset in sorted({r['dataset'] for r in records})},
    limits='Exploratory GPT content labels; same100 train questions, exact same activation and source answer. This is retrospective agreement, not blind prediction, a source-belief measurement or heldout performance.'))
print(json.dumps(results,ensure_ascii=False))
from scripts.export_ao_query_comparison import main as export_comparison
export_comparison()
