"""Aggregate retrospective GPT audits without treating repeats as new questions."""
from collections import Counter
import json
from pathlib import Path

P = Path(__file__).resolve().parents[1] / 'results/nla_ao_parallel_20261012/gepa_reader_pilot/cpu_analysis_20261013'
sources = {r['id']:r for r in json.loads((P/'reader_trajectories.json').read_text())}
records = [json.loads(f.read_text()) for f in (P/'trajectory_audits').glob('*.json')]
assert all(r['valid'] for r in records)
results=[]
for dataset in ['cancer_myth','crepe','all']:
    for reader in ['nla','ao']:
        for label in [1,0]:
            group=[r for r in records if (dataset=='all' or r['dataset']==dataset) and r['reader']==reader and r['label']==label]
            if not group:continue
            seed=[r['audit']['seed'] for r in group]
            current=[r['audit']['current'] for r in group]
            results.append(dict(dataset=dataset,reader=reader,label=label,interpretation_records=len(group),
                unique_questions=len({r['row_id'] for r in group}),
                annotation_alignment=dict(Counter(r['audit']['annotation_alignment'] for r in group)),
                seed_topic_matches=sum(r['reader_topic_match']=='yes' for r in seed),
                current_topic_matches=sum(r['reader_topic_match']=='yes' for r in current),
                seed_specific_target_recognitions=sum(r['reader_specific_target_error_identified'] for r in seed),
                current_specific_target_recognitions=sum(r['reader_specific_target_error_identified'] for r in current),
                recognition_appeared=sum(not a['reader_specific_target_error_identified'] and b['reader_specific_target_error_identified'] for a,b in zip(seed,current)),
                recognition_disappeared=sum(a['reader_specific_target_error_identified'] and not b['reader_specific_target_error_identified'] for a,b in zip(seed,current)),
                seed_correction_intent=sum(r['reader_explicit_correction_intent'] for r in seed),
                current_correction_intent=sum(r['reader_explicit_correction_intent'] for r in current),
                current_generic_correction_language=sum(r['reader_generic_correction_language'] for r in current),
                current_spurious_nfp_correction=sum(r['reader_spurious_correction_on_nfp'] for r in current),
                answer_specific_target_recognitions=sum(r['answer_specific_target_error_identified'] for r in current),
                answer_spurious_nfp_correction=sum(r['answer_spurious_correction_on_nfp'] for r in current),
                added_information_beyond_answer=sum(r['new_information_beyond_answer'] for r in current),
                mismatch=dict(Counter(r['mismatch'] for r in current))))
report=dict(completed_records=len(records),expected_records=59,results=results,
    limitations=['Exploratory GPT judgment with answers visible; not blind behavioral prediction.',
        'Counts are question-candidate records, not independent questions; unique_question denominators are reported.',
        'Sampled training feedback varies by candidate and search branch; no full-population or monotonic improvement claim.',
        'Annotation alignment refers jointly to presupposition and reference fields; inspect reasons to distinguish which field conflicts.',
        'A missing correction signal is not evidence of a normality decision. A positive signal is not proof of source belief.'])
(P/'trajectory_audit_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
interesting=[]
for r in records:
    current=r['audit']['current'];seed=r['audit']['seed']
    if current['reader_specific_target_error_identified'] or current['reader_spurious_correction_on_nfp'] or current['new_information_beyond_answer'] or current['reader_generic_correction_language']:
        interesting.append(dict(source=sources[r['id']],judgment=r,
            tags=[tag for tag,condition in [
                ('specific_target',current['reader_specific_target_error_identified']),
                ('new_recognition',current['reader_specific_target_error_identified'] and not seed['reader_specific_target_error_identified']),
                ('spurious_nfp',current['reader_spurious_correction_on_nfp']),
                ('beyond_answer',current['new_information_beyond_answer']),
                ('generic_correction_language',current['reader_generic_correction_language'])] if condition]))
(P/'interesting_cases.json').write_text(json.dumps(interesting,ensure_ascii=False,indent=2)+'\n')
print('Completed',len(records),'of59')
for r in results:
    if r['dataset']=='all':print(json.dumps(r,ensure_ascii=False))
