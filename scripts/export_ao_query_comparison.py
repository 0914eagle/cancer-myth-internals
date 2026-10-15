"""Export the fixed-activation query comparison with actual answer provenance."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import OUT,P
from scripts.reader_gepa_io import save


def main():
    source=OUT/'ao_query_probe'
    target=ROOT.parents[1]/'results/ao_query_comparison_20261014'
    raw=json.loads((source/'query_outputs.json').read_text())
    assert len(raw)==100 and len({r['row_id'] for r in raw})==100
    audits={r['row_id']:r for r in (json.loads(p.read_text()) for p in (source/'trajectory_audits').glob('*.json'))}
    rows=[]
    for r in raw:
        n=json.loads((P/'native/results'/(r['source_native_id']+'.json')).read_text())
        assert r['stance']['source_activation_path']==r['source_activation_path']==n['ao_activation_path']
        assert r['stance']['source_text_provided'] is False
        assert r['neutral']['bytewise_verified'] is True
        judgment=P/'judgments'/(n['id']+'.json')
        judge=json.loads(judgment.read_text()) if judgment.exists() else {}
        rows.append(dict(dataset=r['dataset'],question_id=r['row_id'],question_type='FPQ' if r['label'] else 'NFP',
            question=r['question'],neutral_text=r['neutral']['explanation'],stance_text=r['stance']['explanation'],
            actual_answer=r['actual_answer'],source_prompt=n.get('candidate_prompt'),well_score=judge.get('score'),
            well_reason=judge.get('reason'),source_native_id=n['id'],
            activation_sha256=hashlib.sha256(Path(r['source_activation_path']).read_bytes()).hexdigest(),
            source_layer_zero_based=31,token_window_size=len(n['ao_source_token_ids']),
            neutral_original_output_id=r['neutral']['original_output_id'],stance_output_id=r['stance']['id'],
            exact_activation_match_verified=True,source_text_given_to_ao=False,audit=audits.get(r['row_id'])))
    for ds in ['crepe','cancer_myth']:
        assert sum(r['dataset']==ds and r['question_type']=='FPQ' for r in rows)==25
        assert sum(r['dataset']==ds and r['question_type']=='NFP' for r in rows)==25
    target.mkdir(parents=True,exist_ok=True)
    for name,selected in [('all100',rows),('crepe50',[r for r in rows if r['dataset']=='crepe'])]:
        (target/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        fields=[k for k in selected[0] if k!='audit']
        with (target/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(selected)
    for name in ['protocol.json','comparison_summary.json','trajectory_audit_protocol.json']:
        if (source/name).exists():shutil.copyfile(source/name,target/name)
    save(target/'snapshot.json',dict(questions=100,crepe_questions=50,audited=len(audits),
        source=str(source),source_generation=False,
        caveats=['Same saved Plain activations and same actual answer; only AO query changes.',
            'Any GPT audit is retrospective, has answer/reference access, and is not human verification.',
            'AO attribution of acceptance/rejection is not proof of source belief or causal influence.']))
    for name in ['run_ao_stance_query_probe.py','run_ao_query_probe_now.py','audit_reader_gepa_trajectories.py',
                 'summarize_ao_query_probe.py','export_ao_query_comparison.py']:
        code=target/'code';code.mkdir(exist_ok=True);shutil.copyfile(ROOT/'scripts'/name,code/name)
    print(json.dumps(dict(output=str(target),questions=len(rows),audited=len(audits))))


if __name__=='__main__':main()
