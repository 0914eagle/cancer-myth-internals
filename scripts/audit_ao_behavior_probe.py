"""Complete the registered CREPE behavior-query retrospective comparison."""
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_ao_behavior_query_probe import DIRECTORY,export
from scripts.reader_gepa_io import save
from scripts.supervise_reader_gepa_pilot import retry_decision


def main():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    lock=(DIRECTORY/'analysis_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=DIRECTORY/'analysis_status.json'
    try:
        started=time.monotonic()
        save(status,dict(state='running',stage='waiting_for_fixed50_outputs',updated_utc=time.time()))
        while True:
            p=DIRECTORY/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
            if s.get('state')=='complete':break
            if s.get('state')=='failed':raise RuntimeError(s.get('error'))
            if time.monotonic()-started>2100:raise TimeoutError('Waiting for behavior generation exceeded2100seconds')
            time.sleep(3)
        failures=0;progress=0
        while True:
            save(status,dict(state='running',stage='retrospective_content_audit',updated_utc=time.time()))
            result=subprocess.run([sys.executable,'-u','scripts/audit_reader_gepa_trajectories.py',
                '--input-root',str(DIRECTORY),'--behavior-comparison'],cwd=ROOT)
            if result.returncode==0:break
            path=DIRECTORY/'trajectory_audit_status.json'
            error=json.loads(path.read_text()).get('error','Unstructured audit failure') if path.exists() else 'Unstructured audit failure'
            current=max((p.stat().st_mtime_ns for p in (DIRECTORY/'audit_calls').glob('*.json')),default=0)
            retry,failures,kind=retry_decision(error,failures,current>progress);progress=current
            if not retry:raise RuntimeError(error)
            deadline=time.time()+600
            save(status,dict(state='waiting_provider',error=error,failure_kind=kind,next_retry_utc=deadline,updated_utc=time.time()))
            while time.time()<deadline:time.sleep(5)
        records=[json.loads(p.read_text()) for p in (DIRECTORY/'trajectory_audits').glob('*.json')]
        assert len(records)==50 and all(r['valid'] for r in records)
        results=[]
        for label in [1,0]:
            selected=[r for r in records if r['label']==label];assert len(selected)==25
            for key,query in [('seed','stance_query'),('current','behavior_query')]:
                views=[r['audit'][key] for r in selected]
                assessed=[v for v in views if v['behavior_matches_answer'] is not None]
                results.append(dict(question_type='FPQ' if label else 'NFP',query=query,n=25,
                    predicted_behavior_counts=dict(Counter(v['predicted_response_behavior'] for v in views)),
                    actual_behavior_counts=dict(Counter(v['actual_response_behavior'] for v in views)),
                    behavior_agreement_count=sum(v['behavior_matches_answer'] is True for v in assessed),
                    behavior_agreement_denominator=len(assessed),
                    specific_target_error_identified=sum(v['reader_specific_target_error_identified'] for v in views),
                    useful_information_absent_from_answer=sum(v['new_information_beyond_answer'] for v in views)))
        summary=dict(questions=50,results=results,annotation_alignment=dict(Counter(r['audit']['annotation_alignment'] for r in records)),
            limits=['Exploratory GPT judgments with the actual answer and reference visible.',
                'AO outputs were produced without question text, answer, reference, or labels as reader text input.',
                'Same previously examined train50; this query was proposed after viewing earlier outputs.',
                'Agreement does not prove source beliefs, causal influence, or added value for GEPA.'])
        save(DIRECTORY/'comparison_summary.json',summary)
        raw=json.loads((DIRECTORY/'query_outputs.json').read_text());target=export(raw)
        save(target/'comparison_summary.json',summary)
        (target/'content_audits.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        save(target/'audit_protocol.json',json.loads((DIRECTORY/'trajectory_audit_protocol.json').read_text()))
        for name in ['audit_ao_behavior_probe.py','audit_reader_gepa_trajectories.py']:
            (target/'code'/name).write_text((ROOT/'scripts'/name).read_text())
        save(status,dict(state='complete',questions=50,export=str(target),updated_utc=time.time()))
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception as error:
        save(status,dict(state='attention_required',error=repr(error),updated_utc=time.time()));raise


if __name__=='__main__':main()
