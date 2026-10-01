"""Recover task-schema duplicates and two literal quote defects, without new judgments."""
import json
from pathlib import Path
import time
from scripts import audit_correction_boundaries_20261001 as audit
from src.pilot import output_lock

RUN=audit.ROOT/'results/audits/correction_boundaries_20261001_v1'
QUOTE_FIXES={
    'item_0455': 'Pleuropulmonary blastoma (PPB) is an extremely rare primary lung tumor that occurs almost exclusively in children and adolescents.',
    'item_0464': 'Chronic Myeloid Leukemia (CML) is actually very rare in pediatric patients and is primarily a disease of adults.',
}


def main():
    with output_lock(RUN):
        jobs=json.loads((RUN/'jobs.json').read_text())
        proposed=[]
        for job in jobs:
            folder=RUN/'records'/job['job'];p=folder/'record.json'
            rec=json.loads(p.read_text())
            if rec['status']=='complete':continue
            assert rec['status']=='failed'
            assert len(job['items'])==1
            inp=job['items'][0]
            field='relation' if inp['task']=='target_alignment' else 'stance'
            original=json.loads((folder/'answer.txt').read_text())
            matching=[r for r in original['items'] if r['key']==inp['key'] and field in r]
            assert len(matching)==1, 'Do not choose between conflicting judgments'
            chosen=None
            for name in ['answer.txt','quote_repair.txt']:
                source=folder/name
                if not source.exists():continue
                value=json.loads(source.read_text())
                rows=[r for r in value['items'] if r['key']==inp['key'] and field in r]
                assert len(rows)==1
                candidate=dict(rows[0])
                kind='select_assigned_task_schema'
                if job['job'] in QUOTE_FIXES:
                    candidate['answer_quote']=QUOTE_FIXES[job['job']]
                    kind='restore_parenthesis_and_full_name_in_verbatim_quote'
                strip=lambda r:{k:v for k,v in r.items() if k not in {'question_quote','answer_quote','later_quote'}}
                assert strip(candidate)==strip(matching[0]), 'Judgment or explanation changed'
                try: result=audit.validate(json.dumps({'items':[candidate]}),job['items'])
                except AssertionError:continue
                chosen=(name,kind,result);break
            assert chosen is not None,job['job']
            proposed.append((p,rec,chosen))
        assert len(proposed) in (0,28), 'Unexpected recovery cohort; inspect before mutation'
        archive=RUN/'final_validation_recovery';archive.mkdir(exist_ok=True)
        changes=[]
        for p,rec,(name,kind,result) in proposed:
            backup=archive/(rec['job']+'.json');assert not backup.exists()
            audit.tr.write(backup,rec,frozen=True)
            record=dict(rec,status='complete',result=result,
                validation_recovery=dict(source=name,kind=kind,original_failure=rec.get('error'),
                    judgment_unchanged=True,recovered_unix=time.time()))
            record.pop('error',None);audit.tr.write(p,record)
            changes.append(dict(job=rec['job'],kind=kind,source=name,judgment_unchanged=True))
        if changes:
            audit.tr.write(archive/'manifest.json',dict(n=28,new_model_calls=0,changes=changes,
                script_sha256=audit.tr.sha(Path(__file__))),frozen=True)
        records=[json.loads((RUN/'records'/j['job']/'record.json').read_text()) for j in jobs]
        assert len(records)==598 and all(r['status']=='complete' for r in records)
        for j,r in zip(jobs,records):audit.validate(json.dumps({'items':r['result']}),j['items'])
        audit.tr.write(RUN/'status.json',dict(state='complete',expected=598,complete=598,failed=0,
            final_validation_recoveries=28,updated_unix=time.time()))
    from scripts.report_correction_boundaries_20261001 import main as report
    report()


if __name__=='__main__':main()
