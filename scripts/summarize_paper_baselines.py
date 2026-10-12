"""Export new-protocol Table 1/2/3 CSV/JSON. Never overwrite historical tables."""
import argparse
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import DETECTORS,FIXED,digest,write_json
from scripts.run_paper_baselines import check_bundle


def aggregate(out,bundle,config,judgments_out=None):
    tables={1:[],2:[],3:[]}
    lookup={r['id']:r for r in bundle['rows']}
    if config.get('final_judge') and judgments_out is None:
        raise ValueError('New protocol requires explicit final-judge ledger; no legacy score fallback')
    for model in config['models']:
        root=out/model
        identities=[]
        identity_path=root/'identity.json'
        if identity_path.exists(): identities.append(json.loads(identity_path.read_text()))
        if identities and identities[0]['bundle_sha256']!=bundle['content_sha256']:
            raise ValueError('Output/bundle mismatch')
        for dataset in ('cancer_myth','crepe'):
            splits=[s for s in bundle['splits'] if s['dataset']==dataset]
            expected={i for s in splits for i in s['test']}
            for method in config['methods']:
                records=[]
                failures=[]
                for s in splits:
                    for p in (root/'table12'/dataset/s['name']/method).glob('*.json'):
                        r=json.loads(p.read_text())
                        if r['id'] not in s['test']: raise ValueError('Non-test record in Table 1/2')
                        if judgments_out is not None:
                            for key in ('rating','judge_reason','judge_config'):r.pop(key,None)
                            grade=judgments_out/p.relative_to(out)
                            if grade.exists():
                                from scripts.judge_paper_queue import evaluation_key
                                g=json.loads(grade.read_text())
                                if g.get('evaluation_key')==evaluation_key(r,config['final_judge'],lookup[r['id']]):
                                    r.update({k:g[k] for k in ('rating','judge_reason','judge_config')})
                        records.append(r)
                    for p in (root/'failures/table12'/dataset/s['name']/method).glob('*.json'):
                        r=json.loads(p.read_text())
                        if r['id'] not in s['test']: raise ValueError('Non-test failure in Table 1/2')
                        failures.append(r)
                if len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate OOF records')
                if len({r['id'] for r in records+failures})!=len(records)+len(failures):
                    raise ValueError('Duplicate/overlapping result and failure IDs')
                lookup={r['id']:r for r in bundle['rows']}
                item={'model':model,'dataset':dataset,'method':method,'expected':len(expected),
                      'generated':len(records),'scored':sum('rating' in r for r in records),
                      'execution_failures':len(failures)}
                for y,name in ((1,'FPQ'),(0,'NFP')):
                    den=sum(lookup[i]['label']==y for i in expected)
                    scored=[r for r in records if r['label']==y and 'rating' in r]
                    item[name+'_n']=den;item[name+'_scored']=len(scored)
                    failed=sum(r['label']==y for r in failures)
                    item[name+'_execution_failures']=failed
                    # Partial runs never masquerade as complete performance percentages.
                    item[name+'_Well_ge4']=100*sum(r['rating']>=4 for r in scored)/den if len(scored)+failed==den and den else None
                complete=item['scored']+len(failures)==len(expected)
                item['state']=('complete_with_errors' if failures else 'complete') if complete else 'pending' if not records and not failures else 'partial'
                tables[2].append(item)
                if method in DETECTORS:
                    detection_records=records
                    if config.get('response_protocol')=='fresh_response_with_saved_detection_v1':
                        detection_records=[]
                        for split in splits:
                            for path in (root/'table1'/dataset/split['name']/method).glob('*.json'):
                                record=json.loads(path.read_text())
                                if record['id'] not in split['test']:raise ValueError('Non-test detection')
                                detection_records.append(record)
                        if len({r['id'] for r in detection_records})!=len(detection_records):raise ValueError('Duplicate detections')
                    d={k:item[k] for k in ('model','dataset','method','expected')}
                    d['generated']=len(detection_records)
                    d['missing_predictions']=len(expected)-len(detection_records)
                    d['state']='complete' if len(detection_records)==len(expected) else 'partial' if detection_records else 'pending'
                    for y,name in ((1,'TPR'),(0,'FPR')):
                        den=sum(lookup[i]['label']==y for i in expected)
                        rr=[r for r in detection_records if r['label']==y]
                        d[name]=100*sum(r.get('verdict')==1 for r in rr)/den if len(rr)==den and den else None
                        d[name+'_invalid']=sum(r.get('verdict') is None for r in rr)
                    d['AUROC']=None
                    if len(detection_records)==len(expected) and detection_records and all('probability' in r for r in detection_records):
                        from sklearn.metrics import roc_auc_score
                        d['AUROC']=roc_auc_score([r['label'] for r in detection_records],[r['probability'] for r in detection_records])
                    tables[1].append(d)
        # Both adaptation sources evaluated on every native QA dataset.
        for source in ('cancer_myth','crepe'):
            splits=[s for s in bundle['splits'] if s['dataset']==source]
            if config.get('qa_cancer_system') and source=='cancer_myth':splits=[s for s in splits if s['name']=='fold0']
            for dataset in config['qa_datasets']:
                qa=[r for r in bundle['qa'] if r['dataset']==dataset]
                for method in config['methods']:
                    scores=[];n_completed=0;n_failures=0;unique_failures=set()
                    for split in splits:
                        folder_source=source;name=split['name']
                        if method in FIXED:
                            name='fixed';folder_source=source if method in ('prewome','extract_verify') else 'shared'
                        records=[]
                        failures=[]
                        for row in qa:
                            p=root/'table3'/folder_source/name/method/(digest([row['dataset'],row['id']])+'.json')
                            failure=root/'failures/table3'/folder_source/name/method/p.name
                            if p.exists() and failure.exists(): raise ValueError('QA answer/failure overlap')
                            if p.exists():records.append(json.loads(p.read_text()))
                            elif failure.exists():
                                failures.append(json.loads(failure.read_text()))
                                unique_failures.add(str(failure))
                        n_completed+=len(records)
                        n_failures+=len(failures)
                        if qa and len(records)+len(failures)==len(qa): scores.append(100*sum(r['correct'] for r in records)/len(qa))
                    complete=bool(qa) and bool(splits) and len(scores)==len(splits)
                    tables[3].append({'model':model,'source':source,'dataset':dataset,'method':method,
                        'expected_questions':len(qa),'systems':len(splits),'completed_system_questions':n_completed,
                        'failed_system_questions':n_failures,'unique_failed_generations':len(unique_failures),
                        'accuracy':sum(scores)/len(scores) if complete else None,
                        'per_system_accuracy':scores,'state':('complete_with_errors' if n_failures else 'complete') if complete else 'pending'})
    for rows in tables.values():
        for row in rows:
            if row['model']=='luna6' and row['method']=='probe_gate':row['state']='not_applicable'
    destination=out/'tables'
    for number,rows in tables.items():
        write_json(destination/f'table{number}.json',rows)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (destination/f'table{number}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(destination/'provenance.json',{'protocol':config['protocol_id'],'bundle':bundle['content_sha256'],
        'audit':bundle['audit'],'historical_values_reused':(out/'reuse_audit.json').exists(),'final_judge':config.get('final_judge'),'judgments_out':str(judgments_out) if judgments_out else None,'qa_cancer_system':config.get('qa_cancer_system','mean across three systems'),
        'note':'Table1 suppresses metrics with missing generation predictions. Table2/3 retain failed generations in full denominators (unsuccessful, no invented Well score). Failures reported separately. QA system count is reported explicitly. Independent detections are not held back by answer judging.'})
    return tables


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_baselines/local_4090.json')
    p.add_argument('--judgments-out',type=Path)
    a=p.parse_args(); aggregate(a.out,check_bundle(a.bundle),json.loads(a.config.read_text()),a.judgments_out)
    print(a.out/'tables')
