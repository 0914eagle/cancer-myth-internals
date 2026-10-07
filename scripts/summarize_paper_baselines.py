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


def aggregate(out,bundle,config):
    tables={1:[],2:[],3:[]}
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
                for s in splits:
                    for p in (root/'table12'/dataset/s['name']/method).glob('*.json'):
                        r=json.loads(p.read_text())
                        if r['id'] not in s['test']: raise ValueError('Non-test record in Table 1/2')
                        records.append(r)
                if len({r['id'] for r in records})!=len(records): raise ValueError('Duplicate OOF records')
                lookup={r['id']:r for r in bundle['rows']}
                item={'model':model,'dataset':dataset,'method':method,'expected':len(expected),
                      'generated':len(records),'scored':sum('rating' in r for r in records)}
                for y,name in ((1,'FPQ'),(0,'NFP')):
                    den=sum(lookup[i]['label']==y for i in expected)
                    scored=[r for r in records if r['label']==y and 'rating' in r]
                    item[name+'_n']=den;item[name+'_scored']=len(scored)
                    # Partial runs never masquerade as complete performance percentages.
                    item[name+'_Well_ge4']=100*sum(r['rating']>=4 for r in scored)/den if len(scored)==den and den else None
                item['state']='complete' if item['scored']==len(expected) else 'pending' if not records else 'partial'
                tables[2].append(item)
                if method in DETECTORS:
                    d={k:item[k] for k in ('model','dataset','method','expected','generated')}
                    d['state']='complete' if len(records)==len(expected) else 'partial' if records else 'pending'
                    for y,name in ((1,'TPR'),(0,'FPR')):
                        den=sum(lookup[i]['label']==y for i in expected)
                        rr=[r for r in records if r['label']==y]
                        d[name]=100*sum(r.get('verdict')==1 for r in rr)/den if len(rr)==den and den else None
                        d[name+'_invalid']=sum(r.get('verdict') is None for r in rr)
                    d['AUROC']=None
                    if len(records)==len(expected) and records and all('probability' in r for r in records):
                        from sklearn.metrics import roc_auc_score
                        d['AUROC']=roc_auc_score([r['label'] for r in records],[r['probability'] for r in records])
                    tables[1].append(d)
        # Both adaptation sources evaluated on every native QA dataset.
        for source in ('cancer_myth','crepe'):
            splits=[s for s in bundle['splits'] if s['dataset']==source]
            for dataset in config['qa_datasets']:
                qa=[r for r in bundle['qa'] if r['dataset']==dataset]
                for method in config['methods']:
                    scores=[];n_completed=0
                    for split in splits:
                        folder_source=source;name=split['name']
                        if method in FIXED:
                            name='fixed';folder_source=source if method in ('prewome','extract_verify') else 'shared'
                        records=[]
                        for row in qa:
                            p=root/'table3'/folder_source/name/method/(digest([row['dataset'],row['id']])+'.json')
                            if p.exists():records.append(json.loads(p.read_text()))
                        n_completed+=len(records)
                        if qa and len(records)==len(qa): scores.append(100*sum(r['correct'] for r in records)/len(qa))
                    complete=bool(qa) and len(scores)==len(splits)
                    tables[3].append({'model':model,'source':source,'dataset':dataset,'method':method,
                        'expected_questions':len(qa),'systems':len(splits),'completed_system_questions':n_completed,
                        'accuracy':sum(scores)/len(scores) if complete else None,
                        'per_system_accuracy':scores,'state':'complete' if complete else 'pending'})
    destination=out/'tables'
    for number,rows in tables.items():
        write_json(destination/f'table{number}.json',rows)
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (destination/f'table{number}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(destination/'provenance.json',{'protocol':config['protocol_id'],'bundle':bundle['content_sha256'],
        'audit':bundle['audit'],'historical_values_reused':False,
        'note':'Table1 TPR/FPR use full eligible denominators; invalid counted separately. QA Cancer mean is across 3 systems, not 3x independent questions.'})
    return tables


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_baselines/local_4090.json')
    a=p.parse_args(); aggregate(a.out,check_bundle(a.bundle),json.loads(a.config.read_text()))
    print(a.out/'tables')
