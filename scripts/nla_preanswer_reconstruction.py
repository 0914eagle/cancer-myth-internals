"""Stream released AR scoring; avoid rewriting the growing output per vector."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.nla_connection import medical_package,snapshot

p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True)
a=p.parse_args();c=json.loads(a.config.read_text());os.environ['CUDA_VISIBLE_DEVICES']=str(c['models'][a.model]['gpu'])
import torch
from src.paper_baselines import write_json
out=Path(c['artifact_root'])/a.model
package=medical_package(c['medical_nla_root']);helpers=importlib.import_module(package+'.reconstruction_scoring')
critic_cls=helpers.load_nla_critic_class(str(ROOT/'external/nla-inference-official'))
critic=critic_cls(snapshot(c['cache_root'],c['models'][a.model]['ar']),device='cuda:0',dtype=torch.bfloat16)
manifest={r['id']:r for r in helpers.read_jsonl(out/'manifest.jsonl')}
answers={r['case_id']:r for r in helpers.read_jsonl(out/'answers.jsonl')}
saved=out/'scored.jsonl';done=set();total=parsed=0;sum_mse=sum_control=0.0
if saved.exists():
    for row in helpers.read_jsonl(saved):
        if row['id'] in done:raise RuntimeError('Duplicate AR score')
        done.add(row['id']);total+=1;parsed+=int(row['parsed_explanation_tag'])
        sum_mse+=row['recon_mse'];sum_control+=row['question_text_control_mse']
with saved.open('a') as stream:
    for line in (out/'av_results.jsonl').open():
        row=json.loads(line)
        if row['id'] in done:continue
        target=manifest[row['id']];activation=helpers.load_activation(row['activation_path'])
        mse,cos=critic.score(row['nla_output'],activation)
        control_mse,control_cos=critic.score(target['prompt'],activation)
        helpers.validate_mse(float(mse),row['id']);helpers.validate_mse(float(control_mse),row['id'])
        enriched={**row,**{k:target[k] for k in ['source','condition','category','split','partition','activation_mode','case_id','selected_token_text','answer_completed']},
            'recon_mse':float(mse),'recon_cos':float(cos),'question_text_control_mse':float(control_mse),
            'question_text_control_cos':float(control_cos),'interpretation':'direction fidelity, not explanation correctness'}
        stream.write(json.dumps(enriched,ensure_ascii=False)+'\n');stream.flush()
        done.add(row['id']);total+=1;parsed+=int(row['parsed_explanation_tag']);sum_mse+=mse;sum_control+=control_mse
        if total%100==0:
            write_json(out/'ar_status.json',dict(state='running',scored_vectors=total,total_vectors=len(manifest)))
            print('AR scored',total,'/',len(manifest),flush=True)
fields=['model','base_id','source','category','condition','position_family','selected_token_text','question','answer','answer_completed','nla_output','recon_mse','recon_cos','question_text_control_mse']
with (out/'comparison.csv').open('w') as file:
    writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
    for line in saved.open():
        row=json.loads(line);answer=answers[row['case_id']]
        row.update(model=a.model,question=answer['question'],answer=answer['answer'])
        writer.writerow({k:row.get(k) for k in fields})
write_json(out/'summary.json',dict(state='complete',n=total,parsed_explanations=parsed,
    mean_recon_mse=sum_mse/total,mean_question_control_mse=sum_control/total,
    scope=c['analysis_scope']))
