"""Read-only AR timing on completed AV outputs, including question-text control."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.nla_connection import medical_package,snapshot
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
p.add_argument('--model',required=True);p.add_argument('--gpu',type=int,required=True)
p.add_argument('--samples',type=int,default=128)
a=p.parse_args();c=json.loads(a.config.read_text());os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
import torch
package=medical_package(c['medical_nla_root']);helpers=importlib.import_module(package+'.reconstruction_scoring')
critic_cls=helpers.load_nla_critic_class(str(ROOT/'external/nla-inference-official'))
out=Path(c['artifact_root'])/a.model
rows=[]
with (out/'av_results.jsonl').open() as file:
    for line in file:
        row=json.loads(line)
        # Cover an initial block of all position families and both conditions.
        rows.append(row)
        if len(rows)==a.samples:break
began=time.perf_counter()
critic=critic_cls(snapshot(c['cache_root'],c['models'][a.model]['ar']),device='cuda:0',dtype=torch.bfloat16)
torch.cuda.synchronize();load_seconds=time.perf_counter()-began
activation=helpers.load_activation(rows[0]['activation_path'])
critic.score(rows[0]['nla_output'],activation);critic.score(rows[0]['prompt'],activation)
torch.cuda.synchronize();started=time.perf_counter()
with (out/'ar_benchmark_scores.jsonl').open('w') as file:
    for row in rows:
        activation=helpers.load_activation(row['activation_path'])
        mse,cos=critic.score(row['nla_output'],activation)
        control_mse,control_cos=critic.score(row['prompt'],activation)
        helpers.validate_mse(float(mse),row['id']);helpers.validate_mse(float(control_mse),row['id'])
        file.write(json.dumps(dict(id=row['id'],recon_mse=mse,recon_cos=cos,
            question_text_control_mse=control_mse,question_text_control_cos=control_cos))+'\n')
torch.cuda.synchronize();seconds=time.perf_counter()-started
total=json.loads((out/'generation_status.json').read_text())['vectors']
result=dict(model=a.model,physical_gpu=a.gpu,samples=len(rows),load_seconds=load_seconds,
    measured_seconds=seconds,seconds_per_vector_including_control=seconds/len(rows),
    total_vectors=total,estimated_full_ar_hours=(load_seconds+seconds/len(rows)*total)/3600,
    caveat='Small initial-block timing estimate; throughput varies with explanation length; AV runs concurrently')
(out/'ar_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
