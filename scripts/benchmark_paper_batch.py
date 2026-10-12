"""Small train-only batch validation on an explicitly assigned spare GPU."""
import argparse,json,os,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paper_baselines import write_json
from scripts.paper_batch_inference import BatchedHFTask
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--bundle',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default='qwen25');p.add_argument('--gpu',default='3');p.add_argument('--n',type=int,default=4);a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
cfg=json.load(open(a.config));bundle=json.load(open(a.bundle));reg=json.load(open(cfg['registry']))
lookup={r['id']:r for r in bundle['rows']};ids=bundle['splits'][0]['train'][:a.n]
messages=[[{'role':'system','content':reg['transport_common']+'\n\n'+reg['prompts']['plain']},{'role':'user','content':lookup[i]['question']}] for i in ids]
b=BatchedHFTask(cfg,a.model)
# Compare the original singleton transport against actual padded batch transport.
t=time.monotonic();single=[b(m) for m in messages];single_seconds=time.monotonic()-t
print('single finished',single_seconds,flush=True)
t=time.monotonic();batch=b.batch(messages);batch_seconds=time.monotonic()-t
result={'ids':ids,'split':'train_only','model':a.model,'n':a.n,'single_seconds':single_seconds,'batch_seconds':batch_seconds,'speedup':single_seconds/batch_seconds,'equal_text':[not isinstance(y,Exception) and x[0]==y[0] for x,y in zip(single,batch)],'single':single,'batch':[str(x) if isinstance(x,Exception) else x for x in batch]}
write_json(a.out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('single','batch')}),flush=True)
