"""Qwen/Gemma matched Table 1/2/3 baseline worker; Ours is excluded.

preflight: no model calls; smoke: only frozen train IDs; run: complete evaluation.
All artifacts are immutable or content-addressed. STOP drains the current call.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import (FIXED,TRAINED,HFTask,CachedTask,Pipeline,WellBridge,digest,
    fit_gate,gate_predict,qa_question,qa_score,sha,summarize,write_json)


def check_bundle(path):
    b=json.loads(path.read_text()); expected=b.pop('content_sha256')
    if digest(b)!=expected: raise ValueError('Bundle hash mismatch')
    b['content_sha256']=expected
    return b


class Worker:
    def __init__(self,args):
        self.args=args; self.config=json.loads(args.config.read_text()); self.bundle=check_bundle(args.bundle)
        self.reg=json.loads((ROOT/self.config['registry']).read_text()); self.out=args.out/args.model
        self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'.lock').open('a'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.bridge=WellBridge(self.config); self.backend=None
        self.lookup={r['id']:r for r in self.bundle['rows']}
        self.methods=list(FIXED) if args.phase=='fixed' else list(TRAINED) if args.phase=='trained' else self.config['methods']
        if args.methods:
            self.methods=args.methods.split(',')
            if not set(self.methods)<=set(self.config['methods']): raise ValueError('Unknown/excluded method')
        self.external=json.loads((ROOT/self.config['external_config']).read_text())
        self.identity=dict(protocol=self.config['protocol_id'],model=self.config['models'][args.model],
            decoding={k:self.config[k] for k in ('dtype','max_new_tokens','max_input_tokens','seed','replicate')},
            registry_sha256=sha(ROOT/self.config['registry']),bundle_sha256=self.bundle['content_sha256'],
            full_config=self.config, external_config=self.external,
            well_source_hashes={str(p.relative_to(ROOT/self.config['well_root'])):sha(p)
                for p in sorted((ROOT/self.config['well_root']).rglob('*.py'))
                if '/response/' in str(p) or '/data_gen/template/' in str(p)},
            code_sha256={f:sha(ROOT/f) for f in ('src/paper_baselines.py','scripts/run_paper_baselines.py',
                                              'scripts/paper_well_bridge.py','src/fpqa_cli_backend.py','src/fpqa_well_pipelines.py','src/fpqa_prompting.py')},
            versions={k:importlib.metadata.version(k) for k in ('torch','transformers','scikit-learn','accelerate')})
        write_json(self.out/'identity.json',self.identity,immutable=True)
        self.status={}

    def task(self,messages):
        if self.backend is None: self.backend=HFTask(self.config,self.args.model)
        return self.backend(messages)

    def external_call(self,role,messages):
        from src.fpqa_cli_backend import CLIBackend
        identity={'role':role,'config':self.external[role+'_model'],'messages':messages,
                  'code':sha(ROOT/'src/fpqa_cli_backend.py')}
        path=self.out/'external_calls'/(digest(identity)+'.json')
        if path.exists(): return json.loads(path.read_text())['text']
        if (self.out/'STOP').exists(): raise RuntimeError('STOP requested')
        cfg={**self.external[role+'_model'],'exact_well_messages':True}
        backend=CLIBackend(cfg)
        # CLIBackend is a callable returning text and metadata.
        text,meta=backend(messages)
        write_json(path,{'request':identity,'text':text,'metadata':meta},immutable=True)
        return text

    def judge(self,row,answer):
        from src.fpqa_prompting import parse_rating
        messages=self.bridge(op='judge',row=row,answer=answer)
        text=self.external_call('judge',messages)
        rating=parse_rating(text)
        if rating is None: raise ValueError('Unparseable judge output')
        return rating,text

    def feature(self,question):
        import numpy as np
        path=self.out/'features'/(digest({'identity':self.identity,'q':question})+'.npz')
        if path.exists():
            with np.load(path) as f: return {k:f[k] for k in f.files}
        if self.backend is None: self.backend=HFTask(self.config,self.args.model)
        # Same unmodified question + Plain wrapper for all probe candidates.
        messages=[{'role':'system','content':self.reg['transport_common']+'\n\n'+self.reg['prompts']['plain']},
                  {'role':'user','content':question}]
        features=self.backend.features(messages); path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp.npz'); np.savez(temp,**features);temp.replace(path)
        return features

    def classifier(self,split,method):
        import joblib
        # Shared TF-IDF file and lock across models; probe remains model-specific.
        base=self.args.out/'shared' if method=='tfidf_gate' else self.out
        directory=base/'gates'/split['dataset']/split['name'];directory.mkdir(parents=True,exist_ok=True)
        path=directory/(method+'.joblib')
        with (directory/(method+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ident={'bundle':self.bundle['content_sha256'],'kind':method,
                   'implementation':sha(ROOT/'src/paper_baselines.py'),
                   'model':self.identity if method=='probe_gate' else None}
            if path.exists():
                artifact=joblib.load(path)
                if artifact['identity']!=ident: raise ValueError('Classifier cache mismatch')
            else:
                artifact=fit_gate([self.lookup[i] for i in split['train']],
                    [self.lookup[i] for i in split['dev']],method,self.feature)
                artifact['identity']=ident
                temp=path.with_suffix('.tmp');joblib.dump(artifact,temp);temp.replace(path)
                write_json(path.with_suffix('.json'),{k:v for k,v in artifact.items()
                            if k not in ('transform','classifier')},immutable=True)
        return lambda q:gate_predict(artifact,q,self.feature)

    def optimize(self,split,method):
        from gepa.optimize_anything import GEPAConfig,EngineConfig,ReflectionConfig,optimize_anything
        from transformers import AutoTokenizer
        cfg=self.config['gepa']; directory=self.out/'gepa'/split['dataset']/split['name']/method
        saved=directory/'selected.json'
        if saved.exists(): return json.loads(saved.read_text())['candidate']
        if not cfg['tokenizer_revision']: raise ValueError('Pin GEPA prompt tokenizer revision before optimization (prepare_paper_runtime.py)')
        tok=AutoTokenizer.from_pretrained(cfg['prompt_tokenizer'],revision=cfg['tokenizer_revision'],local_files_only=True,
                                         cache_dir=self.config['cache_root'])
        candidate={'answer':self.reg['prompts']['plain']} if method=='gepa' else {'detector':self.reg['prompts']['cot_detector']}
        if method=='gepa_both': candidate['supplement']=''
        def pairs(ids,limit=None):
            ys=[[self.lookup[i] for i in ids if self.lookup[i]['label']==y] for y in (1,0)]
            rng=random.Random(cfg['seed'])
            for x in ys:rng.shuffle(x)
            if limit:
                unique=[]
                for x in ys:
                    seen=set(); selected=[]
                    for row in x:
                        group=row.get('group_id',row['id'])
                        if group not in seen:
                            selected.append(row); seen.add(group)
                    unique.append(selected)
                ys=unique
            n=min(limit,min(map(len,ys))) if limit else max(map(len,ys))
            return [{'pair':[ys[0][i%len(ys[0])],ys[1][i%len(ys[1])]]} for i in range(n)]
        train=pairs(split['train']); dev=pairs(split['dev'],cfg['dev_per_class_max'])
        # One metric call = one FPQ + one normal; budget expressed below in question units.
        if cfg['metric_budget']%2: raise ValueError('Paired GEPA needs an even question budget')
        def evaluate(candidate,example):
            if set(candidate)!=set(seed_candidate): raise ValueError('Candidate components changed')
            if any(len(tok.encode(v,add_special_tokens=False))>cfg['prompt_token_limit'] for v in candidate.values()):
                return 0.,{'error':'Candidate exceeds fixed component token limit; rejected','valid':False}
            feedback=[]
            for row in example['pair']:
                record=self.pipeline.run(method,row['question'],split['dataset'],candidate=candidate)
                rating,reason=self.judge(row,record['answer'])
                feedback.append({'id':row['id'],'label':row['label'],'question':row['question'],
                                 'answer':record['answer'],'rating':rating,'reason':reason,
                                 'detector_raw':record.get('detector_raw')})
            return sum(r['rating']/5 for r in feedback)/2,{'examples':feedback}
        seed_candidate=candidate.copy()
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'plan.json',{'method':method,'train_ids':split['train'],'dev_ids':[r['id'] for p in dev for r in p['pair']],
            'budget_questions':cfg['metric_budget'],'budget_pair_calls':cfg['metric_budget']//2,
            'seed':cfg['seed'],'candidate':candidate,'identity':self.identity},immutable=True)
        def reflect(messages):
            if isinstance(messages,str): messages=[{'role':'user','content':messages}]
            if not any(m['role']=='system' for m in messages):
                messages=[{'role':'system','content':'Improve the supplied prompt using training feedback. Return the requested candidate.'}]+messages
            if [m['role'] for m in messages] != ['system','user']:
                messages=[{'role':'system','content':'Improve the prompt following the supplied optimization conversation.'},
                          {'role':'user','content':json.dumps(messages,ensure_ascii=False)}]
            return self.external_call('reflection',messages)
        result=optimize_anything(seed_candidate=candidate,evaluator=evaluate,dataset=train,valset=dev,
            objective='Maximize balanced FPQ correction and normal-question preservation. Only questions are available at inference.',
            background='Optimize only supplied components. Keep Yes/No verdict format when detecting. No tools or retrieval. Each component must fit 512 tokenizer tokens.',
            config=GEPAConfig(engine=EngineConfig(run_dir=str(directory/'engine'),seed=cfg['seed'],
                max_metric_calls=cfg['metric_budget']//2,parallel=False,max_workers=1,cache_evaluation=True),
                reflection=ReflectionConfig(reflection_lm=reflect,reflection_minibatch_size=2)))
        candidate=result.best_candidate
        if any(len(tok.encode(v,add_special_tokens=False))>cfg['prompt_token_limit'] for v in candidate.values()):
            raise ValueError('Selected candidate violates length limit')
        write_json(saved,{'candidate':candidate,'best_idx':result.best_idx,
                         'selection':'GEPA best_candidate on frozen balanced dev pairs'},immutable=True)
        return candidate

    def execute(self):
        calls=CachedTask(self.task,self.out,self.identity,self.reg['transport_common'])
        self.pipeline=Pipeline(self.reg,calls,self.bridge)
        # Preflight renders both pipeline families and judge without accessing held-out rows.
        for split in self.bundle['splits']:
            row=self.lookup[split['smoke'][0]]
            self.bridge(op='render',family='prewome',template='PresuppositionExtractionTemplate',
                kwargs={'question':row['question'],'passages':[],
                        'few_shot_data':self.reg['fixed_examples'][split['dataset']]['examples']})
            self.bridge(op='judge',row=row,answer='Preflight placeholder')
        if self.args.command=='preflight':
            report={'identity':self.identity,'methods':self.methods,'model_calls':0,
                'qa_status':self.bundle['audit']['qa_status'],
                'gepa_ready':bool(self.config['gepa']['tokenizer_revision']),
                'note':'Rendering passed; GPU and provider authentication not tested.'}
            write_json(self.out/'preflight.json',report); print(json.dumps(report,indent=2));return
        splits=self.bundle['splits']
        if self.args.command=='smoke': splits=splits[:1]
        for split in splits:
            rows=[self.lookup[i] for i in split['smoke' if self.args.command=='smoke' else 'test']]
            for method in self.methods:
                try:
                    gate=self.classifier(split,method) if method in ('tfidf_gate','probe_gate') else None
                    candidate=self.optimize(split,method) if method.startswith('gepa') else None
                    for row in rows:
                        self.process(row,split,method,candidate,gate,qa=False)
                    if self.args.command=='run':
                        for row in self.bundle['qa']:
                            self.process(row,split,method,candidate,gate,qa=True)
                    key=f"{split['dataset']}/{split['name']}/{method}"
                    self.status[key]={'state':'complete' if self.args.command=='run' else 'smoke_complete',
                                      'n_fpqa':len(rows),'n_qa':len(self.bundle['qa']) if self.args.command=='run' else 0}
                except Exception as error:
                    key=f"{split['dataset']}/{split['name']}/{method}"
                    self.status[key]={'state':'failed','error':str(error)}
                    write_json(self.out/'status.json',self.status)
                    raise
                write_json(self.out/'status.json',self.status)
        self.status['table3']={'state':'generated' if self.bundle['qa'] and self.args.command=='run' else 'pending',
                              'reason':self.bundle['audit']['qa_status']}
        write_json(self.out/'status.json',self.status)

    def process(self,row,split,method,candidate,gate,qa):
        # Share zero-shot QA across source/folds, fixed pipeline QA across folds per source.
        source=split['dataset']; name=split['name']
        if qa and method in FIXED:
            name='fixed';source=source if method in ('prewome','extract_verify') else 'shared'
        base='smoke' if self.args.command=='smoke' else 'table3' if qa else 'table12'
        path=self.out/base/source/name/method/(digest([row['dataset'],row['id']])+'.json')
        if path.exists():
            record=json.loads(path.read_text())
        else:
            question=qa_question(row) if qa else row['question']
            result=self.pipeline.run(method,question,split['dataset'],candidate=candidate,gate=gate)
            record={'id':row['id'],'dataset':row['dataset'],'source':source,'split':name,'method':method,
                    'question':question,'label':None if qa else row['label'],**result}
            if qa: record.update(qa_score(row,result['answer']))
            write_json(path,record,immutable=True)
        if not qa and not self.args.generate_only and 'rating' not in record:
            rating,reason=self.judge(row,record['answer'])
            record.update(rating=rating,judge_reason=reason,judge_config=self.external['judge_model'])
            write_json(path,record)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('preflight','smoke','run'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_baselines/local_4090.json')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=('qwen25','gemma3'),required=True)
    p.add_argument('--phase',choices=('fixed','trained','all'),default='fixed')
    p.add_argument('--methods');p.add_argument('--generate-only',action='store_true')
    a=p.parse_args(); cfg=json.loads(a.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,cfg['models'][a.model]['gpus']))
    Worker(a).execute()


if __name__=='__main__':main()
