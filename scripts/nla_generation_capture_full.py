"""Matched train-only answers and sparse original-generation hidden states.

Source settings are identical to the baseline. Capture only the official NLA
layer during the actual generate() call; no saved-answer teacher forcing.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.nla_connection import snapshot
from src.paper_baselines import HFTask, digest, sha, write_json
from scripts.nla_generation_capture import selected_rows as pilot_rows


def selected_rows(bundle, per_class=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    lookup = {r['id']:r for r in bundle['rows']}
    selected=[]
    for dataset,name in [('cancer_myth','fold0'),('crepe','official')]:
        split=next(s for s in bundle['splits'] if (s['dataset'],s['name'])==(dataset,name))
        assert not set(split['train']) & (set(split['dev'])|set(split['test']))
        pool=[lookup[i] for i in sorted(split['train'])]
        vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=50000)
        vectorizer.fit([r['question'] for r in pool])
        features=vectorizer.get_feature_names_out()
        for label in (1,0):
            for row in pool:
                if row['label']!=label:continue
                question=row['question']
                vector=vectorizer.transform([question]).tocoo()
                ranked=sorted(zip(vector.col,vector.data),key=lambda x:(-x[1],features[x[0]]))
                keywords=[];spans=[]
                for idx,_ in ranked:
                    words=features[idx].split()
                    pattern=r'\b'+r'\W+'.join(re.escape(w) for w in words)+r'\b'
                    matches=list(re.finditer(pattern,question,re.I))
                    if not matches:continue
                    match=matches[-1]; span=match.span()
                    if any(span[0]<b and span[1]>a for a,b in spans):continue
                    keywords.append({'text':match.group(),'char_span':list(span)})
                    spans.append(span)
                    if len(keywords)==2:break
                if not keywords:raise ValueError('No label-blind keyword for '+row['id'])
                selected.append({'id':row['id'],'dataset':dataset,'split':name,'label':label,
                    'group_id':row['group_id'],'question':question,'keywords':keywords})
    # Keep the already running pilot as the prefix; then cover every train row.
    prefix=pilot_rows(bundle,5)
    priority={(r['dataset'],r['id']):i for i,r in enumerate(prefix)}
    return sorted(selected,key=lambda r:(priority.get((r['dataset'],r['id']),len(prefix)),r['dataset'],r['id']))


class SparseCapture:
    """Layer output index L-1 equals hidden_states[L] for interior layers."""
    def __init__(self,prompt_length,prompt_positions,special_ids,decode):
        self.prompt_length=prompt_length;self.prompt_positions=prompt_positions
        self.special_ids=set(special_ids);self.decode=decode
        self.seen=0;self.meaningful=0;self.current_ids=None;self.vectors={}

    def before(self,module,args,kwargs):
        ids=kwargs.get('input_ids')
        if ids is None and args:ids=args[0]
        if ids is None:raise RuntimeError('Source forward has no token IDs')
        self.current_ids=ids

    def after(self,module,args,output):
        hidden=output[0] if isinstance(output,tuple) else output
        ids=self.current_ids
        if ids is None or tuple(ids.shape)!=tuple(hidden.shape[:2]):
            raise RuntimeError('Token/activation alignment mismatch')
        n=hidden.shape[1]
        if self.seen==0:
            if n!=self.prompt_length:raise RuntimeError('Unexpected prefill length')
            for family,info in self.prompt_positions.items():
                pos=info['position']
                self.vectors[family]=(hidden[0,pos].detach().float().cpu(),pos,int(ids[0,pos]),info)
        elif self.meaningful<32:
            for local,token in enumerate(ids[0].detach().cpu().tolist()):
                if token in self.special_ids or not any(c.isalnum() for c in self.decode(token)):continue
                self.meaningful+=1
                if self.meaningful in (1,8,32):
                    family=f'answer_content_token_{self.meaningful}'
                    pos=self.seen+local
                    self.vectors[family]=(hidden[0,local].detach().float().cpu(),pos,token,
                        {'generated_token_index':pos-self.prompt_length,'content_token_rank':self.meaningful})
        self.seen+=n


def decoder_layer(model,layer,n_layers):
    for path in ('model.layers','model.language_model.layers','language_model.model.layers'):
        try:layers=model.get_submodule(path)
        except AttributeError:continue
        if len(layers)!=n_layers:raise RuntimeError('Decoder layer count mismatch')
        return layers[layer-1],path
    raise RuntimeError('Cannot locate source decoder layers')


def capture(c,key):
    import torch
    import yaml
    out=Path(c['artifact_root'])/key;out.mkdir(parents=True,exist_ok=True)
    spec=c['models'][key]
    baseline=json.loads(Path(c['baseline_config']).read_text())
    bundle=json.loads(Path(c['bundle']).read_text());expected=bundle.pop('content_sha256')
    if digest(bundle)!=expected:raise ValueError('Bundle hash mismatch')
    selection=selected_rows(bundle)
    registry=json.loads((ROOT/baseline['registry']).read_text())
    policy={'bundle_sha256':expected,'selection':selection,'conditions':['plain','balanced'],
        'keyword_rule':'Top two nonoverlapping TF-IDF terms; fit on this split train questions only, no labels/gold',
        'source':baseline['models'][key],'physical_gpu':spec['gpu'],'nla':spec,
        'source_decoding':{k:baseline[k] for k in ['dtype','seed','max_input_tokens','max_new_tokens']},
        'activation_mode':'captured during original autoregressive generation; no answer replay',
        'positions':'two keyword final subtokens, question last token, assistant boundary, answer alphanumeric token ranks 1/8/32',
        'adapter_sha256':sha(__file__),'original_source_backend_sha256':sha(ROOT/'src/paper_baselines.py'),
        'coverage':'Every Cancer fold0 train and CREPE official train row, including grouped variants',
        'reuse_root':c['reuse_root'],
        'scope':'train-only exploratory comparison, not held-out Table scores'}
    write_json(out/'provenance.json',policy,immutable=True)
    av_cfg={'seed':baseline['seed'],'paths':{'cache_dir':c['cache_root']},
        'nla_model':{'model_id':snapshot(c['cache_root'],spec['av']), 'dtype':'bfloat16',
            'device_map':'cuda','trust_remote_code':False,'expected_d_model':spec['d_model'],
            'sidecar_filename':'nla_meta.yaml'},'generation':{'do_sample':False,'max_new_tokens':256}}
    (out/'av_config.yaml').write_text(yaml.safe_dump(av_cfg,sort_keys=False))
    jobs=[(r,condition) for r in selection for condition in ['plain','balanced']]
    backend=None;manifest_count=0;answer_count=0;complete_count=0
    # Rebuild append-only indexes from sealed per-case records on resume.
    (out/'answers.jsonl').write_text('')
    (out/'manifest.jsonl').write_text('')
    for row,condition in jobs:
        case_id=digest([row['dataset'],row['id'],condition]);case_path=out/'cases'/(case_id+'.json')
        previous=Path(c['reuse_root'])/key/'cases'/(case_id+'.json')
        if not case_path.exists() and previous.exists():
            old=json.loads(previous.read_text())
            if old['answer']['question']!=row['question'] or old['answer']['condition']!=condition:
                raise RuntimeError('Prior case identity mismatch')
            old['reused_from_case']=str(previous)
            write_json(case_path,old,immutable=True)
        if case_path.exists():
            case=json.loads(case_path.read_text())
            for item in case['activations']:
                if not Path(item['activation_path']).exists():raise RuntimeError('Missing saved activation')
        else:
            if backend is None:backend=HFTask(baseline,key)
            messages=[{'role':'system','content':registry['transport_common']+'\n\n'+registry['prompts'][condition]},
                {'role':'user','content':row['question']}]
            text=backend.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            batch=backend.inputs(messages);prompt_len=batch['input_ids'].shape[1]
            mapping=backend.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            if mapping['input_ids']!=batch['input_ids'][0].tolist():raise RuntimeError('Tokenizer offset drift')
            start=text.rfind(row['question']);end=start+len(row['question'])
            if start<0:raise RuntimeError('Question missing in chat text')
            def last_token(a,b):
                pos=[i for i,(x,y) in enumerate(mapping['offset_mapping']) if x<b and y>a and x!=y]
                if not pos:raise RuntimeError('No token overlaps selected text')
                return pos[-1]
            positions={'question_last_token':{'position':last_token(start,end)},
                'assistant_boundary_last_token':{'position':prompt_len-1}}
            for idx,word in enumerate(row['keywords'],1):
                a,b=word['char_span'];positions[f'prompt_keyword_{idx}']={'position':last_token(start+a,start+b),**word}
            collector=SparseCapture(prompt_len,positions,backend.tokenizer.all_special_ids,
                lambda tid:backend.tokenizer.decode([tid],skip_special_tokens=False))
            layer,path=decoder_layer(backend.model,spec['layer'],baseline['models'][key]['n_layers'])
            first=backend.model.register_forward_pre_hook(collector.before,with_kwargs=True)
            second=layer.register_forward_hook(collector.after)
            began=time.time()
            try:
                with torch.inference_mode():
                    output=backend.model.generate(**batch,do_sample=False,max_new_tokens=baseline['max_new_tokens'],
                        pad_token_id=backend.tokenizer.pad_token_id if backend.tokenizer.pad_token_id is not None else backend.tokenizer.eos_token_id)
            finally:first.remove();second.remove()
            tokens=output[0,prompt_len:].detach().cpu();eos=backend.model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            completed=len(tokens)>0 and int(tokens[-1]) in eos
            answer=backend.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            answer_record={'case_id':case_id,'model':key,'id':row['id'],'dataset':row['dataset'],
                'split':row['split'],'partition':'train','label':row['label'],'condition':condition,
                'question':row['question'],'messages':messages,'answer':answer,'answer_token_ids':tokens.tolist(),
                'input_tokens':prompt_len,'output_tokens':len(tokens),'completed':completed,
                'finish_reason':'eos' if completed else 'length','elapsed_seconds':time.time()-began,
                'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            items=[]
            for family,(vector,pos,tid,info) in collector.vectors.items():
                if vector.shape!=(spec['d_model'],) or not torch.isfinite(vector).all():
                    raise RuntimeError('Invalid captured activation')
                if family.startswith('answer_') and int(tokens[pos-prompt_len])!=tid:
                    raise RuntimeError('Generated token/hidden-state mismatch')
                ident=digest([case_id,family]);file=out/'activations'/(ident+'.pt')
                file.parent.mkdir(parents=True,exist_ok=True);torch.save(vector,file)
                items.append({'id':ident,'base_id':row['id'],'case_id':case_id,'source':row['dataset'],
                    'condition':condition,'category':'FPQ' if row['label'] else 'NFP','prompt':row['question'],
                    'activation_path':str(file),'layer':spec['layer'],'hidden_state_index':spec['layer'],
                    'decoder_module':path+f'.{spec["layer"]-1}','position':str(pos),'position_family':family,
                    'position_mode':'token_index','selected_token_id':tid,
                    'selected_token_text':backend.tokenizer.decode([tid],skip_special_tokens=False),
                    'activation_mode':'original_generation_decode' if family.startswith('answer_') else 'original_generation_prefill',
                    'shape':list(vector.shape),'dtype':str(vector.dtype),'split':row['split'],'partition':'train',
                    'source_model_revision':baseline['models'][key]['revision'],'answer_completed':completed,
                    'answer_case_path':str(case_path),'selection':info})
            case={'answer':answer_record,'activations':items,
                'unobserved_answer_positions':[f'answer_content_token_{i}' for i in [1,8,32]
                    if f'answer_content_token_{i}' not in collector.vectors]}
            write_json(case_path,case,immutable=True)
            print(f"Saved {key} {row['dataset']} {row['id']} {condition}: {len(tokens)} answer tokens, {len(items)} activations, completed={completed}",flush=True)
        with (out/'answers.jsonl').open('a') as stream:
            stream.write(json.dumps(case['answer'],ensure_ascii=False)+'\n')
        with (out/'manifest.jsonl').open('a') as stream:
            for item in case['activations']:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
        answer_count+=1;manifest_count+=len(case['activations']);complete_count+=int(case['answer']['completed'])
        write_json(out/'generation_status.json',{'state':'running','completed_cases':answer_count,
            'total_cases':len(jobs),'vectors':manifest_count,'completed_answers':complete_count})
    write_json(out/'generation_status.json',{'state':'complete','completed_cases':answer_count,
        'total_cases':len(jobs),'vectors':manifest_count,'completed_answers':complete_count})


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--model',required=True)
    a=p.parse_args();c=json.loads(a.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES']=str(c['models'][a.model]['gpu'])
    capture(c,a.model)


if __name__=='__main__':main()
