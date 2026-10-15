"""Resume released AV decoding, with checkpoint EOS and explicit batch provenance.

Existing explanations are retained. Source answers/activations, weights, sidecar
prompt/injection and decoding cap stay fixed. Batch changes are recorded rather
than assumed to reproduce legacy singleton decoding.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.nla_connection import medical_package


def trim_eos(tokens,eos_ids):
    for i,token in enumerate(tokens):
        if token in eos_ids:return tokens[:i+1],True
    return tokens,False


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--gpu',type=str,required=True)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--limit',type=int)
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);a=p.parse_args()
    c=json.loads(a.config.read_text());out=Path(c['artifact_root'])/a.model
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    import torch,yaml
    torch.manual_seed(42)
    package=medical_package(c['medical_nla_root'])
    nla=importlib.import_module(package+'.nla');modeling=importlib.import_module(package+'.modeling')
    cfg=yaml.safe_load((out/'av_config.yaml').read_text());model_cfg=cfg['nla_model']
    path=Path(model_cfg['model_id'])
    original_download=nla.hf_hub_download
    def local_sidecar(repo_id,filename,**kwargs):
        local=Path(repo_id)
        return str(local/filename) if local.is_dir() else original_download(repo_id,filename=filename,**kwargs)
    nla.hf_hub_download=local_sidecar
    tokenizer=modeling.load_tokenizer(str(path),cache_dir=c['cache_root'],trust_remote_code=False)
    sidecar=nla.load_nla_sidecar(str(path),tokenizer=tokenizer,cache_dir=c['cache_root'],
        expected_d_model=c['models'][a.model]['d_model'])
    meta=yaml.safe_load((path/'nla_meta.yaml').read_text())
    if meta['extraction_layer_index']!=c['models'][a.model]['layer']:raise RuntimeError('NLA layer mismatch')
    output=a.output or out/'av_results.jsonl';done=set()
    if output.exists():
        with output.open() as file:
            for line in file:
                row=json.loads(line)
                if row['id'] in done:raise RuntimeError('Duplicate saved AV result')
                done.add(row['id'])
    rows=[]
    manifest=a.manifest or out/'manifest_priority.jsonl'
    with manifest.open() as file:
        for line in file:
            row=json.loads(line)
            if row['id'] not in done:rows.append(row)
            if a.limit and len(rows)>=a.limit:break
    if not rows:
        print('No pending AV rows',flush=True);return
    began=time.perf_counter()
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,
        torch_dtype=torch.bfloat16,device_map='auto',
        max_memory={i:'40GiB' for i in range(torch.cuda.device_count())}).eval()
    assert all(str(v) not in ('cpu','disk') for v in model.hf_device_map.values())
    torch.cuda.synchronize();loaded=time.perf_counter()-began
    eos=model.generation_config.eos_token_id
    eos=eos if isinstance(eos,list) else [eos]
    if any(x is None for x in eos):raise RuntimeError('Checkpoint has no EOS')
    policy={'policy':'released_checkpoint_eos_batched_resume_v1','model':a.model,
        'physical_gpu':a.gpu,'requested_batch_size':a.batch_size,'eos_token_ids':eos,
        'max_new_tokens':cfg['generation']['max_new_tokens'],'do_sample':False,
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_path':str(manifest.resolve()),
        'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'legacy_outputs_retained':True,'batch_can_change_generated_description':True,
        'source_answers_activations_unchanged':True,'sidecar_prompt_scale_weights_unchanged':True}
    policy_id=hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()
    (out/f'av_execution_policy_{policy_id}.json').write_text(json.dumps(policy,indent=2)+'\n')
    completed=0;started=time.perf_counter()
    with output.open('a') as file:
        for start in range(0,len(rows),a.batch_size):
            chunk=rows[start:start+a.batch_size];injected=[]
            for row in chunk:
                activation=torch.load(row['activation_path'],map_location='cpu',weights_only=True)
                if not torch.isfinite(activation).all():raise RuntimeError('Nonfinite activation')
                result=nla.build_nla_inputs_embeds(tokenizer=tokenizer,embed_layer=model.get_input_embeddings(),
                    sidecar=sidecar,activation=activation,device=model.device)
                injected.append(result)
            prompt_ids=injected[0].input_ids
            if any(not torch.equal(r.input_ids,prompt_ids) for r in injected):
                raise RuntimeError('AV prompts differ within batch')
            embeds=torch.cat([r.inputs_embeds for r in injected],dim=0)
            mask=torch.cat([r.attention_mask for r in injected],dim=0)
            with torch.inference_mode():
                generated=model.generate(inputs_embeds=embeds,attention_mask=mask,
                    do_sample=False,max_new_tokens=cfg['generation']['max_new_tokens'],
                    pad_token_id=tokenizer.pad_token_id,eos_token_id=eos)
            if generated.shape[0]!=len(chunk):raise RuntimeError('AV batch alignment mismatch')
            token_rows=generated.detach().cpu().tolist()
            for row,result,tokens in zip(chunk,injected,token_rows):
                tokens,ended=trim_eos(tokens,set(eos));raw=tokenizer.decode(tokens,skip_special_tokens=False)
                explanation,parsed=nla.extract_explanation(raw)
                record={**row,'query':result.prompt_text,'actor_prompt_template_file':None,
                    'actor_prompt_suffix_file':None,'adapter_id':None,'nla_output':explanation,
                    'raw_nla_output':raw,'parsed_explanation_tag':parsed,'cjk_fraction':nla.cjk_fraction(raw),
                    'activation_norm':result.activation_norm,'scaled_activation_norm':result.scaled_activation_norm,
                    'injection_position':result.injection_position,'injection_scale':sidecar.injection_scale,
                    'injection_token_id':sidecar.injection_token_id,'sidecar_path':sidecar.path,
                    'gen_config':{'do_sample':False,'max_new_tokens':cfg['generation']['max_new_tokens'],
                        'eos_token_id':eos,'batch_size':len(chunk)},
                    'av_execution_policy_id':policy_id,'output_tokens':len(tokens),
                    'finish_reason':'eos' if ended else 'length','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                file.write(json.dumps(record,ensure_ascii=False)+'\n');completed+=1
            file.flush()
            print('AV saved',completed,'new rows, batch',len(chunk),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    timing={'model':a.model,'batch_size':a.batch_size,'new_rows':completed,'retained_rows':len(done),
        'load_seconds':loaded,'generation_seconds':seconds,'seconds_per_vector':seconds/completed,
        'execution_policy_id':policy_id}
    target=output.with_suffix('.timing.json') if a.output else out/'av_resume_timing.json'
    target.write_text(json.dumps(timing,indent=2)+'\n');print(json.dumps(timing),flush=True)


if __name__=='__main__':main()
