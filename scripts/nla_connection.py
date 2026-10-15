"""Train-only FPQA adapter for the existing medical_nla and released NLA code.

No NLA training. Extract prompt activations (no saved-answer replay) in the
baseline environment. AV uses medical_nla.src.run_nla unchanged; AR uses the
official NLACritic unchanged. Run each stage in a separate process to free VRAM.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def medical_package(root):
    name = 'existing_medical_nla'
    spec = importlib.util.spec_from_file_location(name, Path(root)/'src/__init__.py',
        submodule_search_locations=[str(Path(root)/'src')])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return name


def select_rows(bundle):
    lookup = {r['id']: r for r in bundle['rows']}
    selected = []
    for dataset, name in [('cancer_myth', 'fold0'), ('crepe', 'official')]:
        split = next(s for s in bundle['splits'] if (s['dataset'], s['name']) == (dataset, name))
        assert not (set(split['train']) & (set(split['dev']) | set(split['test'])))
        for label in (1, 0):
            row = next(lookup[i] for i in sorted(split['train']) if lookup[i]['label'] == label)
            # Labels are selection metadata only; never copied into task/NLA prompts.
            selected.append({'id': row['id'], 'dataset': dataset, 'split': name,
                'label': label, 'question': row['question']})
    return selected


def snapshot(cache, spec):
    return str(Path(cache)/('models--'+spec['repo'].replace('/', '--'))/'snapshots'/spec['revision'])


def prepare(c, key, out):
    import yaml
    from src.paper_baselines import digest, sha, write_json
    bundle = json.loads(Path(c['bundle']).read_text())
    expected = bundle.pop('content_sha256')
    assert digest(bundle) == expected
    baseline = json.loads(Path(c['baseline_config']).read_text())
    rows = select_rows(bundle)
    medical = Path(c['medical_nla_root'])
    prov = {'pilot': 'train_only_prompt_activation_connection', 'bundle_sha256': expected,
        'selection': 'lexicographically first FPQ/NFP in Cancer fold0 train and CREPE official train',
        'rows': rows, 'conditions': ['plain','balanced'],
        'activation_mode': 'prompt forward before generation; no original answer hidden states or answer replay',
        'positions': ['question_last_token','assistant_boundary_last_token'],
        'source': baseline['models'][key], 'nla': c['models'][key],
        'existing_code_sha256': {f:sha(medical/f) for f in
            ['src/extract_activations.py','src/nla.py','src/run_nla.py','src/modeling.py','src/reconstruction_scoring.py']},
        'official_ar_sha256': sha(ROOT/'external/nla-inference-official/nla_inference.py'),
        'adapter_sha256': sha(__file__), 'held_out_selection': False}
    write_json(out/'provenance.json', prov, immutable=True)
    spec = c['models'][key]
    cfg = {'seed': baseline['seed'], 'paths': {'cache_dir': c['cache_root']},
        'nla_model': {'model_id': snapshot(c['cache_root'],spec['av']),
            'dtype': 'bfloat16', 'device_map': 'cuda', 'trust_remote_code': False,
            'expected_d_model': spec['d_model'], 'sidecar_filename': 'nla_meta.yaml'},
        'generation': {'do_sample': False,'max_new_tokens': 256}}
    (out/'av_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    return rows, baseline


def extract(c, key, out):
    import torch
    from src.paper_baselines import HFTask, digest, write_json
    medical = medical_package(c['medical_nla_root'])
    select_activation = importlib.import_module(medical+'.extract_activations').select_activation
    rows, baseline = prepare(c, key, out)
    spec = c['models'][key]
    # Reuse the exact baseline model revision, chat renderer and input settings.
    backend = HFTask(baseline, key)
    registry = json.loads((ROOT/baseline['registry']).read_text())
    manifest = []
    for row in rows:
        for condition in ('plain', 'balanced'):
            messages = [{'role':'system','content':registry['transport_common']+'\n\n'+registry['prompts'][condition]},
                        {'role':'user','content':row['question']}]
            text = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = backend.inputs(messages)
            mapped = backend.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            assert mapped['input_ids'] == batch['input_ids'][0].tolist()
            start = text.rfind(row['question'])
            assert start >= 0
            end = start + len(row['question'])
            q_positions = [i for i,(a,b) in enumerate(mapped['offset_mapping']) if a < end and b > start and a != b]
            assert q_positions
            positions = {'question_last_token':q_positions[-1],
                'assistant_boundary_last_token':batch['input_ids'].shape[1]-1}
            with torch.inference_mode():
                states = backend.model(**batch, output_hidden_states=True, use_cache=False).hidden_states
            hidden = states[spec['layer']]
            assert hidden.shape[-1] == spec['d_model']
            for family, pos in positions.items():
                ident = digest([row['dataset'],row['id'],condition,family])
                activation, position = select_activation(hidden, batch['attention_mask'],
                    {'id':ident,'position_mode':'token_index','target_token_position':pos}, {})
                activation = activation.float()
                assert activation.shape == (spec['d_model'],) and torch.isfinite(activation).all()
                path = out/'activations'/(ident+'.pt'); path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(activation,path)
                manifest.append({'id':ident,'base_id':row['id'],'source':row['dataset'],
                    'condition':condition,'category':'FPQ' if row['label'] else 'NFP',
                    'prompt':row['question'],'messages':messages,'chat_text':text,
                    'activation_path':str(path),'model_id':baseline['models'][key]['model_id'],
                    'model_revision':baseline['models'][key]['revision'],
                    'layer':spec['layer'],'hidden_state_index':spec['layer'],
                    'position':position,'position_family':family,'position_mode':'token_index',
                    'selected_token_id':int(batch['input_ids'][0,pos]),
                    'activation_mode':'prompt_forward_before_generation','shape':list(activation.shape),
                    'dtype':str(activation.dtype),'split':row['split'],'partition':'train'})
            del states, hidden
    path = out/'manifest.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest))
    write_json(out/'extraction_status.json',{'state':'complete','prompts':len(rows)*2,'vectors':len(manifest)})
    print(f'Extracted {len(manifest)} prompt vectors using existing medical_nla selector',flush=True)


def score(c, key, out):
    import torch
    from src.paper_baselines import write_json
    medical = medical_package(c['medical_nla_root'])
    helpers = importlib.import_module(medical+'.reconstruction_scoring')
    critic_cls = helpers.load_nla_critic_class(str(ROOT/'external/nla-inference-official'))
    critic = critic_cls(snapshot(c['cache_root'], c['models'][key]['ar']),device='cuda:0',dtype=torch.bfloat16)
    rows = helpers.read_jsonl(out/'av_results.jsonl')
    manifest = {r['id']:r for r in helpers.read_jsonl(out/'manifest.jsonl')}
    scored = []
    for row in rows:
        target = manifest[row['id']]
        activation = helpers.load_activation(row['activation_path'])
        mse,cos = critic.score(row['nla_output'],activation)
        control_mse,control_cos = critic.score(target['prompt'],activation)
        helpers.validate_mse(float(mse),row['id'])
        helpers.validate_mse(float(control_mse),row['id'])
        result = {**row,**{k:target[k] for k in ['source','condition','category','split','partition','activation_mode']},
            'recon_mse':float(mse),'recon_cos':float(cos),
            'question_text_control_mse':float(control_mse),'question_text_control_cos':float(control_cos),
            'interpretation':'direction reconstruction fidelity; not correctness of the AV explanation'}
        scored.append(result)
        helpers.write_jsonl(out/'scored.jsonl',scored)
        print(f"AR {target['base_id']} {target['condition']} {target['position_family']}: mse={mse:.4f}, cos={cos:.4f}",flush=True)
    write_json(out/'summary.json',{'state':'complete','n':len(scored),
        'parsed_explanations':sum(r['parsed_explanation_tag'] for r in scored),
        'mean_recon_mse':sum(r['recon_mse'] for r in scored)/len(scored),
        'mean_question_control_mse':sum(r['question_text_control_mse'] for r in scored)/len(scored),
        'scope':'train-only connection pilot; no Table 1/2 performance claim'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['extract','score'])
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--model',required=True)
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(c['models'][a.model]['gpu'])
    out = Path(c['artifact_root'])/a.model; out.mkdir(parents=True,exist_ok=True)
    {'extract':extract,'score':score}[a.stage](c,a.model,out)


if __name__ == '__main__':
    main()
