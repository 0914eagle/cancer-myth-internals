"""Freeze common CREPE samples; leave official train and classifier dev intact."""
import argparse,copy,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_baselines import check_bundle
from src.paper_baselines import digest,write_json


def sample(ids,lookup,n,seed):
    rng=random.Random(seed);chosen=[]
    for label in (1,0):
        pool=sorted(i for i in ids if lookup[i]['label']==label)
        if len(pool)<n:raise ValueError('Not enough examples')
        chosen+=rng.sample(pool,n)
    return sorted(chosen)


def prepare(source,config,out):
    b=check_bundle(source);old_hash=b.pop('content_sha256');lookup={r['id']:r for r in b['rows']}
    for s in b['splits']:
        if s['dataset']=='crepe':
            s['test']=sample(s['test'],lookup,500,42)
            s['gepa_dev']=sample(s['dev'],lookup,100,43)
        else:s['gepa_dev']=list(s['dev'])
    b['audit'].update(protocol_revision='v4_common_crepe1000',parent_bundle_sha256=old_hash,
        sampling='CREPE: class-stratified random sampling, test seed42, GEPA dev seed43; no length filter',
        test_selection='before new v4 results; prior study exposure is not erased')
    b['content_sha256']=digest(b);write_json(out/'bundle.json',b,immutable=True)
    manifest={'bundle_sha256':b['content_sha256'],'splits':b['splits'],'sampling_seeds':{'crepe_test':42,'crepe_gepa_dev':43}}
    write_json(out/'split_manifest.json',manifest,immutable=True)
    ext=json.loads(Path(config['external_config']).read_text())
    # Optimization feedback and reflection are GPT; final reporting uses Haiku.
    ext['judge_model']=copy.deepcopy(ext['task_model']);ext['reflection_model']=copy.deepcopy(ext['task_model'])
    write_json(out/'external.json',ext,immutable=True)
    c=copy.deepcopy(config);c.update(protocol_id='paper_baselines_v4_shared_luna_crepe1000_20261012',external_config=str(out/'external.json'))
    c['models']['luna6']={'model_id':'gpt-6-luna','backend':'codex','gpus':[],'decoding':'provider defaults; medium reasoning; no claim of HF-equivalent sampling'}
    c['gepa']['dev_per_class_max']=100
    c['gepa']['metric_budget']=2000
    c['batching']={'qwen25':{'max_batch':8,'token_budget':73728},'gemma3':{'max_batch':4,'token_budget':49152},'luna6':{'pipeline_workers':4}}
    c['global_gpt_concurrency']=4
    c['response_protocol']='fresh_response_with_saved_detection_v1'
    c['extraction_protocol']='shared_premise_extraction_v1'
    c['qa_cancer_system']='fold0 preselected; report as single-system transfer, not three-fold mean'
    c['final_judge']={'backend':'claude','model':'claude-haiku-5-5','effort':'medium','timeout_seconds':240}
    write_json(out/'runtime.json',c,immutable=True)
    return b,c


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    b,c=prepare(a.source,json.loads(a.config.read_text()),a.out);print(b['content_sha256'])
