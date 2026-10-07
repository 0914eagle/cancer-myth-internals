"""Freeze QA source revisions and optional deterministic evaluation subsets.

No task/judge calls. QA is never used for adaptation. PubMedQA uses the released
PQA-L pool (HF calls it train); report this scope, not an official test score.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import digest,write_json

SOURCES={
 'medqa':('GBaker/MedQA-USMLE-4-options',None,'test'),
 'pubmedqa':('qiaojin/PubMedQA','pqa_labeled','train'),
 'medbullets':('mkieffer/Medbullets',None,'op4_test'),
 'mmlu':('cais/mmlu','all','test'),
 'gsm8k':('openai/gsm8k','main','test'),
}


def convert(name,r,i,revision,split):
    row={'id':f'{name}:{i}','dataset':name,'question':r['question'],
         'source_revision':revision,'source_split':split,'source_index':i}
    if name=='medqa':row.update(metric='choice',choices=r['options'],gold=r['answer_idx'])
    elif name=='medbullets':row.update(metric='choice',choices={k:v for k,v in r['options'].items() if k in 'ABCD'},gold=r['answer'])
    elif name=='mmlu':row.update(metric='choice',choices=dict(zip('ABCD',r['choices'])),gold='ABCD'[r['answer']],subject=r['subject'])
    elif name=='gsm8k':row.update(metric='numeric',gold=r['answer'].split('####')[-1].strip())
    elif name=='pubmedqa':
        row.update(id=f"pubmedqa:{r['pubid']}",metric='label',gold=r['final_decision'],allowed_labels=['yes','no','maybe'],
                   question='Context:\n'+'\n'.join(r['context']['contexts'])+'\n\nQuestion:\n'+r['question'],
                   evaluation_scope='PQA-L released labeled pool; not official test partition')
    return row


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--datasets',default=','.join(SOURCES))
    p.add_argument('--limit-per-dataset',type=int,default=0,help='0=full split; positive=hash-selected exploratory subset')
    a=p.parse_args()
    if a.limit_per_dataset<0: p.error('negative limit')
    names=a.datasets.split(',')
    if not set(names)<=set(SOURCES):p.error('unknown dataset')
    a.out.mkdir(parents=True,exist_ok=True)
    pin=a.out/'sources.json'
    if pin.exists():
        manifest=json.loads(pin.read_text())
        if manifest['datasets']!=names or manifest['limit']!=a.limit_per_dataset:raise ValueError('Frozen selection differs')
    else:
        manifest={'datasets':names,'limit':a.limit_per_dataset,'selection':'sha256(seed42,dataset,id), independent of answers/scores',
                  'sources':{n:{'repo':SOURCES[n][0],'config':SOURCES[n][1],'split':SOURCES[n][2],
                                'revision':HfApi(token=False).dataset_info(SOURCES[n][0]).sha} for n in names}}
        write_json(pin,manifest,immutable=True)
    output=[]
    for name in names:
        spec=manifest['sources'][name]
        data=load_dataset(spec['repo'],spec['config'],revision=spec['revision'],split=spec['split'],token=False,cache_dir=str(a.out/'dataset_cache'))
        rows=[convert(name,r,i,spec['revision'],spec['split']) for i,r in enumerate(data)]
        rows.sort(key=lambda r:digest([42,name,r['id']]))
        if a.limit_per_dataset:rows=rows[:a.limit_per_dataset]
        output.extend(rows)
    path=a.out/'questions.jsonl'
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output)
    if path.exists() and path.read_text()!=text:raise ValueError('Frozen QA file differs')
    path.write_text(text)
    write_json(a.out/'counts.json',{n:sum(r['dataset']==n for r in output) for n in names},immutable=True)
    print(path)


if __name__=='__main__':main()
