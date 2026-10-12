"""Reuse only unchanged fixed-method answers; preserve original provenance."""
import argparse,json,sys,shutil
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paper_baselines import write_json,sha,digest
from scripts.run_paper_baselines import check_bundle


def adopt(old,new,bundle):
    lookup={r['id']:r for r in bundle['rows']};valid={i for s in bundle['splits'] for i in s['test']}
    counts=Counter()
    for model in ('qwen25','gemma3'):
        a=json.loads((old/model/'identity.json').read_text());b=json.loads((new/model/'identity.json').read_text())
        for key in ('model','decoding','registry_sha256','well_source_hashes','code_sha256'):
            if a[key]!=b[key]:raise ValueError('Unsafe answer reuse: '+key)
        for table in ('table12','table3'):
            for p in (old/model/table).rglob('*.json'):
                r=json.loads(p.read_text())
                if r['method'] not in ('plain','balanced','cot'):continue
                if table=='table12' and (r['id'] not in valid or lookup[r['id']]['question']!=r['question']):continue
                dest=new/model/p.relative_to(old/model)
                if dest.exists():continue
                prior={k:r.pop(k) for k in ('rating','judge_reason','judge_config') if k in r}
                r.update(reused_from=str(p),source_sha256=sha(p),original_batch_size=1)
                if prior:r['archived_evaluation']=prior
                write_json(dest,r,immutable=True);counts[(model,table,r['method'])]+=1
        # Keep known length failures for unchanged fixed prompts; never retry an
        # 8192-token deterministic failure just because the ledger moved.
        for p in (old/model/'failures').rglob('*.json'):
            r=json.loads(p.read_text())
            if r.get('method') not in ('plain','balanced','cot'):continue
            dest=new/model/p.relative_to(old/model)
            if not dest.exists():write_json(dest,r,immutable=True)
    report={str(k):v for k,v in counts.items()};write_json(new/'reuse_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();adopt(a.old,a.new,check_bundle(a.bundle))
