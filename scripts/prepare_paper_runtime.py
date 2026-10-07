"""Pin/download only the GEPA prompt-length tokenizer; no LLM calls or weight download."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.paper_baselines import write_json

if __name__=='__main__':
    from huggingface_hub import HfApi,snapshot_download
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/paper_baselines/local_4090.json')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());g=cfg['gepa']
    g['tokenizer_revision']=g['tokenizer_revision'] or HfApi(token=False).model_info(g['prompt_tokenizer']).sha
    snapshot_download(g['prompt_tokenizer'],revision=g['tokenizer_revision'],cache_dir=cfg['cache_root'],
                      token=False, allow_patterns=['*.json','*.txt','*.model','*.jinja','tokenizer*'])
    write_json(a.out,cfg,immutable=True)
    print(a.out)
