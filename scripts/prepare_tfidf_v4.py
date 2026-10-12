"""CPU-only shared TF-IDF fitting and full Table 1 predictions."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_baselines import Worker,check_bundle
from src.paper_baselines import digest,write_json
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());b=check_bundle(a.bundle);look={r['id']:r for r in b['rows']}
w=SimpleNamespace(args=a,bundle=b,lookup=look,identity={},out=a.out/'shared',feature=None)
for s in b['splits']:
 gate=Worker.classifier(w,s,'tfidf_gate')
 for i in s['test']:
  r=look[i];prob,threshold=gate(r['question'])
  info={'method':'tfidf_gate','question':r['question'],'review':'','detector_raw':'','id':i,'dataset':r['dataset'],'label':r['label'],'split':s['name'],'source':s['dataset'],'probability':prob,'threshold':threshold,'verdict':int(prob>=threshold),'invalid_detection':False}
  for model in c['models']:
   path=a.out/model/'table1'/s['dataset']/s['name']/'tfidf_gate'/(digest([r['dataset'],i])+'.json')
   write_json(path,info,immutable=True)
 print(s['dataset'],s['name'],len(s['test']),flush=True)
