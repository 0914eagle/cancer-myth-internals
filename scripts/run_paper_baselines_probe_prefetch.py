"""Use the existing Gemma model for background probe preparation, sharing its GPU lock."""
import argparse,json,os,sys,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_paper_baselines_v4 import V4Worker
from scripts.prepare_probe_v4 import prepare
from src.paper_baselines import sha,write_json


class PrefetchWorker(V4Worker):
    def __init__(self,args):
        super().__init__(args)
        self.prefetch_guard=threading.Lock();self.prefetch_thread=None;self.prefetch_stop=threading.Event()
    def start_prefetch(self):
        with self.prefetch_guard:
            if self.prefetch_thread is not None:return
            def work():
                try:prepare(self,self.prefetch_stop)
                except Exception as error:print(f'[probe prefetch failed] {error}',flush=True)
            self.prefetch_thread=threading.Thread(target=work,name='probe-prefetch',daemon=True)
            self.prefetch_thread.start()
            write_json(self.out/'probe_device.json',{'mode':'shared existing generation model and GPU lock',
                'gpus':self.config['models'][self.args.model]['gpus'],'wrapper_sha256':sha(Path(__file__))})
    def detect(self,*args,**kwargs):
        self.start_prefetch();return super().detect(*args,**kwargs)
    def process(self,*args,**kwargs):
        self.start_prefetch();return super().process(*args,**kwargs)
    def execute(self):
        try:super().execute()
        finally:
            self.prefetch_stop.set()
            if self.prefetch_thread:self.prefetch_thread.join()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['run','smoke'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',choices=['gemma3'],required=True);p.add_argument('--max-batch',type=int,default=8);p.add_argument('--token-budget',type=int,default=98304)
    p.add_argument('--detect-first',action='store_true');p.add_argument('--methods');p.add_argument('--phase',default='all');p.add_argument('--generate-only',action='store_true',default=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,cfg['models'][a.model]['gpus']))
    PrefetchWorker(a).execute()
