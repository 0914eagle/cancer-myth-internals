"""Move Gemma to physical GPUs 1,2,3 without invalidating model/prompt/fold caches.

The frozen config retains its original placement for provenance. This wrapper
records the actual placement separately before any model is loaded.
"""
import os,sys,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_paper_baselines_sol as entry
from src.paper_baselines import sha,write_json


class PlacementWorker(entry.PrefetchWorker):
    def __init__(self,args):
        if args.model!='gemma3':raise ValueError('This placement is for Gemma only')
        os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
        super().__init__(args)
        self.placement_lock=threading.Lock()
        self.placement_recorded=False
        write_json(self.out/'device_overrides/gpu1_gpu2_gpu3.json',{
            'physical_gpus':[1,2,3],'original_config_gpus':self.config['models']['gemma3']['gpus'],
            'cuda_visible_devices':'1,2,3','wrapper_sha256':sha(Path(__file__)),
            'model_prompt_split_and_decoding_unchanged':True},immutable=True)

    def start_prefetch(self):
        with self.placement_lock:
            super().start_prefetch()
            if not self.placement_recorded:
                write_json(self.out/'probe_device.json',{
                    'mode':'shared existing generation model and GPU lock','gpus':[1,2,3],
                    'wrapper_sha256':sha(Path(__file__))})
                self.placement_recorded=True


if __name__=='__main__':
    if '--model' not in sys.argv or sys.argv[sys.argv.index('--model')+1]!='gemma3':
        raise SystemExit('Pass --model gemma3')
    entry.PrefetchWorker=PlacementWorker
    entry.main()
