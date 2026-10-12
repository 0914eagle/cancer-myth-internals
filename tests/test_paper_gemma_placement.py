import json,os
from types import SimpleNamespace
from scripts.run_paper_gemma_placement import PlacementWorker


def test_placement_applied_before_worker_init_without_changing_config(tmp_path,monkeypatch):
    parent=PlacementWorker.__bases__[0]
    config={'models':{'gemma3':{'gpus':[1,2]}}}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1,2')
    def init(self,args):
        assert os.environ['CUDA_VISIBLE_DEVICES']=='2,3'
        self.out=tmp_path;self.config=config
    monkeypatch.setattr(parent,'__init__',init)
    monkeypatch.setattr(parent,'start_prefetch',lambda self:None)
    worker=PlacementWorker(SimpleNamespace(model='gemma3'))
    worker.start_prefetch()
    assert config['models']['gemma3']['gpus']==[1,2]
    assert json.loads((tmp_path/'device_overrides/gpu2_gpu3.json').read_text())['physical_gpus']==[2,3]
    assert json.loads((tmp_path/'probe_device.json').read_text())['gpus']==[2,3]
