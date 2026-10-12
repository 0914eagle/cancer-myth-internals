from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading
import pytest
from scripts.paper_batch_inference import completed_tokens, MicroBatch, LockedCachedTask


def test_per_row_eos_ignores_batch_padding():
    assert completed_tokens([11,2,0,0],{2})==[11,2]
    assert completed_tokens([11,12,0],{2}) is None
    assert completed_tokens([11,3,2],{2,3})==[11,3]


class Fake:
    class OOM(Exception):pass
    def __init__(self,oom_above=100):
        self.config={'max_input_tokens':20,'max_new_tokens':10}
        self.sizes=[];self.oom_above=oom_above
        self.torch=SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=self.OOM,empty_cache=lambda:None))
    def length(self,m):return len(m)
    def batch(self,ms):
        self.sizes.append(len(ms))
        if len(ms)>self.oom_above:raise self.OOM('fake')
        return [(m.upper(),{'size':len(ms)}) for m in ms]


def test_dispatcher_maps_results_and_enforces_budget():
    fake=Fake();q=MicroBatch(fake,max_batch=8,token_budget=26,wait_seconds=.05)
    try:
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(q,['a','bb','ccc','d']))
        assert [r[0] for r in results]==['A','BB','CCC','D']
        assert max(fake.sizes)<=2
        with pytest.raises(ValueError,match='Input exceeds'):
            q('x'*21)
    finally:q.close()


def test_oom_shrinks_without_dropping_or_changing_requests():
    fake=Fake(oom_above=2);q=MicroBatch(fake,max_batch=4,wait_seconds=.1)
    try:
        with ThreadPoolExecutor(4) as pool:
            results=list(pool.map(q,['a','bb','ccc','d']))
        assert [r[0] for r in results]==['A','BB','CCC','D']
        assert 4 in fake.sizes and q.max_batch==2
    finally:q.close()


def test_identical_concurrent_stages_only_generate_once(tmp_path):
    count=[];lock=threading.Lock()
    def backend(messages):
        with lock:count.append(messages)
        return 'same extraction',{}
    cache=LockedCachedTask(backend,tmp_path,{},'common')
    messages=[{'role':'system','content':'extract'},{'role':'user','content':'q'}]
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(cache,[messages]*4))==['same extraction']*4
    assert len(count)==1


def test_shared_extractor_never_uses_answer_model_for_extraction():
    from scripts.paper_shared_extraction import SharedExtractionPipeline
    extractor_calls=[];answer_calls=[]
    def bridge(**req):
        if req['op']=='render':
            return {'messages':[{'role':'system','content':'s'},{'role':'user','content':req['kwargs']['question']}],'parser':'claims'}
        return ['claim']
    extractor=SimpleNamespace(config={'model':'gpt-6-luna'},call=lambda m:extractor_calls.append(m) or 'claims')
    pipeline=SharedExtractionPipeline({'prompts':{}},lambda m:answer_calls.append(m) or 'review',bridge,extractor)
    raw,claims=pipeline.well_stage('prewome','PresuppositionExtractionTemplate',question='q')
    assert claims==['claim'] and raw=='claims'
    assert len(extractor_calls)==1 and not answer_calls
    pipeline.well_stage('prewome','FeedbackActionTemplate',question='q')
    assert len(answer_calls)==1 and len(extractor_calls)==1
