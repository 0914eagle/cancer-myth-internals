from types import SimpleNamespace
from scripts.run_paper_self_extraction import SelfExtractionPipeline
from scripts.queue_paper_self_extraction import ready_for_handoff
from src.paper_baselines import Pipeline

def test_extraction_uses_same_model_call_and_is_reused_across_methods():
    calls=[];cache={}
    def call(messages):
        key=str(messages)
        if key not in cache:calls.append(messages);cache[key]='claim'
        return cache[key]
    def bridge(**args):
        if args['op']=='render':return {'messages':[{'role':'user','content':args['kwargs']['question']}],'parser':'list'}
        return [args['text']]
    forbidden=SimpleNamespace(call=lambda *a:(_ for _ in ()).throw(AssertionError('external extraction used')))
    pipeline=SelfExtractionPipeline({'prompts':{}},call,bridge,forbidden,SimpleNamespace())
    for _ in ('prewome','extract_verify'):
        assert pipeline.well_stage('prewome','PresuppositionExtractionTemplate',question='Q')==('claim',['claim'])
    assert len(calls)==1

def test_metadata_marks_actual_extraction_model(monkeypatch):
    monkeypatch.setattr(Pipeline,'run',lambda *args,**kw:{'answer':'A'})
    worker=SimpleNamespace(args=SimpleNamespace(model='gemma3'),config={'models':{'gemma3':{'model_id':'google/gemma-3-12b-it'}}})
    p=SelfExtractionPipeline({'prompts':{}},None,None,SimpleNamespace(config={'model':'google/gemma-3-12b-it'}),worker)
    r=p.run('prewome','Q','cancer_myth')
    assert r['extraction_model']=='google/gemma-3-12b-it'
    assert r['extraction_protocol']=='same_model_extraction_v1'

def test_no_handoff_during_gepa_or_failed_progress():
    assert not ready_for_handoff({'state':'running','stage':'table12','method':'gepa_both'})
    assert not ready_for_handoff({'state':'failed','stage':'table3'})
    assert ready_for_handoff({'state':'running','stage':'table3'})
