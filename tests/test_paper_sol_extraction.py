import json
from types import SimpleNamespace
import pytest
from scripts import paper_sol_extraction as module


def test_shared_sol_cache_is_model_independent_and_separate(tmp_path, monkeypatch):
    protocol=tmp_path/'protocol';protocol.mkdir()
    (protocol/'extraction_sol.json').write_text(json.dumps({'backend':'codex','model':'gpt-6-sol','effort':'high','timeout_seconds':240}))
    calls=[]
    monkeypatch.setattr(module,'PooledCLI',lambda cfg,root,stop:calls.append((cfg,root,stop)))
    monkeypatch.setattr(module,'LockedCachedTask',lambda *args:args)
    def worker(model):
        return SimpleNamespace(args=SimpleNamespace(out=tmp_path/'main'),out=tmp_path/'main'/model,
            identity={'registry_sha256':'fixed','well_source_hashes':{'source':'fixed'}},reg={'transport_common':'fixed'})
    a=module.SharedSolExtraction(worker('qwen25'));b=module.SharedSolExtraction(worker('gemma3'))
    assert a.out==b.out==tmp_path/'main/shared/sol_extraction'
    assert a.call[2]==b.call[2]
    assert all(c[0]['model']=='gpt-6-sol' and c[0]['effort']=='high' for c in calls)
    assert not (tmp_path/'main/shared/luna_extraction').exists()
    (protocol/'extraction_sol.json').write_text(json.dumps({'backend':'codex','model':'gpt-6-luna'}))
    with pytest.raises(ValueError,match='GPT-6 Sol'):module.SharedSolExtraction(worker('luna6'))


def test_sol_pipeline_marks_only_affected_methods(monkeypatch):
    monkeypatch.setattr(module.EvaluatedPipeline,'run',lambda *args,**kwargs:{'answer':'answer'})
    pipeline=object.__new__(module.SolPipeline)
    assert pipeline.run('prewome')['extraction_protocol']=='shared_premise_extraction_sol_v1'
    assert pipeline.run('extract_verify')['extraction_protocol']=='shared_premise_extraction_sol_v1'
    assert 'extraction_protocol' not in pipeline.run('plain')
