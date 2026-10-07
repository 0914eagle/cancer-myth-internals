"""Exercise the installed GEPA adapter with fake task/judge/reflection; no API/GPU."""
import json
import sys
from types import SimpleNamespace
import pytest


def test_gepa_response_adapter_pair_budget(tmp_path,monkeypatch):
    pytest.importorskip('gepa.optimize_anything')
    from scripts.run_paper_baselines import Worker
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*a,**k):return cls()
        def encode(self,v,**kw):return v.split()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=Tokenizer))
    w=object.__new__(Worker)
    w.config={'cache_root':str(tmp_path),'gepa':{'metric_budget':24,'seed':42,'dev_per_class_max':2,
                'tokenizer_revision':'test','prompt_tokenizer':'fake','prompt_token_limit':512}}
    w.out=tmp_path;w.identity={'test':True};w.reg={'prompts':{'plain':'seed','cot_detector':'seed'}}
    w.lookup={str(i):{'id':str(i),'label':i%2,'question':f'q{i}'} for i in range(12)}
    w.pipeline=SimpleNamespace(run=lambda method,q,source,candidate:{'answer':candidate['answer']})
    w.judge=lambda row,answer:(5 if answer=='improved' else 1,'test score')
    w.external_call=lambda role,msg:'```\nimproved\n```'
    split={'dataset':'cancer_myth','name':'fold0','train':[str(i) for i in range(8)],'dev':[str(i) for i in range(8,12)]}
    result=w.optimize(split,'gepa')
    assert result['answer']=='improved'
    plan=json.loads((tmp_path/'gepa/cancer_myth/fold0/gepa/plan.json').read_text())
    assert plan['budget_pair_calls']==12
    assert not set(plan['train_ids'])&set(plan['dev_ids'])
