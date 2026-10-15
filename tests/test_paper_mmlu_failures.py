from collections import defaultdict
import json
import subprocess
from types import SimpleNamespace
import pytest
from scripts.run_paper_baselines import Worker
from scripts.run_paper_baselines_mmlu_resilient import MMLUWorker, recoverable_item_error
from scripts.run_paper_baselines_resilient import record_paths


@pytest.mark.parametrize('error', [ValueError('Input exceeds pinned limit; no silent truncation'),
    RuntimeError('CUDA out of memory'), ValueError('Could not parse a model output'), RuntimeError('Generation failed')])
def test_recoverable_mmlu_errors(error):
    assert recoverable_item_error(error)


@pytest.mark.parametrize('error', [RuntimeError("You've hit your limit"),RuntimeError('STOP requested'),
    OSError('No space left'),ValueError('Frozen artifact differs'),RuntimeError('CUDA illegal memory access'),
    RuntimeError('Invalid API key'), json.JSONDecodeError('Broken cache','x',0)])
def test_global_errors_still_stop(error):
    assert not recoverable_item_error(error)


def test_generated_output_parser_error_is_recoverable_but_template_error_is_not():
    parse=subprocess.CalledProcessError(1,['python','paper_well_bridge.py'],
        stderr='model_validate_plain_text: pydantic.ValidationError: invalid generated output')
    assert recoverable_item_error(parse)
    render=subprocess.CalledProcessError(1,['python','paper_well_bridge.py'],stderr='SyntaxError in template')
    assert not recoverable_item_error(render)


def make_worker(tmp_path):
    w=MMLUWorker.__new__(MMLUWorker);w.out=tmp_path;w.config={'max_new_tokens':8192}
    w.args=SimpleNamespace(command='run');w.pending=defaultdict(set);w.status={}
    w.backend=None;w.needs_cuda_cleanup=False
    return w


def test_item_error_is_logged_and_next_item_runs(tmp_path,monkeypatch):
    worker=make_worker(tmp_path);calls=[]
    split={'dataset':'cancer_myth','name':'fold0'}
    row={'id':'mmlu:1','dataset':'mmlu','question':'Q','metric':'choice','choices':{'A':'a'},'gold':'A'}
    def evaluate(self,row,*args,**kwargs):
        calls.append(row['id'])
        if row['id']=='mmlu:1':raise ValueError('Input exceeds pinned limit; no silent truncation')
    monkeypatch.setattr(Worker,'process',evaluate)
    worker.process(row,split,'plain',None,None,True)
    worker.process({**row,'id':'mmlu:2'},split,'plain',None,None,True)
    output,failure=record_paths(tmp_path,row,split,'plain',True)
    assert not output.exists()
    recorded=json.loads(failure.read_text())
    assert recorded['error_type']=='ValueError' and 'Input exceeds' in recorded['error']
    assert not {'answer','rating','correct','gold'}&recorded.keys()
    assert calls==['mmlu:1','mmlu:2']


def test_other_dataset_error_does_not_get_silently_skipped(tmp_path,monkeypatch):
    worker=make_worker(tmp_path)
    monkeypatch.setattr(Worker,'process',lambda *a,**k:(_ for _ in ()).throw(ValueError('error')))
    with pytest.raises(ValueError):
        worker.process({'dataset':'gsm8k','id':'g','question':'Q'},
            {'dataset':'crepe','name':'official'},'plain',None,None,True)


def test_oom_releases_unused_cuda_memory_before_next_question(tmp_path,monkeypatch):
    worker=make_worker(tmp_path);cleanups=[];calls=[]
    worker.backend=SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:cleanups.append(True))))
    row={'id':'bad','dataset':'mmlu','question':'Q','metric':'choice','choices':{'A':'a'},'gold':'A'}
    split={'dataset':'crepe','name':'official'}
    def evaluate(self,row,*args,**kwargs):
        calls.append(row['id'])
        if row['id']=='bad':raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(Worker,'process',evaluate)
    worker.process(row,split,'plain',None,None,True)
    assert worker.needs_cuda_cleanup and not cleanups
    worker.process({**row,'id':'good'},split,'plain',None,None,True)
    assert calls==['bad','good'] and cleanups==[True] and not worker.needs_cuda_cleanup
