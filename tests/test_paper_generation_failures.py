from collections import defaultdict
import json
from types import SimpleNamespace
import pytest
from scripts.run_paper_baselines_pending import PendingWorker, TRUNCATED, record_paths, record_failure
from scripts.run_paper_baselines import Worker


@pytest.fixture
def worker(tmp_path):
    w=PendingWorker.__new__(PendingWorker)
    w.out=tmp_path;w.config={'max_new_tokens':8192};w.args=SimpleNamespace(command='run')
    w.pending=defaultdict(set);w.status={}
    return w


ROW={'id':'q','dataset':'mmlu','question':'QUESTION','metric':'choice','choices':{'A':'a','B':'b'},'gold':'A'}
SPLIT={'dataset':'cancer_myth','name':'fold0'}


def test_truncation_is_pending_without_answer_or_score(worker,monkeypatch):
    def truncated(*args,**kwargs):raise RuntimeError(TRUNCATED)
    monkeypatch.setattr(Worker,'process',truncated)
    worker.process(ROW,SPLIT,'plain',None,None,True)
    output,failure=record_paths(worker.out,ROW,SPLIT,'plain',True)
    assert not output.exists()
    record=json.loads(failure.read_text())
    assert record['max_new_tokens']==8192
    assert not {'answer','rating','correct','gold'}&record.keys()
    assert worker.status['cancer_myth/fold0/plain']['pending_generation_failures']==1


def test_recorded_failure_is_shared_across_fixed_folds_and_not_retried(worker,monkeypatch):
    record_failure(worker.out,worker.config,ROW,SPLIT,'plain',True)
    monkeypatch.setattr(Worker,'process',lambda *a,**k:pytest.fail('Do not regenerate known length failure'))
    worker.process(ROW,{'dataset':'crepe','name':'official'},'plain',None,None,True)
    assert worker.pending['crepe/official/plain']=={('mmlu','q')}


def test_quota_error_still_propagates_to_watchdog(worker,monkeypatch):
    def quota(*a,**k):raise RuntimeError("You've hit your limit")
    monkeypatch.setattr(Worker,'process',quota)
    with pytest.raises(RuntimeError,match='hit your limit'):
        worker.process(ROW,SPLIT,'plain',None,None,True)


def test_status_does_not_claim_complete_with_pending_rows(worker):
    worker.status={'cancer_myth/fold0/plain':{'state':'complete'},'table3':{'state':'generated'}}
    worker.note_pending(ROW,SPLIT,'plain')
    assert worker.status['cancer_myth/fold0/plain']['state']=='partial'
    assert worker.status['table3']['state']=='partial'


def test_successfully_recovered_row_takes_priority_over_old_failure(worker,monkeypatch):
    record_failure(worker.out,worker.config,ROW,SPLIT,'plain',True)
    output,_=record_paths(worker.out,ROW,SPLIT,'plain',True)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text('{}')
    calls=[]
    monkeypatch.setattr(Worker,'process',lambda *a,**k:calls.append(True))
    worker.process(ROW,SPLIT,'plain',None,None,True)
    assert calls==[True] and not worker.pending
