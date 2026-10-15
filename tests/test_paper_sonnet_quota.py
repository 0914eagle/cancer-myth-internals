import json
from pathlib import Path
import pytest
from scripts.watch_paper_sonnet_quota import Supervisor, quota_error


@pytest.mark.parametrize('error', [
    "Claude failed: You've hit your limit · resets 5am (UTC)",
    'CLI exited 1: You’ve hit your limit',
    'HTTP 429 Too Many Requests',
    'rate_limit_error',
    'Usage limit reached',
    'quota exhausted',
])
def test_recognizes_usage_limits(error):
    assert quota_error(error)


@pytest.mark.parametrize('error', [
    'CUDA out of memory', 'STOP requested', 'Input exceeds pinned limit',
    'Unexpected served model(s)', 'Invalid API key', 'Judge must produce exactly one Rating',
    'Generation token limit reached', 'Input token limit exceeded', 'Candidate length limit exceeded',
])
def test_other_errors_are_not_usage_limits(error):
    assert not quota_error(error)


@pytest.fixture
def watcher(tmp_path):
    external=tmp_path/'external.json'
    external.write_text(json.dumps({'judge_model':{'backend':'claude','model':'claude-sonnet-5-5'}}))
    config=tmp_path/'runtime.json'
    config.write_text(json.dumps({'external_config':str(external),'models':{'qwen25':{},'gemma3':{}}}))
    out=tmp_path/'main'; (out/'logs').mkdir(parents=True)
    clock={'now':1000.0}
    supervisor=Supervisor(config,tmp_path/'bundle.json',out,Path('/python'),clock=lambda:clock['now'])
    supervisor.test_clock=clock
    return supervisor


def stopped_suite(watcher,model,error,phase='fixed'):
    pid=987654321
    (watcher.out/'logs'/f'{model}-supervisor.state').write_text(
        f'pid={pid}\nphase={phase}\nstate=running\nexit_code=1\n')
    (watcher.out/'logs'/f'{model}-run-{phase}.log').write_text(
        'Traceback (most recent call last):\nRuntimeError: '+error)
    watcher.adopt(model,pid)


def test_shared_ping_every_ten_minutes_and_resume_same_phase(watcher,monkeypatch):
    stopped_suite(watcher,'qwen25',"You've hit your limit")
    stopped_suite(watcher,'gemma3','usage limit reached',phase='trained')
    calls=[]; launches=[]
    def probe():
        calls.append(watcher.test_clock['now'])
        return {'ok':len(calls)==2,'quota_error':len(calls)==1}
    def launch(model,now):
        launches.append((model,watcher.models[model]['phase']))
        watcher.models[model]['state']='running'
    monkeypatch.setattr(watcher,'probe',probe)
    monkeypatch.setattr(watcher,'launch',launch)
    assert watcher.step(1000)
    assert watcher.next_probe==1600
    watcher.test_clock['now']=1599;watcher.step(1599)
    assert calls==[]
    watcher.test_clock['now']=1600;watcher.step(1600)
    assert calls==[1600] and not launches and watcher.next_probe==2200
    watcher.test_clock['now']=2200;watcher.step(2200)
    assert calls==[1600,2200]
    assert launches==[('qwen25','fixed'),('gemma3','trained')]
    assert watcher.next_probe is None


def test_stop_while_waiting_prevents_ping_and_resume(watcher,monkeypatch):
    stopped_suite(watcher,'qwen25','usage limit reached')
    watcher.step(1000)
    (watcher.out/'qwen25').mkdir();(watcher.out/'qwen25/STOP').touch()
    monkeypatch.setattr(watcher,'probe',lambda:pytest.fail('STOP must prevent provider probe'))
    assert not watcher.step(1600)
    assert watcher.models['qwen25']['state']=='stopped'


def test_unrelated_failure_is_not_retried(watcher,monkeypatch):
    stopped_suite(watcher,'gemma3','CUDA out of memory')
    monkeypatch.setattr(watcher,'probe',lambda:pytest.fail('No quota probe for model errors'))
    assert not watcher.step(1000)
    assert watcher.models['gemma3']['state']=='failed_other'


def test_current_failure_does_not_reuse_old_quota_log(watcher):
    stopped_suite(watcher,'qwen25','usage limit reached')
    item=watcher.models['qwen25']
    item['started_at']=10**12
    assert not quota_error(watcher.failure('qwen25'))


def test_recovered_fixed_phase_advances_to_trained(watcher,monkeypatch):
    stopped_suite(watcher,'qwen25','usage limit reached')
    item=watcher.models['qwen25']; item.update(adopted=False,state='running')
    class Completed:
        def poll(self):return 0
    watcher.children['qwen25']=Completed()
    launches=[]
    monkeypatch.setattr(watcher,'launch',lambda model,now:launches.append((model,watcher.models[model]['phase'])))
    assert watcher.step(1000)
    assert launches==[('qwen25','trained')]
