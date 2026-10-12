import json
from types import SimpleNamespace
import pytest
from scripts.run_paper_baselines_resilient import with_quota_retry, quota_error


def worker(path):
    return SimpleNamespace(out=path, external={'judge_model': {'backend': 'claude'}})


def test_quota_waits_ten_minutes_and_retries_same_request(tmp_path):
    clock = [0.]
    calls = []
    messages = [{'role': 'user', 'content': 'fixed question'}]
    def call(w, role, request):
        calls.append((clock[0], request))
        if len(calls) < 3:
            raise RuntimeError("CLI exited 1: You've hit your limit · resets later")
        return 'Rating: 4'
    result = with_quota_retry(worker(tmp_path), 'judge', messages, call,
                             now=lambda: clock[0], sleep=lambda s: clock.__setitem__(0, clock[0]+s))
    assert result == 'Rating: 4'
    assert [t for t, _ in calls] == [0, 600, 1200]
    assert all(m is messages for _, m in calls)
    assert json.loads((tmp_path/'quota_retry.json').read_text())['state'] == 'recovered'


@pytest.mark.parametrize('message', ['CUDA out of memory', 'Unparseable judge output',
                                   'Invalid API key', 'Unexpected served model(s)',
                                   'Insufficient credits', 'connection timed out'])
def test_other_errors_are_not_retried(tmp_path, message):
    def call(*args): raise RuntimeError(message)
    with pytest.raises(RuntimeError, match=message.replace('(', r'\(').replace(')', r'\)')):
        with_quota_retry(worker(tmp_path), 'judge', [], call,
                         sleep=lambda s: pytest.fail('Should not sleep'))


def test_stop_during_wait_and_resume_preserves_deadline(tmp_path):
    clock = [100.]
    (tmp_path/'quota_retry.json').write_text(json.dumps({'state':'waiting',
         'next_retry_epoch':600, 'quota_failures':1}))
    def sleep(seconds):
        clock[0] += seconds
        (tmp_path/'STOP').touch()
    with pytest.raises(RuntimeError, match='STOP requested'):
        with_quota_retry(worker(tmp_path), 'judge', [], lambda *a: pytest.fail('Too early'),
                         now=lambda:clock[0], sleep=sleep)
    (tmp_path/'STOP').unlink()
    calls=[]
    with_quota_retry(worker(tmp_path), 'judge', [], lambda *a: calls.append(clock[0]) or 'ok',
                     now=lambda:clock[0], sleep=lambda s:clock.__setitem__(0,clock[0]+s))
    assert calls == [600]


def test_rate_error_variants():
    for message in ['rate_limit_error', 'Rate limit exceeded', 'HTTP 429', 'Too many requests',
                    "You've hit your session limit · resets 4:40pm (Australia/Perth)",
                    "You've hit your weekly limit", 'CLI exited 1: {"api_error_status":429}']:
        assert quota_error(ValueError(message))
    assert not quota_error(OSError('HTTP 429'))


def test_recorded_session_limit_recovers_after_delay(tmp_path):
    clock=[0.]
    calls=[]
    def call(*args):
        calls.append(clock[0])
        if len(calls)==1:
            raise RuntimeError('CLI exited 1: {"api_error_status":429,"result":"You\'ve hit your session limit · resets 4:40pm (Australia/Perth)"}')
        return 'Rating: 4'
    assert with_quota_retry(worker(tmp_path),'judge',[],call,
        now=lambda:clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s))=='Rating: 4'
    assert calls==[0,600]
