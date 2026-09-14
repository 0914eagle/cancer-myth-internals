import json
from pathlib import Path
import numpy as np
import pytest

from scripts import overnight_c_diagnostics as runner
from src.pilot import digest, file_digest, make_manifest


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from scripts import quick_pilot, quick_pilot_readout
    pilot=tmp_path/'pilot'; quick=pilot/'quick'; out=pilot/'night'
    for part in ('split','fit','quick','external/data'):(pilot/part).mkdir(parents=True)
    qs=[{'id':f'{part}_{i}', 'question':f'{part} question {i}', 'group_id':f'{part}_{i}',
         'set':'fpq','partition':part} for part in ('fit','dev','test') for i in range(4)]
    qs += [{'id':f'nfp_{i}','question':f'normal {i}','group_id':f'nfp_{i}','set':'nfp','partition':'dev'} for i in range(2)]
    raw=[{'example_question':q['question'],'answers':{'A':'Correct answer','B':'Following answer'},'scores':{'A':1,'B':-1}} for q in qs if q['set']=='fpq']
    raw_path=pilot/'external/data/all_data.json';raw_path.write_text(json.dumps(raw))
    manifest=make_manifest(qs,seed=17,folds=5,test_fold=0,dev_fold=1,reference_hash=file_digest(raw_path))
    (pilot/'split/manifest.json').write_text(json.dumps(manifest))
    source={'model_id':'fake','d_model':4,'max_memory':{0:'22GiB'}}
    identity={'source_model':json.loads(json.dumps(source)), 'implementation_hash':'same'}
    fit={'manifest_hash':manifest['manifest_hash'],'identity':identity,'pair_ids':[q['id'] for q in qs if q['partition']=='fit'], 'prefix_tokens':32}
    (pilot/'fit/fit.json').write_text(json.dumps(fit))
    arrays={}
    for layer in (14,21,28):arrays.update({f'L{layer}_c_pair32':np.ones(4)/2,f'L{layer}_norm_scale':np.array(3.)})
    np.savez(pilot/'fit/directions.npz',**arrays)
    dev=[q for q in qs if q['partition']=='dev']
    plan={'manifest_hash':manifest['manifest_hash'],'identity':identity,'questions':dev,
          'baseline_responses':{'plain':{q['id']:{'response':'Baseline answer'} for q in dev}}}
    spec={'budget':1,'mapping':{'plain':{q['id']:'key' for q in dev}},
          'cases':{'key':{'answer_hash':digest('Baseline answer')}}}
    for name,obj in (('plan.json',plan),('judge_plan.json',spec)):(quick/name).write_text(json.dumps(obj))
    (quick/'attempts.jsonl').write_text('')
    monkeypatch.setattr(quick_pilot,'judge_plan',lambda _: (plan,spec))
    monkeypatch.setattr(quick_pilot_readout,'read_scores',lambda *a: ({},{'key':1},[],[]))
    cfg={'source_model':source,'data':{'cancer_myth_repo':str(pilot/'external')}}
    monkeypatch.setattr(runner,'load_config',lambda _:cfg)
    result=runner.prepare(pilot,quick,out,tmp_path/'config.yaml',pair_limit=4,fit_limit=4)
    return pilot,quick,out,result


def test_plan_fit_dev_separation_controls_budget_and_no_judge(prepared):
    _,_,out,plan=prepared
    assert plan['judge_calls']==0
    assert not set(plan['own_fit_ids']) & set(plan['dev_pair_ids'])
    assert all(not t.get('question_id','').startswith('test') for t in plan['tasks'])
    assert {t['kind'] for t in plan['tasks']}=={'dose','project','generate','extract'}
    assert {t.get('policy') for t in plan['tasks'] if t['kind']=='dose'}=={'all','first_k','prefill','ablate'}
    assert sum(t['group']=='03_fit_generation' for t in plan['tasks'])==8
    seen=set()
    for t in plan['tasks']:
        assert set(t.get('dependencies',[]))<=seen
        seen.add(t['id'])
    runner.check_sources(plan)
    assert (out/'reference_pairs.json').exists()


def test_claims_unique_and_failed_dependencies_do_not_stop_other_work(tmp_path):
    tasks=[{'id':'a','kind':'dose'}, {'id':'b','kind':'dose','dependencies':['a']}, {'id':'c','kind':'dose'}]
    plan={'tasks':tasks}
    assert runner.claim_next(tmp_path,plan,0)['id']=='a'
    assert runner.claim_next(tmp_path,plan,1)['id']=='c'
    runner.save_result(tmp_path,tasks[0],'failed',error='failure',result={})
    assert runner.claim_next(tmp_path,plan,0) is None
    assert runner.task_result(tmp_path,tasks[1])['status']=='skipped'
    runner.save_result(tmp_path,tasks[2],'complete',result={})
    assert runner.claim_next(tmp_path,plan,1) is None


def test_instruction_extract_uses_only_finished_distinct_pairs(tmp_path):
    pairs=[]
    for i in range(3):
        p={'id':str(i),'question':'q','positive_task':f'p{i}','negative_task':f'n{i}'};pairs.append(p)
        for side in ('p','n'):
            task={'id':f'{side}{i}','kind':'generate'}
            runner.save_result(tmp_path,task,'complete',result={'response':side if i<2 else 'identical'})
    resolved=runner.resolve_task(tmp_path,{'kind':'extract','pairs_from':pairs})
    assert len(resolved['pairs'])==2
    assert [p['id'] for p in resolved['pairs']]==['0','1']
    with pytest.raises(ValueError,match='Fewer than two'):
        runner.resolve_task(tmp_path,{'kind':'extract','pairs_from':pairs[-1:]})


def test_source_change_and_task_tampering_fail(prepared):
    _,_,out,plan=prepared
    task=plan['tasks'][0]
    runner.save_result(out,task,'complete',result={})
    changed={**task,'alpha':99}
    with pytest.raises(ValueError,match='provenance'):runner.task_result(out,changed)
    Path(next(iter(plan['sources']))).write_text('changed')
    with pytest.raises(ValueError,match='Frozen source'):runner.check_sources(plan)


def test_gpu_allowlist_and_time_limit_checked_before_launch(tmp_path):
    for hours,gpus in [(10,[0,1]),(0,[0]),(1,[2]),(1,[0,0]),(1,[])]:
        with pytest.raises(ValueError,match='physical GPUs'):
            runner.run(tmp_path,hours,gpus)


def test_coordinator_lock_excludes_duplicate_start(tmp_path):
    with runner.run_lock(tmp_path):
        with pytest.raises(ValueError,match='active coordinator'):
            with runner.run_lock(tmp_path):pass


def test_coordinator_two_gpu_completion_includes_failure_and_real_report(tmp_path, monkeypatch):
    tasks=[{'id':str(i),'kind':'generate','partition':'dev','question_id':str(i),
            'question':'A question','baseline_response':'Plain','direction':None,'alpha':0.,'policy':'all'}
           for i in range(3)]
    runner.atomic_json(tmp_path/'plan.json',{'tasks':tasks,'judge_calls':0})
    monkeypatch.setattr(runner,'check_sources',lambda _:None)
    launched=[]

    class FinishedWorker:
        def __init__(self,cmd,**kwargs):
            gpu=kwargs['env']['CUDA_VISIBLE_DEVICES']; launched.append(gpu)
            assert kwargs['start_new_session'] is True
            assert cmd[cmd.index('--gpu')+1]==gpu
            self.pid=999900+int(gpu)
            for task in (tasks[:2] if gpu=='0' else tasks[2:]):
                if task['id']=='1':runner.save_result(tmp_path,task,'failed',result={},error='Synthetic OOM')
                else:runner.save_result(tmp_path,task,'complete',result={'response':'New answer','output_tokens':2,'hit_token_cap':False})
        def poll(self):return 0

    monkeypatch.setattr(runner.subprocess,'Popen',FinishedWorker)
    runner.run(tmp_path,9,[0,1])
    assert launched==['0','1']
    state=json.loads((tmp_path/'run_status.json').read_text())
    assert state['counts']=={'complete':2,'failed':1}
    assert state['status']=='complete' and state['judge_calls']==0
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['statuses']['failed']==1 and summary['statuses']['complete']==2
    assert 'Synthetic OOM' in (tmp_path/'report.md').read_text()
    assert 'New answer' in (tmp_path/'generated_examples.md').read_text()


def test_exhausted_budget_does_not_launch_gpu_and_reports_pending(tmp_path,monkeypatch):
    runner.atomic_json(tmp_path/'plan.json',{'tasks':[{'id':'unstarted','kind':'generate'}]})
    runner.atomic_json(tmp_path/'run_status.json',{'elapsed_seconds':9*3600})
    monkeypatch.setattr(runner,'check_sources',lambda _:None)
    def forbidden(*a,**kw):raise AssertionError('A spent budget must not load another model')
    monkeypatch.setattr(runner.subprocess,'Popen',forbidden)
    runner.run(tmp_path,9,[0,1])
    state=json.loads((tmp_path/'run_status.json').read_text())
    assert state['status']=='time_limit' and state['counts']=={'pending':1}
    assert (tmp_path/'report.md').exists()
