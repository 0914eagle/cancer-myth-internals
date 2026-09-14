import json
from pathlib import Path

import numpy as np
import pytest

from scripts import quick_pilot as mod
from scripts.run_pilot import generation_rows
from src.jsonl import load_json, write_jsonl
from src.pilot import digest, file_digest, make_manifest


@pytest.fixture
def sample(tmp_path, monkeypatch):
    pilot=tmp_path/'pilot';out=tmp_path/'quick';(pilot/'split').mkdir(parents=True);(pilot/'dev').mkdir();(pilot/'fit').mkdir()
    qs=[{'id':f'{kind}{i}', 'set':kind,'partition':'dev','question':f'{kind} question {i}',
         'correction':f'FP target {i}','hallucination_text':f'NFP target {i}','group_id':f'g{kind}{i}'}
        for kind,n in [('fpq',35),('nfp',20)] for i in range(n)]
    test={'id':'locked','set':'fpq','partition':'test','question':'Never read for inference','group_id':'test'}
    manifest=make_manifest(qs+[test],seed=17,folds=5,test_fold=0,dev_fold=1,reference_hash='ref')
    (pilot/'split/manifest.json').write_text(json.dumps(manifest))
    ident={'source_model':{'model_id':'google/gemma-2-9b-it'},'implementation_hash':digest({n:file_digest(mod.ROOT/'src'/n) for n in ('pilot.py','pilot_model.py','steering.py')})}
    for method in mod.BASE:
        run={'method':method,'partition':'dev','manifest_hash':manifest['manifest_hash'],'identity':ident,'max_new_tokens':512,'review_tokens':128,'batch_size':1}
        path=pilot/'dev'/f'{method}.jsonl'
        write_jsonl(path,[{'id':q['id'],'question':q['question'],'set':q['set'], 'run_hash':digest(run),
                          'response':'FP response' if method=='fp_identification' else 'Plain response'} for q in qs])
        Path(str(path)+'.run.json').write_text(json.dumps(run))
    (pilot/'fit/fit.json').write_text(json.dumps({'manifest_hash':manifest['manifest_hash'],'identity':ident,'prefix_tokens':32}))
    np.savez(pilot/'fit/directions.npz',L21_c_pair32=np.ones(4)/2,L21_norm_scale=3.)
    examples=tmp_path/'examples.json';examples.write_text(json.dumps([{'example_question':'Example','example_assumption':'Assumption','answer':'Answer','score':{'Sharpness':1}}]))
    cfg={'source_model':ident['source_model'],'judge':{'examples_fpq':str(examples),'examples_nfp':str(examples)}}
    monkeypatch.setattr(mod,'load_config',lambda _:cfg)
    mod.prepare(pilot,out,tmp_path/'config.yaml')
    return pilot,out,manifest


def fake_generate(monkeypatch, calls):
    def run(cmd,**kwargs):
        calls.append(cmd)
        arg=lambda name:cmd[cmd.index(name)+1]
        path=Path(arg('--output'));plan=load_json(path.parent/'plan.json')
        method=arg('--method')
        spec={'method':method,'partition':'dev','manifest_hash':plan['manifest_hash'],'identity':plan['identity'],
              'review_tokens':int(arg('--review-tokens')),'max_new_tokens':512,'batch_size':1,
              'question_ids':[q['id'] for q in plan['questions']]}
        if method=='steering':spec.update(layer=21,alpha=.1,direction_hash=file_digest(Path(plan['pilot'])/'fit/directions.npz'),
                                         fit_spec_hash=arg('--reuse-fit-sha256'),fit_source_identity=plan['fit_identity'])
        Path(str(path)+'.run.json').write_text(json.dumps(spec))
        write_jsonl(path,[{'id':q['id'],'question':q['question'],'set':q['set'],'run_hash':digest(spec),
                          'response':'Plain response' if method=='steering' else 'Long review answer'} for q in plan['questions']])
    monkeypatch.setattr(mod.subprocess,'run',run)


def test_fixed_subset_generation_and_uniform_deduplicated_judge(sample,monkeypatch):
    pilot,out,_=sample;plan=mod.get_plan(out)
    assert len(plan['questions'])==45 and 'locked' not in [q['id'] for q in plan['questions']]
    before=(pilot/'dev/plain.jsonl').read_bytes()
    calls=[];fake_generate(monkeypatch,calls)
    mod.generate(out);mod.generate(out)
    assert len(calls)==2
    assert '--question-ids' in calls[1] and calls[1][calls[1].index('--alpha')+1]=='0.1'
    spec=mod.plan_score(out)
    assert spec['budget']==135  # 3 unique answers x 45; plain, old CoT and steering share answers
    for job in spec['cases'].values():
        assert 'Score (in JSON):' in job['prompt']
        if job['set']=='nfp':assert 'Possible hallucination' in job['prompt']
    judged=[]
    monkeypatch.setattr(mod,'make_caller',lambda *a,**k:lambda prompt:(judged.append(prompt) or json.dumps({'Sharpness':1,'Reason':'ok'},indent=2),mod.MODEL))
    mod.score(out);mod.score(out)
    assert len(judged)==135
    values=mod.report(out)
    assert values['plain']==values['premise_cot']==values['steering_L21_a0.1']
    assert (pilot/'dev/plain.jsonl').read_bytes()==before
    assert '100.0' in (out/'report.md').read_text()


def test_interrupted_call_consumes_budget_no_imputation(sample,monkeypatch):
    _,out,_=sample;fake_generate(monkeypatch,[]);mod.generate(out);spec=mod.plan_score(out)
    def interrupt(_):raise KeyboardInterrupt()
    monkeypatch.setattr(mod,'make_caller',lambda *a,**k:interrupt)
    with pytest.raises(KeyboardInterrupt):mod.score(out)
    calls=[]
    monkeypatch.setattr(mod,'make_caller',lambda *a,**k:lambda prompt:(calls.append(prompt) or json.dumps({'Sharpness':1,'Reason':'ok'},indent=2),mod.MODEL))
    mod.score(out);mod.score(out)
    assert len(calls)==spec['budget']-1
    values=mod.report(out)
    assert any(v is None for row in values.values() for v in row.values())
    assert 'incomplete' in (out/'report.md').read_text()


def test_invalid_parse_does_not_use_original_plus_one_fallback(sample,monkeypatch):
    _,out,_=sample;fake_generate(monkeypatch,[]);mod.generate(out);mod.plan_score(out)
    calls=[]
    monkeypatch.setattr(mod,'make_caller',lambda *a,**k:lambda p:(calls.append(p) or 'not JSON',mod.MODEL))
    mod.score(out)
    assert len(calls)==3
    assert all(v is None for row in mod.report(out).values() for v in row.values())


def test_source_tampering_fails_before_calls(sample):
    pilot,out,_=sample
    with (pilot/'dev/plain.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed'):mod.get_plan(out)


def test_dev_steering_subset_does_not_unlock_test(tmp_path):
    path=tmp_path/'ids.json';path.write_text('["a"]')
    manifest={'questions':[{'id':'a','partition':'dev'},{'id':'b','partition':'test'}]}
    assert generation_rows(manifest,'dev','steering',path)==[manifest['questions'][0]]
    with pytest.raises(ValueError,match='only available'):generation_rows(manifest,'test','steering',path)
    path.write_text('["b"]')
    with pytest.raises(ValueError,match='unique nonempty'):generation_rows(manifest,'dev','steering',path)


def test_code_change_refreshes_all_baselines_without_rewriting_sources(sample, monkeypatch):
    pilot, original_out, _ = sample
    for tag in mod.BASE:
        path = pilot / 'dev' / f'{tag}.jsonl'
        meta = Path(str(path) + '.run.json')
        run = load_json(meta)
        run['identity']['implementation_hash'] = 'old-generation-code'
        rows = list(mod.read_jsonl(path))
        for row in rows:
            row['run_hash'] = digest(run)
        meta.write_text(json.dumps(run))
        write_jsonl(path, rows)
    # Fit may have been created after a different alignment-code fix.
    fp = pilot / 'fit/fit.json'
    fit = load_json(fp); fit['identity']['implementation_hash'] = 'old-fit-code'
    fp.write_text(json.dumps(fit))
    out = original_out.parent / 'refresh'
    plan = mod.prepare(pilot, out, original_out.parent / 'config.yaml')
    before = {p: p.read_bytes() for p in (pilot/'dev').iterdir()}
    before[fp] = fp.read_bytes()
    assert plan['refresh_baselines'] and plan['generation_budget'] == 225
    assert plan['identity'] != plan['baseline_identity']
    calls = []; fake_generate(monkeypatch, calls)
    mod.generate(out); mod.generate(out)
    assert len(calls) == 5
    assert all('--baseline-refresh' in c for c in calls[:3])
    spec = mod.plan_score(out)
    assert spec['budget'] <= 225
    # Refreshed Plain differs from old Plain and is used for actual judging.
    qid = plan['questions'][0]['id']
    job = spec['cases'][spec['mapping']['plain'][qid]]
    assert job['answer_hash'] == digest('Long review answer')
    assert load_json(out/'baseline_refresh_audit.json')['plain']['answer_changed_ids']
    assert all(p.read_bytes() == content for p, content in before.items())
    path = out/'plain.jsonl.run.json'
    run = load_json(path); run['identity']['implementation_hash'] = 'unexpected-code'
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match='differs from frozen'):
        mod.plan_score(out)


def test_fit_reuse_requires_exact_artifact_and_unchanged_model_runtime_split(sample):
    from scripts.run_pilot import check_fit_compatibility
    pilot, _, manifest = sample
    path = pilot/'fit/fit.json'; fit = load_json(path)
    current = {**fit['identity'], 'implementation_hash': 'new-code'}
    sha = file_digest(path)
    with pytest.raises(ValueError, match='model/split'):
        check_fit_compatibility(path, current, manifest['manifest_hash'])
    assert check_fit_compatibility(path, current, manifest['manifest_hash'], reuse_sha=sha) == fit
    for changed in ({**current, 'source_model': 'different-model'}, {**current, 'torch_version': 'different-runtime'}):
        with pytest.raises(ValueError, match='model/split'):
            check_fit_compatibility(path, changed, manifest['manifest_hash'], reuse_sha=sha)
    with pytest.raises(ValueError, match='model/split'):
        check_fit_compatibility(path, current, 'different-split', reuse_sha=sha)
    for opts in ({'reuse_sha': 'wrong-hash'}, {'reuse_sha': sha, 'partition': 'test'}):
        with pytest.raises(ValueError, match='exact frozen fit'):
            check_fit_compatibility(path, current, manifest['manifest_hash'], **opts)


def test_refresh_subset_remains_dev_only(tmp_path):
    path = tmp_path/'ids.json'; path.write_text('["a"]')
    manifest = {'questions': [{'id': 'a', 'partition': 'dev'}, {'id': 'b', 'partition': 'test'}]}
    for method in mod.BASE:
        assert generation_rows(manifest, 'dev', method, path, True) == [manifest['questions'][0]]
        with pytest.raises(ValueError, match='only available'):
            generation_rows(manifest, 'test', method, path, True)


@pytest.mark.parametrize('method', [*mod.BASE, 'steering'])
def test_real_runner_refresh_and_fit_provenance(sample, monkeypatch, method):
    from types import SimpleNamespace
    from scripts import run_pilot as runner
    from src import pilot_model
    pilot, out, _ = sample
    plan = mod.get_plan(out)
    current = {**plan['identity'], 'implementation_hash': 'changed-code'}
    monkeypatch.setattr(runner, 'load_config', mod.load_config)
    monkeypatch.setattr(pilot_model, 'load_model', lambda _: (None, None))
    monkeypatch.setattr(pilot_model, 'model_identity', lambda *a: current)
    calls = []
    def fake_batch(model, tokenizer, questions, **kwargs):
        calls.extend(q['id'] for q in questions)
        assert kwargs['method'] == method
        return ['new answer' for _ in questions], [{} for _ in questions]
    monkeypatch.setattr(pilot_model, 'generate_batch', fake_batch)
    args = SimpleNamespace(config=plan['config_path'], manifest=pilot/'split/manifest.json',
                           partition='dev', method=method, max_new_tokens=512, review_tokens=128,
                           batch_size=1, question_ids=out/'question_ids.json', baseline_refresh=method in mod.BASE,
                           identity_reference=pilot/'dev/plain.jsonl.run.json', output=out/f'runner_{method}.jsonl',
                           fit_dir=pilot/'fit', layer=21, alpha=.1, reuse_fit_sha256=file_digest(pilot/'fit/fit.json'))
    runner.generate(args); runner.generate(args)
    assert len(calls) == 45 and set(calls) == set(load_json(out/'question_ids.json'))
    meta = load_json(str(args.output)+'.run.json')
    assert meta['identity'] == current
    if method == 'steering':
        assert meta['fit_source_identity'] == plan['fit_identity']
        assert meta['alpha_abs'] == pytest.approx(.3)
        assert meta['direction_hash'] == file_digest(pilot/'fit/directions.npz')
