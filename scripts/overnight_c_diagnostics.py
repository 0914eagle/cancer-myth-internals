"""Bounded, resumable two-GPU C diagnostics. No judge/API calls or test evaluation."""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import json
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.jsonl import load_json
from src.pilot import FP_CORRECT, digest, file_digest, frozen_json, load_manifest

VERSION = "overnight-c-v1"


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


@contextlib.contextmanager
def run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'coordinator.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This output directory already has an active coordinator')
        yield


def pick(items, n, seed=17):
    ordered = sorted(items, key=lambda x: x['id'])
    random.Random(seed).shuffle(ordered)
    return ordered[:n]


def check_sources(plan):
    for path, expected in plan['sources'].items():
        if file_digest(path) != expected:
            raise ValueError(f'Frozen source changed: {path}. Use a new output directory.')
    if digest(load_config(plan['config_path'])) != plan['config_hash']:
        raise ValueError('Resolved config changed')


def prepare(pilot, quick_dir, out, config, pair_limit=24, fit_limit=64):
    from scripts.make_paired_rows import pick_references
    from scripts import quick_pilot
    from scripts.quick_pilot_readout import read_scores
    from scripts.run_pilot import comparison_identity
    import numpy as np

    pilot, quick_dir, out, config = map(lambda p: Path(p).resolve(), (pilot, quick_dir, out, config))
    if out == quick_dir or out == pilot or out == pilot/'fit':
        raise ValueError('Use a new, separate overnight output directory')
    if pair_limit < 1 or fit_limit < 2:
        raise ValueError('Positive diagnostic pair limit and at least two fit questions required')
    cfg = load_config(config)
    manifest_path = pilot/'split/manifest.json'; manifest = load_manifest(manifest_path)
    fit_path = pilot/'fit/fit.json'; fit = load_json(fit_path)
    old_path = pilot/'fit/directions.npz'
    quick_plan, judge_plan = quick_pilot.judge_plan(quick_dir)
    _, scores, _, missing = read_scores(quick_dir, judge_plan)
    if missing or len(scores) != judge_plan['budget']:
        raise ValueError('Quick pilot judging must already be complete; overnight never calls a judge')
    plain_rows = (quick_pilot.new_rows(quick_dir, quick_plan, 'plain')
                  if 'plain' in quick_plan.get('generation_conditions',[]) else quick_plan['baseline_responses']['plain'])
    if plain_rows is None:
        raise ValueError('Quick Plain generation is incomplete')
    for q in quick_plan['questions']:
        key=judge_plan['mapping']['plain'][q['id']]
        if digest(plain_rows[q['id']]['response']) != judge_plan['cases'][key]['answer_hash']:
            raise ValueError('Plain answer differs from the answer actually judged')
    if (fit['manifest_hash'] != manifest['manifest_hash'] or quick_plan['manifest_hash'] != manifest['manifest_hash']
            or comparison_identity(fit['identity']) != comparison_identity(quick_plan['identity'])
            or digest(fit['identity']['source_model']) != digest(cfg['source_model'])):
        raise ValueError('Fit, quick pilot, split and configured model must match')
    raw_path = Path(cfg['data']['cancer_myth_repo'])/'data/all_data.json'
    if file_digest(raw_path) != manifest['reference_hash']:
        raise ValueError('Reference answers differ from the original split')
    references = load_json(raw_path)
    all_q = manifest['questions']
    fit_q = [q for q in all_q if q['partition'] == 'fit' and q['set'] == 'fpq']
    dev_q = [q for q in all_q if q['partition'] == 'dev' and q['set'] == 'fpq']
    if {q['group_id'] for q in fit_q} & {q['group_id'] for q in dev_q}:
        raise ValueError('Fit/dev source groups overlap')

    def pairs_for(qs):
        texts = {q['question'].strip() for q in qs}
        chosen, _ = pick_references([r for r in references if r['example_question'].strip() in texts])
        return [{'id': q['id'], 'question': q['question'], 'group_id': q['group_id'],
                 'positive': chosen[q['question'].strip()]['corr'][1],
                 'negative': chosen[q['question'].strip()]['follow'][1],
                 'positive_author': chosen[q['question'].strip()]['corr'][0],
                 'negative_author': chosen[q['question'].strip()]['follow'][0]}
                for q in qs if len(chosen.get(q['question'].strip(), {})) == 2]

    fit_pairs = pairs_for(fit_q)
    if set(fit['pair_ids']) != {p['id'] for p in fit_pairs}:
        raise ValueError('Reconstructed fit pairs differ from the saved direction fit')
    dev_pairs = pick(pairs_for(dev_q), pair_limit)
    if not dev_pairs:
        raise ValueError('No held-out dev +1/-1 reference pairs are available')
    authors = {side: dict(collections.Counter(p[f'{side}_author'] for p in fit_pairs))
               for side in ('positive', 'negative')}
    layers = []
    vector_audit = {}
    with np.load(old_path, allow_pickle=False) as vectors:
        for layer in (14, 21, 28):
            key = f'L{layer}_c_pair{fit["prefix_tokens"]}'
            if key not in vectors or f'L{layer}_norm_scale' not in vectors:
                continue
            v = vectors[key]; scale = float(vectors[f'L{layer}_norm_scale'])
            if v.shape != (cfg['source_model']['d_model'],) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1., atol=1e-4) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(f'Invalid original vector/scale at L{layer}')
            layers.append(layer)
            vector_audit[str(layer)] = {'norm': float(np.linalg.norm(v)), 'dimension': len(v), 'scale': scale}
    if 21 not in layers:
        raise ValueError('Need original L21 direction')
    tasks = []; comparisons = []

    def add(kind, group, **kw):
        task = dict(kind=kind, group=group, **kw)
        task['id'] = f'{len(tasks):05d}_{kind}_{digest(task)[:12]}'
        tasks.append(task)
        return task['id']

    def old(layer=21):
        return {'name': 'original_prefix32', 'path': str(old_path), 'key': f'L{layer}_c_pair{fit["prefix_tokens"]}',
                'scale_key': f'L{layer}_norm_scale', 'layer': layer}

    def dose(pair, direction, alpha, group, policy='all', dependencies=()):
        return add('dose', group, question_id=pair['id'], question=pair['question'],
                   positive=pair['positive'], negative=pair['negative'], direction=direction,
                   alpha=alpha, policy=policy, first_k=32, max_answer_tokens=512,
                   dependencies=list(dependencies), max_seconds=180)

    # Priority 1: a bounded causal response of the existing direction.
    for alpha in (0., .1, -.1, .3, .6, 1., -.3):
        for pair in dev_pairs:
            dose(pair, old(), alpha, '01_original_dose')

    # Priority 2: reference extraction diagnostics and fit-only own-model contrast.
    reextract_path = out/'directions/reference_reextract.npz'
    ref_job = add('extract', '02_reference_reextract', pairs=fit_pairs, layers=layers,
                  output_path=str(reextract_path), cache_dir=str(out/'extract_cache/reference'),
                  max_answer_tokens=512, max_seconds=5400, dependencies=[])
    own_q = pick(fit_q, fit_limit)
    own_sources = []
    for q in own_q:
        common = dict(question_id=q['id'], question=q['question'], direction=None, alpha=0.,
                      policy='all', max_new_tokens=512, max_seconds=240, dependencies=[], partition='fit')
        plain = add('generate', '03_fit_generation', **common, method='fit_plain')
        corrected = add('generate', '03_fit_generation', **common, method='fit_correct_prompt',
                        prompt=FP_CORRECT.format(question=q['question']))
        own_sources.append({'id': q['id'], 'question': q['question'], 'positive_task': corrected, 'negative_task': plain})
    own_path = out/'directions/gemma_instruction_contrast.npz'
    own_job = add('extract', '04_own_model_extract', pairs_from=own_sources, layers=layers,
                  output_path=str(own_path), cache_dir=str(out/'extract_cache/own_model'),
                  max_answer_tokens=512, max_seconds=5400,
                  dependencies=[v for p in own_sources for k,v in p.items() if k.endswith('_task')],
                  allow_partial_dependencies=True)

    # Priority 3: proper controls, location changes, and other already fitted layers.
    for seed in (17, 29, 43):
        direction = {**old(), 'name': f'random_{seed}', 'random_seed': seed}
        for alpha in (0., .1, .3, .6):
            for pair in dev_pairs[:12]:
                dose(pair, direction, alpha, '05_random_controls')
    for layer in layers:
        if layer != 21:
            for alpha in (0., -.1, .1, .3, .6):
                for pair in dev_pairs[:12]:
                    dose(pair, old(layer), alpha, '06_layer_comparison')
    for policy in ('prefill', 'first_k', 'ablate'):
        for alpha in ((0.,) if policy == 'ablate' else (.1, .3, .6)):
            for pair in dev_pairs[:12]:
                dose(pair, old(), alpha, '07_position_and_ablation', policy)

    # Own-answer projection: a readout diagnostic, not an input gate.
    for q in quick_plan['questions']:
        if q['set'] == 'fpq':
            label = scores[judge_plan['mapping']['plain'][q['id']]]
            add('project', '08_own_answer_readout', question_id=q['id'], question=q['question'],
                response=plain_rows[q['id']]['response'], label=label,
                direction=old(), max_answer_tokens=512, dependencies=[], max_seconds=180)

    # Distinguish extraction pooling from author/model differences.
    alternatives = []
    for name, path, pool, dependency in (
        ('reference_full', reextract_path, 'full', ref_job),
        ('gemma_instruction_prefix32', own_path, 'prefix32', own_job),
        ('gemma_instruction_full', own_path, 'full', own_job)):
        direction = {'name': name, 'path': str(path), 'key': f'L21_{pool}',
                     'scale_path': str(old_path), 'scale_key': 'L21_norm_scale', 'layer': 21}
        alternatives.append((direction, dependency))
        for alpha in (0., -.1, .1, .3, .6, 1.):
            for pair in dev_pairs[:12]:
                dose(pair, direction, alpha, '09_alternative_directions', dependencies=[dependency])
        comparisons.append({'name': name, 'left': old(), 'right': direction})
    comparisons.append({'name': 'original_vs_reextracted_prefix32', 'left': old(),
                        'right': {'path': str(reextract_path), 'key': 'L21_prefix32'}})

    # Final priority: readable free-generation examples, WITHOUT automatic grading.
    smoke = pick([q for q in quick_plan['questions'] if q['set'] == 'fpq'], 12) + pick([q for q in quick_plan['questions'] if q['set'] == 'nfp'], 6)
    settings = [(None, 0., 'all', [])]
    settings += [(old(), a, pol, []) for a,pol in ((.1,'all'),(.3,'all'),(.6,'all'),(-.1,'all'),(.3,'prefill'),(.3,'first_k'),(0.,'ablate'))]
    settings += [(d, a, 'all', [dep]) for d,dep in alternatives for a in (.1,.3)]
    for direction, alpha, policy, deps in settings:
        for q in smoke:
            add('generate', '10_free_generation_examples', question_id=q['id'], question=q['question'],
                set=q['set'], partition='dev', direction=direction, alpha=alpha, policy=policy, first_k=32,
                baseline_response=plain_rows[q['id']]['response'],
                max_new_tokens=512, dependencies=deps, max_seconds=240)

    sources = {str(path): file_digest(path) for path in (manifest_path, fit_path, old_path, raw_path,
               quick_dir/'plan.json', quick_dir/'judge_plan.json', quick_dir/'attempts.jsonl')}
    for rel in ('scripts/overnight_c_diagnostics.py','src/overnight_c_gpu.py','src/overnight_c_report.py',
                'src/pilot.py','src/pilot_model.py','src/steering.py','src/modeling.py','src/config.py','scripts/make_paired_rows.py'):
        path=ROOT/rel; sources[str(path)] = file_digest(path)
    plan = dict(version=VERSION, config_path=str(config), config_hash=digest(cfg), expected_identity=fit['identity'],
                manifest_hash=manifest['manifest_hash'], pilot=str(pilot), quick_dir=str(quick_dir), tasks=tasks,
                sources=sources, layers=layers, vector_audit=vector_audit, fit_pair_count=len(fit_pairs),
                fit_pair_authors=authors, dev_pair_ids=[p['id'] for p in dev_pairs], own_fit_ids=[q['id'] for q in own_q],
                judge_calls=0, direction_comparisons=comparisons,
                warnings=['No GPT/judge calls; teacher-forced likelihood is not PCR or clinical correctness.',
                          'Dev is exploratory and previously inspected. Test is never evaluated.',
                          'Gemma instruction-contrast pairs are unjudged, not verified correction-success/failure pairs.',
                          'Author-matched NFP reference pairs are unavailable here; no pure-style cosine is claimed.',
                          'Ablation removes the whole axis, not selectively the compliance component.',
                          '512-token answer cap retained. High alpha can degrade both reference likelihoods.'])
    frozen_json(out/'plan.json',plan)
    frozen_json(out/'reference_pairs.json', {'fit':fit_pairs,'dev':dev_pairs})
    print(f'Prepared {len(tasks)} tasks; {len(dev_pairs)} held-out reference pairs; {len(own_q)} fit-only own-model questions. GPT calls: 0.')
    return plan


def task_result(out, task):
    p=out/'tasks'/f'{task["id"]}.json'
    if not p.exists(): return None
    result=load_json(p)
    if result['id']!=task['id'] or result.get('task_hash')!=digest(task):
        raise ValueError(f'Task provenance mismatch: {task["id"]}')
    return result


def save_result(out, task, status, **kw):
    atomic_json(out/'tasks'/f'{task["id"]}.json', dict(id=task['id'],task_hash=digest(task),task=task,status=status,**kw))


def claim_next(out, plan, gpu):
    by_id={t['id']:t for t in plan['tasks']}
    for task in plan['tasks']:
        if task_result(out, task) is not None: continue
        deps=[task_result(out, by_id[i]) for i in task.get('dependencies',[])]
        if any(d is None for d in deps): continue
        claim=out/'claims'/task['id']; claim.parent.mkdir(parents=True,exist_ok=True)
        try: claim.mkdir()
        except FileExistsError: continue
        if task_result(out,task) is not None:
            claim.rmdir(); continue
        atomic_json(claim/'owner.json',{'pid':os.getpid(),'gpu':gpu,'started':time.time(),'id':task['id']})
        if any(d['status']!='complete' for d in deps) and not task.get('allow_partial_dependencies'):
            save_result(out,task,'skipped',error='Prerequisite task did not complete successfully',result={})
            continue
        return task
    return None


def resolve_task(out, task):
    if 'pairs_from' not in task: return task
    pairs=[]
    for pair in task['pairs_from']:
        paths=[out/'tasks'/f'{pair[k]}.json' for k in ('positive_task','negative_task')]
        results=[load_json(p) for p in paths]
        if any(r['status']!='complete' for r in results): continue
        pos,neg=[r['result']['response'] for r in results]
        if not pos.strip() or not neg.strip() or pos==neg: continue
        pairs.append({'id':pair['id'],'question':pair['question'],'positive':pos,'negative':neg})
    if len(pairs)<2: raise ValueError('Fewer than two distinct, completed fit-only instruction pairs')
    return {**task,'pairs':pairs}


def worker(out, gpu, deadline):
    plan=load_json(out/'plan.json'); check_sources(plan)
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from src.pilot_model import load_model,model_identity
    from scripts.run_pilot import comparison_identity
    from src.overnight_c_gpu import DiagnosticsRuntime
    if gpu not in (0,1) or torch.cuda.device_count()!=1:
        raise ValueError('Each worker must see exactly one of physical GPUs 0 and 1')
    cfg=load_config(plan['config_path']); model,tok=load_model(cfg)
    identity=model_identity(model,tok,cfg)
    if comparison_identity(identity)!=comparison_identity(plan['expected_identity']):
        raise ValueError('Loaded model/tokenizer/runtime differs from original fit')
    runtime=DiagnosticsRuntime(model,tok,cfg)
    while time.time()<deadline and not (out/'STOP').exists():
        task=claim_next(out,plan,gpu)
        if task is None:
            if all(task_result(out,t) is not None for t in plan['tasks']): break
            time.sleep(2); continue
        started=time.time()
        atomic_json(out/f'worker_{gpu}.json',{'pid':os.getpid(),'gpu':gpu,'task':task['id'],'started':started})
        try:
            runtime.deadline = time.monotonic() + max(0., min(deadline-started, task.get('max_seconds',300)-5))
            result=runtime.execute(resolve_task(out,task))
            status = 'timed_out' if result.get('stopped_by_deadline') else 'complete'
            if task['kind']=='generate' and not result.get('response','').strip():
                status = 'failed'
            save_result(out,task,status,result=result,elapsed_seconds=time.time()-started,gpu=gpu)
        except Exception as exc:
            save_result(out,task,'failed',result={},error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-started,gpu=gpu)
            print(f'FAILED {task["id"]}: {exc}',flush=True)
            if isinstance(exc,torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
            elif 'CUDA' in str(exc): raise
        atomic_json(out/f'worker_{gpu}.json',{'pid':os.getpid(),'gpu':gpu,'task':None,'started':time.time()})
    print(f'GPU {gpu} worker finished',flush=True)


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGTERM)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL); proc.wait(timeout=10)


def record_abandoned(out,plan,pid,reason):
    for task in plan['tasks']:
        owner=out/'claims'/task['id']/'owner.json'
        if owner.exists() and load_json(owner)['pid']==pid and task_result(out,task) is None:
            save_result(out,task,'timed_out',result={},error=reason)


def run(out,hours,gpus):
    from src.overnight_c_report import write_report
    if not 0<hours<=9 or not gpus or set(gpus)-{0,1} or len(set(gpus))!=len(gpus):
        raise ValueError('Use at most 9 hours and physical GPUs 0 and/or 1 only')
    with run_lock(out):
        plan=load_json(out/'plan.json'); check_sources(plan)
        status_path=out/'run_status.json'
        previous=load_json(status_path) if status_path.exists() else {}
        used=previous.get('elapsed_seconds',0.)
        start=time.time(); remaining=max(0.,hours*3600-used); deadline=start+remaining
        # An exclusive coordinator owns cleanup. Preserve completed/failed records.
        (out/'STOP').unlink(missing_ok=True)
        for claim in (out/'claims').glob('*') if (out/'claims').exists() else []:
            owner=claim/'owner.json'
            if owner.exists():
                old=load_json(owner)
                try: os.kill(old['pid'],0)
                except ProcessLookupError: pass
                else: raise ValueError(f'An old worker may still be running: PID {old["pid"]}. Stop it before resuming.')
                owner.unlink()
            claim.rmdir()
        (out/'logs').mkdir(parents=True,exist_ok=True)
        procs={}; logs={}; starts=collections.Counter(); reason='complete'; last_print=0
        def launch(gpu):
            env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu);env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
            log=(out/'logs'/f'gpu{gpu}.log').open('a');logs[gpu]=log
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--out-dir',str(out),
                                   '--gpu',str(gpu),'--deadline',str(deadline)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            procs[gpu]=proc;starts[gpu]+=1
            atomic_json(out/f'worker_{gpu}.json',{'pid':proc.pid,'gpu':gpu,'task':None,'started':time.time(),'loading':True})
        try:
            if remaining>0:
                for gpu in gpus: launch(gpu)
            while procs:
                now=time.time(); counts=collections.Counter((task_result(out,t) or {}).get('status','pending') for t in plan['tasks'])
                state=dict(pid=os.getpid(),status='running',started=start,deadline=deadline,elapsed_seconds=used+now-start,
                           hours_limit=hours,gpus=gpus,counts=dict(counts),judge_calls=0,worker_pids={g:p.pid for g,p in procs.items()})
                atomic_json(status_path,state)
                if now-last_print>=60:
                    print(f'Progress {dict(counts)}; remaining {max(0,int(deadline-now))}s; GPT calls 0',flush=True);last_print=now
                if counts['pending']==0: break
                if now>=deadline or (out/'STOP').exists():
                    reason='time_limit' if now>=deadline else 'stopped';break
                for gpu,proc in list(procs.items()):
                    w=load_json(out/f'worker_{gpu}.json')
                    active=next((t for t in plan['tasks'] if t['id']==w.get('task')),None)
                    limit=active.get('max_seconds',300) if active else 900
                    if now-w['started']>limit:
                        stop_process(proc)
                    if proc.poll() is not None:
                        record_abandoned(out,plan,proc.pid,'Worker exit or per-task timeout')
                        logs[gpu].close();del procs[gpu]
                        if starts[gpu]<3: launch(gpu)
                if not procs: reason='workers_failed';break
                time.sleep(5)
        except KeyboardInterrupt:
            reason='interrupted'
        except Exception:
            reason='coordinator_failed'
            raise
        finally:
            for gpu,proc in procs.items():
                stop_process(proc);record_abandoned(out,plan,proc.pid,f'Run ended: {reason}');logs[gpu].close()
            if remaining<=0: reason='time_limit'
            counts=collections.Counter((task_result(out,t) or {}).get('status','pending') for t in plan['tasks'])
            atomic_json(status_path,dict(pid=os.getpid(),status=reason,elapsed_seconds=used+time.time()-start,
                        hours_limit=hours,gpus=gpus,counts=dict(counts),judge_calls=0))
            write_report(out)
        print(f'Finished ({reason}). Read {out / "report.md"}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run','worker','report','status','stop'])
    p.add_argument('--out-dir',required=True);p.add_argument('--pilot-dir');p.add_argument('--quick-dir')
    p.add_argument('--config',default='configs/gemma2_9b.yaml');p.add_argument('--hours',type=float,default=9)
    p.add_argument('--gpus',type=int,nargs='+',default=[0,1]);p.add_argument('--gpu',type=int);p.add_argument('--deadline',type=float)
    p.add_argument('--pairs',type=int,default=24);p.add_argument('--fit-questions',type=int,default=64)
    a=p.parse_args();out=Path(a.out_dir).resolve()
    if a.stage=='prepare':
        if not a.pilot_dir or not a.quick_dir:p.error('prepare requires --pilot-dir and --quick-dir')
        prepare(a.pilot_dir,a.quick_dir,out,a.config,a.pairs,a.fit_questions)
    elif a.stage=='run':run(out,a.hours,a.gpus)
    elif a.stage=='worker':worker(out,a.gpu,a.deadline)
    elif a.stage=='report':
        from src.overnight_c_report import write_report
        print(write_report(out))
    elif a.stage=='status':
        print((out/'run_status.json').read_text() if (out/'run_status.json').exists() else 'Not started')
    elif a.stage=='stop':
        out.mkdir(parents=True,exist_ok=True);(out/'STOP').touch();print('Stop requested; coordinator will end its own workers.')


if __name__=='__main__':main()
