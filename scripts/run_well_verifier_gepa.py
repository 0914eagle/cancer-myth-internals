#!/usr/bin/env python3
"""Optimize the D+E verifier only, then compare frozen seed/winner on exposed test IDs.

Uses Well's GEPA version/engine settings, official extraction/few-shot/answer
parser and response judge. Custom evaluator/trace feedback: NOT unchanged Well GEPA.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import freeze, load_rows, response_metrics
from src.fpqa_well_upstream import UncachedCaller, FatalTransportError, load_upstream, make_args, verify_source
from src.fpqa_well_pipelines import load_templates, extract
from src.fpqa_scope_factorial import GROUND_SYSTEM, E_ACTION, ground_messages, parse_grounded
from src.fpqa_uncertainty import VERIFY_SYSTEM
from src.fpqa_verifier_gepa import (OBJECTIVE, BACKGROUND, validate_splits, execute_pipeline,
                                     feedback, paired_metrics, demonstrations_for)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['preflight', 'smoke', 'run', 'evaluate'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--previous-gepa', type=Path, default=ROOT/'results/fpqa_prompting/well_upstream_cli_v3_20261003/cancer_myth')
    p.add_argument('--scope-source', type=Path, default=ROOT/'results/fpqa_prompting/well_scope_factorial_luna6_20261004_v1')
    p.add_argument('--test', type=Path, default=ROOT/'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl')
    p.add_argument('--well-root', type=Path, default=ROOT/'external/fpqa_20261002/Well')
    p.add_argument('--workers', type=int, default=4, help='Preparation and fixed-candidate evaluation only; GEPA stays serial')
    p.add_argument('--test-repeats', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.workers <= 4 or a.test_repeats != 2:
        p.error('workers 1..4; fixed two evaluation repeats')
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((a.previous_gepa/'run.json').read_text())
    scope = json.loads((a.scope_source/'run.json').read_text())
    config = previous['config']
    for role in ['task', 'judge', 'reflection']:
        if config[f'{role}_model'] != scope['config'][f'{role}_model']:
            raise ValueError('Source model configurations differ')
    if importlib.metadata.version('gepa') != previous['gepa_version'] or previous['gepa_version'] != '0.1.1':
        raise ValueError('Need pinned GEPA 0.1.1')
    source_hashes = verify_source(a.well_root)
    if source_hashes != previous['source_sha256'] or source_hashes != scope['well_source_sha256']:
        raise ValueError('Upstream source mismatch')
    if sha(a.test) != scope['test_sha256']:
        raise ValueError('Test changed')
    splits = {}
    input_hashes = {}
    for split in ['train', 'val']:
        groups = []
        for kind, label in [('fpq', 1), ('tpq', 0)]:
            path = a.previous_gepa/'inputs'/f'{kind}_{split}.jsonl'
            input_hashes[str(path)] = sha(path)
            rows = [{**r, 'label': label, 'reference_answer': r['answer'], 'split': split}
                    for r in load_rows(path)]
            groups.append(rows)
        if len(groups[0]) != len(groups[1]):
            raise ValueError('Do not silently truncate unequal groups')
        splits[split] = [r for pair in zip(*groups) for r in pair]
    test_all = load_rows(a.test)
    splits['test'] = [{**r, 'split': 'test'} for r in test_all if r['id'] in scope['ids']]
    validate_splits(splits['train'], splits['val'], test_all)
    if [len(splits[s]) for s in ['train', 'val', 'test']] != [60, 36, 199]:
        raise ValueError('Unexpected comparison splits')
    fewpath = a.well_root/'data_gen/CancerMyth/few_shot_data.json'
    few = json.loads(fewpath.read_text())
    uncertainty = json.loads((Path(scope['source'])/'run.json').read_text())
    saved_few = Path(uncertainty['source'])/'few_shot_data.json'
    if sha(saved_few) != scope['few_shot_sha256'] or json.loads(saved_few.read_text()) != few:
        raise ValueError('Few-shot examples changed')
    norm = lambda x: ' '.join(x.casefold().split())
    overlap = {s: [r['id'] for r in rows if norm(r['question']) in {norm(f['question']) for f in few}]
               for s, rows in splits.items()}
    if overlap['test']:
        raise ValueError('Unexpected test demonstration overlap')
    code_files = ['scripts/run_well_verifier_gepa.py', 'src/fpqa_verifier_gepa.py',
                  'src/fpqa_scope_factorial.py', 'src/fpqa_uncertainty.py', 'src/fpqa_well_upstream.py',
                  'src/fpqa_well_pipelines.py', 'src/fpqa_cli_backend.py']
    bank_sources = {r['id']: sha(a.scope_source/'items'/r['id']/'D_verification/record.json') for r in splits['test']}
    budget = previous['upstream_args']['max_metric_calls']
    plan = {'protocol': 'well_verifier_gepa_DE_v1', 'config': config, 'gepa_version': '0.1.1',
            'previous_gepa': str(a.previous_gepa.resolve()), 'scope_source': str(a.scope_source.resolve()),
            'input_sha256': input_hashes, 'test_sha256': sha(a.test), 'few_shot_sha256': sha(fewpath),
            'well_sha256': source_hashes, 'code_sha256': {f: sha(ROOT/f) for f in code_files},
            'test_claim_source_sha256': bank_sources,
            'split_ids': {s: [r['id'] for r in rows] for s, rows in splits.items()},
            'counts': {s: dict(Counter('fpq' if r['label'] else 'nfp' for r in rows)) for s, rows in splits.items()},
            'max_metric_calls': budget, 'max_candidate_proposals': previous['upstream_args']['max_candidate_proposals'],
            'seed': previous['upstream_args']['seed'], 'test_repeats': a.test_repeats,
            'demonstration_overlap_ids': overlap,
            'demonstration_policy': 'Remove exact normalized question matches from few-shot examples per item, for every candidate. Preserve train/val IDs. Test still uses all four original examples.',
            'seed_prompt': VERIFY_SYSTEM, 'fixed_grounding_prompt': GROUND_SYSTEM, 'fixed_answer_action': E_ACTION,
            'objective': OBJECTIVE, 'background': BACKGROUND,
            'feedback': 'Original score/reason plus factual intermediate traces. No automated blame, target filtering or gold sentences added as fields.',
            'frozen_claims': 'Once-only official extraction + D grounding for train/val; historical D claims for test. No truth labels used to prepare claims.',
            'selection': 'Same exposed development/test questions as prior studies; not untouched generalization. Test annotations never enter optimization.',
            'weighting': 'Equal group counts within train/val => equal mean weights. Reflection mini-batches use original epoch-shuffled default (3), not necessarily balanced.',
            'budget_note': '500 end-to-end GEPA evaluations, NOT 500 model calls. Claim preparation and two seed/best test repeats are additional, logged separately.',
            'comparison': 'Custom verifier-only evaluator with fixed DE pipeline; original direct-GEPA used a different zero-shot answer pipeline.',
            'caching': 'Only immutable preparation/stage resume; fresh verification, answers and scores on every GEPA evaluation. No label-based reuse.',
            'format': 'At most 3 same-input attempts for structured formatting. Invalid candidate after retries => optimization score 0, no clinical Well score. Infrastructure errors abort.'}
    if a.command != 'preflight':
        freeze(out/'run.json', plan)
    local = threading.local()
    balanced, _, _ = load_upstream(a.well_root, lambda role, messages: local.judge(role, messages))
    templates = load_templates(a.well_root)
    args = make_args(balanced, 'CancerMyth', a.previous_gepa/'inputs', out, config)
    from constant.constant import MODEL_FINAL_ANSWER_KEY
    progress_lock = threading.Lock()
    stopped = threading.Event()

    def status(state, **extra):
        with progress_lock:
            data = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                    'prepared_train_val': len(list((out/'claims').glob('*/record.json'))),
                    'target_prepare': 96, 'optimization_evaluations': len(list((out/'evaluations').glob('*/record.json'))),
                    'test_evaluations': len(list((out/'test').glob('*/*/*/record.json'))),
                    'target_test_evaluations': 796, **extra}
            tmp = out/'status.tmp'; tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); tmp.replace(out/'status.json')

    def stage_call(folder, stage, role, messages, parser=None):
        if stopped.is_set() or (out/'STOP').exists():
            raise FatalTransportError('STOP requested')
        loc = folder/'stages'/stage
        freeze(loc/'request.json', {'role': role, 'messages': messages, 'config': config[f'{role}_model']})
        if (loc/'result.json').exists():
            raw = json.loads((loc/'result.json').read_text())['text']
            return parser(raw) if parser else raw
        for i in range(1, 4):
            attempt = loc/'attempts'/f'{i:03d}'
            if (attempt/'rejected.json').exists():
                continue
            rawpath = attempt/'raw.json'
            if rawpath.exists():
                raw = json.loads(rawpath.read_text())['text']
            else:
                raw = UncachedCaller(config, attempt)(role, messages)
                freeze(rawpath, {'text': raw})
            try:
                if not raw.strip():
                    raise ValueError('Empty output')
                parsed = parser(raw) if parser else raw
            except (ValueError, TypeError, KeyError) as e:
                freeze(attempt/'rejected.json', {'type': type(e).__name__, 'message': str(e)})
                continue
            freeze(loc/'result.json', {'text': raw, 'attempt': i})
            return parsed
        raise ValueError(f'Invalid structured output after three attempts: {stage}')

    def prepare(row):
        folder = out/'claims'/row['id']
        if (folder/'record.json').exists():
            rec = json.loads((folder/'record.json').read_text())
            if rec['question'] != row['question']:
                raise ValueError('Cached preparation question mismatch')
            return
        def call(stage, role, messages):
            return stage_call(folder, stage, role, messages)
        claims = extract(row['question'], demonstrations_for(row['question'], few), templates, call)
        grounded = stage_call(folder, 'ground', 'task', ground_messages(row['question'], claims),
                              lambda raw: parse_grounded(raw, row['question'], claims))
        freeze(folder/'record.json', {'id': row['id'], 'question': row['question'],
                                     'original_claims': claims, 'grounded_claims': grounded})
        print(f'PREPARED {row["id"]} claims={len(grounded)}', flush=True)
        status('preparing')

    def claim_bank(row):
        if row['split'] == 'test':
            path = a.scope_source/'items'/row['id']/'D_verification/record.json'
            if sha(path) != bank_sources[row['id']]:
                raise ValueError('Frozen test claims changed')
        else:
            path = out/'claims'/row['id']/'record.json'
        rec = json.loads(path.read_text())
        if rec['question'] != row['question']:
            raise ValueError('Claim question mismatch')
        return rec['grounded_claims']

    def evaluate(candidate, row, folder):
        freeze(folder/'input.json', {'candidate': candidate, 'id': row['id'], 'split': row['split']})
        if (folder/'record.json').exists():
            record = json.loads((folder/'record.json').read_text())
            return record['normalized_score'], record['feedback']
        def call(stage, role, messages, parser=None):
            return stage_call(folder, stage, role, messages, parser)
        grounded = claim_bank(row)
        template = templates['atomic'].FactCheckFinalAnswerTemplate(question=row['question'],
            few_shot_data=demonstrations_for(row['question'], few),
            factcheck_results=[1]*len(grounded), model_detected_presuppositions=[c['text'] for c in grounded])
        try:
            if set(candidate) != {'verifier_prompt'}:
                raise ValueError('Only verifier_prompt may be optimized')
            generated = execute_pipeline(candidate['verifier_prompt'], row['question'], grounded, template, call)
        except (ValueError, TypeError, KeyError) as error:
            info = {'question': row['question'], 'question_type': 'fpq' if row['label'] else 'tpq',
                    'format_error': str(error), 'note': 'Structural pipeline failure; not a clinical Well judgement.'}
            record = {'id': row['id'], 'label': row['label'], 'score': None,
                      'normalized_score': 0.0, 'feedback': info, 'valid': False}
        else:
            local.judge = lambda role, messages: call('judge', role, messages)
            evaluator = balanced.BalancedDirectQAGEPAEvaluator(args, 'gemini')
            evaluator.ScoreTemplateClass = evaluator.template_classes['fpq' if row['label'] else 'tpq']
            dp = {'id': row['id'], 'question': row['question'], 'answer': row['reference_answer'],
                  'presuppositions': row['presuppositions'], 'few_shot_data': [], 'passages': [],
                  MODEL_FINAL_ANSWER_KEY: generated['answer']}
            score, explanation = evaluator.score_response(dp)
            if not 0 <= score <= 5:
                raise ValueError('Invalid Well score')
            info = feedback(row, generated, score, explanation)
            record = {'id': row['id'], 'label': row['label'], 'score': score,
                      'normalized_score': score/5, 'feedback': info, 'valid': True, **generated}
        freeze(folder/'record.json', record)
        print(f'EVALUATED {row["split"]} {row["id"]} score={record["score"]}', flush=True)
        return record['normalized_score'], record['feedback']

    def parallel(fn, rows):
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(fn, row) for row in rows]
            for f in as_completed(futures):
                try:
                    f.result()
                except BaseException:
                    stopped.set()
                    raise

    if a.command == 'preflight':
        print(json.dumps({'counts': plan['counts'], 'budget': budget, 'prepare_calls': '192 minimum',
                          'fixed': ['D claims', 'E answer policy', 'Well judge'],
                          'optimized': 'verifier_prompt', 'test_repeats': 2,
                          'leave_one_out_ids': overlap}, indent=2)); return
    try:
        if a.command in {'run', 'smoke'}:
            status('preparing')
            rows = splits['train'][:2] if a.command == 'smoke' else splits['train'] + splits['val']
            parallel(prepare, rows)
            if a.command == 'smoke':
                for row in rows:
                    evaluate({'verifier_prompt': VERIFY_SYSTEM}, row, out/'smoke'/row['id'])
                status('smoke_complete'); return
            freeze(out/'claim_manifest.json', {r['id']: sha(out/'claims'/r['id']/'record.json') for r in rows})
            if not (out/'optimized_prompt.json').exists():
                # GEPA checkpoints in run_dir can resume. Every evaluator invocation
                # gets a new location, even if the same candidate/example reappears.
                n = max([int(q.name) for q in (out/'evaluations').glob('*') if q.is_dir()] or [0])
                def objective_eval(candidate, example):
                    nonlocal n
                    n += 1
                    result = evaluate(candidate, example, out/'evaluations'/f'{n:06d}')
                    status('optimizing', last_id=example['id'])
                    return result
                from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
                reflection_n = len(list((out/'reflection').glob('*')))
                def reflection(prompt):
                    nonlocal reflection_n
                    reflection_n += 1
                    messages = ([{'role': 'system', 'content': None}, {'role': 'user', 'content': prompt}]
                                if isinstance(prompt, str) else prompt)
                    return stage_call(out/'reflection'/f'{reflection_n:06d}', 'reflection', 'reflection', messages)
                status('optimizing')
                result = optimize_anything(seed_candidate={'verifier_prompt': VERIFY_SYSTEM},
                    evaluator=objective_eval, dataset=splits['train'], valset=splits['val'],
                    objective=OBJECTIVE, background=BACKGROUND,
                    config=GEPAConfig(engine=EngineConfig(run_dir=str(out/'gepa'), max_metric_calls=budget,
                        max_candidate_proposals=plan['max_candidate_proposals'], parallel=False,
                        cache_evaluation=False, seed=plan['seed']),
                        reflection=ReflectionConfig(reflection_lm=reflection)))
                freeze(out/'gepa_result.json', result.to_dict())
                freeze(out/'optimized_prompt.json', result.best_candidate)
                freeze(out/'selection.json', {'best_idx': result.best_idx,
                    'best_score': result.val_aggregate_scores[result.best_idx],
                    'meaning': 'Mean validation Well score / 5, NOT success rate',
                    'total_metric_calls': result.total_metric_calls,
                    'test_used_for_selection': False})
        # Winner is locked on validation before any test answer is produced.
        best = json.loads((out/'optimized_prompt.json').read_text())
        freeze(out/'test_selection.json', {'best': best, 'seed': {'verifier_prompt': VERIFY_SYSTEM},
                                         'repeats': 2, 'selection_sha256': sha(out/'selection.json')})
        def test_one(row):
            for repeat in range(1, 3):
                methods = ['seed', 'best']
                if (int(hashlib.sha256(row['id'].encode()).hexdigest(), 16)+repeat) % 2:
                    methods.reverse()
                for method in methods:
                    candidate = best if method == 'best' else {'verifier_prompt': VERIFY_SYSTEM}
                    evaluate(candidate, row, out/'test'/str(repeat)/method/row['id'])
                    status('evaluating_test')
        status('evaluating_test'); parallel(test_one, splits['test'])
        metrics = {}
        for repeat in range(1, 3):
            records = {m: [json.loads(q.read_text()) for q in (out/'test'/str(repeat)/m).glob('*/record.json')]
                       for m in ['seed', 'best']}
            invalid = {m: sum(not r['valid'] for r in rows) for m, rows in records.items()}
            # Invalid outputs remain failures in the primary denominator, with
            # their clinical score kept null in the underlying records.
            all_records = {m: [{**r, 'score': r['score'] if r['valid'] else 0} for r in rows]
                           for m, rows in records.items()}
            metrics[str(repeat)] = {'invalid': invalid,
                'primary_invalid_as_failure': {m: response_metrics(rows) for m, rows in all_records.items()},
                'valid_only_metrics': {m: response_metrics([r for r in rows if r['valid']]) for m, rows in records.items()},
                'paired_invalid_as_failure': paired_metrics(all_records['seed'], all_records['best'])}
        freeze(out/'metrics.json', metrics)
        # Full per-call records retain transport token/latency metadata for budget audit.
        costs = Counter()
        for q in out.glob('**/calls/*.request.json'):
            record = json.loads(q.read_text()); costs[(q.relative_to(out).parts[0], record['role'])] += 1
        freeze(out/'call_counts.json', {f'{phase}/{role}': count for (phase, role), count in costs.items()})
        status('complete')
    except BaseException as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
