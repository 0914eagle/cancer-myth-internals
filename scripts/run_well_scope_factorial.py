#!/usr/bin/env python3
"""Paired B/D/E/DE response experiment; fixed verifier reuse, no retrieval or gold in inference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import freeze, load_rows, response_metrics
from src.fpqa_well_upstream import UncachedCaller, FatalTransportError, load_upstream, make_args, verify_source
from src.fpqa_well_pipelines import load_templates
from src.fpqa_uncertainty import parse_verdict, VERIFY_SYSTEM, ANSWER_ACTION
from src.fpqa_scope_factorial import (GROUND_SYSTEM, E_ACTION, ground_messages, parse_grounded,
                                     grounded_verify_messages, response_messages, route)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', type=Path, default=ROOT/'results/fpqa_prompting/well_uncertainty_luna6_20261004_v1')
    p.add_argument('--test', type=Path, default=ROOT/'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl')
    p.add_argument('--well-root', type=Path, default=ROOT/'external/fpqa_20261002/Well')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        p.error('workers must be 1..4')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'STOP').exists():
        raise RuntimeError('STOP exists')
    original = json.loads((a.source/'run.json').read_text())
    config = original['config']
    if sha(a.test) != original['test_sha256']:
        raise ValueError('Test file changed')
    rows = [r for r in load_rows(a.test) if r['id'] in set(original['ids'])]
    if Counter(r['label'] for r in rows) != {1: 99, 0: 100} or len({r['id'] for r in rows}) != 199:
        raise ValueError('Need same 199 unique IDs')
    baselines, hashes = {}, {}
    for r in rows:
        path = a.source/'items'/r['id']/'B/record.json'
        rec = json.loads(path.read_text())
        if rec['question'] != r['question'] or rec['label'] != r['label']:
            raise ValueError('Baseline mismatch')
        if len(rec['claims']) != len(rec['verdicts']):
            raise ValueError('Incomplete baseline verdicts')
        for v in rec['verdicts']:
            parse_verdict(json.dumps(v))
        baselines[r['id']] = rec
        hashes[r['id']] = sha(path)
    upstream = verify_source(a.well_root)
    if upstream != original['well_source_sha256']:
        raise ValueError('Well source changed')
    local, stopped = threading.local(), threading.Event()
    balanced, _, _ = load_upstream(a.well_root, lambda role, messages: local.judge(role, messages))
    templates = load_templates(a.well_root)
    fewpath = Path(original['source'])/'few_shot_data.json'
    few = json.loads(fewpath.read_text())
    args = make_args(balanced, 'CancerMyth', out/'unused', out, config)
    from constant.constant import MODEL_FINAL_ANSWER_KEY
    repeat_ids = []
    for label in [1, 0]:
        ids = [r['id'] for r in rows if r['label'] == label]
        repeat_ids += sorted(ids, key=lambda s: hashlib.sha256(('scope-rejudge-v1:'+s).encode()).hexdigest())[:10]
    permutations = list(itertools.permutations(['B', 'D', 'E', 'DE']))
    orders = {r['id']: list(permutations[int(hashlib.sha256(('scope-order-v1:'+r['id']).encode()).hexdigest(), 16) % 24]) for r in rows}
    code = ['src/fpqa_scope_factorial.py', 'scripts/run_well_scope_factorial.py',
            'src/fpqa_uncertainty.py', 'src/fpqa_cli_backend.py', 'src/fpqa_well_upstream.py', 'src/fpqa_well_pipelines.py']
    plan = {
        'protocol': 'scope_answer_factorial_v1', 'config': config, 'source': str(a.source.resolve()),
        'test_sha256': sha(a.test), 'ids': [r['id'] for r in rows], 'counts': {'fpq': 99, 'nfp': 100},
        'baseline_sha256': hashes, 'well_source_sha256': upstream, 'few_shot_sha256': sha(fewpath),
        'source_sha256': {f: sha(ROOT/f) for f in code},
        'prompts': {'grounding': GROUND_SYSTEM, 'verification': VERIFY_SYSTEM,
                    'answer_B': ANSWER_ACTION, 'answer_E': E_ACTION},
        'methods': ['B', 'D', 'E', 'DE'], 'response_order': orders, 'rejudge_ids': repeat_ids,
        'B_E': 'Exact same saved B claims and verdicts; fresh response and unchanged Well judge in both cells.',
        'D_DE': 'One grounding pass per question and one verification per grounded claim; exact same outputs reused.',
        'D_change': 'Grounded candidate construction plus candidate grounding data to identical B verifier system. Not an isolated type-label ablation.',
        'E_change': 'Only answer Action changes. System/fewshot and complete claim/verdict feedback unchanged within each pair.',
        'baseline_control': 'B_saved historical; B freshly generated from identical saved verification, same run as D/E/DE.',
        'measurement': '20 hash-selected saved B answers (10/10), two fresh unchanged judge calls per answer; no answer regeneration.',
        'judge': 'Original Well, same Luna model, zero judge examples, no method/verification visible; scores not replaced.',
        'scope_audit': 'Exact quote check automatic only. Entailment/target coverage require separate direct audit; gold used only for scoring/audit.',
        'selection': 'All 199 are previously inspected development items. No untouched test or generalization claim.',
        'retrieval': 'None', 'format_retry': 'At most 3 identical-prompt attempts for invalid structured output, all retained; transport errors abort.',
    }
    freeze(out/'run.json', plan)
    for r in rows:
        freeze(out/'items'/r['id']/'B_saved/record.json', baselines[r['id']])
    if a.preflight:
        print(json.dumps({'counts': plan['counts'], 'saved_claims': sum(len(b['claims']) for b in baselines.values()),
                          'fresh_answers': 796, 'answer_judges': 796, 'grounding_calls': 199,
                          'verifier_calls': 'one per D claim; shared by D and DE', 'repeat_judges': 40})); return

    def status(state):
        data = {'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                'target_primary_records': 796, 'methods': {}, 'paired': {}, 'grounding': {}}
        all_records = {}
        for method in ['B_saved', 'B', 'D', 'E', 'DE', 'judge_repeat_1', 'judge_repeat_2']:
            recs = [json.loads(q.read_text()) for q in (out/'items').glob(f'*/{method}/record.json')]
            all_records[method] = {r['id']: r for r in recs}
            data['methods'][method] = {'completed': len(recs), 'metrics': response_metrics(recs)}
        for before, after in [('B_saved', 'B'), ('B', 'D'), ('B', 'E'), ('D', 'DE'), ('E', 'DE'), ('B', 'DE')]:
            ids = set(all_records[before]) & set(all_records[after])
            data['paired'][f'{before}->{after}'] = {
                name: {'n': sum(all_records[before][i]['label'] == label for i in ids),
                       'gain': sum(all_records[before][i]['label'] == label and all_records[before][i]['score'] < 4 <= all_records[after][i]['score'] for i in ids),
                       'loss': sum(all_records[before][i]['label'] == label and all_records[after][i]['score'] < 4 <= all_records[before][i]['score'] for i in ids)}
                for label, name in [(1, 'fpq'), (0, 'nfp')]}
        gs = [json.loads(q.read_text()) for q in (out/'items').glob('*/D_verification/record.json')]
        data['grounding'] = {'completed': len(gs), 'claims': sum(len(g['claims']) for g in gs),
                             'kinds': dict(Counter(c['kind'] for g in gs for c in g['grounded_claims'])),
                             'note': 'Model-assigned categories, not validated causal labels.'}
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
        tmp.replace(out/'status.json')
        return data

    def process(row):
        folder = out/'items'/row['id']
        baseline = baselines[row['id']]

        def call(stage, role, messages, parser=None):
            if stopped.is_set() or (out/'STOP').exists():
                raise FatalTransportError('STOP requested')
            loc = folder/'stages'/stage
            freeze(loc/'request.json', {'role': role, 'messages': messages, 'config': config[f'{role}_model']})
            if (loc/'result.json').exists():
                raw = json.loads((loc/'result.json').read_text())['text']
                return parser(raw) if parser else raw
            for attempt_number in range(1, 4):
                attempt = loc/'attempts'/f'{attempt_number:03d}'
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
                except (ValueError, TypeError, KeyError) as error:
                    freeze(attempt/'rejected.json', {'type': type(error).__name__, 'message': str(error)})
                    continue
                freeze(loc/'result.json', {'text': raw, 'accepted_attempt': attempt_number})
                return parsed
            raise ValueError(f'{stage}: three invalid outputs; see retained attempts')

        def score(method, generated, claims, verdicts, extra):
            local.judge = lambda role, messages: call(method+'/judge', role, messages)
            evaluator = balanced.BalancedDirectQAGEPAEvaluator(args, 'gemini')
            evaluator.ScoreTemplateClass = evaluator.template_classes['fpq' if row['label'] else 'tpq']
            dp = {'id': row['id'], 'question': row['question'], 'answer': row['reference_answer'],
                  'presuppositions': row['presuppositions'], 'few_shot_data': [], 'passages': [],
                  MODEL_FINAL_ANSWER_KEY: generated['answer']}
            value, explanation = evaluator.score_response(dp)
            rec = {'id': row['id'], 'label': row['label'], 'question': row['question'], 'method': method,
                   'claims': claims, 'verdicts': verdicts, **generated, 'score': value,
                   'score_explanation': explanation, **extra}
            freeze(folder/method/'record.json', rec)
            print(f"COMPLETE {row['id']} {method} score={value}", flush=True)

        try:
            dpath = folder/'D_verification/record.json'
            if dpath.exists():
                d = json.loads(dpath.read_text())
            else:
                grounded = call('D_grounding', 'task', ground_messages(row['question'], baseline['claims']),
                                lambda raw: parse_grounded(raw, row['question'], baseline['claims']))
                verdicts = [call(f'D_verification/check_{i:03d}', 'task', grounded_verify_messages(row['question'], c), parse_verdict)
                            for i, c in enumerate(grounded)]
                d = {'id': row['id'], 'question': row['question'], 'grounded_claims': grounded,
                     'claims': [c['text'] for c in grounded], 'verdicts': verdicts}
                freeze(dpath, d)
                print(f"VERIFIED {row['id']} D claims={len(grounded)}", flush=True)
            for method in orders[row['id']]:
                if (folder/method/'record.json').exists():
                    continue
                claims, verdicts, enhanced = route(method, baseline, d)
                t = templates['atomic'].FactCheckFinalAnswerTemplate(question=row['question'], few_shot_data=few,
                    factcheck_results=[1]*len(claims), model_detected_presuppositions=claims)
                raw = call(method+'/answer', 'task', response_messages(row['question'], claims, verdicts, t, enhanced))
                generated = {'raw_answer': raw, 'answer': t.ResponseClass.model_validate_plain_text(raw).get()}
                score(method, generated, claims, verdicts,
                      {'verification_source': 'B_saved' if method in {'B', 'E'} else 'D_verification',
                       'verification_sha256': hashes[row['id']] if method in {'B', 'E'} else sha(dpath),
                       'answer_policy': 'E' if enhanced else 'B'})
            if row['id'] in repeat_ids:
                for n in [1, 2]:
                    method = f'judge_repeat_{n}'
                    if not (folder/method/'record.json').exists():
                        score(method, {'answer': baseline['answer'], 'raw_answer': baseline['raw_answer']},
                              baseline['claims'], baseline['verdicts'], {'answer_source': 'B_saved', 'no_new_generation': True})
        except BaseException as error:
            stopped.set()
            errors = folder/'errors'
            freeze(errors/f'{len(list(errors.glob("*")))+1:03d}.json',
                   {'type': type(error).__name__, 'message': str(error)})
            raise

    groups = [[r for r in rows if r['label'] == y] for y in [1, 0]]
    selected = [r for pair in zip(*groups) for r in pair] + groups[1][99:]
    if a.smoke:
        selected = [groups[0][0], groups[1][0]]
    status('running')
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(process, r) for r in selected]
            for future in as_completed(futures):
                future.result()
                status('running')
        result = status('smoke_complete' if a.smoke else 'complete')
        if not a.smoke:
            if any(result['methods'][m]['completed'] != 199 for m in ['B', 'D', 'E', 'DE']):
                raise ValueError('Incomplete primary records')
            if any(result['methods'][m]['completed'] != 20 for m in ['judge_repeat_1', 'judge_repeat_2']):
                raise ValueError('Incomplete repeat scoring')
            freeze(out/'metrics.json', result)
    except BaseException:
        stopped.set()
        status('failed')
        raise


if __name__ == '__main__':
    main()
