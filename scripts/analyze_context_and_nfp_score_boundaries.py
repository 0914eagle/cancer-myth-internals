"""Frozen, exhaustive joins for two distinct descriptive analyses; no new inference.

Keep historical Direct decisions separate from independent Review explanations
and answer-generation conditions. Original labels, codings and scores stay intact.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/analysis'
CONTEXT = BASE / 'full_behavior_census_20261001_v1/context_all_items.json'
SCOPE = BASE / 'full_behavior_completion_20260930_v1/question_scope_items.json'
PAIRS = BASE / 'completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
MANUAL = BASE / 'completed_primary_behavior_20261001_v2/clinician_scope_manual_check.json'
RUBRIC = ROOT / 'prompts/well/nfp_user.txt'


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csvout(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read(path):
    return json.loads(path.read_text())


def grouping(rows, fields):
    groups = {}
    for r in rows:
        key = tuple(r[f] for f in fields)
        groups.setdefault(key, []).append(r)
    return groups


def judge_loader():
    cache = {}
    def get(answer):
        p = Path(answer['well_dir']) / 'attempts.jsonl'
        if p not in cache:
            values = {}
            for line in p.open():
                r = json.loads(line)
                if r.get('valid'):
                    values[r['id']] = r
            cache[p] = values
        r = cache[p][answer['job']]
        assert r['score'] == answer['score']
        return dict(reason=r['raw'], score=r['score'], path=str(p), job=answer['job'])
    return get, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    context = read(CONTEXT)
    scope = {r['id']: r for r in read(SCOPE)}
    manual = read(MANUAL)
    bad_attribution = set(manual['unsupported_clinician_attribution_ids'])
    assert len(context) == len(scope) == 732
    assert len({r['id'] for r in context}) == 732
    context_rows = []
    for r in context:
        s = scope[r['id']]
        assert r['dataset'] == s['dataset'] and r['transition'] == s['transition']
        features = dict(s['features'])
        # Explicit sensitivity correction only; retain original AI annotation.
        features['clinician_individual_sensitivity'] = (
            features['has_clinician_reported_individual_claim'] and r['id'] not in bad_attribution)
        context_rows.append(dict(**r, features=features, question_scope_coding=s['coding']))
    dump(out / 'all_732_context_items.json', context_rows)
    transitions = []
    for (ds, tr), rr in sorted(grouping(context_rows, ['dataset', 'transition']).items()):
        review = [r['separate_review_code'] for r in rr if r.get('separate_review_code')]
        fpq = [r['lost_fpq_code'] for r in rr if r.get('lost_fpq_code')]
        transitions.append(dict(dataset=ds, transition=tr, n=len(rr), ids=[r['id'] for r in rr],
            separate_review_n=len(review), separate_review_reading=Counter(x['reading'] for x in review),
            separate_review_reason=Counter(x['reason_type'] for x in review),
            separate_review_stance=Counter(x['stance'] for x in review),
            lost_fpq_audit_n=len(fpq), reference_scope=Counter(x['reference_scope'] for x in fpq),
            review_reference_match=Counter(x['match'] for x in fpq)))
    dump(out / 'context_transitions.json', transitions)
    feature_rows = []
    for ds in ['fpq', 'nfp']:
        for feature in context_rows[0]['features']:
            for present in [False, True]:
                rr = [r for r in context_rows if r['dataset'] == ds and r['features'][feature] == present]
                before = sum(r['before'] for r in rr)
                lost = sum(r['before'] and not r['after'] for r in rr)
                feature_rows.append(dict(dataset=ds, feature=feature, present=present, n=len(rr),
                    before_yes=before, after_yes=sum(r['after'] for r in rr), yes_to_no=lost,
                    no_to_yes=sum(not r['before'] and r['after'] for r in rr),
                    yes_to_no_among_before_yes=lost / before if before else None))
    csvout(out / 'context_feature_controls.csv', feature_rows)
    # Same rubric/condition controls; no pooling them into an alleged model ranking.
    pairs = [json.loads(l) for l in PAIRS.open()]
    nfp = [r for r in pairs if r['dataset'] == 'nfp']
    assert len(nfp) == 745
    get_judge, judge_cache = judge_loader()
    joined = []
    for r in nfp:
        judges = {side: get_judge(r[side]) for side in ['plain', 'alternative']}
        p, a = [r[side]['coding'] for side in ['plain', 'alternative']]
        lost = r['transition'] == '1->0'
        reasons = []
        if lost and a['stance'] != 'challenge':
            reasons.append('low_score_without_coded_challenge')
        if lost and a['response'] == 'substantive':
            reasons.append('low_score_with_coded_substantive_help')
        if lost and p['stance'] == a['stance']:
            reasons.append('score_loss_with_same_stance_category')
        # A search queue, not automatic semantic adjudication or corrected score.
        pattern = r'[^.!?\n]*(?:not dominated|not dominate|doesn.t dominate|does not dominate|not dominant|not\s+the\s+dominant)[^.!?\n]*[.!?]?'
        matched = re.findall(pattern, judges['alternative']['reason'], flags=re.I)
        if lost and r['alternative']['score'] == 2 and matched:
            reasons.append('rating_2_reason_explicitly_says_not_dominated')
        joined.append(dict(**r, original_judges=judges, review_queue_reasons=reasons,
            literal_not_dominated_matches=matched))
    with (out / 'all_745_nfp_pairs_with_judges.jsonl').open('w') as f:
        for r in joined:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    loss = [r for r in joined if r['transition'] == '1->0']
    assert len(loss) == 121
    dump(out / 'all_121_nfp_losses_with_judges.json', loss)
    groups = []
    for (model, tr), rr in sorted(grouping(joined, ['model', 'transition']).items()):
        groups.append(dict(model=model, method=rr[0]['method'], transition=tr, n=len(rr),
            response_after=Counter(r['alternative']['coding']['response'] for r in rr),
            stance_pairs=Counter(r['plain']['coding']['stance'] + ' -> ' + r['alternative']['coding']['stance'] for r in rr),
            flags_after=Counter(f['tag'] for r in rr for f in r['alternative']['coding']['flags']),
            new_challenge=sum(r['plain']['coding']['stance'] != 'challenge' and r['alternative']['coding']['stance'] == 'challenge' for r in rr),
            repeated_challenge=sum(r['plain']['coding']['stance'] == r['alternative']['coding']['stance'] == 'challenge' for r in rr),
            ids=[r['id'] for r in rr]))
    dump(out / 'nfp_loss_and_retained_controls.json', groups)
    flags = ['personal_context_reopened', 'scope_strengthening', 'context_shift', 'request_to_assertion']
    flag_controls = []
    for model in sorted({r['model'] for r in joined}):
        for flag in flags:
            for present in [False, True]:
                rr = [r for r in joined if r['model'] == model and r['plain']['score'] >= 4
                    and (flag in {f['tag'] for f in r['alternative']['coding']['flags']}) == present]
                n = len(rr)
                losses = sum(r['transition'] == '1->0' for r in rr)
                flag_controls.append(dict(model=model, flag=flag, present=present,
                    plain_high_n=n, loss_n=losses, loss_fraction=losses/n if n else None))
    csvout(out / 'nfp_flag_controls.csv', flag_controls)
    csvout(out / 'nfp_121_loss_index.csv', [dict(model=r['model'],id=r['id'],method=r['method'],
        plain_score=r['plain']['score'],alternative_score=r['alternative']['score'],
        plain_stance=r['plain']['coding']['stance'],alternative_stance=r['alternative']['coding']['stance'],
        response=r['alternative']['coding']['response'],
        review_queue=';'.join(r['review_queue_reasons'])) for r in loss])
    # Descriptive item-ID connection only: these answers were NOT generated under
    # the personal-context Direct instruction.
    byid = {r['id']:r for r in context_rows}
    bridge = []
    for r in joined:
        c = byid[r['id']]
        bridge.append(dict(model=r['model'], id=r['id'], answer_method=r['method'],
            answer_score_transition=r['transition'], luna_direct_context_transition=c['transition'],
            connection='same benchmark question only; different experimental conditions'))
    dump(out / 'cross_task_item_link_NOT_causal.json', bridge)
    sources = [CONTEXT,SCOPE,PAIRS,MANUAL,RUBRIC,Path(__file__),*judge_cache]
    counts = dict(questions=732, fpq=583,nfp=149, original_pairs=len(pairs), nfp_pairs=745,
        nfp_score_losses=121, nfp_loss_unique_questions=len({r['id'] for r in loss}),
        nfp_loss_responses=Counter(r['alternative']['coding']['response'] for r in loss),
        score_loss_review_queues=Counter(k for r in loss for k in r['review_queue_reasons']),
        nfp_score_loss_stance_pairs=Counter(r['plain']['coding']['stance']+' -> '+r['alternative']['coding']['stance'] for r in loss))
    dump(out / 'summary.json', dict(**counts, sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        limitations=[
            'Frozen historical run; not repeated, randomized or contemporaneous causal measurement.',
            'Direct has no rationale; independent Review codes are not Direct reasons.',
            'All scope and answer codes are existing AI annotations, not clinical adjudication.',
            'Question feature may refer to a different claim from the error target.',
            'Flags can overlap or miss errors; absence is not proof of a correct answer.',
            'Text-pattern queues and stance mismatches are review signals, not corrected Well scores.',
            'Model-question instances share questions; model settings and prompts differ.',
            'No new answers, scores, clinical labels, or gate thresholds are generated or changed.']))
    dump(out / 'validation.json', dict(passed=True, all_732_context_joined=True,
        all_745_nfp_pairs_have_both_original_judge_reasons=True,
        original_judge_scores_match_saved_scores=True, all_121_losses_included=True,
        original_artifacts_unchanged=True))
    print(json.dumps(counts,ensure_ascii=False,indent=2))
    print('OUTPUT',out)


if __name__ == '__main__':
    main()
