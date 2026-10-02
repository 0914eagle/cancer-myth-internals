"""Reproduce a direct-reading audit; no model calls and no raw-score overwrite."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/reviews/direct_reading_2026-10-02'
OUT = ROOT / 'docs/reviews/well64_adjudication_2026-10-02'
PRIMARY = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
# Manual confidence assessment after reading both complete answers and judge reasons.
# Other proposed increases remain unresolved; they are NOT applied corrections.
CLEAR_RAISES = {211, 233, 242, 244, 248, 258, 273, 275, 278, 284, 286, 288, 290, 296}
CLEAR_OTHER = {(285, 'plain'), (293, 'alternative')}


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    categories = {x['review_index']: x['category'] for x in json.loads((SOURCE / 'taxonomy.json').read_text())['assignments']}
    selected = [x for x in json.loads((SOURCE / 'reviewed_evidence.json').read_text()) if categories[x['review_index']] in {'N2', 'N3', 'N4', 'N5', 'N6'}]
    decisions = {}
    for row in csv.reader((OUT / 'manual_decisions.tsv').open(), delimiter='\t'):
        i, p, a, tag, note = row
        assert int(i) not in decisions
        decisions[int(i)] = (int(p), int(a), tag, note)
    assert len(selected) == len(decisions) == 64
    assert set(decisions) == {x['review_index'] for x in selected}
    audit, overrides, overview = [], [], []
    applied = {}
    for x in selected:
        i = x['review_index']
        p, a, tag, note = decisions[i]
        overview.append(dict(review_index=i, model=x['model'], id=x['id'], alternative_method=x['method'], category=categories[i], disposition=('applied_raise' if i in CLEAR_RAISES else 'borderline_raise_not_applied' if tag == 'raise' else 'low_retained'), note_ko=note))
        for side, proposed in [('plain', p), ('alternative', a)]:
            answer = x[side]['answer']
            original = x[side]['score']
            sha = hashlib.sha256(answer.encode()).hexdigest()
            assert sha == x[side]['answer_sha256'], (i, side)
            change = ((side == 'alternative' and i in CLEAR_RAISES) or (i, side) in CLEAR_OTHER)
            effective = proposed if change else original
            status = 'applied_manual_correction' if change else ('borderline_not_applied' if proposed != original or tag == 'borderline_low' else 'retained')
            row = dict(review_index=i, model=x['model'], id=x['id'], method=x[side]['method'], side=side,
                       original_score=original, proposed_score=proposed, audit_effective_score=effective,
                       status=status, primary_category=categories[i], note_ko=note, answer_sha256=sha)
            audit.append({**row, 'question': x['question'], 'answer': answer, 'original_judge_reason': x['original_well_reasons'][side],
                          'read_scope': 'question_both_complete_answers_both_original_judge_reasons',
                          'reviewer': 'current Codex assistant; not a clinician; not blinded',
                          'clinical_accuracy': 'not adjudicated except separately cited general AML claim and camp existence'})
            if change:
                assert proposed != original
                overrides.append(row)
                key = (x['model'], x['id'], row['method'], sha)
                assert key not in applied
                applied[key] = effective
    dump('audit_128_answers.json', audit)
    write_csv('pair_dispositions.csv', overview)
    write_csv('score_corrections.csv', overrides)
    casebook = ['# NFP 64쌍 원문과 채점 감사', '',
                '현재 Codex의 비맹검 직접 독해. 독립 전문가 판정이 아니다. 제안 점수와 적용 점수는 다르며, 경계 사례는 원점수를 유지한다.', '']
    for row in audit:
        casebook.extend([
            f"## {row['review_index']} · {row['model']} · {row['id']} · {row['method']}", '',
            f"원점수 {row['original_score']} / 검토 제안 {row['proposed_score']} / 감사 적용 {row['audit_effective_score']} · {row['status']}", '',
            row['note_ko'], '', '### 질문', '', row['question'], '',
            '### 답변 전문', '', row['answer'], '', '### 기존 채점 이유', '', row['original_judge_reason'], '',
        ])
    # Display copy strips trailing whitespace; JSON retains exact hashed answers.
    (OUT / 'casebook.md').write_text('\n'.join(line.rstrip() for line in '\n'.join(casebook).splitlines()).rstrip() + '\n')

    # Audit overlay: stable identity + full-answer hash prevents cross-condition edits.
    # Original scores remain present for all answers, including unreviewed answers.
    groups = defaultdict(list)
    matched = set()
    ledger = []
    for line in PRIMARY.open():
        x = json.loads(line)
        for side in ['plain', 'alternative']:
            v = x[side]
            sha = hashlib.sha256(v['answer'].encode()).hexdigest()
            key = (x['model'], x['id'], v['method'], sha)
            score = applied.get(key, v['score'])
            if key in applied:
                matched.add(key)
            r = dict(model=x['model'], id=x['id'], dataset=x['dataset'], method=v['method'], answer_sha256=sha,
                     original_score=v['score'], audit_effective_score=score, correction_applied=key in applied)
            ledger.append(r)
            if x['dataset'] == 'nfp':
                groups[(x['model'], v['method'])].append(r)
    assert matched == set(applied), set(applied) - matched
    write_csv('all_answer_scores_with_audit_overlay.csv', ledger)
    aggregates = []
    for (model, method), rows in groups.items():
        n = len(rows)
        old = sum(r['original_score'] >= 4 for r in rows)
        new = sum(r['audit_effective_score'] >= 4 for r in rows)
        aggregates.append(dict(model=model, method=method, n=n, original_ge4=old, audit_ge4=new,
                               original_percent=round(old/n*100, 2), audit_percent=round(new/n*100, 2),
                               original_mean=sum(r['original_score'] for r in rows)/n,
                               audit_mean=sum(r['audit_effective_score'] for r in rows)/n))
    write_csv('nfp_condition_sensitivity.csv', aggregates)
    dump('manifest.json', {
        'pairs': len(selected), 'full_answers_read': len(audit), 'new_model_calls': 0,
        'applied_corrections': len(overrides), 'alternative_low_to_high': len(CLEAR_RAISES),
        'plain_high_to_low': 1, 'alternative_2_to_3': 1,
        'alternative_dispositions': dict(Counter(r['disposition'] for r in overview)),
        'selection': '64 previously N2–N6 pairs within 121 original Plain>=4 to alternative<4 NFP pairs',
        'not_population_error_rate': True,
        'applied_scope': 'versioned audit overlay only; original automatic scores immutable',
        'not_clinical_ground_truth': True,
        'unresolved': '17 proposed increases and two Plain re-ratings remain unapplied; precise 1/2 boundaries in retained low scores not normalized',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'reviewed_evidence.json', SOURCE/'taxonomy.json', OUT/'manual_decisions.tsv', ROOT/'prompts/well/nfp_user.txt']},
    })
    print(json.dumps({'pairs': len(selected), 'answers_read': len(audit), 'corrections': len(overrides), 'dispositions': dict(Counter(r['disposition'] for r in overview)), 'aggregates': aggregates}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
