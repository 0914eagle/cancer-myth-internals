#!/usr/bin/env python3
"""Reproduce factorial counts and bind the direct reading of 21 FPQ losses to source records.

No model calls, no rescoring, no changes to original outputs. Manual notes are observations,
not causal ground truth or new clinical labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'results/fpqa_prompting/well_scope_factorial_luna6_20261004_v1')
    ap.add_argument('--test', type=Path, default=ROOT/'results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl')
    ap.add_argument('--out', type=Path, default=ROOT/'docs/reviews/well_scope_factorial_loss_audit_2026-10-04')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((a.run/'run.json').read_text())
    assert digest(a.test) == plan['test_sha256']
    gold = {x['id']: x for x in map(json.loads, a.test.read_text().splitlines())}
    data, manifest = [], {str(a.test.relative_to(ROOT)): digest(a.test)}
    for i in plan['ids']:
        folder = a.run/'items'/i
        rs = {}
        for method in ['B_saved', 'B', 'D', 'E', 'DE', 'D_verification']:
            path = folder/method/'record.json'
            rs[method] = json.loads(path.read_text())
            manifest[str(path.relative_to(ROOT))] = digest(path)
            assert rs[method]['id'] == i and rs[method]['question'] == gold[i]['question']
        for x, y in [('B_saved', 'B'), ('B', 'E'), ('D', 'DE'), ('D', 'D_verification')]:
            assert rs[x]['claims'] == rs[y]['claims'] and rs[x]['verdicts'] == rs[y]['verdicts']
        for c in rs['D_verification']['grounded_claims']:
            assert c['quotes'] and all(q in gold[i]['question'] for q in c['quotes'])
        data.append({'id': i, 'label': gold[i]['label'], 'gold': gold[i], 'records': rs})
    assert Counter(x['label'] for x in data) == {1: 99, 0: 100}
    losses = [x for x in data if x['label'] == 1 and x['records']['B']['score'] >= 4 > x['records']['DE']['score']]
    losses.sort(key=lambda x: int(x['id'].split('_')[1]))
    notespath = a.out/'manual_notes.jsonl'
    notes = [json.loads(x) for x in notespath.read_text().splitlines()]
    assert len(losses) == len(notes) == 21
    assert {x['id'] for x in losses} == {x['id'] for x in notes}
    byid = {x['id']: x for x in losses}
    for n in notes:
        assert n['scores'] == {m: byid[n['id']]['records'][m]['score'] for m in ['B', 'D', 'E', 'DE']}

    def breakdown(xs):
        groups = {'n': len(xs)}
        for method in ['B', 'D']:
            vs = [v for x in xs for v in x['records'][method]['verdicts']]
            groups[method] = {
                'claims': len(vs), 'verdicts': dict(Counter(v['verdict'] for v in vs)),
                'items_any_false': sum(any(v['verdict'] == 'false' for v in x['records'][method]['verdicts']) for x in xs),
                'items_eligible_false': sum(any(v['verdict'] == 'false' and v['claim_relation'] in ['explicit', 'implied'] for v in x['records'][method]['verdicts']) for x in xs)}
        grounded = [x['records']['D_verification'] for x in xs]
        groups['D_extraction_kinds'] = dict(Counter(c['kind'] for x in grounded for c in x['grounded_claims']))
        groups['D_all_supported_items'] = sum(bool(x['verdicts']) and all(v['verdict'] == 'supported' for v in x['verdicts']) for x in grounded)
        groups['D_only_personal_report_items'] = sum(bool(x['grounded_claims']) and all(c['kind'] == 'personal_report' for c in x['grounded_claims']) for x in grounded)
        return groups

    summary = {
        'model': plan['config']['task_model'], 'n': len(data), 'selection': plan['selection'],
        'counts': {}, 'paired': {}, 'groups': {
            'loss21': breakdown(losses),
            'FPQ_all': breakdown([x for x in data if x['label'] == 1]),
            'NFP_all': breakdown([x for x in data if x['label'] == 0]),
            'DE_success_FPQ': breakdown([x for x in data if x['label'] == 1 and x['records']['DE']['score'] >= 4]),
            'DE_failure_FPQ': breakdown([x for x in data if x['label'] == 1 and x['records']['DE']['score'] < 4])},
        'manual_answer_change': dict(Counter(n['answer_change'] for n in notes)),
        'loss_ids': [x['id'] for x in losses],
        'loss_D_E_score_patterns': dict(Counter(
            f"D_ge4={x['records']['D']['score']>=4},E_ge4={x['records']['E']['score']>=4}" for x in losses)),
        'limitations': ['All 199 are development items; no held-out generalization claim.',
                       'Only 21 losses were semantically read here; whole-cohort comparisons are mechanical.',
                       'Direct reading unblinded to condition, annotation and score; no new clinical fact checking.',
                       'Extraction kinds and verdicts are model outputs, not ground-truth mechanism labels.',
                       'No source scores or labels changed; similar-content cases are audit candidates, not confirmed scoring errors.',
                       'Prompts are not hard routing; high scores can occur without an eligible false verdict.']}
    for m in ['B_saved', 'B', 'D', 'E', 'DE']:
        summary['counts'][m] = {name: {'n': len(xs), 'ge4': sum(x['records'][m]['score'] >= 4 for x in xs)}
                               for lab, name in [(1, 'FPQ'), (0, 'NFP')]
                               for xs in [[x for x in data if x['label'] == lab]]}
    for before, after in [('B', 'D'), ('B', 'E'), ('B', 'DE'), ('D', 'DE'), ('E', 'DE')]:
        summary['paired'][f'{before}->{after}'] = {
            name: {'gain': sum(x['records'][before]['score'] < 4 <= x['records'][after]['score'] for x in xs),
                   'loss': sum(x['records'][after]['score'] < 4 <= x['records'][before]['score'] for x in xs)}
            for lab, name in [(1, 'FPQ'), (0, 'NFP')] for xs in [[x for x in data if x['label'] == lab]]}
    for filename, obj in [('summary.json', summary), ('source_manifest.json', manifest)]:
        (a.out/filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    (a.out/'evidence.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in losses))
    text = ['# B→D+E FPQ 점수 하락 21문항 원문', '',
            '질문·주석과 네 조건의 전체 답변을 직접 읽었다. 임상 진위 재검증이나 재채점은 수행하지 않았다.', '']
    note_by_id = {x['id']: x for x in notes}
    for c in losses:
        text += [f"## {c['id']}", '', '### 질문', c['gold']['question'], '', '### 주석',
                 json.dumps({'presuppositions': c['gold']['presuppositions'], 'reference_answer': c['gold']['reference_answer']}, ensure_ascii=False, indent=2),
                 '', '### 직접 독해 기록', json.dumps(note_by_id[c['id']], ensure_ascii=False, indent=2),
                 '', '### D 주장 추출과 검증', json.dumps(c['records']['D_verification'], ensure_ascii=False, indent=2)]
        for m in ['B', 'D', 'E', 'DE']:
            r = c['records'][m]
            text += ['', f"### {m}: {r['score']}점", '', '검증 입력·출력:',
                     json.dumps(list(zip(r['claims'], r['verdicts'])), ensure_ascii=False, indent=2),
                     '', '답변:', r['answer'], '', '기존 채점 이유:', r['score_explanation']]
    (a.out/'casebook.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'items': len(data), 'losses_read': len(losses), 'answer_change': summary['manual_answer_change'],
                      'D_E_patterns': summary['loss_D_E_score_patterns'], 'source_files_hashed': len(manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
