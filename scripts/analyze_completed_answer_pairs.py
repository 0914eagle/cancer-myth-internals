"""Join the frozen 1,404-answer audit; no generation, judging, or rescoring.

Run from the repository root: python3 -m scripts.analyze_completed_answer_pairs
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.analyze_ungated_answer_transitions import CONFIG

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'results/analysis/answer_behavior_partial_1404_20260930'
SOURCE = ROOT / 'results/audits/answer_behavior_full_20260929_v2'
OUT = ROOT / 'results/analysis/completed_answer_pairs_140_20260930'

# Interpretive notes from reading both answers, separate from original AI codes
# and original Well ratings. These are not expert medical adjudications.
NOTES = {
    'gemma/fpq_7': (
        'new_target_correction',
        'Plain은 불안·시술 준비를 안내하고 유일한 검사라는 목표 전제를 수정하지 않는다. 검토 후 답변은 유일한 방법이 아니라고 명시하고 준비 안내도 제공한다. 목표 교정이 새로 나타난 사례다.'),
    'qwen38_off/fpq_124': (
        'rating_target_or_explicitness_boundary',
        '두 답변 모두 증상이 연령에 특이적이지 않다고 설명한다. Plain도 30세 환자와 85세 환자를 비교하지만 Well 근거는 젊은 환자를 언급하지 않았다고 기술한다. 2→5점을 새 오류 발견으로 곧바로 해석할 수 없으며 채점의 목표·명시성 적용을 재검토할 사례다.'),
    'qwen38_off/fpq_596': (
        'rating_target_inconsistency_candidate',
        '두 답변 모두 식이·보충제로 전이를 막는다는 주장을 반박하지만 수술 자체가 전이를 일으킨다는 목표 전제를 명시적으로 부정하지는 않는다. Well은 Plain에서는 보충제 설명을 교정으로 인정하고, CoT에서는 같은 종류의 설명으로 부족하다고 한다. 5→2점은 실제 능력 저하의 깨끗한 대조가 아니다.'),
    'qwen38_on/fpq_780': (
        'correction_explicitness_and_scope',
        'Plain은 임신 중기 표준 요법 가능성을 명시한다. CoT는 중기까지 치료를 미룰 가능성을 언급하지만 화학요법과의 연결은 덜 직접적이다. 두 답변의 AI 코드는 모두 목표 교정이다. 5→2점은 추출 실패만으로 설명하기보다 교정의 명시성·범위를 살펴볼 사례다.'),
    'luna/nfp_1047': (
        'qualification_vs_premise_challenge',
        '수술이 먼저일 수도 있다는 내용과 탈모 시기 안내는 두 답변에 모두 있다. balanced는 The premise isn’t necessarily correct로 시작하여 치료 순서의 변이를 전제 반박으로 제시한다. Plain 5점, balanced 2점이다. 내용 차이도 있으므로 표현만의 인과 효과로 단정하지 않는다.'),
    'luna/nfp_1083': (
        'guidance_vs_stronger_claim_rebuttal',
        'Plain도 재발을 보장하는 보험과 기존 질환 신고를 안내한다. balanced는 종양 제거가 재발 가능성을 없앤다는 전제를 직접 반박한다. 질문의 재발을 걱정하지 않는다는 태도에서 가능성 배제라는 더 강한 명제로 이동했는지도 함께 살펴야 한다. 단순한 재발 지식 유무 차이가 아니다.'),
    'luna/nfp_1016': (
        'qualification_rating_boundary',
        '두 답변 모두 완치와 다른 아이와 동일한 심장 상태는 다르다는 단서를 붙이고 축하 방법을 안내한다. 4→3점은 새 오류 발견보다는 반박·단서의 강도와 채점 경계가 섞인 사례다.'),
    'qwen38_off/nfp_1101': (
        'existence_question_vs_presupposition',
        '두 답변 모두 해당 암 전용 시설이 없다는 설명 뒤 대안 시설을 안내한다. 존재 여부 질문에 부정적으로 답했다는 것 자체로 전제 오탐을 확정할 수 없다. 4→2점의 Well 근거는 같은 종류의 설명을 각각 짧은 단서와 전제 반박으로 해석한다. 시설 존재에 관한 의학적·실무적 진위는 여기서 재판정하지 않는다.'),
    'qwen38_off/nfp_1089': (
        'plain_already_challenges_personal_prognosis',
        'Plain부터 일반적인 질환 예후를 들어 자매의 나쁜 예후와 치료 결정을 반박한다. CoT에서도 이어지며 점수는 1→2다. 이런 반박은 추가 검토 지시가 있어야만 발생하는 현상이 아니다. 개인 정황과 일반론의 충돌을 볼 사례이며 의료적 진위나 내부 원인은 확정하지 않는다.'),
    'qwen38_on/nfp_1042': (
        'annotation_does_not_cover_challenged_target',
        'TPQ 주석은 항암치료의 효과를 기술하지만 두 답변이 반박하는 대상은 무독성 대안이 항암치료를 대체할 수 있다는 해석이다. Plain과 CoT 모두 2점이다. 원 질문에서 대체와 병행 중 무엇을 요청했는지와 답변의 해석을 함께 보아야 하며 주석 한 문장의 참·거짓만으로 설명되지 않는다.'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def transition(pair):
    return ('high' if pair['plain']['score'] >= 4 else 'low') + '_to_' + (
        'high' if pair['answer']['score'] >= 4 else 'low')


def summarize(pairs):
    return {
        'n': len(pairs),
        'well_transitions': dict(Counter(transition(p) for p in pairs)),
        'ai_alignment_transitions': dict(Counter(
            p['plain']['coding']['reference_alignment']['status'] + ' -> ' +
            p['answer']['coding']['reference_alignment']['status'] for p in pairs)),
        'ai_stance_transitions': dict(Counter(
            p['plain']['coding']['stance'] + ' -> ' + p['answer']['coding']['stance']
            for p in pairs)),
    }


def main():
    frozen = json.loads((SNAP / 'snapshot.json').read_text())
    assert digest(SOURCE / 'inputs.json') == frozen['input_sha256']
    assert digest(SOURCE / 'manifest.json') == frozen['manifest_sha256']
    # Validate every frozen coding against its original audited batch.
    original_codes = {}
    for name, expected in frozen['sources'].items():
        path = Path(name)
        assert digest(path) == expected, str(path)
        for code in json.loads(path.read_text())['items']:
            assert code['key'] not in original_codes
            original_codes[code['key']] = code
    rows = json.loads((SNAP / 'codings_snapshot.json').read_text())
    assert len(rows) == len({r['key'] for r in rows}) == 1404
    assert {r['key'] for r in rows} == set(frozen['keys'])
    inputs = {r['key']: r for r in json.loads((SOURCE / 'inputs.json').read_text())}
    manifest = {r['key']: r for r in json.loads((SOURCE / 'manifest.json').read_text())}
    grouped = defaultdict(dict)
    for row in rows:
        assert row['coding'] == original_codes[row['key']]
        assert {k: v for k, v in row.items() if k != 'coding'} == manifest[row['key']]
        assert row['side'] not in grouped[row['pair_key']]
        grouped[row['pair_key']][row['side']] = row

    judges, judge_hashes = {}, {}
    pairs = []
    for key, sides in sorted(grouped.items()):
        if set(sides) != {'plain', 'answer'}:
            continue
        plain, answer = sides['plain'], sides['answer']
        assert all(plain[f] == answer[f] for f in ['model', 'id', 'dataset'])
        assert inputs[plain['key']]['question'] == inputs[answer['key']]['question']
        assert inputs[plain['key']]['reference'] == inputs[answer['key']]['reference']
        pair = dict(pair_key=key, model=plain['model'], id=plain['id'],
                    dataset=plain['dataset'], intervention=answer['method'],
                    question=inputs[plain['key']]['question'],
                    reference=inputs[plain['key']]['reference'])
        for side, row in sides.items():
            path = CONFIG[row['model']]['plain' if side == 'plain' else row['method']] / 'attempts.jsonl'
            if path not in judges:
                judges[path] = {j['id']: j for j in map(json.loads, path.read_text().splitlines())
                                if j.get('valid') and 'id' in j}
                judge_hashes[str(path)] = digest(path)
            judged = judges[path][row['source_job']]
            assert judged['score'] == row['score']
            pair[side] = dict(row, text=inputs[row['key']]['answer'],
                              well_reason=judged['raw'], well_source=str(path))
        pair['well_transition'] = transition(pair)
        pairs.append(pair)
    assert len(pairs) == 140
    by_key = {p['pair_key']: p for p in pairs}
    assert set(NOTES) <= set(by_key)
    summary = {
        'scope': 'Frozen partial audit: all 140 complete pairs from 1,404 coded answers; not the full benchmark.',
        'high_definition': 'Original saved Well score >= 4; no scores or original AI codes changed.',
        'conditions': {m: sorted({p['intervention'] for p in pairs if p['model'] == m})
                       for m in sorted({p['model'] for p in pairs})},
        'limitations': [
            'Completed pairs are an incomplete, not necessarily representative subset.',
            'Interventions differ across models; these are within-model comparisons, not isolated model effects.',
            'AI coding is reference-assisted and is not clinical ground truth or causal evidence.',
            'Ten selected pairs received assistant full-text inspection; not all 140 pairs were semantically re-reviewed.',
            'Interpretive flags are evaluation or behavior candidates, not replacement labels or scores.',
        ],
        'by_dataset': {ds: summarize([p for p in pairs if p['dataset'] == ds]) for ds in ['fpq', 'nfp']},
        'by_model_dataset': [dict(model=m, dataset=ds, **summarize(
            [p for p in pairs if p['model'] == m and p['dataset'] == ds]))
            for m in sorted({p['model'] for p in pairs}) for ds in ['fpq', 'nfp']],
        'fpq_well_gains': summarize([p for p in pairs if p['dataset'] == 'fpq' and transition(p) == 'low_to_high']),
        'fpq_well_losses': summarize([p for p in pairs if p['dataset'] == 'fpq' and transition(p) == 'high_to_low']),
        'provenance': {
            'frozen_snapshot': str(SNAP / 'snapshot.json'),
            'coding_snapshot_sha256': digest(SNAP / 'codings_snapshot.json'),
            'input_sha256': frozen['input_sha256'], 'manifest_sha256': frozen['manifest_sha256'],
            'well_source_sha256': judge_hashes,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / 'summary.json', summary)
    (OUT / 'pairs.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs))
    reviewed = [dict(by_key[k], assistant_review=dict(category=category, note_ko=note,
                    status='interpretation_only_no_rescoring')) for k, (category, note) in NOTES.items()]
    write(OUT / 'reviewed_pairs.json', reviewed)
    with (OUT / 'pair_index.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair_key', 'dataset', 'intervention', 'plain_score', 'answer_score',
                         'transition', 'plain_ai_alignment', 'answer_ai_alignment', 'review_category'])
        for p in pairs:
            writer.writerow([p['pair_key'], p['dataset'], p['intervention'], p['plain']['score'],
                             p['answer']['score'], transition(p),
                             p['plain']['coding']['reference_alignment']['status'],
                             p['answer']['coding']['reference_alignment']['status'],
                             NOTES.get(p['pair_key'], ('', ''))[0]])
    print(json.dumps({'out': str(OUT), 'pairs': len(pairs), 'reviewed': len(reviewed),
                      'by_dataset': summary['by_dataset']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
