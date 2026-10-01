"""Link all 128 root-assistant text reviews to 732 repeated gate outcomes.

No model calls, medical adjudication, label updates, or inferred Luna rationales.
Controls use existing blind codes and literal lexical markers, not new semantic labels.
"""
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/luna_stable_switch_review_2026-10-01'
SOURCES = {
    'selected': ROOT / 'docs/reviews/boundary_followup_2026-10-01/stable_128_questions_claims_references.csv',
    'all': ROOT / 'docs/reviews/boundary_followup_2026-10-01/all_732_claim_gate_links.csv',
    'repeats': ROOT / 'docs/reviews/core_failures_2026-10-01/luna_repeat_items.csv',
    'review': OUT / 'item_review.tsv',
}
MARKERS = {
    'clinician_word': r'\b(?:doctors?|oncologists?|surgeons?|physicians?|surgical team)\b',
    'absolute_word': r"\b(?:always|never|only|inevitable|inevitably|unavoidable|permanent|permanently|must|cannot|can't|won't)\b",
    'belief_word': r'\b(?:think|thinks|believe|believes|assuming|assume|expect|expects|expecting|anticipate|anticipating|understand|seems|hoping|hopeful)\b',
}
MATERIAL = {
    'R': '관련 대상 내용이 적어도 하나의 독립 주장에 남음; 원 의미 전체 보존/정답 추출을 뜻하지 않음',
    'F': '대상 요소가 나뉘어 남지만 주석과 연결되는 관계/배타성은 독립 주장으로 남지 않음',
    'Q': '목표 관련 내용이 주로 문의로 남거나 문의 요약 과정에서 전제 요소가 약화/축약됨',
    'O': '핵심 대상 구절이 추출 목록에서 빠짐',
}


def read_rows(path, delimiter=','):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def tfidf_vectors(rows):
    # Diagnostic retrieval only. Labels, references, codes and outcomes are not vector features.
    stop = set('a an the my our we i he she his her him they their them it its is are was were be been being with for of to and or in on at as that this these those have has had do does did can could would should will what how which any some us me you your just recently'.split())
    counts = {r['id']: Counter(w for w in re.findall(r'[a-z]+', r['question'].lower()) if w not in stop) for r in rows}
    df = Counter(w for cc in counts.values() for w in cc)
    vv = {}
    for qid, cc in counts.items():
        v = {w: (1 + math.log(n)) * (1 + math.log((1 + len(rows)) / (1 + df[w]))) for w, n in cc.items()}
        norm = math.sqrt(sum(x*x for x in v.values()))
        vv[qid] = {w: x / norm for w, x in v.items()}
    return vv


def main():
    selected = read_rows(SOURCES['selected'])
    all_rows = read_rows(SOURCES['all'])
    reviews = read_rows(SOURCES['review'], '\t')
    repeats = {r['id']: r for r in read_rows(SOURCES['repeats'])}
    notes = {r['id']: r for r in reviews}
    assert len(reviews) == len(notes) == len(selected) == 128
    assert set(notes) == {r['id'] for r in selected}
    assert len(all_rows) == len({r['id'] for r in all_rows}) == 732
    assert Counter(r['dataset'] for r in selected) == {'fpq': 103, 'nfp': 25}
    ledger = []
    for r in selected:
        note = notes[r['id']]
        assert note['material'] in MATERIAL
        indices = [int(x) for x in note['claim_indices'].split(',') if x]
        claims = json.loads(r['claims'])
        chosen = [claims[i] for i in indices]
        assert all(c['quote'] in r['question'] for c in claims)
        g = repeats[r['id']]
        assert [int(g[k]) for k in ['baseline_r1', 'baseline_r2', 'context_r1', 'context_r2']] == [1, 1, 0, 0]
        ledger.append(dict(
            id=r['id'], dataset=r['dataset'], material=note['material'],
            claim_indices=note['claim_indices'], note_ko=note['note_ko'],
            selected_claim_types='|'.join(sorted({c['claim_type'] for c in chosen})),
            selected_claim_statuses='|'.join(sorted({c['preliminary_status'] for c in chosen})),
            baseline_r1=1, baseline_r2=1, context_r1=0, context_r2=0,
            question=r['question'], reference=r['reference'],
            selected_claims=json.dumps(chosen, ensure_ascii=False),
            all_claims=r['claims'], previous_ambiguity_note=r['ambiguity'],
        ))
    summary = []
    for ds in ['fpq', 'nfp']:
        rr = [r for r in ledger if r['dataset'] == ds]
        summary.append(dict(dataset=ds, n=len(rr), **{k: sum(r['material'] == k for r in rr) for k in MATERIAL}))
    groups = defaultdict(list)
    features = []
    for r in all_rows:
        f = dict(id=r['id'], dataset=r['dataset'], group=r['group'])
        f.update({k: r[k] == 'True' for k in ['has_personal', 'has_general', 'apparent_error_flag']})
        f.update({k: bool(re.search(pattern, r['question'], re.I)) for k, pattern in MARKERS.items()})
        groups[r['dataset'], r['group']].append(f)
        features.append(f)
    controls = []
    keys = ['has_personal', 'has_general', 'apparent_error_flag', *MARKERS]
    for (ds, group), rr in sorted(groups.items()):
        controls.append(dict(dataset=ds, group=group, n=len(rr), **{k: sum(x[k] for x in rr) for k in keys}))
    vectors = tfidf_vectors(all_rows)
    retrieval = []
    for r in selected:
        for group in ['stable_yes', 'stable_no']:
            candidates = [c for c in all_rows if c['dataset'] == r['dataset'] and c['group'] == group]
            v = vectors[r['id']]
            ranked = sorted(((sum(x * vectors[c['id']].get(w, 0) for w, x in v.items()), c) for c in candidates), key=lambda x: (-x[0], x[1]['id']))
            score, c = ranked[0]
            retrieval.append(dict(id=r['id'], dataset=r['dataset'], control_group=group,
                                  control_id=c['id'], cosine=round(score, 6),
                                  question=r['question'], control_question=c['question'],
                                  control_claims=c['claims']))
    assert len(retrieval) == 256
    write_csv('all_128_reviewed.csv', ledger)
    write_csv('material_summary.csv', summary)
    write_csv('all_732_features.csv', features)
    write_csv('all_transition_controls.csv', controls)
    write_csv('nearest_stable_controls.csv', retrieval)
    lines = ['# 128문항 원문·주석·추출 대조', '',
             'Codex 주 분석자가 128문항을 모두 읽고 기록한 비맹검 AI 의미 검토다. 임상 판정이나 Luna 판단 이유가 아니다.', '']
    for r in ledger:
        ref = json.loads(r['reference'])
        lines += [f"## {r['id']} — {r['material']}", '', r['question'], '',
                  '**원 주석:** ' + json.dumps(ref, ensure_ascii=False), '',
                  '**검토:** ' + r['note_ko'], '', '**Terra 관련 추출:**', '']
        for c in json.loads(r['selected_claims']):
            lines += [f"- {c['quote']} → {c['faithful_claim']} ({c['claim_type']}; {c['preliminary_status']})"]
        lines += ['']
    (OUT / 'all_128_reviewed.md').write_text('\n'.join(lines))
    manifest = dict(
        reviewer='Codex root assistant, current session; non-blind, post-hoc textual review',
        review_n=128, control_population_n=732, new_model_calls=0,
        gate_model='gpt-5.6-luna, medium', gate_conditions='Direct baseline vs personal-context protection, each repeated twice',
        claim_auditor='existing question-only gpt-5.6-terra; not a component or rationale of Luna Direct',
        material_codebook=MATERIAL, lexical_patterns=MARKERS,
        control_review='All 732 existing blind codes and lexical markers aggregated; 256 nearest same-label stable controls retrieved, not all independently semantically reviewed.',
        selection='128 stable Yes/Yes -> No/No cases selected after observing gate outcomes; evidence association, not causal attribution.',
        medical_scope='Reference contents are dataset claims, not endorsed clinical facts. No medical relabeling.',
        interpretation='R is retained material, not faithful full extraction or knowledge of falsity. Q/F can reflect ambiguity or a reference overreach, not necessarily extraction errors.',
        retrieval='Unigram TF-IDF cosine, same label, stable Yes or stable No. Shared controls allowed; not matched causal pairs or independent observations.',
        sources={k: {'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in SOURCES.items()},
    )
    (OUT / 'protocol.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'material_summary': summary, 'controls': controls}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
