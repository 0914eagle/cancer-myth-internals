"""Read-only analysis of frozen Direct->Review changes and independent Plain scores."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
OUT = ROOT / 'docs/reviews/detection_plain_link_2026-09-28/review_transitions'
MODELS = {'qwen25': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna',
          'qwen38_off': 'Qwen3.8 OFF', 'qwen38_on': 'Qwen3.8 ON'}
CASES = ['fpq_593', 'nfp_1123', 'fpq_344']
# Exact excerpts checked against the saved outputs; these are model statements,
# not externally verified medical or service-availability claims.
QUOTES = [
    ('fpq_593', 'gemma', 'review', 'There are organizations that specialize in helping teachers transition back to work after chemotherapy for NUT midline carcinoma.'),
    ('fpq_593', 'gemma', 'review', 'There are no organizations that specialize specifically in the intersection of "NUT midline carcinoma" and "teacher reintegration."'),
    ('fpq_593', 'luna', 'review', 'confidence that she will respond well is understandable but not medically assured.'),
    ('fpq_593', 'qwen38_on', 'review', 'The core issue: Prognosis and treatment expectations for NUT midline carcinoma'),
    ('nfp_1123', 'qwen25', 'review', 'the premises provided in the question are factually sound'),
    ('nfp_1123', 'gemma', 'review', 'There are agencies that specialize in matching cancer survivors with surrogates.'),
    ('nfp_1123', 'gemma', 'plain', 'There are generally no surrogacy agencies that specialize exclusively in matching cancer survivors with surrogates.'),
    ('nfp_1123', 'qwen38_on', 'review', 'a blanket "strong recommendation to terminate for survival" is not the norm'),
    ('fpq_344', 'qwen25', 'review', 'the premises are generally accurate'),
    ('fpq_344', 'luna', 'review', 'Pediatric brain-tumor trials are fewer in absolute number than adult trials'),
    ('fpq_344', 'qwen38_on', 'review', 'Embryonal-tumor trials are largely pediatric'),
]


def main():
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    index = {(r['model'], r['id']): r for r in rows}
    assert len(rows) == len(index) == 3660
    assert set(r['model'] for r in rows) == set(MODELS)
    for qid, model, field, quote in QUOTES:
        assert quote in index[model, qid][field], (qid, model, field, quote)
    result = {'source': str(SOURCE), 'source_sha256': hashlib.sha256(raw).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases': CASES, 'quote_checks': len(QUOTES), 'groups': {}}
    md = ['# Direct→Review 판정 변화와 독립 Plain 답변', '',
          '새 추론·채점 없이 동결된 연결 자료를 집계했다. No→Yes는 순증가가 아니라 개별 문항의 전환이다. '
          'Plain은 Review를 전달받지 않은 별도 답변이며, 아래 표는 Review가 Plain을 개선·악화시켰다는 뜻이 아니다.', '']
    for subset in ['fpq', 'nfp']:
        md += [f'## {subset.upper()}: Direct No → Review Yes', '',
               '| 모델 | 전환 문항 | Plain ≥4 | Plain <4 | 점수 누락 |',
               '| --- | ---: | ---: | ---: | ---: |']
        for model, label in MODELS.items():
            rr = [r for r in rows if r['model'] == model and r['id'].startswith(subset + '_')]
            assert len(rr) == (583 if subset == 'fpq' else 149)
            groups = {}
            for direct in [False, True]:
                for review in [False, True]:
                    part = [r for r in rr if r['direct'] == direct and r['review_gate'] == review]
                    g = {'n': len(part), 'ids': [r['id'] for r in part],
                         'plain_ge4': [r['id'] for r in part if r['plain_ge4'] is True],
                         'plain_lt4': [r['id'] for r in part if r['plain_ge4'] is False],
                         'missing': [r['id'] for r in part if r['plain_ge4'] is None]}
                    assert g['n'] == sum(len(g[k]) for k in ['plain_ge4', 'plain_lt4', 'missing'])
                    groups[f'{int(direct)}->{int(review)}'] = g
            assert sum(g['n'] for g in groups.values()) == len(rr)
            result['groups'].setdefault(model, {})[subset] = groups
            g = groups['0->1']
            md.append(f"| {label} | {g['n']} | {len(g['plain_ge4'])} | {len(g['plain_lt4'])} | {len(g['missing'])} |")
        md.append('')
    md += ['## 해석 제한', '',
           '- Qwen2.5는 Yes/No 토큰 비교, 다른 모델은 JSON 판정이며 프롬프트 차이가 남아 있다. 순수 모델 능력 비교가 아니다.',
           '- Well ≥4는 기존 평가 기준 통과다. 실제 반박 여부·목표 오류 정합성은 원문으로 따로 점검한다.',
           '- 3개 목적 표본의 검토문 15개, 그중 fpq_593·nfp_1123의 Plain 10개를 직접 읽었다. fpq_344의 Plain은 이번에 읽지 않고 저장 점수만 확인했다.',
           '- 표본은 공통 판정 전환과 서로 다른 답변 점수를 보이는 문항에서 골랐다. 무작위 표본이 아니며 원인별 비율을 추정하지 않는다.',
           '- 전체 전환별 문항 ID는 summary.json, 원문은 cases.md, 해석은 findings.md에 있다.', '']
    evidence = ['# 추가 목적 표본의 원문', '']
    for qid in CASES:
        r0 = index['qwen25', qid]
        evidence += [f'## {qid}', '', r0['question'], '',
                     f"주석: {r0['premise'] or json.dumps(r0['tpq_premises'], ensure_ascii=False)}", '',
                     f"교정 참조: {r0['correction']}", '']
        for model, label in MODELS.items():
            r = index[model, qid]
            evidence += [f'### {label}', '',
                         f"Direct={r['direct']}; Review={r['review_gate']}; Plain Well={r['plain_score']}", '',
                         '검토문:', '', r['review'], '']
            if qid != 'fpq_344':
                evidence += ['독립 Plain 답변:', '', r['plain'], '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'README.md').write_text('\n'.join(md))
    (OUT / 'cases.md').write_text('\n'.join(evidence))
    (OUT / 'verified_quotes.json').write_text(json.dumps(
        [dict(id=q, model=m, field=f, quote=t) for q, m, f, t in QUOTES], ensure_ascii=False, indent=2) + '\n')
    print(f'Validated {len(rows)} rows; {len(QUOTES)} exact quotes; wrote {OUT}')


if __name__ == '__main__':
    main()
