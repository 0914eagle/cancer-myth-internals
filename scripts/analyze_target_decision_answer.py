"""Trace saved review targets, decisions, and answer branches; no inference calls."""
import csv
import hashlib
import json
from pathlib import Path

from src import well_eval as we

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/detection_plain_link_2026-09-28/target_decision_answer'
LEDGER = ROOT / 'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
GEMMA = ROOT / 'results/gemma4/full_20260926_v2'
METHODS = ['plain', 'premise_review_answer', 'fp_unconditional']
CASES = ['fpq_593', 'nfp_1123', 'fpq_344', 'fpq_320', 'nfp_1048']
LABELS = {'qwen25': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna',
          'qwen38_off': 'Qwen3.8 OFF', 'qwen38_on': 'Qwen3.8 ON'}
SOURCES = {}


def read(path, lines=False):
    raw = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(raw).hexdigest()
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()] if lines else json.loads(raw)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def main():
    rows = read(LEDGER, True)
    index = {(r['model'], r['id']): r for r in rows}
    assert len(index) == len(rows) == 3660
    annotations = read(OUT / 'annotations.json')
    assert len(annotations) == 10
    quotes = 0
    for a in annotations:
        r = index[a['model'], a['id']]
        for field in ['question', 'review', 'plain']:
            for suffix in ['_quote', '_quote_2']:
                if field + suffix in a:
                    assert a[field + suffix] in r[field], (a['id'], a['model'], field)
                    quotes += 1
    csvpath = GEMMA / 'analysis/question_links.csv'
    SOURCES[str(csvpath)] = hashlib.sha256(csvpath.read_bytes()).hexdigest()
    links = list(csv.DictReader(csvpath.open()))
    assert len(links) == len({r['id'] for r in links}) == 732
    plans, judges = {}, {}
    for method in METHODS:
        well = GEMMA / 'well' / method
        read(well / 'judge_plan.json')
        events = read(well / 'attempts.jsonl', True)
        plan, ph = we.load_plan(well)
        _, scores = we.read_ledger(well, plan, ph)
        assert plan['model'] == 'claude-sonnet-5'
        if plans:
            assert plan['sources']['templates']['derived_templates'] == plans['plain']['sources']['templates']['derived_templates']
        plans[method] = plan
        answers = {}
        for source in plan['sources']['answers']:
            path = Path(source['path'])
            if path.name == method + '.jsonl':
                rr = read(path, True)
                assert SOURCES[str(path)] == source['sha256']
                answers.update({r['id']: r for r in rr})
        assert len(answers) == 732
        for r in links:
            qid = r['id']
            job = plan['mapping'][method][qid]
            assert scores[job] == int(r[method + '_well'])
            assert we.digest(answers[qid]) == plan['answer_hashes'][method][qid]
            if qid in CASES:
                record = read(GEMMA / 'records' / method / (qid + '.json'))
                assert record['response'] == answers[qid]['response']
                assert record['status'] == 'complete' and not record.get('truncated', False)
                valid = [e for e in events if e.get('id') == job and e.get('valid') and e.get('score') == scores[job]]
                assert valid
                judges[qid, method] = {'job': job, 'score': scores[job], 'explanation': valid[-1]['raw']}
    # Verify the actual text dependency for both branches on every item.
    for r in links:
        qid = r['id']
        review, gate, answer = [read(GEMMA / 'records' / s / (qid + '.json'))
                                for s in ['review', 'cot', 'premise_review_answer']]
        assert all(z['status'] == 'complete' for z in [review, gate, answer])
        assert review['response'] == index['gemma', qid]['review']
        assert gate['result']['has_false_premise'] == (r['cot_gate'] == 'True')
        digest = we.digest(review['response'])
        assert gate['review_sha256'] == answer['review_sha256'] == digest
        assert review['response'] in gate['prompt'] and review['response'] in answer['prompt']
        routed = int(r['fp_unconditional_well'] if r['cot_gate'] == 'True' else r['plain_well'])
        assert routed == int(r['cot_routed_well'])
    groups = {}
    for subset in ['fpq', 'nfp']:
        for gate in ['True', 'False']:
            eligible = [r for r in links if r['set'] == subset and r['cot_gate'] == gate]
            cells = {}
            for answer_pass in [False, True]:
                for route_pass in [False, True]:
                    cells[f'{int(answer_pass)}{int(route_pass)}'] = [r['id'] for r in eligible
                        if (int(r['premise_review_answer_well']) >= 4) == answer_pass
                        and (int(r['cot_routed_well']) >= 4) == route_pass]
            assert sum(map(len, cells.values())) == len(eligible)
            groups[subset + '_' + gate] = {'n': len(eligible), 'cells': cells}
    selected = {r['id']: r for r in links if r['id'] in CASES}
    md = ['# 검토 대상·판정·답변 연결 자료', '',
          '동일 질문의 독립 Plain과, Gemma에서 같은 검토문을 실제로 입력받은 답변을 구분한다. '
          'No/Yes는 후속 답변에 전달된 라벨이 아니라 분기 선택 또는 사후 집계에 쓰인 판정이다.', '',
          '## Gemma 전체: 같은 검토문 → 답변과 gate → 답변 선택', '',
          table(['집단', 'gate', '문항 수', '둘 다 ≥4', '검토 답변만 ≥4', '선택 답변만 ≥4', '둘 다 <4'],
                [[s, g, groups[s+'_'+g]['n']] + [len(groups[s+'_'+g]['cells'][k]) for k in ['11','10','01','00']]
                 for s in ['fpq','nfp'] for g in ['True','False']]), '',
          '선택 답변은 gate Yes면 원 질문+무조건 교정 지시 답변, No면 Plain이다. '
          '같은 검토문→답변은 검토문을 실제로 받지만 Yes/No 판정을 받지 않는다. '
          '두 답변 방식의 차이는 검토 정보 제공과 답변 지시 차이를 포함하므로 검토문 하나의 인과 효과가 아니다.', '',
          '## 직접 읽은 Gemma 답변 비교', '',
          table(['문항','Review gate','Plain','검토문→답변','무조건 교정','gate 선택'],
                [[q,selected[q]['cot_gate']] + [selected[q][k] for k in
                 ['plain_well','premise_review_answer_well','fp_unconditional_well','cot_routed_well']] for q in CASES]), '',
          '## 네 모델 추가 독해: 같은 원 질문 2개', '',
          table(['문항','모델','대상 해석 코드','Direct','Review','Plain Well'],
                [[a['id'],LABELS[a['model']],a['target_relation'],index[a['model'],a['id']]['direct'],
                  index[a['model'],a['id']]['review_gate'],index[a['model'],a['id']]['plain_score']] for a in annotations]), '',
          '해석 코드는 이 분석에서 원문을 읽어 붙인 서술적 코드이며 독립 전문가 검증 라벨이 아니다. '
          '원 FPQ/NFP 라벨과 점수를 유지한다. [해석](findings.md), [원문](cases.md), [채점 설명](judge_explanations.json).', '',
          f'검증: 3,660행 연결 자료; Gemma 732개×3조건 점수·답변 해시; 732개 동일 검토문 입력; {quotes}개 정확 인용. '
          '새 생성·재채점·GPU 작업 없음. Qwen3.8의 두 후속 답변 조건은 확인 시점에 기록이 없어 이번 실제 분기 비교에 포함하지 않았다.', '']
    evidence = ['# 직접 확인한 출력과 입력 의존성', '']
    for q in ['fpq_320','nfp_1048']:
        r0 = index['gemma',q]
        evidence += [f'## {q}: 네 모델', '', r0['question'], '',
                     '참조: ' + json.dumps({k:r0[k] for k in ['premise','correction','tpq_premises']},ensure_ascii=False), '']
        for m,label in LABELS.items():
            r = index[m,q]
            evidence += [f'### {label}', '',f"Direct={r['direct']}, Review={r['review_gate']}, Plain={r['plain_score']}",
                         '', '검토:', '', r['review'], '', 'Plain:', '', r['plain'], '']
    for q in CASES:
        evidence += [f'## {q}: Gemma 후속 답변', '']
        for method in ['premise_review_answer','fp_unconditional']:
            record = read(GEMMA/'records'/method/(q+'.json'))
            evidence += [f'### {method} · Well {judges[q,method]["score"]}', '',
                         '실제 입력:', '', record['system'], '', record['prompt'], '',
                         '생성 답변:', '', record['response'], '']
    summary = {'groups':groups, 'selected':selected, 'quote_checks':quotes,
               'sources':SOURCES, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'judge_explanations.json').write_text(json.dumps(
        [dict(id=q,method=m,**v) for (q,m),v in judges.items()],ensure_ascii=False,indent=2)+'\n')
    (OUT/'README.md').write_text('\n'.join(md))
    (OUT/'cases.md').write_text('\n'.join(evidence))
    print(f'Checked {quotes} excerpts, 2196 answer scores/hashes, 732 review dependencies; wrote {OUT}')


if __name__ == '__main__':
    main()
