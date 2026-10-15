"""Join four models' independent detection tasks to frozen Plain responses.

Qwen3.8 OFF/ON are two conditions, not two independent model families.
No generation, judgment, routing, or fitted decision thresholds.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src import well_eval as we

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/reviews/detection_plain_link_2026-09-28'
OUT = ROOT / 'results/analysis/detection_plain_link_20260928_v1'
QWEN = Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1')
GEMMA = ROOT / 'results/gemma4/full_20260926_v2'
LUNA = ROOT / 'results/frontier_cli/luna_direct_cot_20260923_v1'
Q38 = ROOT / 'results/qwen38/full_transformers_20260926_v1'
LABELS = {'qwen25': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna',
          'qwen38_off': 'Qwen3.8 OFF', 'qwen38_on': 'Qwen3.8 ON'}
SOURCES = {}


def read(path, lines=False):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return [json.loads(x) for x in data.decode().splitlines() if x.strip()] if lines else json.loads(data)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    questions = {r['id']: r for r in read(GEMMA / 'evaluation_questions.json')}
    assert len(questions) == 732
    oldp = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/decisions.csv'
    SOURCES[str(oldp)] = hashlib.sha256(oldp.read_bytes()).hexdigest()
    old = {r['id']: r for r in csv.DictReader(oldp.open())}
    old_answers = read(ROOT / 'docs/reviews/advisor_presentation_2026-09-23/answer_summary.json')
    refs = {r['id']: r for r in read(ROOT / 'docs/reviews/annotated_premises_2026-09-22/run_v1/source_annotations.jsonl', True)}
    configs = {
        'qwen25': (QWEN, QWEN / 'well_judge_claude'),
        'gemma': (GEMMA, GEMMA / 'well/plain'),
        'luna': (LUNA, ROOT / 'results/frontier_cli/luna_plain_20260922_v1/well_judge_sonnet_v1'),
        'qwen38_off': (Q38 / 'thinking_off', Q38 / 'thinking_off/well/plain'),
        'qwen38_on': (Q38 / 'thinking_on', Q38 / 'thinking_on/well/plain')}
    qcache = {}
    for path in (QWEN / 'model/detection/cache').glob('*.json'):
        r = read(path)
        assert r['id'] not in qcache
        qcache[r['id']] = (r, str(path))
    rows = []
    rubric = None
    for model, (run, well) in configs.items():
        read(well / 'judge_plan.json')
        read(well / 'attempts.jsonl', True)
        plan, ph = we.load_plan(well)
        _, scores = we.read_ledger(well, plan, ph)
        assert plan['model'] == 'claude-sonnet-5'
        current = plan['sources']['templates']['derived_templates']
        if rubric is None:
            rubric = current
        assert current == rubric
        answers = {}
        for source in plan['sources']['answers']:
            path = Path(source['path'])
            if path.name not in ['plain.jsonl', 'answers_eval732.jsonl']:
                continue
            data = read(path, True)
            assert SOURCES[str(path)] == source['sha256']
            answers.update({r['id']: r for r in data})
        assert answers
        for qid, q in questions.items():
            judged_q = plan['questions'][qid]
            assert (q['question'], q['set']) == (judged_q['question'], judged_q['set'])
            answer = answers.get(qid)
            job = plan['mapping']['plain'].get(qid)
            score = scores.get(job)
            if answer is not None:
                assert answer['question'] == q['question']
                assert we.digest(answer) == plan['answer_hashes']['plain'][qid]
            if model == 'qwen25':
                cache, review_source = qcache[qid]
                assert cache['question'] == q['question'] and cache['status'] == 'complete'
                review = cache['details']['review']['text']
                assert not cache['details']['review']['cap_hit']
                direct = old[qid]['qwen_direct'] == 'True'
                gate = old[qid]['qwen_review'] == 'True'
                gate_source = str(oldp)
            elif model == 'luna':
                values = {s: read(run / s / qid / 'record.json') for s in ['direct', 'review', 'cot']}
                assert all(v['status'] == 'complete' and q['question'] in v['prompt'] for v in values.values())
                review = values['review']['result']['review']
                assert review in values['cot']['prompt']
                direct = values['direct']['result']['has_false_premise']
                gate = values['cot']['result']['has_false_premise']
                review_source = str(run / 'review' / qid / 'record.json')
                gate_source = str(run / 'cot' / qid / 'record.json')
            else:
                values = {s: read(run / 'records' / s / (qid+'.json')) for s in ['direct', 'review', 'cot']}
                assert all(v['status'] == 'complete' and q['question'] in v['prompt'] for v in values.values())
                review = values['review']['response']
                assert review in values['cot']['prompt']
                direct = values['direct']['result']['has_false_premise']
                gate = values['cot']['result']['has_false_premise']
                review_source = str(run / 'records/review' / (qid+'.json'))
                gate_source = str(run / 'records/cot' / (qid+'.json'))
            assert isinstance(direct, bool) and isinstance(gate, bool)
            if model in ['gemma', 'luna']:
                assert direct == (old[qid][model+'_direct'] == 'True')
                assert gate == (old[qid][model+'_review'] == 'True')
            if qid in refs:
                assert refs[qid]['question'] == q['question']
            rows.append(dict(model=model, id=qid, set=q['set'], question=q['question'],
                             premise=q.get('premise_text',''), correction=q.get('correction',''),
                             tpq_premises=refs.get(qid,{}).get('premises',[]),
                             direct=direct, review_gate=gate, review=review,
                             plain=answer['response'] if answer else None,
                             plain_score=score, plain_ge4=None if score is None else score >= 4,
                             review_source=review_source, gate_source=gate_source,
                             well_job=job, well_dir=str(well)))
    assert len(rows) == 3660 and len({(r['model'],r['id']) for r in rows}) == 3660
    summary = {}
    for model in LABELS:
        summary[model] = {}
        for cl in ['fpq','nfp']:
            subset = [r for r in rows if r['model']==model and r['set']==cl]
            valid = [r for r in subset if r['plain_score'] is not None]
            res = dict(expected=len(subset), valid=len(valid), missing=[r['id'] for r in subset if r['plain_score'] is None],
                       score_counts=dict(Counter(r['plain_score'] for r in valid)), plain_ge4=sum(r['plain_ge4'] for r in valid))
            for stage, key in [('direct','direct'),('review','review_gate')]:
                res[stage] = {}
                for decision in [True,False]:
                    for passed in [True,False]:
                        ids = sorted(r['id'] for r in valid if r[key]==decision and r['plain_ge4']==passed)
                        res[stage][f'{"yes" if decision else "no"}_{"pass" if passed else "fail"}'] = ids
                assert sum(len(v) for v in res[stage].values()) == len(valid)
            summary[model][cl] = res
            if model in ['qwen25','luna']:
                previous = old_answers[('qwen' if model=='qwen25' else model)+'/plain'][cl]
                assert res['valid'] == previous['valid'] and res['plain_ge4'] == previous['ge4']
    # The paired OFF/ON transitions are descriptive; both detection and answer
    # generation changed their thinking setting, so they are not a causal chain.
    idx = {(r['model'],r['id']):r for r in rows}
    transitions = {}
    for cl in ['fpq','nfp']:
        transitions[cl] = {}
        for key in ['direct','review_gate','plain_ge4']:
            counts = {k:[] for k in ['00','01','10','11']}
            for qid,q in questions.items():
                if q['set'] != cl:continue
                a,b = idx['qwen38_off',qid][key],idx['qwen38_on',qid][key]
                counts[f'{int(a)}{int(b)}'].append(qid)
            transitions[cl][key] = counts
    common = sorted(q for q in questions if all(idx[m,q]['plain_score'] is not None for m in LABELS))
    assert len(common)==731
    # Existing discussed cases are selected deliberately for explanation, not
    # as an unbiased estimate of causes. Additional case review is separate.
    selected = ['fpq_18','fpq_114','fpq_272','fpq_316','nfp_1044','nfp_1024','nfp_1083','nfp_1142']
    cases = ['# 동일 문항: 탐지·검토·Plain 원문', '',
             '기존에 논의한 FPQ 4개·NFP 4개의 목적 표본. 아래는 저장 원문으로, 의료적 진위 재판정이나 내부 사고의 기록이 아니다. 별도 case_checks.md에서 원문 해석을 구분한다.', '']
    for qid in selected:
        q=questions[qid]
        cases += [f'## {qid}', '', q['question'], '', '**참조**', '', json.dumps(dict(premise=q.get('premise_text'), correction=q.get('correction'),tpq_premises=refs.get(qid,{}).get('premises',[])),ensure_ascii=False), '']
        for model in LABELS:
            r=idx[model,qid]
            cases += [f"### {LABELS[model]} · Direct {'Yes' if r['direct'] else 'No'} / Review {'Yes' if r['review_gate'] else 'No'} / Plain Well {r['plain_score']}", '', '**검토문**', '', r['review'], '', '**Plain 답변**', '', r['plain'] or '(missing)', '']
    report = ['# 네 모델의 명시적 탐지와 독립적인 Plain 답변 연결', '',
              '연구 목표: 거짓 전제 교정과 정상 전제 보존을 함께 개선한다. 이번 질문: 명시적 탐지 성공/실패가 일반 답변의 평가와 어떻게 연결되는가? Gate로 선택한 답변이 아니라 각 모델이 독립적으로 생성한 Plain을 연결했다.', '',
              'Qwen2.5·Qwen3.8·Gemma·Luna 네 모델이며 Qwen3.8 OFF/ON은 두 조건이다. 원 질문 732개, 총 3,660 모델조건–문항 중 유효 Plain 점수 3,659개. Qwen2.5 fpq_389만 누락이며 실패로 채우지 않는다. 5조건 공통 비교 집합은 731개다.', '',
              'Well ≥4는 답변 평가 기준 통과이지 내부 인식 여부나 실제 반박 여부의 직접 라벨이 아니다. FPQ <4에도 부분 교정이 있을 수 있으며, NFP <4도 원문 없이 과잉 반박으로 확정하지 않는다. 주석은 평가·사후 대조에만 사용됐다.', '',
              '## 먼저 읽을 결과', '',
              '탐지 성공·Plain 저점뿐 아니라 탐지 실패·Plain 고점도 전 모델조건에서 나타난다. 따라서 명시적 탐지 결과를 일반 답변의 전제 처리 능력과 동일시할 수 없다. 특히 Luna fpq_316은 Direct/Review 모두 No지만 Plain은 탈모의 필연성을 제한하면서 5점을 받았다. Gemma의 같은 질문은 Review Yes지만 Plain은 1점이다. [원문 해석과 반례](case_checks.md).', '',
              'NFP에서는 Qwen2.5와 Gemma의 모든 Plain이 4점 이상이지만 Review 오탐은 각각 24개와 49개다. Qwen3.8 OFF/ON에서 탐지는 ON이 더 많이 잡지만 Plain의 FPQ ≥4는 290→237, NFP ≥4는 119→132다. thinking이 어느 과제에서든 같은 방향으로 FPQ/NFP를 움직인다고 일반화할 수 없다.', '',
              '자동 평가 점수와 텍스트 의미도 분리해야 한다. nfp_1142의 Qwen3.8 Plain은 4점이지만 원 질문의 마지막 기능 신장 제거 상황을 수술 뒤 한 신장이 남는 일반 상황으로 바꿔 설명한다. 원 라벨·점수는 그대로 보존하며, 고점을 질문 의미의 완벽한 보존으로 해석하지 않는다.', '',
              '## 0. 각 과제의 전체 수치', '',
              table(['모델조건','FPQ Direct Yes','FPQ Review Yes','FPQ Plain ≥4','NFP Direct Yes','NFP Review Yes','NFP Plain ≥4'], [[LABELS[m]]+[f"{sum(1 for r in rows if r['model']==m and r['set']==cl and r[key] is True)}/{sum(1 for r in rows if r['model']==m and r['set']==cl and r[key] is not None)}" for cl in ['fpq','nfp'] for key in ['direct','review_gate','plain_ge4']] for m in LABELS]), '',
              '## 1. FPQ: 별도 탐지와 Plain 점수', '',
              table(['모델','탐지 과제','유효 FPQ','Yes·≥4','Yes·<4','No·≥4','No·<4'], [[LABELS[m],s,summary[m]['fpq']['valid']]+[len(summary[m]['fpq'][s][k]) for k in ['yes_pass','yes_fail','no_pass','no_fail']] for m in LABELS for s in ['direct','review']]), '',
              'Yes·<4는 명시적 과제에서는 FPQ를 탐지했지만 Plain 답변은 목표 기준에 미달한 문항이다. Plain 안에서 이미 오류를 인식했다는 뜻은 아니다. No·≥4는 탐지 판정이 Plain의 교정 성과를 완전히 대변하지 않음을 보여주는 대조군이다.', '',
              '## 2. NFP: 별도 오탐과 Plain 점수', '',
              table(['모델','탐지 과제','유효 NFP','Yes·≥4','Yes·<4','No·≥4','No·<4'], [[LABELS[m],s,summary[m]['nfp']['valid']]+[len(summary[m]['nfp'][s][k]) for k in ['yes_pass','yes_fail','no_pass','no_fail']] for m in LABELS for s in ['direct','review']]), '',
              'Yes·≥4는 별도 탐지 과제의 오탐이 Plain의 저점과 반드시 같은 문항에서 발생하는 것은 아니라는 뜻이다. Yes·<4도 같은 원인이나 같은 반박 대상을 증명하지 않는다.', '',
              '## 3. Qwen3.8 thinking OFF→ON', '',
              '동일 문항을 연결했다. 탐지와 답변을 별도로 생성한 두 과제 모두 설정이 달라지므로, 탐지 변화가 Plain 변화의 원인이라고 해석하지 않는다.', '',
              table(['집단','변수','0→0','0→1','1→0','1→1'], [[cl,k]+[len(transitions[cl][k][v]) for v in ['00','01','10','11']] for cl in ['fpq','nfp'] for k in ['direct','review_gate','plain_ge4']]), '',
              'direct/review_gate의 1은 거짓 전제 있음, plain_ge4의 1은 Well ≥4다. NFP에서 gate의 0→1은 오탐 증가지만 plain_ge4의 0→1은 답변 개선이다.', '',
              '## 4. 원문 점검 범위', '',
              '[cases.md](cases.md): 이전에 논의한 FPQ 4개·NFP 4개의 모든 모델조건 검토문과 Plain 답변. 목적 표본이므로 빈도로 일반화하지 않는다. [case_checks.md](case_checks.md)는 이 원문에서 확인한 차이와 한계를 별도로 기술한다.', '',
              '## 5. 조건 차이와 해석 제한', '',
              '- Qwen2.5는 Yes>No 토큰 점수(동점 No), 나머지는 생성 JSON boolean이다. 후속 통일 Yes/No 작업과 섞지 않았다.',
              '- Plain과 탐지는 질문 본문이 같지만 과제 지시가 다르다. Review는 생성된 검토문도 추가한다. 두 과제의 차이는 내부 억제의 직접 증거가 아니다.',
              '- 모델의 시스템 지시·추론 설정·출력 제한이 완전히 같지 않다. 순수 모델 능력 순위를 주장하지 않는다.',
              '- Plain은 자유롭게 생성한 저장 답변이다. 정답 라벨로 고르거나 gate로 교정 답변을 선택한 결과가 아니다.',
              '- 동일 Sonnet 판정기와 동일 Well 템플릿 해시를 확인했다. 자동 채점의 타당성 한계는 남는다.',
              '- Qwen3.8 전체 suite는 진행 중이어도 이 분석에 사용하는 Direct/Review/Plain/Plain 채점은 완료되어 있다. 다른 생성·채점 작업은 수정하지 않았다.', '',
              '## 재현', '',
              '`python3 -m scripts.analyze_detection_plain_link`', '',
              '[summary.json](summary.json): 집단별 문항 ID와 누락 목록. [links.csv](links.csv): 3,660행 판정·점수. [sources.json](sources.json): 입력 해시와 검증 범위.', '',
              f'전체 질문·주석·검토·Plain·채점 작업 ID는 `{OUT / "ledger.jsonl"}`에 저장했다. 새 모델 호출·GPU 작업·채점·문턱 변경 없음.']
    DOC.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (DOC/'summary.json').write_text(json.dumps(dict(models=summary,qwen38_transitions=transitions,common_ids=common,selected_case_ids=selected),ensure_ascii=False,indent=2)+'\n')
    with (DOC/'links.csv').open('w') as f:
        fields=['model','id','set','direct','review_gate','plain_score','plain_ge4']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    (DOC/'sources.json').write_text(json.dumps(dict(source_sha256=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rubric=rubric,checks=['exact question and class alignment','original answer file hash and per-answer judge hash','same rubric and judge','complete detection records','review text included in second decision prompt','existing decision/score crosschecks','missing score excluded']),ensure_ascii=False,indent=2)+'\n')
    (DOC/'cases.md').write_text('\n'.join(cases)+'\n')
    (DOC/'README.md').write_text('\n'.join(report)+'\n')
    # Human-readable analyst notes are kept separate from the frozen model
    # outputs. Quote matching is a provenance check, not semantic validation.
    check_path = DOC/'case_checks.json'
    if check_path.exists():
        checks=json.loads(check_path.read_text())
        lines=['# 원문 점검: 탐지와 Plain 사이의 차이','',
               f"{len({c['id'] for c in checks})}개 목적 선정 문항의 {len(checks)}개 모델조건 기록을 확인했다. 분석 에이전트의 원문 점검이며 새 임상 판정·독립 감사·원인 빈도 추정이 아니다. 원 점수와 라벨은 변경하지 않았다. 인용문은 저장 원문과 정확히 일치하는지 검사했다.",'']
        quote_count=0
        for c in checks:
            r=idx[c['model'],c['id']]
            lines += [f"## {c['id']} · {c['model']}",'',f"Direct {r['direct']} / Review {r['review_gate']} / Plain Well {r['plain_score']}",'']
            for field in ['question','review','plain']:
                quote=c.get(field+'_quote')
                if quote:
                    assert quote in r[field], (c['id'],c['model'],field)
                    quote_count+=1
                    lines += [f'**{field} 인용**','', '> '+quote,'']
            lines += [c['observation'],'','**해석 한계:** '+c['limit'],'']
        lines += ['## 이번 결과로 주장하지 않는 것','',
                  '- 명시적 탐지 성공을 Plain 내부의 오류 인식으로 해석하지 않는다.',
                  '- Well ≥4를 모든 의료 정보와 상황 해석이 정확하다는 보증으로 해석하지 않는다.',
                  '- 서로 다른 과제의 탐지와 답변을 하나의 실행에서 일어난 연속 판단으로 해석하지 않는다.',
                  '- 이 목적 표본의 해석 유형을 전체 오류의 원인 비율로 보고하지 않는다.']
        (DOC/'case_checks.md').write_text('\n'.join(lines)+'\n')
        print(f'Validated {quote_count} exact quotes in {len(checks)} analyst notes.')
    print(json.dumps({m:{cl:{s:{k:len(v) for k,v in summary[m][cl][s].items()} for s in ['direct','review']} for cl in ['fpq','nfp']} for m in LABELS},ensure_ascii=False))


if __name__=='__main__':
    main()
