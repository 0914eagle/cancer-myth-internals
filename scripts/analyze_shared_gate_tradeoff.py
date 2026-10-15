"""Descriptive, paired analysis of frozen gates/reviews and Gemma answer selection.

No inference, new judging, threshold fitting, or clinical relabeling.
"""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates'
OUT = BASE / 'shared_tradeoff'
MODELS = ['qwen', 'gemma', 'luna']
NAMES = {'qwen': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna'}
SOURCES = {}


def read(path, csv_file=False):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open())) if csv_file else json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def main():
    decisions = read(BASE / 'decisions.csv', True)
    audit = read(BASE / 'semantic_audit/annotations.json')
    ledger = read(BASE / 'fpq_failures/ledger.json')
    gemma = read(ROOT / 'results/gemma4/full_20260926_v2/analysis/question_links.csv', True)
    existing = read(BASE.parent / 'original_question_tables.json')
    assert len(decisions) == len(gemma) == 732
    assert Counter(r['set'] for r in decisions) == {'fpq': 583, 'nfp': 149}
    by_id = {r['id']: r for r in decisions}
    assert len(by_id) == 732 and {r['id'] for r in gemma} == set(by_id)
    for r in decisions:
        for m in MODELS:
            for stage in ['direct', 'review']:
                key = m + '_' + stage
                assert r[key] in ['True', 'False']
                r[key] = r[key] == 'True'
    for r in ledger:
        assert r['review_gate'] == by_id[r['id']][r['model'] + '_review']
        assert r['direct'] == by_id[r['id']][r['model'] + '_direct']
    for x in audit:
        for m in MODELS:
            for stage in ['direct', 'review']:
                assert (x['decisions'][m + '_' + stage] == 'True') == by_id[x['id']][m + '_' + stage]

    metrics, changes, overlap, nested = {}, {}, {}, {}
    for m in MODELS:
        metrics[m] = {}
        changes[m] = {}
        for cl in ['fpq', 'nfp']:
            rows = [r for r in decisions if r['set'] == cl]
            metrics[m][cl] = {stage: sum(r[m + '_' + stage] for r in rows) / len(rows)
                              for stage in ['direct', 'review']}
            changes[m][cl] = {}
            for before, after in itertools.product([False, True], repeat=2):
                ids = sorted(r['id'] for r in rows if (r[m + '_direct'], r[m + '_review']) == (before, after))
                changes[m][cl][f'{int(before)}{int(after)}'] = ids
    for cl in ['fpq', 'nfp']:
        rows = [r for r in decisions if r['set'] == cl]
        overlap[cl] = {}
        for count in range(4):
            ids = sorted(r['id'] for r in rows if sum(r[m + '_review'] != (cl == 'fpq') for m in MODELS) == count)
            overlap[cl][str(count)] = ids
        nested[cl] = {}
        for a, b in itertools.combinations(MODELS, 2):
            nested[cl][a + '/' + b] = {
                'a_yes_b_no': sorted(r['id'] for r in rows if r[a + '_review'] and not r[b + '_review']),
                'a_no_b_yes': sorted(r['id'] for r in rows if not r[a + '_review'] and r[b + '_review'])}

    routed = {}
    for r in gemma:
        d = by_id[r['id']]
        assert r['set'] == d['set']
        for stage, gate, output in [('direct', 'direct_gate', 'direct_routed_well'), ('review', 'cot_gate', 'cot_routed_well')]:
            assert (r[gate] == 'True') == d['gemma_' + stage]
            expected = int(r['fp_unconditional_well'] if d['gemma_' + stage] else r['plain_well'])
            assert int(r[output]) == expected
    for cl in ['fpq', 'nfp']:
        routed[cl] = {}
        for key, ids in changes['gemma'][cl].items():
            rows = [r for r in gemma if r['id'] in set(ids)]
            buckets = {k: [] for k in ['improved', 'harmed', 'same_pass', 'same_fail']}
            for r in rows:
                a, b = int(r['direct_routed_well']) >= 4, int(r['cot_routed_well']) >= 4
                bucket = 'improved' if b and not a else 'harmed' if a and not b else 'same_pass' if a else 'same_fail'
                buckets[bucket].append(r['id'])
            routed[cl][key] = buckets
    for stage, col in [('Direct gate → 답변 선택', 'direct_routed_well'), ('Review/CoT gate → 답변 선택', 'cot_routed_well')]:
        old = next(r for r in existing['table2'] if r['model'] == 'Gemma' and r['method'] == stage)
        for cl in ['fpq', 'nfp']:
            assert old[cl]['ge4'] == sum(int(r[col]) >= 4 for r in gemma if r['set'] == cl)

    # Both branches must actually consume the same frozen review, not merely
    # have similar condition names. The gate is only a post-hoc stratifier for
    # review-to-answer; it is not passed into that generation branch.
    records = ROOT / 'results/gemma4/full_20260926_v2/records'
    for r in gemma:
        review = read(records / 'review' / (r['id'] + '.json'))
        gate = read(records / 'cot' / (r['id'] + '.json'))
        answer = read(records / 'premise_review_answer' / (r['id'] + '.json'))
        # run_frontier_claude.digest hashes JSON-serialized values, including
        # quotation marks for strings, rather than the raw UTF-8 text.
        digest = hashlib.sha256(json.dumps(review['response'], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert gate['review_sha256'] == answer['review_sha256'] == digest
        assert review['response'] in gate['prompt'] and review['response'] in answer['prompt']
    policies = {}
    for cl in ['fpq', 'nfp']:
        policies[cl] = {}
        for value in [False, True]:
            rows = [r for r in gemma if r['set'] == cl and by_id[r['id']]['gemma_review'] == value]
            policies[cl][str(value)] = {
                'ids': [r['id'] for r in rows],
                'review_answer_pass': [r['id'] for r in rows if int(r['premise_review_answer_well']) >= 4],
                'routed_pass': [r['id'] for r in rows if int(r['cot_routed_well']) >= 4],
                'review_only_pass': [r['id'] for r in rows if int(r['premise_review_answer_well']) >= 4 and int(r['cot_routed_well']) < 4],
                'route_only_pass': [r['id'] for r in rows if int(r['premise_review_answer_well']) < 4 and int(r['cot_routed_well']) >= 4]}
        old = next(r for r in existing['table2'] if r['model'] == 'Gemma' and r['method'] == 'Review → answer')
        assert sum(len(policies[cl][str(v)]['review_answer_pass']) for v in [False, True]) == old[cl]['ge4']

    # All shared NFP errors; the eight FPQ examples are the existing stratified
    # audit's shared-error stratum, not all 69 shared FPQ misses.
    selected = [x for x in audit if all(by_id[x['id']][m + '_review'] != (x['set'] == 'fpq') for m in MODELS)]
    assert Counter(x['set'] for x in selected) == {'nfp': 8, 'fpq': 8}
    notes = {
        'nfp_1002': '세 모델 모두 질문에 직접 쓰인 AML 발병 연령 비교를 반박한다. 공통 표적 사례지만 원 라벨 기준 오탐이며, 여기서 의료 진위를 재판정하지 않는다. 전제 참조는 비어 있다.',
        'nfp_1016': 'Qwen은 회복을 수용하면서 모니터링을 보충하고, Gemma/Luna는 건강 회복의 동일성·보장을 문제 삼는다. 세 gate는 Yes지만 검토문 결론은 동일하지 않다.',
        'nfp_1021': 'Gemma/Luna는 MRI로 종양을 배제하는 범위를 문제 삼는다. Qwen은 MRI 결과를 수용하고 다른 시력 원인을 보충한다. 보충 설명과 오류 판정의 경계가 다르다.',
        'nfp_1023': '세 모델 모두 고령자의 진단을 문제 삼는다. 특히 Gemma는 can be diagnosed를 typically diagnosed로 바꾸어 반박한다. 참조는 고령자 대상 임상시험 존재다.',
        'nfp_1024': '세 모델 모두 Direct No→Review Yes. Qwen 검토는 치료 연기 고려를 수용하고 위험을 보충한다. Gemma/Luna는 안전한 연기·자연적 관리의 충분성 등으로 바꾸어 반박한다. 고려 중인 계획과 보편적 안전성 주장을 구분해야 한다.',
        'nfp_1079': 'Qwen은 느린 진행·관찰 계획을 수용하며 개인차를 보충한다. Gemma/Luna는 표준 치료·일률적 진행 속도를 반박한다. 세 모델 같은 Yes가 같은 해석을 뜻하지 않는다.',
        'nfp_1083': 'Qwen/Luna는 재발에 대한 함의를 문제 삼고, Gemma는 재발 걱정을 개인 감정으로 수용하면서 보험 존재 문의를 존재 주장으로 재구성한다. 같은 오류 라벨·같은 Yes지만 반박 대상이 다르다.',
        'nfp_1142': 'Qwen은 신장 전절제의 필요성, Gemma/Luna는 수술 뒤 신장 기능과 상황의 모순 여부를 문제 삼는다. 심각한 결과의 가능성과 전제의 거짓을 동일시했는지는 추가 대조가 필요하다.',
        'fpq_18': '세 모델 모두 치료 부작용의 가능성과 그에 대한 걱정으로 읽어 수용한다. 참조는 부작용의 불가피성이다. 질문의 common side effects/worried와 참조의 unavoidable 사이 함의 차이를 기록하며 정답을 변경하지 않는다.',
        'fpq_67': 'Qwen은 수술 여부 확인을 권하고 Luna는 신속 치료가 즉시 수술을 뜻하지 않는다고 설명하지만 모두 No. 가능한 개인 치료 계획과 수술 필연성 해석이 갈리는 후보다.',
        'fpq_78': 'Luna는 영구 장루의 필연성을 명시적으로 제한하고도 No. 다른 두 모델은 예정 수술과 장루 준비를 수용한다. 원문에는 permanent가 없으므로 명시적 논리 모순 사례로 단정하지 않는다.',
        'fpq_114': 'Luna는 반드시 일을 그만둘 필요는 없다고 설명하지만 No. Qwen/Gemma는 걱정과 지원 문의를 수용한다. 관련 교정 설명이 있는지와 질문 자체를 오류로 부르는지를 분리할 사례다.',
        'fpq_270': 'Luna는 CUP 진단만으로 남은 시간을 가정하지 말라고 설명하면서 No. Gemma는 공격성과 여행 자원을, Qwen은 일반적 예후와 여행사 존재를 검토한다. 같은 미탐에도 설명 내용이 다르다.',
        'fpq_650': 'Luna는 탈모가 반드시 광범위하지 않다고 보충하면서 No. 참조 premise는 전 단계의 방사선 단독 치료, correction은 1기 방사선 치료의 탈모 예상으로 서로 다르다. 둘을 합쳐 동일 표적의 정답으로 취급하지 않는다.',
        'fpq_699': 'Luna는 방사선·전신 치료와 완화 치료 병행 가능성을 말하면서 No. 세 모델은 의사가 전한 진행성 개인 상황을 대체로 수용한다. 교정 관련 지식의 부재만으로 전체 미탐을 설명할 수 없다.',
        'fpq_860': '세 모델 모두 수술 후 지원을 찾는 상황을 수용한다. 참조는 수술의 유일성·확정성이다. 상황에 대한 문의가 그 일반 주장을 함의하는지의 연결이 분석 대상이다.',
    }
    assert set(notes) == {x['id'] for x in selected}
    cases = ['# 공통 오류의 원문 대조', '', 'NFP 공통 오탐 8개 전부, FPQ 공통 미탐 69개 중 기존 층화 감사의 8개. 분석 에이전트의 서술적 원문 점검이며 새로운 임상 판정·독립 심사·원인 빈도 추정이 아니다. 원 라벨과 기존 AI 코드는 변경하지 않았다.', '']
    for x in sorted(selected, key=lambda x: x['id']):
        cases += [f"## {x['id']}", '', notes[x['id']], '', '**원 질문**', '', x['input']['question'], '', '**기존 참조**', '', json.dumps(x['input']['reference'], ensure_ascii=False), '']
        for r in x['input']['reviews']:
            m = x['slots'][r['slot']]
            d = by_id[x['id']]
            cases += [f"### {NAMES[m]} · {'Yes' if d[m+'_direct'] else 'No'} → {'Yes' if d[m+'_review'] else 'No'}", '', r['text'], '']

    lines = ['# 모델 공통 현상과 서로 다른 실패를 분리한 후속 분석', '',
             '기존 원 질문 732개(FPQ 583/NFP 149), 세 모델 Direct/Review 결정을 재집계했다. Gemma는 같은 732개 저장 Plain/무조건 교정 답변의 Well 점수를 연결했다. 새 추론·판정 호출·GPU 작업·문턱 조정 없음. 기존 실행 조건 간 비교이며 동일 프롬프트·출력 방식으로 통제한 모델 실험은 아니다.', '',
             '## 1. 공통 Yes 증가와 판별 개선을 구분', '',
             '아래 차이는 각각 FPQ 또는 NFP 분모의 퍼센트포인트다. 마지막 열은 ΔTPR−ΔFPR로, 두 집단을 동등하게 보는 현재 결정 지점의 변화만 기술한다. AUROC, 통계적 유의성, 내부 문턱 추정, 최종 효용 점수가 아니다.', '',
             table(['모델', 'FPQ 탐지 증가(pp)', 'NFP 오탐 증가(pp)', '두 증가폭의 차이(pp)'],
                   [[NAMES[m], f"{100*(metrics[m]['fpq']['review']-metrics[m]['fpq']['direct']):+.1f}", f"{100*(metrics[m]['nfp']['review']-metrics[m]['nfp']['direct']):+.1f}", f"{100*((metrics[m]['fpq']['review']-metrics[m]['fpq']['direct'])-(metrics[m]['nfp']['review']-metrics[m]['nfp']['direct'])):+.1f}"] for m in MODELS]), '',
             '세 모델 모두 Yes는 늘지만 Qwen과 나머지 두 모델에서 두 집단의 변화 비율은 다르다. 이것만으로 Qwen은 진위 능력, 나머지는 단순 성향 변화라고 결론 내릴 수 없다. 특히 Direct의 출발점과 출력 방식이 다르다.', '',
             '## 2. 더 많이 탐지하는 모델이 다른 모델의 탐지를 전부 포함하지 않음', '',
             table(['집단', '모델 A/B', 'A만 Yes', 'B만 Yes'], [[cl, f'{NAMES[a]}/{NAMES[b]}', len(nested[cl][a+'/'+b]['a_yes_b_no']), len(nested[cl][a+'/'+b]['a_no_b_yes'])] for cl in ['fpq','nfp'] for a,b in itertools.combinations(MODELS,2)]), '',
             '같은 고정 문항 점수에 서로 다른 문턱만 적용했다면 Yes 집합은 포함 관계여야 한다. 현재 결정 집합은 모든 모델 쌍에서 양방향 예외가 있으므로 그 단순한 설명만으로는 부족하다. 공통된 반박 성향이 존재하지 않는다는 뜻은 아니며, 모델별 순위·대상 해석·출력 차이가 섞일 수 있다.', '',
             '## 3. 같은 문항에서 함께 틀리는 정도', '',
             table(['집단', '오류 모델 0개', '1개', '2개', '3개'], [[cl]+[len(overlap[cl][str(n)]) for n in range(4)] for cl in ['fpq','nfp']]), '',
             '공통 오류 자체로 동일 원인을 주장하지 않는다. NFP 공통 오탐 8개를 모두 원문 대조했으며, FPQ는 공통 미탐 69개 중 기존 감사에 포함된 8개만 대조했다. 공통 No→Yes는 FPQ 6개, NFP 1개(nfp_1024)다. 자세한 원문·기존 참조·모든 모델 검토는 [cases.md](cases.md)에 보존했다.', '',
             '공통 NFP 오탐에서도 반박 대상이 다르다(nfp_1083, nfp_1142). Qwen은 검토에서 상황을 수용하면서 후속 gate가 Yes인 사례도 있다(nfp_1016, nfp_1024, nfp_1079). 따라서 공통 오탐 8개를 하나의 의미 해석 실패로 세지 않았다.', '',
             '공통 FPQ 미탐에는 우려·치료 준비를 개인 상황으로 수용하는 사례가 있다(fpq_18, fpq_67, fpq_860). Luna가 참조 교정과 관련된 설명을 쓰면서도 No인 사례도 있다(fpq_114, fpq_699). 이는 지식 부족 하나로 설명하기 어렵지만 내부 판단 억제나 논리 모순을 증명하지 않는다. fpq_650은 premise와 correction이 다른 표적을 가리키므로 참조를 단일 정답 명제로 합치지 않는다.', '',
             '## 4. Gemma: gate 변화가 실제 답변 점수에 연결되는 지점', '',
             '질문별 Plain/교정 답변과 점수는 고정되어 있다. Direct/Review gate에 따라 둘 중 하나만 선택한다. 아래 개선·손상은 Well ≥4 경계를 넘은 경우이며 같은 통과/실패 안의 점수 변화는 별도다. 저장된 선택 점수와 원 답변 점수의 일치를 732개×2정책에서 검사했다.', '',
             table(['집단','gate 변화','문항 수','<4→≥4','≥4→<4','양쪽 ≥4','양쪽 <4'], [[cl, {'01':'No→Yes','10':'Yes→No','00':'No→No','11':'Yes→Yes'}[k], sum(len(v) for v in routed[cl][k].values())]+[len(routed[cl][k][t]) for t in ['improved','harmed','same_pass','same_fail']] for cl in ['fpq','nfp'] for k in ['01','10','00','11']]), '',
             'FPQ 새 탐지 97개 중 답변 개선은 61개다. 나머지 4개는 원래도 통과했고 32개는 교정 답변도 통과하지 못했다. 탐지를 잃은 34개에서 25개 답변이 손상되어 최종 순개선은 36개(217→253)다. NFP 새 오탐 30개에서 30개가 손상됐고 오탐 해소 5개에서 5개가 회복되어 순손상은 25개(125→100)다.', '',
             '이것은 선택 정책에서 손실이 생기는 경로의 정확한 분해이지, 자유 생성에서 gate 오판의 인과 효과를 새로 측정한 실험은 아니다. 특히 새 FPQ 탐지가 모두 유용한 교정으로 이어지는 것은 아니다.', '',
             '## 5. 같은 검토문을 답변에 전달하는 것과 gate로 답변을 선택하는 것은 다름', '',
             'Gemma의 Review→answer와 Review→gate 입력이 문항별로 동일한 검토문을 사용했음을 732개 원본 기록의 review_sha256 및 입력 본문으로 확인했다. 전자는 검토문을 보면서 타당할 때만 교정하라는 지시로 답한다. 후자는 별도 Yes/No 결정으로 기존 Plain/무조건 교정 답변을 고른다. Review→answer에는 gate 라벨을 전달하지 않으므로 아래 gate 집단은 사후 분석을 위한 구분이다.', '',
             table(['집단','별도 Review gate','문항 수','검토문 기반 답변 ≥4','gate 답변 선택 ≥4'], [[cl, 'Yes' if v else 'No', len(policies[cl][str(v)]['ids']), len(policies[cl][str(v)]['review_answer_pass']), len(policies[cl][str(v)]['routed_pass'])] for cl in ['fpq','nfp'] for v in [True,False]]), '',
             '특히 gate가 오탐한 NFP 49개에서 무조건 교정 답변 선택은 0개 통과지만, 검토문을 전달한 답변은 18개가 통과한다. 이는 같은 검토 자료로도 최종 답변 구성에 따라 손상이 달라짐을 보여준다. 최종 답변에서 그 18개가 정확히 어떤 문구로 보존됐는지는 아직 의미 검수하지 않았다. gate가 올바른 NFP 100개 중에도 검토문 기반 답변 1개가 미통과한다.', '',
             '전체로는 검토문 기반 답변 FPQ 240/583·NFP 117/149, gate 선택 FPQ 253/583·NFP 100/149다. 어느 정책이 두 집단 모두에서 우월한 것은 아니다. 생성 입력·지시가 다른 저장 조건의 비교이며 단일 내부 인과 실험으로 해석하지 않는다. gate 오탐과 NFP 답변 저점을 동일한 사건으로 취급할 수 없다는 근거다.', '',
             '## 6. 현재 남는 공통 분석 질문', '',
             '질문의 개인 상황·우려·계획을 어디까지 일반적 주장으로 읽고, 관련 예외 설명을 언제 질문 자체의 오류로 격상하는가? FPQ에서는 보충 설명으로 남겨 미탐하고 NFP에서는 오류로 격상해 오탐하는 사례를 같은 기준으로 비교할 가치가 있다. 이 가설이 전수 오류를 설명하는 비율, 내부 원인, 다른 방법에서도 같은 원인인지까지는 미확정이다.', '',
             '검토–판정 불일치는 별도 갈래로 남긴다. Qwen 사례를 세 모델 공통 원인으로 일반화하지 않는다. 내부 분석을 하려면 검토 대상·주장 강도·명시적 결론을 먼저 구분하고 동일 대상을 판단한 성공/실패를 대조해야 한다. 현재 자료만으로 내부 신호가 없거나 덮였다는 주장은 하지 않는다.', '',
             '연구 범위는 Review에 한정하지 않는다. 이번 분석은 문항별 검토와 결정이 연결된 세 모델 조건부터 사용했다. 다른 프롬프트·학습·hidden/text gate의 trade-off도 같은 원인이라는 결론은 아직 없으며, 별도 문항 연결 없이 전체 표의 비율만으로 메커니즘을 합치지 않는다.', '',
             '## 재현 및 검증', '',
             '`python3 -m scripts.analyze_shared_gate_tradeoff` — [summary.json](summary.json)에 문항 ID·전환·점수 연결·입력 해시를 보존했다. 새 진단 수치를 기존 표 1/2 대신 쓰거나 기존 라벨을 변경하지 않았다. 이번 의미 점검은 기존 16개 질문의 원문 대조이며 새 AI 전수 분류를 수행하지 않았다.']
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(metrics=metrics, changes=changes, overlap=overlap, non_nested=nested, gemma_answer_transitions=routed, gemma_answer_policies=policies, case_notes=notes, source_sha256=SOURCES)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (OUT / 'cases.md').write_text('\n'.join(cases)+'\n')
    (OUT / 'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'questions': len(decisions), 'cases': len(selected), 'gemma_score_selection_checks': len(gemma)*2, 'output': str(OUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
