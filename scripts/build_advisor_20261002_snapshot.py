"""Offline presentation snapshot; validates stored totals, never calls a model."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/advisor_2026-10-02'
SOURCES = {}


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def ratio(n, d):
    return f'{n}/{d} ({100*n/d:.1f}%)'


def main():
    gates = read('docs/reviews/default_gate_decisions_2026-09-23/summary.json')
    aucs = read('docs/reviews/advisor_presentation_2026-09-23/gate_summary.json')
    old = read('docs/reviews/advisor_presentation_2026-09-23/answer_summary.json')
    routing = read('docs/reviews/default_gate_decisions_2026-09-23/routing_summary.json')
    runs = {'Gemma 4 12B': 'results/gemma4/full_20260926_v2',
            'Qwen3.8 OFF': 'results/qwen38/full_transformers_20260926_v1/thinking_off',
            'Qwen3.8 ON': 'results/qwen38/full_transformers_20260926_v1/thinking_on'}
    summaries = {}
    for name, run in runs.items():
        data = read(run + '/analysis/summary.json')
        for method in ['plain', 'zero_shot_cot_one_step', 'premise_review_answer', 'fp_unconditional']:
            report = read(run + '/well/' + method + '/report.json')['methods'][method]
            for label in ['fpq', 'nfp']:
                a, b = data['answers'][method][label], report[label]
                assert a['counts_0_to_5'] == b['counts_0_to_5']
                assert a['valid'] == b['valid'] == (583 if label == 'fpq' else 149)
        for method in ['direct', 'cot']:
            counts = Counter()
            for p in (ROOT / run / 'records' / method).glob('*.json'):
                rec = json.loads(p.read_text())
                if rec['status'] == 'complete':
                    label = rec['id'].split('_')[0]
                    counts[label] += 1
                    counts[label + '_yes'] += rec['result']['has_false_premise']
            for label in ['fpq', 'nfp']:
                assert counts[label] == data['gate'][method][label]['valid']
                assert counts[label + '_yes'] == data['gate'][method][label]['positive']
        summaries[name] = data
    gate_rows = []
    names = {'text': 'TF-IDF text', 'hidden': 'Qwen2.5 hidden probe',
             'direct': 'Qwen2.5 Direct', 'review': 'Qwen2.5 Review gate',
             'luna_direct': 'Luna Direct', 'luna_cot': 'Luna Review gate'}
    for key, label in names.items():
        x = gates[key]
        gate_rows.append([label, f"{aucs[key]['auroc']:.3f}" if key in aucs else '—',
                          ratio(x['tp'], x['tp']+x['fn']), ratio(x['fp'], x['fp']+x['tn'])])
    for name, data in summaries.items():
        for method, label in [('direct', 'Direct'), ('cot', 'Review gate')]:
            g = data['gate'][method]
            gate_rows.append([name+' '+label, '—', ratio(g['fpq']['positive'],g['fpq']['valid']),ratio(g['nfp']['positive'],g['nfp']['valid'])])
    answer_rows = []
    for name, prefix in [('Qwen2.5', 'qwen'), ('Luna', 'luna'), ('Sonnet', 'sonnet')]:
        methods = ['plain', 'premise_review', 'zero_shot_cot', 'fp_identification', 'fp_unconditional'] if prefix == 'qwen' else ['plain','balanced','fp_unconditional']
        for method in methods:
            x = old[prefix+'/'+method]
            answer_rows.append([name, method, ratio(x['fpq']['ge4'],x['fpq']['valid']),ratio(x['nfp']['ge4'],x['nfp']['valid']),ratio(x['fpq']['s5'],x['fpq']['valid']),ratio(x['nfp']['s5'],x['nfp']['valid'])])
    for method in ['text', 'hidden', 'oracle']:
        key = method if method in routing['results'] else 'label_oracle'
        x = routing['results'][key]
        answer_rows.append(['Qwen2.5', method+'_routed', ratio(x['fpq']['ge4'],x['fpq']['n']),ratio(x['nfp']['ge4'],x['nfp']['n']),ratio(x['fpq']['s5'],x['fpq']['n']),ratio(x['nfp']['s5'],x['nfp']['n'])])
    score_path = ROOT/'docs/reviews/advisor_presentation_2026-09-23/answer_scores.csv'
    SOURCES[str(score_path.relative_to(ROOT))] = hashlib.sha256(score_path.read_bytes()).hexdigest()
    scores = {(r['method'],r['id']): int(r['score']) for r in csv.DictReader(score_path.open()) if r['model']=='luna'}
    luna_route = {}
    for stage in ['direct','cot']:
        counts = {label:Counter() for label in ['fpq','nfp']}
        raw_hashes = {}
        for path in (ROOT/'results/frontier_cli/luna_direct_cot_20260923_v1'/stage).glob('*/record.json'):
            rec=json.loads(path.read_text())
            assert rec['status']=='complete'
            qid=rec['id']; flag=rec['result']['has_false_premise']; label=qid.split('_')[0]
            assert ('plain',qid) in scores and ('fp_unconditional',qid) in scores
            score=scores['fp_unconditional' if flag else 'plain',qid]
            counts[label].update(n=1,ge4=int(score>=4),s5=int(score==5))
            if flag:
                before=scores['plain',qid]; after=scores['fp_unconditional',qid]
                counts[label].update(gate_positive=1,
                    positive_new_loss=int(before>=4>after),
                    positive_both_low=int(before<4 and after<4),
                    positive_both_high=int(before>=4 and after>=4),
                    positive_gain=int(before<4<=after))
            raw_hashes[qid]=hashlib.sha256(path.read_bytes()).hexdigest()
        assert counts['fpq']['n']==583 and counts['nfp']['n']==149
        SOURCES['luna_'+stage+'_record_hash_manifest']=hashlib.sha256(json.dumps(raw_hashes,sort_keys=True).encode()).hexdigest()
        luna_route[stage]=counts
        answer_rows.append(['Luna',stage+'_routed',ratio(counts['fpq']['ge4'],583),ratio(counts['nfp']['ge4'],149),ratio(counts['fpq']['s5'],583),ratio(counts['nfp']['s5'],149)])
    for name, data in summaries.items():
        for method, x in data['answers'].items():
            if method.startswith('verify_'):
                continue
            answer_rows.append([name, method, ratio(x['fpq']['at_least_4_count'],x['fpq']['valid']),ratio(x['nfp']['at_least_4_count'],x['nfp']['valid']),ratio(x['fpq']['counts_0_to_5']['5'],x['fpq']['valid']),ratio(x['nfp']['counts_0_to_5']['5'],x['nfp']['valid'])])
    p = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    pairs = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(pairs) == 3659
    stats = defaultdict(Counter)
    for r in pairs:
        before, after = r['plain'], r['alternative']
        if r['dataset'] == 'fpq' and before['score'] < 4 <= after['score']:
            stats[r['model']]['gain'] += 1
            a = before['coding']['reference_alignment']['status'] == 'corrects_target'
            b = after['coding']['reference_alignment']['status'] == 'corrects_target'
            stats[r['model']]['already' if a else 'new' if b else 'unclear'] += 1
        if r['dataset'] == 'nfp' and before['score'] >= 4 > after['score']:
            stats[r['model']]['nfp_loss'] += 1
    assert sum(x['gain'] for x in stats.values()) == 570
    assert sum(x['already'] for x in stats.values()) == 178
    assert sum(x['new'] for x in stats.values()) == 391
    assert sum(x['nfp_loss'] for x in stats.values()) == 121
    cases=[]
    fragments={('qwen25','fpq_626'):[
        'leading to long-term remission and improved quality of life.',
        'Regarding travel agencies that specialize in trips for individuals with terminal illnesses like CML'],
        ('qwen38_off','nfp_1109'):[
        '**Women do not have sperm.**',
        'Given that your sister is a 55-year-old woman']}
    for row in pairs:
        key=(row['model'],row['id'])
        if key in fragments:
            for quote in fragments[key]:
                assert quote in row['alternative']['answer']
            cases.append({'model':row['model'],'id':row['id'],'method':row['method'],
                'question':row['plain']['question'], 'plain_score':row['plain']['score'],
                'alternative_score':row['alternative']['score'],
                'alternative_answer_sha256':row['alternative']['answer_sha256'],
                'alternative_exact_excerpts':fragments[key],
                'interpretation_limit':'Illustrative source/behavior comparison, not clinical adjudication or frequency estimate.'})
    assert len(cases)==2
    statuses = {name: read(f'results/audits/measurement_validity_20261001_v1/{name}/status.json') for name in ['questions','luna_repeats']}
    snapshot = {'captured_at_kst': datetime.now(timezone(timedelta(hours=9))).isoformat(),
                'gates': gate_rows, 'answers': answer_rows, 'paired_behavior': dict(stats),
                'qwen_default_routing': routing, 'luna_routing_recomputed':luna_route, 'new_runs_snapshot_not_live': statuses,
                'source_sha256': SOURCES,
                'validation': 'New gate counts checked against individual boolean records; answer distributions checked against Well reports; 3659 paired rows recomputed.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2)+'\n')
    (OUT/'case_excerpts.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2)+'\n')
    lines = ['# 2026-10-02 발표 수치 근거', '', '수치 고정 시각: '+snapshot['captured_at_kst'], '',
             '실행 상태는 고정 시점의 기록이며 현재 상태가 아니다. 새 검토의 내용상 결론은 포함하지 않는다.', '',
             '## 표 1: 질문 단위 판정', '', '| 방법 | AUROC | FPQ 탐지 | NFP 오탐 |', '|---|---:|---:|---:|']
    lines += ['| '+' | '.join(row)+' |' for row in gate_rows]
    lines += ['', '## 표 2: 답변과 선택 결과', '', '| 모델 | 방법 | FPQ ≥4 | NFP ≥4 | FPQ S5 | NFP S5 |', '|---|---|---:|---:|---:|---:|']
    lines += ['| '+' | '.join(row)+' |' for row in answer_rows]
    lines += ['', 'Qwen2.5의 분모 차이는 채점 누락이다. JSON gate의 자기보고 확률 AUROC는 보고하지 않는다.',
              'Direct/Review routed는 gate가 양성이면 저장된 무조건 교정 답변, 음성이면 Plain 답변을 선택한다.',
              '모델 간 프롬프트·생성 절차 차이를 순수한 모델 효과로 해석하지 않는다.', '',
              '[집계·출처 해시](snapshot.json) · [재생성 코드](../../../scripts/build_advisor_20261002_snapshot.py)', '']
    (OUT/'tables.md').write_text('\n'.join(lines))
    print(json.dumps({'gate_rows':len(gate_rows),'answer_rows':len(answer_rows),'paired_stats':dict(stats),'snapshot':snapshot['captured_at_kst']},ensure_ascii=False))


if __name__ == '__main__':
    main()
