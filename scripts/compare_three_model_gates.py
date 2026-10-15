"""Compare saved original-question gate decisions; no inference or judging."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates'
MODELS = ('qwen', 'gemma', 'luna')
LABELS = {'qwen': 'Qwen2.5', 'gemma': 'Gemma', 'luna': 'Luna'}


def main():
    source = ROOT / 'docs/reviews/default_gate_decisions_2026-09-23/decisions.csv'
    gemma_source = ROOT / 'results/gemma4/full_20260926_v2/analysis/question_links.csv'
    table = {}
    names = {'direct': ('qwen', 'direct'), 'review': ('qwen', 'review'),
             'luna_direct': ('luna', 'direct'), 'luna_cot': ('luna', 'review')}
    for row in csv.DictReader(source.open()):
        if row['method'] not in names:
            continue
        model, condition = names[row['method']]
        x = table.setdefault(row['id'], {'id': row['id'], 'set': row['set']})
        assert x['set'] == row['set'] and f'{model}_{condition}' not in x
        assert row['predicted_fpq'] in ('0', '1')
        x[f'{model}_{condition}'] = row['predicted_fpq'] == '1'
    for row in csv.DictReader(gemma_source.open()):
        x = table[row['id']]
        assert x['set'] == row['set'] and 'gemma_direct' not in x
        for stage, col in [('direct', 'direct_gate'), ('review', 'cot_gate')]:
            assert row[col] in ('True', 'False')
            x[f'gemma_{stage}'] = row[col] == 'True'
    assert len(table) == 732
    assert Counter(x['set'] for x in table.values()) == {'fpq': 583, 'nfp': 149}
    assert all(len(x) == 8 for x in table.values())
    result = {'n': 732, 'models': MODELS, 'metrics': {}, 'transitions': {}, 'error_overlap': {},
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source, gemma_source]},
              'limitations': 'Descriptive comparison of existing protocols. Qwen uses Yes>No token scores; Gemma/Luna generated JSON. Not a matched-prompt model comparison, medical adjudication, or shared-cause proof.'}
    for model in MODELS:
        result['metrics'][model] = {}
        result['transitions'][model] = {}
        for split in ['fpq', 'nfp']:
            rows = [r for r in table.values() if r['set'] == split]
            result['metrics'][model][split] = {
                stage: {'positive': sum(r[f'{model}_{stage}'] for r in rows), 'n': len(rows)}
                for stage in ['direct', 'review']}
            transitions = {k: [] for k in ['No→No', 'No→Yes', 'Yes→No', 'Yes→Yes']}
            for r in rows:
                key = ('Yes' if r[f'{model}_direct'] else 'No') + '→' + ('Yes' if r[f'{model}_review'] else 'No')
                transitions[key].append(r['id'])
            result['transitions'][model][split] = {k: {'n': len(v), 'ids': sorted(v)} for k,v in transitions.items()}
    for stage in ['direct', 'review']:
        result['error_overlap'][stage] = {}
        for split in ['fpq', 'nfp']:
            rows = [r for r in table.values() if r['set'] == split]
            errors = {m: {r['id'] for r in rows if r[f'{m}_{stage}'] != (split == 'fpq')} for m in MODELS}
            patterns = {}
            for r in rows:
                pattern = ''.join('1' if r['id'] in errors[m] else '0' for m in MODELS)
                patterns.setdefault(pattern, []).append(r['id'])
            pairwise = {}
            for a,b in itertools.combinations(MODELS,2):
                shared, union = errors[a] & errors[b], errors[a] | errors[b]
                pairwise[f'{a}/{b}'] = dict(shared=len(shared),a_only=len(errors[a]-errors[b]),b_only=len(errors[b]-errors[a]),
                                           jaccard=len(shared)/len(union) if union else None,ids=sorted(shared))
            result['error_overlap'][stage][split] = dict(
                patterns={k: dict(n=len(v),ids=sorted(v)) for k,v in patterns.items()},
                pattern_order=MODELS,pattern_definition='1=error, 0=correct',pairwise=pairwise,
                all_wrong=len(set.intersection(*errors.values())),all_correct=len(rows)-len(set.union(*errors.values())))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'decisions.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','set']+[f'{m}_{s}' for m in MODELS for s in ['direct','review']])
        writer.writeheader();writer.writerows(table[k] for k in sorted(table))
    lines=['# 원본 질문의 세 모델 gate 비교','',
           '같은 FPQ 583/NFP 149개의 기존 저장 판정 비교. 새 추론·채점 없음. 현재 프롬프트와 판정 출력 방식이 완전히 같지 않으므로 모델 자체의 우열이나 동일 원인을 입증하는 비교가 아니다. Qwen2.5는 Yes>No 토큰 판정(동점 No), Gemma/Luna는 생성 JSON 판정이다.','',
           '## 탐지·오탐','',
           '| 모델 | Direct FPQ 탐지 | Review FPQ 탐지 | Direct NFP 오탐 | Review NFP 오탐 |',
           '|---|---:|---:|---:|---:|']
    for m in MODELS:
        vals=[]
        for split,stage in [('fpq','direct'),('fpq','review'),('nfp','direct'),('nfp','review')]:
            d=result['metrics'][m][split][stage];vals.append(f"{d['positive']}/{d['n']} ({100*d['positive']/d['n']:.1f}%)")
        lines.append('| '+LABELS[m]+' | '+' | '.join(vals)+' |')
    lines+=['','## Direct → Review의 네 전환','',
            '| 모델 | FPQ 새 탐지 No→Yes | FPQ 탐지 손실 Yes→No | NFP 새 오탐 No→Yes | NFP 오탐 해소 Yes→No |',
            '|---|---:|---:|---:|---:|']
    for m in MODELS:
        t=result['transitions'][m]
        vals=[t[s][k]['n'] for s,k in [('fpq','No→Yes'),('fpq','Yes→No'),('nfp','No→Yes'),('nfp','Yes→No')]]
        lines.append('| '+LABELS[m]+' | '+' | '.join(map(str,vals))+' |')
    lines+=['','## 같은 문항에서 함께 틀리는가','',
            '| 조건 | 세 모델 모두 놓친 FPQ | 세 모델 모두 맞힌 FPQ | 세 모델 모두 오탐한 NFP | 세 모델 모두 보존한 NFP |',
            '|---|---:|---:|---:|---:|']
    for stage in ['direct','review']:
        d=result['error_overlap'][stage]
        lines.append(f"| {stage} | {d['fpq']['all_wrong']} | {d['fpq']['all_correct']} | {d['nfp']['all_wrong']} | {d['nfp']['all_correct']} |")
    lines+=['','둘 이상의 모델이 함께 틀렸다는 사실은 같은 원인으로 틀렸다는 뜻이 아니다. Qwen의 많은 No 판정, Luna의 많은 Yes 판정 때문에 공통 오류의 크기도 영향을 받는다. 아래의 교집합은 분모와 함께 읽는다.','',
            '| 조건 | 집단의 오류 | 모델 쌍 | 함께 오류 | 앞 모델만 오류 | 뒤 모델만 오류 |',
            '|---|---|---|---:|---:|---:|']
    for stage in ['direct','review']:
        for split in ['fpq','nfp']:
            for pair,x in result['error_overlap'][stage][split]['pairwise'].items():
                a,b=pair.split('/')
                lines.append(f"| {stage} | {split} | {LABELS[a]} / {LABELS[b]} | {x['shared']} | {x['a_only']} | {x['b_only']} |")
    lines+=['','## 해석 범위와 다음 분석 대상','',
            '우선 세 모델 공통 실패와 모델 사이 판정 불일치 문항을 구분해 같은 질문·주석·저장 Review를 대조한다. FPQ에서는 주석의 목표 오류를 실제 지적했는지, NFP에서는 각 모델이 어떤 주장을 오류로 삼았는지 비교한다. 검토문이 없는 Direct의 실제 이유는 복원하지 않는다. 판정이 유지된 문항과 양방향 전환도 함께 남긴다.',
            '이 비교는 현재 실행 조건의 기술적 분석이다. 통일된 프롬프트·출력 규칙을 사용하는 모델 간 비교를 대체하지 않는다. Qwen3.8은 전체 조건 완료 후 공통 기준 비교에 추가한다. [기존 표 1·2](../original_question_tables.md), [문항별 판정](decisions.csv), [집계·문항 ID·출처 해시](summary.json).']
    if (OUT/'semantic_audit/README.md').exists():
        lines += ['', '후속 [세 모델 검토문 의미 비교](semantic_audit/README.md): NFP 149개 전체와 FPQ 층화 표본 59개의 저장 검토문·주석·최종 gate를 대조했다. AI 코딩과 원문 사례 검수를 구분한다.']
    if (OUT/'fpq_failures/README.md').exists():
        lines += ['', '후속 [FPQ 미탐 분석](fpq_failures/README.md): FPQ 583개 × 세 모델의 판정 전환을 전수 집계하고, 탐지 손실 64건 및 명시적 오류 문구 후보 7건의 저장 검토문을 추가 비교했다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ['metrics']},ensure_ascii=False))
    for m in MODELS:print(m,{s:{k:v['n'] for k,v in x.items()} for s,x in result['transitions'][m].items()})
    for s,d in result['error_overlap'].items():print(s,{k:{f:x[f] for f in ['all_wrong','all_correct']} for k,x in d.items()})


if __name__ == '__main__':
    main()
