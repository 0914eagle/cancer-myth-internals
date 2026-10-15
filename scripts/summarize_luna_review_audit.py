"""Join completed blind semantic audits to frozen gate decisions, without inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/audits/luna_review_grounding_20260924_v1'
OUT = ROOT / 'docs/reviews/gate_diagnosis_2026-09-24/full_audit'

def main():
    plan = json.loads((RUN / 'plan.json').read_text())
    inputs = {r['key']: r for r in json.loads((RUN / 'inputs.json').read_text())}
    manifest = {r['key']: r for r in plan['source_manifest']}
    audit = json.loads((RUN / 'audit.json').read_text())
    assert audit['completed'] == audit['expected'] == 732, 'Full audit incomplete'
    assert len({r['key'] for r in audit['items']}) == 732
    assert {r['key'] for r in audit['items']} == set(inputs)
    decisions_path = ROOT / 'docs/reviews/default_gate_decisions_2026-09-23/decisions.csv'
    decisions = {(r['method'], r['id']): r for r in csv.DictReader(decisions_path.open())}
    output, cells, claims = [], {}, Counter()
    lines = ['# 732문항 AI 의미 감사', '',
             'Sonnet의 라벨 비공개 검토 결과다. 의학적 진위 정답·검증된 원인 비율이 아니다.',
             'grounded는 질문에 충실하다는 뜻이며 참이라는 뜻이 아니다. 다중 오류는 중복 집계한다.', '']
    for item in sorted(audit['items'], key=lambda r: manifest[r['key']]['id']):
        meta, source = manifest[item['key']], inputs[item['key']]
        qid = meta['id']
        direct = int(decisions['luna_direct', qid]['predicted_fpq'])
        cot = int(decisions['luna_cot', qid]['predicted_fpq'])
        unsupported = any(t['grounding'] == 'unsupported' for t in item['targets'])
        unclear = any(t['grounding'] == 'unclear' for t in item['targets'])
        grounded = any(t['grounding'] == 'grounded' for t in item['targets'])
        issues = sorted(set(t['issue'] for t in item['targets'] if t['grounding'] == 'unsupported'))
        row = dict(id=qid, set=meta['set'], direct=direct, cot=cot,
                   stance=item['stance'], n_targets=len(item['targets']),
                   has_unsupported=int(unsupported), has_unclear=int(unclear),
                   has_grounded=int(grounded), unsupported_issues='|'.join(issues))
        output.append(row)
        for bucket in (meta['set'], meta['set'] + '/%s->%s' % (direct, cot),
                       meta['set'] + '/cot=%s' % cot):
            cell = cells.setdefault(bucket, Counter())
            cell.update(n=1, any_unsupported=int(unsupported), any_unclear=int(unclear),
                        any_grounded=int(grounded), no_targets=int(not item['targets']))
            cell.update('unsupported/' + issue for issue in issues)
        for target in item['targets']:
            claims[meta['set'] + '/' + target['grounding']] += 1
        lines.extend(['## ' + qid, '',
                      '라벨: %s; Direct %s → CoT %s; stance: %s' % (meta['set'], direct, cot, item['stance']),
                      '', '**질문**', '', source['question'], '', '**저장된 검토문**', '', source['review'], '',
                      '**Sonnet 감사**', '', item['note'], ''])
        for j, target in enumerate(item['targets'], 1):
            lines.extend(['%s. %s / %s' % (j, target['grounding'], target['issue']),
                          '   - 질문 인용: ' + (target['question_quote'] or '(없음)'),
                          '   - 검토 인용: ' + target['review_quote'],
                          '   - 이유: ' + target['reason'], ''])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'items.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(output)
    summary = dict(n=len(output), model=plan['model'], counts=cells, target_counts=claims,
                   limitations='AI semantic labels, not clinical truth or causal attribution; multi-label counts',
                   sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (RUN / 'plan.json', RUN / 'audit.json', RUN / 'inputs.json', decisions_path)})
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'items.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'annotations.json').write_text(json.dumps([
        dict(id=manifest[r['key']]['id'], **r) for r in audit['items']], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary['counts'], ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
