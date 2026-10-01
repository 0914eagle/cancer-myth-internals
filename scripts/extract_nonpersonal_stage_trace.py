"""Export saved regression cases and review stages; makes no model calls."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT = ROOT / 'docs/reviews/nonpersonal_stage_trace_2026-10-02'
QWEN = Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/answers/premise_review.jsonl')


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    pairs = [json.loads(line) for line in SOURCE.open()]
    selected = [x for x in pairs if x['dataset'] == 'fpq'
                and x['plain']['score'] >= 4 and x['alternative']['score'] < 4
                and x['plain']['coding']['reference_alignment']['status'] == 'corrects_target'
                and x['alternative']['coding']['reference_alignment']['status'] != 'corrects_target']
    assert len(selected) == 45
    assert len({(x['model'], x['id']) for x in selected}) == 45
    qwen = {x['id']: x for x in map(json.loads, QWEN.open())}
    OUT.mkdir(parents=True, exist_ok=True)
    rows, stages, truncation = [], [], Counter()
    for x in selected:
        m, i = x['model'], x['id']
        rows.append(dict(model=m, id=i, method=x['method'], plain_score=x['plain']['score'],
                         alternative_score=x['alternative']['score'], question=x['plain']['question'],
                         plain_code=x['plain']['coding']['reference_alignment']['status'],
                         alternative_code=x['alternative']['coding']['reference_alignment']['status']))
        if m == 'qwen25':
            record, source = qwen[i], str(QWEN)
            review = record['details']['review']['text']
        elif m == 'gemma':
            source = f'results/gemma4/full_20260926_v2/records/premise_review_answer/{i}.json'
            record = json.loads((ROOT / source).read_text())
            review = record['prompt'].split('\n\nPremise review:\n', 1)[1]
        else:
            if m.startswith('qwen38'):
                mode = 'thinking_on' if m.endswith('on') else 'thinking_off'
                path = ROOT / f'results/qwen38/full_transformers_20260926_v1/{mode}/records/zero_shot_cot_one_step/{i}.json'
                record = json.loads(path.read_text())
                assert record['response'] == x['alternative']['answer']
                truncation[(m, str(record.get('truncated')))] += 1
            continue
        assert record['response'] == x['alternative']['answer']
        if m == 'gemma':
            truncation[(m, str(record.get('truncated')))] += 1
        stages.append(dict(model=m, id=i, source=source, question=x['plain']['question'],
                           reference=x['plain']['reference'], plain=x['plain']['answer'], review=review,
                           final=x['alternative']['answer'], review_sha256=digest(review),
                           final_sha256=digest(record['response']),
                           plain_score=x['plain']['score'], final_score=x['alternative']['score'],
                           generation_metadata=record.get('details', {k: record.get(k) for k in ['truncated', 'output_tokens', 'max_new_tokens']})))
    with (OUT / 'regression45_index.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    with (OUT / 'two_step_six.jsonl').open('w') as f:
        for x in stages:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    assert len(stages) == 6
    focus_ids = {'fpq_8', 'fpq_30', 'fpq_102', 'fpq_547', 'fpq_745'}
    focus = [x for x in selected if x['model'] == 'qwen38_off' and x['id'] in focus_ids]
    assert len(focus) == 5
    with (OUT / 'cot_five.jsonl').open('w') as f:
        for x in focus:
            f.write(json.dumps(x, ensure_ascii=False) + '\n')
    summary = dict(selection_counts=dict(Counter(x['model'] for x in selected)),
                   saved_stage_count=len(stages),
                   truncation_metadata=[dict(model=m, truncated=t, count=n) for (m,t),n in truncation.items()],
                   warning='Selection uses existing Well and AI codes, not independently established causal failures. All 45 questions were screened; this is not a full reading of all 90 answers.')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
