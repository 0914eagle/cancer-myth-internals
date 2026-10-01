"""Validate saved Luna Table 2 generation/judging and publish coverage-aware counts.

Read-only with respect to experimental records. No model calls or retries.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/frontier_cli/luna_table2_completion_20261001_v1'
OUT = ROOT / 'docs/reviews/table2_completion_2026-10-01'


def main():
    judge = SOURCE / 'well_judge_sonnet_v1'
    plan = json.loads((judge / 'judge_plan.json').read_text())
    report = json.loads((judge / 'report.json').read_text())
    attempts = [json.loads(x) for x in (judge / 'attempts.jsonl').read_text().splitlines()]
    valid = {}
    for row in attempts:
        assert row['plan_hash'] == plan['plan_hash']
        if row.get('valid'):
            assert row['id'] not in valid, 'Duplicate valid judgment'
            assert isinstance(row['score'], int) and 0 <= row['score'] <= 5
            valid[row['id']] = row
    assert len(valid) == report['unique_valid']
    questions = {r['id']: r for r in map(json.loads, (SOURCE / 'evaluation_questions.jsonl').read_text().splitlines())}
    scores, groups, generation = [], [], {}
    for method, mapping in plan['plan']['mapping'].items():
        answers = {r['id']: r for r in map(json.loads, (SOURCE / method / 'answers_eval732.jsonl').read_text().splitlines())}
        assert set(answers) == set(questions) == set(mapping)
        for qid, answer in answers.items():
            rec = json.loads((SOURCE / method / 'records' / f'{qid}.json').read_text())
            assert rec['status'] == answer['status'] == 'complete'
            assert rec['response'] == answer['response']
            assert answer['question'] == questions[qid]['question']
        generation[method] = len(answers)
        for label in ['fpq', 'nfp']:
            ids = [qid for qid in questions if questions[qid]['set'] == label]
            found = []
            for qid in ids:
                result = valid.get(mapping[qid])
                if result:
                    found.append(result['score'])
                    scores.append(dict(method=method, id=qid, label=label, score=result['score'], judgment_id=mapping[qid]))
            c = Counter(found)
            expected = report['methods'][method][label]
            assert len(found) == expected['valid'] and len(ids) == expected['expected']
            assert {str(i): c[i] for i in range(6)} == expected['counts_0_to_5']
            groups.append(dict(method=method, label=label, generated=len(ids), judged=len(found),
                               unjudged=len(ids)-len(found), ge4=c[4]+c[5],
                               ge4_pct_among_judged=100*(c[4]+c[5])/len(found), s5=c[5],
                               score_counts={str(i): c[i] for i in range(6)}))
    summary = dict(status='partial_judging_not_full_evaluation' if len(valid) < 1464 else 'complete',
                   generator='gpt-5.6-luna', effort='medium', judge='claude-sonnet-5',
                   generation=generation, valid_judgments=len(valid), expected_judgments=1464,
                   remaining_judgments=1464-len(valid), groups=groups,
                   last_saved_judge_status=json.loads((judge/'status.json').read_text()),
                   last_stop_reason='session quota in saved failed attempts; current quota not probed',
                   source_files_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in [judge/'attempts.jsonl', judge/'judge_plan.json', judge/'report.json',
                                 SOURCE/'manifest.json', SOURCE/'evaluation_questions.jsonl']},
                   limitations=['Percentages use judged subsets, not all 583/149 questions.',
                                'Judged subsets differ between conditions; no paired improvement claim.',
                                'Historical balanced-prompt behavior cohorts are unchanged.',
                                'No new generation, judging, or retries.'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'luna_answer_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    with (OUT/'luna_answer_scores.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['method','id','label','score','judgment_id'], lineterminator='\n')
        w.writeheader(); w.writerows(scores)
    lines=['# Luna Table 2 추가 답변: 저장 결과 재집계', '',
           f'저장 원장 재집계. 두 조건 모두 732개씩 생성 완료. Sonnet 유효 채점 {len(valid)}/1,464개, 미완료 {1464-len(valid)}개. 마지막 중단 기록은 세션 한도이며 현재 한도를 새로 확인하지 않았다.', '',
           '**아래는 채점된 부분집합의 중간 결과다. 전체 표의 583/149 분모와 다르며 조건 간 평가 문항도 다르다. 미채점은 0점으로 처리하지 않았다.**', '',
           '| 조건 | 집합 | 생성 | 채점 | ≥4 | 채점분 ≥4 비율 |', '|---|---|---:|---:|---:|---:|']
    for r in groups:
        label='일반 CoT (1-step)' if r['method']=='zero_shot_cot_one_step' else '전제 검토 CoT (2-step) → 답변'
        lines.append(f"| {label} | {r['label'].upper()} | {r['generated']} | {r['judged']} | {r['ge4']} | {r['ge4_pct_among_judged']:.1f}% |")
    lines += ['', '일반 CoT는 원 질문 뒤 `Let\'s think step by step.`를 붙였다. 전제 검토 조건은 기존 Luna 검토문을 재사용하여 질문+검토문에서 답변을 생성했다. 두 조건의 새 답변은 gate로 저장 답변을 고른 결과와 다르다. 기존 균형 지시 74.6/55.7은 별도 조건이며 이번 일반 CoT 결과로 부르지 않는다.', '',
              '기존 3,659쌍·299쌍 분석에서 Luna 대안은 여전히 균형 지시다. 이 추가 결과를 그 행동 분석에 소급해서 섞지 않는다.', '',
              '재집계: `python3 scripts/summarize_luna_table2_completion.py`. [요약·원본 해시](luna_answer_summary.json) · [문항별 점수](luna_answer_scores.csv).']
    (OUT/'luna_answer_progress.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(generation=generation, valid=len(valid), groups=groups),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
