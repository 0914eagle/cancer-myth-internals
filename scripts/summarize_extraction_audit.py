#!/usr/bin/env python3
"""Validate and summarize a fixed, manually annotated extraction audit (no inference)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len({r['id'] for r in rows}) == len(rows), f'Duplicate IDs: {path}'
    return {r['id']: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    args = parser.parse_args()
    review = args.review_dir
    question_path = args.suite_dir / 'questions.jsonl'
    answer_path = args.suite_dir / 'answers/extract_verify.jsonl'
    questions, answers = read_rows(question_path), read_rows(answer_path)
    assert set(answers) <= set(questions)
    parse_issues = []
    fenced = 0
    for key, row in answers.items():
        assert row['status'] == 'complete'
        assert row['method'] == 'extract_verify'
        assert all(row[k] == questions[key][k] for k in ('question', 'set', 'partition'))
        checked = row['details']['premises']
        for premise in checked:
            assert 0 <= premise['true_score'] <= 1
            assert premise['predicted_true'] == (premise['true_score'] >= .5)
        try:
            text = row['details']['extraction']['text'].strip()
            if text.startswith('```'):
                match = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
                if not match:
                    raise ValueError('Malformed fence')
                text = match.group(1).strip()
                fenced += 1
            raw = json.loads(text)['premises']
            assert isinstance(raw, list) and all(isinstance(p, str) and p.strip() for p in raw)
            raw = list(dict.fromkeys(p.strip() for p in raw))
        except (ValueError, KeyError, TypeError):
            parse_issues.append({'id': key, 'issue': 'not_strict_json_premises'})
        else:
            if raw != [p['premise'] for p in checked]:
                parse_issues.append({'id': key, 'issue': 'raw_checked_difference'})

    sample = json.loads((review / 'sample.json').read_text())
    annotations = json.loads((review / 'annotations.json').read_text())['items']
    assert len(sample) == len(annotations) == 40
    assert len({r['id'] for r in annotations}) == 40
    assert [r['id'] for r in sample] == [r['id'] for r in annotations]
    selected = []
    for label in ('fpq', 'nfp'):
        eligible = [key for key in answers if questions[key]['partition'] == 'fit'
                    and questions[key]['set'] == label]
        selected.extend(sorted(eligible, key=lambda key: hashlib.sha256(
            ('extract-audit-20260922:' + key).encode()).hexdigest())[:20])
    assert selected == [r['id'] for r in sample], 'Sample selection changed'
    counts, targets, verifier = {}, Counter(), []
    lines = ['# 40문항 항목별 추출 감사', '',
             'AI 의미 검토이며 임상 진위 평가가 아니다. 판정 기준은 [protocol.md](protocol.md).', '',
             'F=입력에 충실, U=원문에 없는 단정, ?=보류. C=상황, P=주장, M=전언/믿음/걱정, R=요청.', '',
             'True/False는 내용 주석을 고정한 뒤 붙인 **기존 verifier 출력**이며 정답이 아니다.', '']
    for row, annotation in zip(sample, annotations):
        key = row['id']
        assert all(row[k] == questions[key].get(k) for k in ('question', 'set', 'premise_text', 'correction'))
        checked = answers[key]['details']['premises']
        assert row['premises'] == [p['premise'] for p in checked]
        assert len(row['premises']) == len(annotation['validity']) == len(annotation['kind'])
        assert set(annotation['validity']) <= {'F', 'U', '?'}
        assert set(annotation['kind']) <= {'C', 'P', 'M', 'R'}
        assert all(1 <= i <= len(checked) for i in annotation['target_indices'])
        bucket = counts.setdefault(row['set'], {'questions': 0, 'validity': Counter(), 'kind': Counter()})
        bucket['questions'] += 1
        bucket['validity'].update(annotation['validity'])
        bucket['kind'].update(annotation['kind'])
        if row['set'] == 'fpq':
            targets[annotation['target']] += 1
        lines.extend([f'## {key}', '', row['question'], ''])
        if row['set'] == 'fpq':
            lines.extend([f"참조 premise: {row['premise_text']}", '',
                          f"참조 correction: {row['correction']}", '',
                          f"목표 표현: {annotation['target']}; 원문–참조 관계 보류: {annotation['reference_question_issue']}", ''])
        for i, (p, validity, kind) in enumerate(zip(checked, annotation['validity'], annotation['kind']), 1):
            lines.append(f"{i}. [{validity}/{kind}] {p['premise']} — 기존 verifier: {p['predicted_true']} ({p['true_score']:.6f})")
            verifier.append({'id': key, 'index': i, 'premise': p['premise'],
                             'validity': validity, 'kind': kind,
                             'target': i in annotation['target_indices'],
                             'predicted_true': p['predicted_true'], 'true_score': p['true_score']})
        lines.extend(['', annotation['note'], ''])

    summary = {
        'scope': 'AI semantic audit; fit only; no medical truth assessment or new inference',
        'inventory': {
            'questions': len(questions), 'saved_complete': len(answers),
            'saved_by_set': Counter(r['set'] for r in answers.values()),
            'available_fit_by_set': Counter(r['set'] for r in answers.values() if r['partition'] == 'fit'),
            'missing_ids': sorted(set(questions) - set(answers)),
            'premise_count_distribution': Counter(len(r['details']['premises']) for r in answers.values()),
            'total_extracted_statements': sum(len(r['details']['premises']) for r in answers.values()),
            'json_code_fenced_records': fenced,
            'json_or_alignment_issues': parse_issues,
        },
        'sample_by_set': counts,
        'fpq_target_expression': targets,
        'fpq_reference_question_issue': sum(r['reference_question_issue'] is True for r in annotations),
        'fpq_absent_without_reference_issue': sum(r['target'] == 'absent' and r['reference_question_issue'] is False for r in annotations),
        'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            question_path, answer_path, review / 'sample.json', review / 'annotations.json', review / 'protocol.md')},
    }
    (review / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (review / 'verifier_comparison.json').write_text(json.dumps(verifier, ensure_ascii=False, indent=2) + '\n')
    (review / 'items.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
