"""Join four source conditions without deriving Well scores from reader outputs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'results/nla_ao_parallel_20261012'
OUT = BASE / 'source_false_premise_probe'
POSITIONS = ['question_last_token', 'prompt_keyword_1', 'assistant_boundary_last_token']
CONDITIONS = ['plain', 'balanced', 'assert_false_premise', 'assert_and_correct']


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    config = json.loads((OUT / 'runtime.json').read_text())
    questions = {r['id']: r for r in config['cohort']}
    annotations = json.loads((OUT / 'qualitative_annotations.json').read_text())
    notes = {(r['model'], r['id'], r['condition']): r for r in annotations['answer_reviews']}
    answers = {}
    nla = {}
    verification = {}
    files = [OUT / 'runtime.json', OUT / 'protocol.md', OUT / 'qualitative_annotations.json']
    for model, spec in config['models'].items():
        old_nla = [r for r in read_rows(BASE / model / 'av_results.jsonl') if r['base_id'] in questions]
        new_nla = read_rows(OUT / model / 'av_results.jsonl')
        new_answers = read_rows(OUT / model / 'answers.jsonl')
        assert len(new_answers) == 16 and len(new_nla) == 48 and len(old_nla) == 48
        assert len({r['id'] for r in new_nla}) == 48
        assert all(r['finish_reason'] == 'eos' and r['parsed_explanation_tag'] for r in new_nla)
        canonical = {r['query'] for r in old_nla}
        assert len(canonical) == 1 and {r['query'] for r in new_nla} == canonical
        assert all(r['layer'] == spec['layer'] and r['hidden_state_index'] == spec['layer'] + 1 for r in new_nla)
        assert all(r['activation_mode'] == 'actual_native_answer_generation_prefill' for r in new_nla)
        for row in old_nla + new_nla:
            key = (model, row['base_id'], row['condition'], row['position_family'])
            assert key not in nla
            nla[key] = row
        if model == 'gemma27':
            old_answers = read_rows(BASE / 'ao_analysis/native_answers/answers.jsonl')
        else:
            old_answers = [r for r in read_rows(Path(config['reference_root']) / model / 'answers.jsonl') if r['id'] in questions]
        assert len(old_answers) == 16
        for row in old_answers + new_answers:
            key = (model, row['id'], row['condition'])
            assert key not in answers
            answers[key] = row
        for row in new_answers:
            key = (model, row['id'], row['condition'])
            assert row['completed'] and row['metadata']['finish_reason'] == 'eos' and key in notes
            plain = answers[(model, row['id'], 'plain')]
            # The native Gemma27 audit stores only messages_sha256, not messages.
            if 'messages' in plain:
                assert row['messages'][1] == plain['messages'][1]
                assert row['messages'][0]['content'] == plain['messages'][0]['content'] + '\n\n' + row['source_prompt_addition']
        verification[model] = dict(new_answers=16, new_nla=48, baseline_nla=48,
            canonical_reader_prompt_exact_match=True, all_new_answers_eos=True, all_new_nla_eos=True,
            source_layer=spec['layer'], hf_hidden_state_index=spec['layer'] + 1,
            baseline_source_answer_provenance='Prior native answer files; Gemma27 additionally has bytewise prefill audit',
            new_source_answer_provenance='Each activation captured in that native answer generation prefill')
        files += [OUT / model / 'answers.jsonl', OUT / model / 'av_results.jsonl', OUT / model / 'manifest.jsonl']
    ao = {}
    ao_files = [BASE / 'ao_prompt_probe/gemma27/ao_results.jsonl', OUT / 'gemma27/ao_results.jsonl']
    for path in ao_files:
        rows = read_rows(path)
        assert len(rows) == 64
        for row in rows:
            key = (row['base_id'], row['condition'], row['prompt_variant'])
            assert key not in ao
            ao[key] = row
            query = next(x['text'] for x in config['queries'] if x['id'] == row['prompt_variant'])
            assert row['oracle_question'] == query and row['layer'] == 31
            assert row['window_size'] == min(20, row['shape'][0])
    files += ao_files
    paired = []
    for model in config['models']:
        for ident, question in questions.items():
            for condition in CONDITIONS:
                native = answers[(model, ident, condition)]
                row = dict(model=model, id=ident, source=question['dataset'],
                    category='FPQ' if question['label'] else 'NFP', condition=condition,
                    question=question['question'], answer=native['answer'], completed=native['completed'],
                    answer_review=notes.get((model, ident, condition)),
                    source_prompt_addition=config['source_prompt_additions'].get(condition),
                    nla=[nla[(model, ident, condition, position)] for position in POSITIONS],
                    ao=[ao[(ident, condition, query['id'])] for query in config['queries']] if model == 'gemma27' else [])
                paired.append(row)
    assert len(paired) == 96
    (OUT / 'paired_comparison.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in paired))
    with (OUT / 'paired_comparison.csv').open('w', newline='') as stream:
        keys = ['model', 'id', 'category', 'condition', 'question', 'answer', 'manual_answer_rating', 'manual_answer_note_ko']
        keys += ['nla_' + p for p in POSITIONS] + ['ao_' + x['id'] for x in config['queries']]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in paired:
            record = {k: row[k] for k in ['model', 'id', 'category', 'condition', 'question', 'answer']}
            record['manual_answer_rating'] = (row['answer_review'] or {}).get('rating', 'see_existing_audit')
            record['manual_answer_note_ko'] = (row['answer_review'] or {}).get('note_ko', '')
            record.update({'nla_' + x['position_family']: x['nla_output'] for x in row['nla']})
            record.update({'ao_' + x['prompt_variant']: x['ao_output'] for x in row['ao']})
            writer.writerow(record)
    summary = dict(state='complete', independent_questions=8, new_native_answers=48,
        new_nla_explanations=144, new_ao_explanations=64, total_paired_answer_records=96,
        source_conditions=CONDITIONS, verification=verification,
        new_answer_manual_ratings={model: {condition: dict(Counter(r['rating'] for r in annotations['answer_reviews']
            if r['model'] == model and r['condition'] == condition)) for condition in config['conditions']}
            for model in config['models']},
        reader_findings=annotations['reader_findings'], limits=annotations['limits'])
    summary['nfp_premise_challenges'] = {
        model: {condition: sum(r['model'] == model and r['condition'] == condition and
                              r['rating'] == 'nfp_premise_challenge' for r in annotations['answer_reviews'])
                for condition in CONDITIONS} for model in config['models']}
    summary['nfp_denominator_per_model_condition'] = 4
    summary['ao_descriptive_counts'] = annotations['ao_descriptive_counts']
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    report = ['# 원 모델 오류 선언 대조', '', annotations['conclusion_ko'], '',
              '## 실행 범위', '',
              '고정 학습8문항×추가2조건×모델3종으로 실제 답변48개/NLA144개, Gemma27 AO64개를 새로 생성했다. Plain/Balanced를 합친96개 모델-문항-조건 레코드를 CSV/JSONL에 연결했다. 신규 활성은 각 실제 답변 생성의 prefill에서 저장했다. NLA 기본 프롬프트 및 AO 질문4종은 기존과 동일하다. 공식 Well/Haiku 채점은 수행하지 않았다.', '',
              '## 원 모델에 추가한 지시문', '']
    for condition, text in config['source_prompt_additions'].items():
        report += [f'**{condition}**', '', text, '']
    report += ['## 정상 질문의 전제를 문제삼은 답변', '',
               '각 칸의 분모는4개 NFP 문항이다. 명시적 오류 선언 또는 사용자 상황을 전제로 지목하여 반박한 경우를 단일 검토자가 분류했다. 상황을 받아들이면서 일반적인 조건을 부연한 답변은 별도 qualification으로 분류한다. 공식 Well 오반박 점수가 아니다.', '',
               '| 모델 | Plain | Balanced | 오류 존재 선언 | 선언+교정 요구 |', '|---|---:|---:|---:|---:|']
    for model, counts in summary['nfp_premise_challenges'].items():
        report.append('| ' + model + ' | ' + ' | '.join(f'{counts[c]}/4' for c in CONDITIONS) + ' |')
    report += ['', '## 주석 대조 범위', ''] + ['- ' + s for s in annotations['annotation_scope_notes']] + ['']
    report += ['## 설명 검토', ''] + ['- ' + s for s in annotations['reader_findings']] + ['', '## 실제 답변 검토', '']
    for model in config['models']:
        report += [f'### {model}', '', '| 문항 | 조건 | 검토 |', '|---|---|---|']
        for r in annotations['answer_reviews']:
            if r['model'] == model:
                report.append(f"| {r['id']} | {r['condition']} | {r['note_ko']} |")
        report.append('')
    report += ['## 설명의 직접 예시', '']
    for example in annotations['reader_examples']:
        record = next(r for r in paired if (r['model'], r['id'], r['condition']) ==
                      (example['model'], example['id'], example['condition']))
        reader = example['reader']
        selected = next(x for x in record[reader] if x['position_family' if reader == 'nla' else 'prompt_variant'] == example['selector'])
        report += [f"### {example['model']} / {example['id']} / {example['condition']} / {reader} / {example['selector']}", '',
                   selected['nla_output' if reader == 'nla' else 'ao_output'], '', example['note_ko'], '']
    report += ['## 해석의 한계', ''] + ['- ' + s for s in annotations['limits']] + ['']
    (OUT / 'inspection_report.md').write_text('\n'.join(report))
    files += [OUT / 'paired_comparison.jsonl', OUT / 'paired_comparison.csv', OUT / 'summary.json', OUT / 'inspection_report.md']
    hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT / 'artifact_checksums.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'verification'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
