"""Publish observed examples and explicit limits of the preliminary NLA analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    summary = json.loads((root / 'analysis_summary.json').read_text())
    annotations = json.loads((root / 'qualitative_annotations.json').read_text())
    comparisons = [json.loads(line) for line in
                   (root / 'question_answer_explanation_comparison.jsonl').open()]
    lookup = {(r['model'], r['dataset'], r['id']): r for r in comparisons}
    lines = ['# NLA 답변 전 활성 설명: 예비 확인', '',
             'AO는 보류했다. Cancer fold0 train 385문항과 CREPE 공식 train의 고정 표본 '
             '200 FPQ + 200 NFP를 사용했다. 모델별 785문항의 Plain 조건에서 '
             '답변 경계 마지막 토큰과 질문 내 TF-IDF 키워드 토큰을 읽었다.', '',
             '## 완료 범위', '',
             '| 모델 | 설명 | 기존 설명 재사용 | 새 배치 생성 | AR 검증 | 설명 MSE | 질문 텍스트 대조 MSE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for model, s in summary['models'].items():
        lines.append(f"| {model} | {s['completed_vectors']}/{s['expected_vectors']} | "
                     f"{s['reused_vectors']} | {s['new_vectors']} | {s['ar_scored_vectors']} | "
                     f"{s.get('mean_recon_mse', float('nan')):.4f} | "
                     f"{s.get('mean_question_control_mse', float('nan')):.4f} |")
    lines += ['', '낮은 MSE는 활성 방향 복원 성능이다. 의미적 정확성이나 FPQ 교정 성공을 뜻하지 않는다.',
              '기존 단일 생성과 새 배치 생성의 설정은 각 행의 출처·실행 정책으로 구분해 보존했다.', '',
              '## 직접 읽은 설명의 경향', '']
    lines += ['- ' + s for s in annotations['conclusions_ko']]
    control_path = root / 'activation_direction_control.json'
    if control_path.exists():
        control = json.loads(control_path.read_text())
        lines += ['', '## 공통 활성 방향 대조', '',
                  '해당 토큰 위치에서 다른 784문항의 단위 활성 벡터를 평균하고, '
                  '그 방향으로 원래 활성을 복원했다. 자기 문항은 제외했고 질문·설명·정답 라벨은 사용하지 않았다. '
                  '학습 표본 내부의 기하학적 진단이며 최종 test 평가가 아니다.', '',
                  '| 모델 | 위치 | AR 검증 수 | NLA 설명 복원 MSE | 다른 문항 평균 방향 MSE |',
                  '|---|---|---:|---:|---:|']
        for model in ('qwen25', 'gemma3'):
            scores = [json.loads(line) for line in (root / model / 'scored.jsonl').open()] \
                if (root / model / 'scored.jsonl').exists() else []
            for position, label in [('assistant_boundary_last_token', '답변 경계'),
                                    ('prompt_keyword_1', '키워드')]:
                group = [r for r in scores if r['position_family'] == position]
                ar = sum(r['recon_mse'] for r in group) / len(group) if group else float('nan')
                mse = control['models'][model][position]['mean_leave_one_out_direction_mse']
                lines.append(f'| {model} | {label} | {len(group)}/785 | {ar:.4f} | {mse:.4f} |')
        lines += ['', 'Qwen 답변 경계에서는 설명으로 복원한 오차보다 다른 문항의 평균 방향 오차가 낮다. '
                  '따라서 답변 경계 설명의 낮은 복원 오차만으로 질문별 정보를 잘 설명했다고 판단하기 어렵다. '
                  '반면 Qwen 키워드에서는 설명 복원이 평균 방향보다 낫다. 이는 키워드 정보와 방향의 대응을 '
                  '보여 주지만 오류 전제 탐지나 GEPA 성능 개선을 증명하지 않는다. '
                  '모델 사이의 MSE 크기만으로 설명 품질 순위를 매기지 않는다.', '']
    lines += ['', '## 실제 사례', '',
              '아래 사례는 설명의 형태와 실패 양상을 보여 주기 위해 직접 확인했다. '
              '무작위 정확도 감사나 전체 문항의 성공률 추정은 아니다. '
              'Plain·Balanced 답변에는 아직 Well 채점을 하지 않았다.', '']
    evidence = []
    for case in annotations['cases']:
        lines += [f"### {case['dataset']} / {case['id']}", '', case['note_ko'], '']
        for model in ('qwen25', 'gemma3'):
            row = lookup[model, case['dataset'], case['id']]
            evidence.append(row)
            lines += [f"**{model}**", '', '질문: ' + row['question'], '',
                      '**답변 경계 설명**', '', row['boundary_explanation'] or '(아직 미생성)', '',
                      '**키워드 설명**', '', f"토큰: `{row['keyword_token']}`", '',
                      row['keyword_explanation'] or '(아직 미생성)', '',
                      '**실제 Plain 답변 앞부분**', '', row['plain_answer'][:900], '',
                      '**실제 Balanced 답변 앞부분**', '', row['balanced_answer'][:900], '']
    lines += ['## 다음 비교에서 확인할 점', '',
              '- 질문 마지막 토큰과 답변 경계 토큰을 구분해, 채팅 형식의 설명이 지배하는지 확인한다.',
              '- TF-IDF 키워드가 soon·lasting처럼 과제와 거리가 먼 단어를 고를 수 있으므로, '
              '학습 자료에서만 질문 기반 위치 선택 규칙을 비교한다.',
              '- NLA 설명은 검증되지 않은 가설로 취급한다. 실제 답변의 교정·오반박 판정을 '
              '설명 자체로 대신하지 않는다.',
              '- 이후 텍스트 피드백만 사용한 GEPA와 NLA 설명을 더한 조건을 같은 train/dev·'
              'GPT 모델·평가 예산으로 비교한다. 이번 확인에는 최종 test를 사용하지 않았다.', '',
              '전체 문항의 질문·Plain/ Balanced 답변·두 위치의 설명·AR 점수는 '
              '`question_answer_explanation_comparison.csv`에 저장했다.', '']
    (root / 'inspection_report.md').write_text('\n'.join(lines))
    with (root / 'reviewed_examples.jsonl').open('w') as stream:
        for row in evidence:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    files = ['runtime_nla_only.json', 'analysis_summary.json', 'qualitative_annotations.json',
             'inspection_report.md', 'question_answer_explanation_comparison.csv',
             'activation_direction_control.json']
    files += [f'{m}/{n}' for m in ('qwen25', 'gemma3')
              for n in ('manifest.jsonl', 'av_results.jsonl', 'scored.jsonl')]
    audit = {name: dict(sha256=hashlib.sha256((root / name).read_bytes()).hexdigest(),
                       bytes=(root / name).stat().st_size)
             for name in files if (root / name).exists()}
    (root / 'artifact_checksums.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(root / 'inspection_report.md')


if __name__ == '__main__':
    main()
