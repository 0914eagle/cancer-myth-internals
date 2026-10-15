"""Export a paired position comparison and a leave-one-out direction control."""
import argparse
import collections
import csv
import json
from pathlib import Path

POSITIONS = ['question_last_token', 'assistant_role_token', 'assistant_boundary_last_token']
LABELS = ['질문 마지막 토큰', '줄바꿈 직전 역할 토큰', '마지막 줄바꿈']


def read(path):
    return [json.loads(line) for line in path.open()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    summaries = {}
    comparisons = []
    examples = []
    config = json.loads((args.root / 'runtime.json').read_text())
    chosen = []
    for dataset in ['cancer_myth', 'crepe']:
        for label in [1, 0]:
            chosen.append(min((r for r in config['cohort'] if r['dataset'] == dataset and r['label'] == label),
                              key=lambda r: r['id']))
    for model in ['qwen25', 'gemma3']:
        directory = args.root / model
        targets = read(directory / 'manifest.jsonl')
        outputs = read(directory / 'scored.jsonl')
        assert len(targets) == len(outputs) == 192
        assert {r['id'] for r in targets} == {r['id'] for r in outputs}
        by_id = {r['id']: r for r in outputs}
        groups = collections.defaultdict(list)
        for row in targets:
            assert row['partition'] == 'train' and row['condition'] == 'plain'
            vector = torch.load(row['activation_path'], map_location='cpu', weights_only=True).numpy().astype(np.float64)
            assert np.isfinite(vector).all() and np.linalg.norm(vector) > 0
            groups[row['position_family']].append((row, vector / np.linalg.norm(vector)))
        summaries[model] = {}
        for position, group in groups.items():
            unit = np.stack([v for _, v in group])
            other = unit.sum(axis=0, keepdims=True) - unit
            other /= np.linalg.norm(other, axis=1, keepdims=True)
            mse = 2 * (1 - np.einsum('ij,ij->i', unit, other))
            scores = [by_id[r['id']] for r, _ in group]
            summaries[model][position] = dict(n=len(group),
                mean_recon_mse=sum(r['recon_mse'] for r in scores) / len(scores),
                mean_leave_one_out_direction_mse=float(mse.mean()),
                tokens=dict(collections.Counter(repr(r['selected_token_text']) for r, _ in group)))
        pairs = collections.defaultdict(dict)
        for row in outputs:
            pairs[row['source'], row['base_id']][row['position_family']] = row
        for (dataset, ident), pair in sorted(pairs.items()):
            assert set(pair) == set(POSITIONS)
            result = dict(model=model, dataset=dataset, id=ident, question=pair[POSITIONS[0]]['prompt'])
            for position in POSITIONS:
                result[position + '_token'] = pair[position]['selected_token_text']
                result[position + '_explanation'] = pair[position]['nla_output']
                result[position + '_recon_mse'] = pair[position]['recon_mse']
            comparisons.append(result)
        for item in chosen:
            examples.append(next(r for r in comparisons if r['model'] == model and
                                 r['dataset'] == item['dataset'] and r['id'] == item['id']))
    summary = dict(scope=config['analysis_scope'], models=summaries,
                   caution='Direction MSE and illustrative readings do not establish semantic accuracy or FPQ correction performance',
                   source_pass='All three positions extracted together in one new forward; no answer generated',
                   ao_status='deferred')
    (args.root / 'position_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (args.root / 'position_comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    with (args.root / 'review_examples.jsonl').open('w') as stream:
        for row in examples:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    lines = ['# 줄바꿈 직전 토큰의 NLA 설명 비교', '',
             '같은 64개 학습 문항(각 데이터셋·라벨 16개)을 고정해 Plain 프롬프트를 새로 forward했다. '
             '세 위치를 같은 forward에서 추출했다. 답변을 생성하지 않았고 NLA/AR는 공개 체크포인트를 사용했다.', '',
             '## 직접 확인한 결과', '',
             '줄바꿈 직전은 Qwen의 assistant, Gemma의 model 역할 토큰이었다. '
             '이 토큰이 앞의 질문을 반영할 수는 있지만, 직접 읽은 예시에서는 '
             'NLA 설명이 AI의 일반적인 답변 형식·면책 문구에 집중하거나 무관한 주제를 추가했다.', '',
             'Gemma의 소아 안암 수술 예시에서는 질문 끝 토큰이 아이의 수술과 심리적 준비라는 주제를 더 잘 짚었다. '
             '역할 토큰은 일반 AI 답변 형식을, 줄바꿈은 원문에 없는 반려동물 상실을 설명했다. '
             'Qwen은 질문 끝에서도 임신 맥락을 덧붙였고, Gemma 역시 뇌진탕 예시에서 수면무호흡 등의 '
             '원문에 없는 맥락을 추가했다. 위치를 바꾸는 것만으로 의미적 정확성이 보장되지는 않는다.', '',
             '이 검토는 아래 8개 모델·문항 조합의 24개 설명을 직접 읽은 예비 확인이다. '
             '전체 설명의 의미적 정확도나 오류 전제 탐지 성능을 채점하지 않았다.', '',
             '## 활성 방향 복원 대조', '',
             '| 모델 | 위치 | NLA 설명 복원 MSE | 다른 63문항 평균 방향 MSE |',
             '|---|---|---:|---:|']
    for model, positions in summaries.items():
        for position, label in zip(POSITIONS, LABELS):
            s = positions[position]
            lines.append(f"| {model} | {label} | {s['mean_recon_mse']:.4f} | {s['mean_leave_one_out_direction_mse']:.4f} |")
    lines += ['', 'MSE는 활성 방향 복원 오차이며 설명의 정확도·답변 교정 성공률이 아니다. '
              '아래 예시는 각 데이터셋·라벨에서 표본 ID 순으로 첫 문항을 선정했다.', '']
    for row in examples:
        lines += [f"## {row['model']} / {row['id']}", '', row['question'], '']
        for position, label in zip(POSITIONS, LABELS):
            lines += [f"**{label} — 토큰 {repr(row[position + '_token'])}**", '',
                      row[position + '_explanation'], '']
    (args.root / 'position_report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
