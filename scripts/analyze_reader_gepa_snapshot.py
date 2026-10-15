"""CPU-only snapshot analysis; no model generation or original artifacts changed."""
import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import html
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / 'results/nla_ao_parallel_20261012/gepa_reader_pilot'
OUT = PILOT / 'cpu_analysis_20261013'


def load(p):
    return json.loads(p.read_text())


def key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def write(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = load(PILOT / 'plan.json')
    audits = {r['row_id']: r for r in (load(f) for f in (PILOT / 'analysis').glob('*.json'))
              if 'row_id' in r}
    assert len(audits) == 100
    natives = {r['id']: r for r in (load(f) for f in (PILOT / 'native/results').glob('*.json'))
               if 'row_id' in r}
    by_input = {(r['candidate_prompt'], r['row_id']): r for r in natives.values()}
    assert len(natives) == len(by_input)
    candidate_rows, selection_rows, dev_rows, trajectory = [], [], [], []
    for dataset in ['cancer_myth', 'crepe']:
        expected = set(plan['splits'][dataset]['candidate_dev_ids'])
        for arm in ['gepa', 'text', 'nla', 'ao']:
            d = PILOT / 'optimization' / dataset / arm
            f = d / 'evaluations.jsonl'
            events = [json.loads(s) for s in f.read_text().splitlines()] if f.exists() else []
            selected = load(d / 'selected.json') if (d / 'selected.json').exists() else None
            pool = load(d / 'engine/candidates.json') if (d / 'engine/candidates.json').exists() else []
            pool_index = {r['answer']: i for i, r in enumerate(pool)}
            groups = {}
            delivered = set()
            for event in events:
                prompt = event['candidate']['answer']
                group = groups.setdefault(prompt, {'dev': {}, 'train': {}, 'first_time': event['timestamp']})
                for ex in event['examples']:
                    native = by_input[prompt, ex['id']]
                    assert native['answer'] == ex['answer']
                    previous = group[event['partition']].get(ex['id'])
                    if previous: assert previous['rating'] == ex['rating']
                    group[event['partition']][ex['id']] = ex
                    if 'additional_diagnostic' in ex:
                        assert event['partition'] == 'train' and ex['diagnostic_source'] == arm
                        delivered.add(native['id'])
            for attempt, (prompt, group) in enumerate(groups.items()):
                dev = list(group['dev'].values())
                assert set(group['dev']) <= expected
                complete = set(group['dev']) == expected
                fpq = [x['rating'] for x in dev if x['label'] == 1]
                nfp = [x['rating'] for x in dev if x['label'] == 0]
                row = dict(dataset=dataset, arm=arm, candidate_key=key(prompt), attempt_order=attempt,
                    engine_pool_index=pool_index.get(prompt), starting_prompt=prompt == plan['starting_prompt'],
                    selected=bool(selected and selected['candidate']['answer'] == prompt),
                    dev_complete=complete, dev_n=len(dev), fpq_n=len(fpq), nfp_n=len(nfp),
                    fpq_mean=statistics.mean(fpq) if complete else None,
                    nfp_mean=statistics.mean(nfp) if complete else None,
                    balanced_normalized=statistics.mean(fpq + nfp) / 5 if complete else None,
                    unique_train_questions=len(group['train']), recorded_pair_evaluations=sum(e['candidate']['answer'] == prompt for e in events),
                    prompt=prompt)
                candidate_rows.append(row)
                for ex in dev:
                    dev_rows.append(dict(dataset=dataset, arm=arm, candidate_key=key(prompt),
                        selected=row['selected'], seed=row['starting_prompt'], row_id=ex['id'],
                        label=ex['label'], rating=ex['rating'], question=ex['question'],
                        answer=ex['answer'], grading_reason=ex['reason']))
                if row['selected']:
                    assert complete and len(fpq) == len(nfp) == 4
                    selection_rows.append(row)
            if arm in ['nla', 'ao']:
                for f in (d / 'extras').glob('*.json'):
                    native = natives[f.stem]
                    assert native['dataset'] == dataset
                    seed = audits[native['row_id']]
                    prompt = native['candidate_prompt']
                    extra = load(f)
                    assert native['row_id'] in plan['splits'][dataset]['train_ids']
                    initial_reader = seed[arm]['explanation']
                    trajectory.append(dict(id=f'{dataset}/{arm}/{native["id"]}', dataset=dataset,
                        reader=arm, row_id=native['row_id'], label=seed['label'], question=seed['question'],
                        candidate_key=key(prompt), engine_pool_index=pool_index.get(prompt),
                        starting_prompt=prompt == plan['starting_prompt'], supplied_in_completed_gepa_evaluation=native['id'] in delivered,
                        source_prompt=prompt, seed_reader_explanation=initial_reader,
                        current_reader_explanation=extra['supplied_diagnostic'], raw_reader_explanation=extra['explanation'],
                        seed_answer=seed['native']['answer'], current_answer=native['answer'],
                        seed_rating=seed['judgment']['score'], current_rating=load(PILOT / 'judgments' / f'{native["id"]}.json')['score'],
                        seed_audit=seed['audit'], source_native_id=native['id'],
                        source_gpu_indices=native.get('source_gpu_indices', '0,1 (original run)'),
                        diagnostic_clipped=extra['clipped']))
    assert len(selection_rows) == 6
    write('candidate_scores.json', candidate_rows)
    write('selected_dev_scores.json', selection_rows)
    write('dev_question_scores.json', dev_rows)
    write('reader_trajectories.json', trajectory)
    matches = []
    for arm in ['nla', 'ao']:
        current = {r['source_native_id']: r for r in trajectory if r['reader'] == arm}
        for dataset in ['cancer_myth', 'crepe']:
            d = PILOT / 'optimization' / dataset / 'text/extras'
            for f in d.glob('*.json'):
                if f.stem in current:
                    r = current[f.stem]; text = load(f)
                    matches.append(dict(reader=arm, dataset=dataset, row_id=r['row_id'], source_native_id=f.stem,
                        candidate_key=r['candidate_key'], starting_prompt=r['starting_prompt'],
                        question=r['question'], answer=r['current_answer'], reader_explanation=r['current_reader_explanation'],
                        text_diagnostic=text['supplied_diagnostic'],
                        limit='Text diagnostic sees grading reason and label; unequal information access. This is a GEPA-feedback comparison, not a clean activation-versus-answer-only control.'))
    write('matched_text_diagnostics.json', matches)
    initial = {}
    for dataset in ['cancer_myth', 'crepe', 'all']:
        data = [r for r in audits.values() if dataset == 'all' or r['dataset'] == dataset]
        initial[dataset] = dict(n=len(data), fpq_n=sum(r['label'] == 1 for r in data),
            annotation_issue_flags=sum(r['audit']['annotation_issue'] for r in data),
            behaviors=dict(Counter(r['audit']['actual_behavior'] for r in data)),
            readers={k: {field: sum(r['audit'][k][field] for r in data)
                for field in ['topic_match', 'specific_target_error_identified', 'adds_diagnostic_information_beyond_text']}
                for k in ['nla', 'ao']})
    issues = [dict(row_id=r['row_id'], dataset=r['dataset'], label=r['label'], question=r['question'],
                   review_reason=r['audit']['annotation_issue_reason'])
              for r in audits.values() if r['audit']['annotation_issue']]
    write('annotation_review_queue.json', issues)
    scalar = [r for r in trajectory if not r['starting_prompt']]
    manifest = dict(snapshot_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        selected_conditions=len(selection_rows), expected_conditions=8,
        final_haiku_scores=len(list((PILOT / 'heldout_haiku').glob('*.json'))),
        native_answers=len(natives), native_eos=sum(r['completed'] for r in natives.values()),
        initial_audit=initial, reader_records=len(trajectory), optimized_reader_records=len(scalar),
        optimized_reader_by_kind=dict(Counter(r['reader'] for r in scalar)),
        recorded_feedback_by_kind=dict(Counter(r['reader'] for r in scalar if r['supplied_in_completed_gepa_evaluation'])),
        matched_text_diagnostics=len(matches),
        fixed_split_checks='All selected scores use exactly the same8 candidate-dev IDs per dataset; all reader trajectories are train only.',
        limitations=['GPT-selected candidate-dev scores are selection data, not heldout performance.',
            'FPQ4 and NFP4 per dataset: estimates are highly variable; no main-table claim.',
            'AO is unfinished; do not compare its partial best candidate as a completed arm.',
            'Candidate search branches and rejections are not a monotonic time series.',
            'Reader audit opinions are not human labels or causal evidence of source beliefs.',
            'Candidate-dependent sampled training items do not form a fixed longitudinal panel.',
            'Annotation flags require review; labels/scores were not changed or selectively removed.'])
    write('snapshot_summary.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['selected_conditions','native_answers','native_eos','reader_records','optimized_reader_records','optimized_reader_by_kind','matched_text_diagnostics']}, indent=2))


def report():
    snapshot = load(OUT / 'snapshot_summary.json')
    selected = load(OUT / 'selected_dev_scores.json')
    candidates = load(OUT / 'candidate_scores.json')
    audits = [load(f) for f in (OUT / 'trajectory_audits').glob('*.json')]
    audits = [r for r in audits if r.get('valid')]
    rows = []
    for dataset in ['cancer_myth', 'crepe']:
        seed = next(r for r in candidates if r['dataset'] == dataset and r['arm'] == 'gepa' and r['starting_prompt'])
        for r in [seed] + [x for x in selected if x['dataset'] == dataset]:
            rows.append(f'<tr><td>{dataset}</td><td>{"Plain seed" if r is seed else r["arm"]}</td><td>{r["fpq_mean"]:.2f}</td><td>{r["nfp_mean"]:.2f}</td><td>{r["balanced_normalized"]:.3f}</td></tr>')
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>NLA·AO 파일럿 CPU 분석</title>',
        '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;color:#203047;font:16px/1.7 sans-serif}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;background:#f3f6f9;padding:16px}details{margin:18px 0}img{width:100%}.note{background:#fff4db;padding:16px}h2{margin-top:40px}</style>',
        '<h1>NLA·AO 피드백 GEPA: 중단 시점 분석</h1>',
        f'<p>스냅샷: {html.escape(snapshot["snapshot_kst"])}. GPU 장애로 생성 중단. 최적화6/8 완료, Haiku 보류 문항 평가0건.</p>',
        '<p class="note">아래 점수는 후보 선택에 사용한 GPT dev 점수입니다. 데이터셋별 FPQ4 + NFP4이며 최종 성능·Table 결과가 아닙니다. AO는 미완료입니다.</p>',
        '<h2>1. 완료된 조건의 dev 점수</h2><table><tr><th>데이터셋</th><th>조건</th><th>FPQ Well 평균 /5</th><th>NFP Well 평균 /5</th><th>균형 점수 /1</th></tr>',
        ''.join(rows), '</table><p>Cancer에서 기본 GEPA는 FPQ 평균이 가장 높고, NLA와 텍스트 대조는 같은 FPQ/NFP 평균입니다. CREPE에서 NLA의 선택 점수는 seed보다 약간 높지만4개 FPQ 문항에서 개선과 악화가 섞여 있습니다.</p>',
        '<h2>2. 초기 설명 분석</h2>',
        '<p>100문항(50FPQ·50NFP)의 기존 GPT 분석: NLA 주제 일치77/100, AO97/100. 구체적 목표 오류 식별은 FPQ에서 두 reader 모두0/50. 기존 답변·채점 이유를 넘어서는 추가 정보는 둘 다0/100. 이 결과는 고정한 중립 reader 설정과 GPT의 탐색 판정에 한정됩니다.</p>',
        '<p>GPT가 주석·질문 불일치 가능성을 표시한 문항은46/100입니다. 오류 확정이나 제외 기준이 아니라 별도 확인 목록이며, 원래 점수는 수정하지 않았습니다.</p>',
        '<h2>3. 최적화 중 설명 변화</h2>',
        f'<p>저장된 reader 피드백{snapshot["reader_records"]}건 중 seed가 아닌 후보의 설명은{snapshot["optimized_reader_records"]}건입니다. 같은 질문의 seed 설명·답변과 대응시켰습니다. 후보별로 선택된 학습 문항이 달라 전체 문항에서 설명이 점차 좋아졌다고 일반화할 수 없습니다.</p>',
        f'<p>추가 GPT 대조 분석 완료: {len(audits)}건. 질문·정답 주석·답변·설명 원문에 근거한 판정이며 reader 이름과 후보 순서는 판정 입력에서 숨겼습니다.</p>']
    for kind in ['nla', 'ao']:
        data = [r for r in audits if r['reader'] == kind]
        if data:
            counts = Counter(r['audit']['mismatch'] for r in data)
            new = sum(r['audit']['new_information_beyond_current_answer'] for r in data)
            parts.append(f'<p><b>{kind.upper()}</b>: 분석{len(data)}건, 실제 답변에 없는 추가 정보 판정{new}건. 분류: {html.escape(str(dict(counts)))}</p>')
    summary_path = OUT / 'trajectory_audit_summary.json'
    if summary_path.exists():
        audit_summary = load(summary_path)
        parts += ['<table><tr><th>Reader</th><th>문항 유형</th><th>설명 기록 / 고유 문항</th><th>구체 오류 식별: 초기→현재</th><th>현재 교정 의향</th><th>현재 NFP 오반박</th><th>답변에 없는 추가 단서</th></tr>']
        for r in audit_summary['results']:
            if r['dataset'] != 'all': continue
            parts += [f'<tr><td>{r["reader"].upper()}</td><td>{"FPQ" if r["label"] else "NFP"}</td><td>{r["interpretation_records"]} / {r["unique_questions"]}</td><td>{r["seed_specific_target_recognitions"]}→{r["current_specific_target_recognitions"]}</td><td>{r["current_correction_intent"]}</td><td>{r["current_spurious_nfp_correction"]}</td><td>{r["added_information_beyond_answer"]}</td></tr>']
        parts += ['</table><p>교정 의향에는 일반적인 전제 검토 표현도 포함됩니다. 구체 오류 식별과 동일한 지표가 아닙니다. 설명 기록은 동일 문항의 여러 후보를 포함하므로 독립 문항 수와 구분합니다.</p>',
            '<p>최초100문항 분석과 이번 전후 대조는 판정 입력·지침이 다릅니다. 이번 대조에서는 NLA 초기 설명에도 구체 오류 식별1건이 판정됐고, 현재 설명에서도1건입니다. 출현1건과 소실1건이 함께 있어 일관된 상승은 확인되지 않았습니다. 자동 판정 차이는 별도 원문 검토 대상으로 남깁니다.</p>']
    parts += ['<h2>해석에 필요한 구분</h2>',
        '<ul><li>초기에는 없던 교정 단서가 최적화 중 나타날 수 있습니다. 폐암 통증 문항에서 AO는 모든 환자의 심한 통증이라는 가정을 오해라고 설명했고, NLA도 통증의 불가피성에 대한 잘못된 가정을 언급했습니다. 두 사례 모두 실제 답변도 그 내용을 교정했으므로 텍스트에 없는 추가 정보는 아닙니다.</li>',
        '<li>NLA는 같은 사례에서 질문에 없는 말기 질환 맥락도 도입했습니다. 특정 단서의 정확성과 전체 설명의 정확성은 구분해야 합니다.</li>',
        '<li>NFP 의족 선택 문항에서 AO는 의족 선택 주제를 요약했지만 실제 답변은 절단의 불가피성을 반박했습니다. 데이터셋 NFP 채점에서는1점이었지만, 주제 요약 자체는 내부 정상성 판단의 증거가 아닙니다.</li>',
        '<li>텍스트 진단과 reader가 정확히 같은 답변에 대응하는 기록은16건(8개 seed 문항×두 reader)입니다. 텍스트 진단은 이미 라벨과 GPT 채점 이유를 받으므로 활성과 답변 텍스트의 순수 정보량 비교가 아닙니다.</li>',
        '<li>주석 충돌을 구분해야 합니다. fpq_44의 presuppositions는 수술·회복을 언급하지만 실제 질문과 reference_answer는 통증 불가피성을 다룹니다. 주석 불일치 표시가 곧 정답 reference가 틀렸다는 뜻은 아닙니다.</li></ul>',
        '<p><b>다음 검증:</b> GPU 복구 후 남은 AO 최적화와 별도 Haiku 평가를 완료합니다. 이미 저장한 동일 후보의 NLA·AO용 활성을 양쪽 reader로 해석하면, 서로 다른 후보를 비교하는 혼동 없이 설명 차이를 확인할 수 있습니다. 실제 답변에 없는 신호가 나오면 문항별 설명 셔플과 답변을 숨긴 행동 예측으로 검증합니다.</p>']
    if (OUT / 'selected_dev_comparison.png').exists():
        parts += ['<h2>4. 점수 비교 그림</h2><img src="selected_dev_comparison.png" alt="Selected GPT dev scores">']
    parts += ['<h2>사례별 원문과 판정</h2>']
    all_cases = load(OUT / 'reader_trajectories.json')
    indexed = {r['id']: r for r in audits}
    for r in all_cases:
        if r['starting_prompt']: continue
        a = indexed.get(r['id'])
        parts += [f'<details><summary>{html.escape(r["dataset"])} · {r["reader"].upper()} · {html.escape(r["row_id"])} · 후보{r["candidate_key"]} · Well {r["seed_rating"]}→{r["current_rating"]}</summary>',
            f'<p><b>질문:</b> {html.escape(r["question"])}</p>',
            '<b>초기 reader 설명</b><pre>' + html.escape(r['seed_reader_explanation']) + '</pre>',
            '<b>현재 reader 설명</b><pre>' + html.escape(r['current_reader_explanation']) + '</pre>',
            '<b>초기 답변</b><pre>' + html.escape(r['seed_answer']) + '</pre>',
            '<b>현재 답변</b><pre>' + html.escape(r['current_answer']) + '</pre>']
        if a: parts += ['<b>추가 GPT 판정</b><pre>' + html.escape(json.dumps(a['audit'],ensure_ascii=False,indent=2)) + '</pre>']
        parts += ['</details>']
    parts += ['</html>']
    (OUT / 'report.html').write_text('\n'.join(parts))
    print(OUT / 'report.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not args.report_only: export()
    report()
