"""Extend the frozen 234-pair descriptive coding to every primary paired item.

No sampling by class or Well score. Reuse only byte-matched, validated old
coding; keep the identical blinded prompt/schema for newly submitted pairs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from scripts import audit_ungated_failure_census as base

ROOT = base.ROOT
OUT = ROOT / 'results/audits/ungated_full_cohort_20260929_v1'
DOC = ROOT / 'docs/reviews/detection_plain_link_2026-09-28/ungated_full_cohort'
OLD_OUT, OLD_DOC = base.OUT, base.DOC


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    rows = [json.loads(s) for s in base.SOURCE.read_text().splitlines()]
    rows = [r for r in rows if base.PRIMARY[r['model']] == r['method']]
    assert len({(r['model'], r['id']) for r in rows}) == len(rows)
    # Validate the completed old audit's provenance before reuse.
    provenance = json.loads((OLD_DOC / 'completed_audit_sources.json').read_text())
    for path, expected in provenance['sources'].items():
        assert sha(Path(path)) == expected, path
    old_plan = json.loads((OLD_OUT / 'plan.json').read_text())
    assert (old_plan['prompt'], old_plan['system'], old_plan['model']) == (base.PROMPT, base.SYSTEM, base.MODEL)
    old = {(r['model'], r['id']): r for r in json.loads((OLD_DOC / 'annotations.json').read_text())}
    rows.sort(key=lambda r: base.digest('full-cohort-v1:' + r['model'] + ':' + r['id']))
    manifest, inputs, reused, pending = [], [], [], []
    for i, r in enumerate(rows):
        key = f'full_{i:04d}'
        flip = int(base.digest('slots:' + r['model'] + r['id'])[:8], 16) % 2
        slots = dict(A='answer' if flip else 'plain', B='plain' if flip else 'answer')
        inp = dict(key=key, question=r['question'], answers=[dict(slot=s, text=r[v]) for s, v in slots.items()])
        group = f"{int(r['plain_score'] >= 4)}->{int(r['answer_score'] >= 4)}"
        row = dict(**r, key=key, slots=slots, score_transition=group)
        inputs.append(inp)
        manifest.append(row)
        previous = old.get((r['model'], r['id']))
        if previous:
            for field in ['question', 'plain', 'answer', 'method', 'plain_job', 'answer_job', 'slots']:
                assert previous[field] == row[field], (key, field)
            audit = deepcopy(previous['audit'])
            audit['key'] = key
            valid, repairs = base.validate([audit], [inp])
            assert not repairs
            reused.append(dict(key=key, old_key=previous['key'], audit=valid[0]))
        else:
            pending.append(inp)
    assert len(reused) == len(old) == 234
    canonical = {r['id'] for r in rows if r['model'] == 'gemma'}
    counts = {m: dict(Counter(r['set'] for r in rows if r['model'] == m)) for m in base.PRIMARY}
    missing = {m: sorted(canonical - {r['id'] for r in rows if r['model'] == m}) for m in base.PRIMARY}
    plan = dict(model=base.MODEL, prompt=base.PROMPT, system=base.SYSTEM,
                source=str(base.SOURCE), source_sha256=sha(base.SOURCE),
                primary_conditions=base.PRIMARY, counts=counts, missing=missing,
                total_pairs=len(rows), reused_pairs=len(reused), new_pairs=len(pending),
                old_annotations_sha256=sha(OLD_DOC/'annotations.json'), batch_size=3,
                scope='All available primary-condition paired FPQ/NFP items, including every score transition. Not all experimental conditions.',
                blinded=['model', 'condition', 'label', 'scores', 'references', 'gates'],
                limits='Descriptive AI coding, not medical adjudication or causal evidence. Same question repeats across model settings. Prompts differ.')
    for name, value in [('plan.json', plan), ('manifest.json', manifest), ('inputs.json', inputs),
                        ('reused.json', reused), ('pending_inputs.json', pending)]:
        base.write(OUT/name, value, frozen=True)
    return plan, manifest, inputs, reused, pending


def collect(plan, manifest, inputs, reused, pending, state):
    results = {r['key']: r['audit'] for r in reused}
    for i in range(0, len(pending), 3):
        path = OUT/'batches'/f'{i//3:03d}'/'result.json'
        if path.exists():
            valid, repairs = base.validate(json.loads(path.read_text())['items'], pending[i:i+3])
            assert not repairs
            results.update({x['key']: x for x in valid})
    strata = defaultdict(lambda: dict(expected=0, coded=0, stance_transitions=Counter(),
                                     response_transitions=Counter(), plain_flags=Counter(), answer_flags=Counter()))
    annotations = []
    oldkeys = {r['key'] for r in reused}
    for row in manifest:
        s = strata[(row['model'], row['set'], row['score_transition'])]
        s['expected'] += 1
        audit = results.get(row['key'])
        if not audit:
            continue
        coded = {row['slots'][a['slot']]: a for a in audit['answers']}
        annotations.append(dict(**row, audit=audit, coded=coded,
                                coding_source='reused' if row['key'] in oldkeys else 'new'))
        s['coded'] += 1
        s['stance_transitions'][coded['plain']['stance']+' -> '+coded['answer']['stance']] += 1
        s['response_transitions'][coded['plain']['response']+' -> '+coded['answer']['response']] += 1
        for side in ['plain', 'answer']:
            s[side+'_flags'].update(f['tag'] for f in coded[side]['flags'])
    complete = len(results) == len(inputs)
    summary = dict(state='complete' if complete else state, complete=complete, valid=len(results),
                   expected=len(inputs), reused=len(reused), new_valid=len(results)-len(reused),
                   remaining=len(inputs)-len(results), pid=os.getpid(), at=time.time(),
                   strata=[dict(model=k[0], dataset=k[1], score_transition=k[2], **v) for k,v in sorted(strata.items())])
    base.write(OUT/'status.json', summary)
    base.write(DOC/'summary.json', summary)
    base.write(DOC/'annotations.json', annotations)
    lines = ['# 네 모델의 전체 문항 행동 분류', '',
             '**일부 실패 사례를 고르는 대신, 아래 고정된 비교 조건의 모든 FPQ/NFP 문항을 포함한다.**', '',
             '| 모델 조건 | 비교 답변 | FPQ 쌍 | NFP 쌍 | 분류 완료 |', '|---|---|---:|---:|---:|']
    for model, method in base.PRIMARY.items():
        n = sum(a['model'] == model for a in annotations)
        c = plan['counts'][model]
        lines.append(f"| {model} | Plain → {method} | {c.get('fpq',0)} | {c.get('nfp',0)} | {n} |")
    lines += ['', f"전체 **{len(inputs):,}쌍**, 기존 분류 재사용 **{len(reused)}쌍**, 신규 완료 **{summary['new_valid']:,}쌍**, 남음 **{summary['remaining']:,}쌍**. 상태: `{summary['state']}`.", '',
              '## 분석 범위', '',
              '- Qwen3.8 thinking OFF/ON을 나누어 네 모델·다섯 설정을 집계한다. 동일 문항의 반복 측정이므로 모든 쌍을 독립 질문처럼 합쳐 통계 검정하지 않는다.',
              '- Qwen2.5는 fpq_389의 유효한 답변 쌍이 없어 제외했다. 나머지 설정은 FPQ 583 + NFP 149 전체다.',
              '- 모든 점수 전이(<4→<4, <4→≥4, ≥4→<4, ≥4→≥4)를 포함한다. 성공 사례를 더 이상 다섯 개씩 표본 추출하지 않는다.',
              '- 기존 234쌍은 입력·프롬프트·출력 원본 해시 및 인용 검증 후 재사용한다. 남은 문항도 동일한 Claude Sonnet 분류 지시와 검증기를 사용한다.',
              '- 분류기에 제공하는 것은 질문과 전체 답변 두 개뿐이다. 모델명·조건명·정답 라벨·주석·Well 점수·gate 결과를 숨기고 A/B 순서를 섞는다.',
              '- 반박/단서/개입 없음, 요청에 답하는 정도, 의미 범위 강화·개인 정황 재검토·존재 문의의 단정화·불확실성을 거짓으로 처리하는 표현을 근거 인용과 함께 분류한다.',
              '- 실패 집단뿐 아니라 성공 집단에서도 같은 행동이 얼마나 나타나는지 모델별로 비교한다. 미완료 집계는 처리 순서에 따른 편향이 있으므로 전체 빈도로 보고하지 않는다.',
              '- 전수 AI 분류도 의미 판정의 정확성 인증이나 내부 원인 증명은 아니다. 인용 일치는 기계적으로 검증하되 의학적 정답 라벨과 기존 Well 점수는 수정하지 않는다.',
              '- 이번 전수는 위의 주 비교 조건에 대한 것이다. 저장된 무조건 교정·일반 CoT 등 나머지 비교 조건 전체를 본문 분류했다는 뜻은 아니다.', '',
              '## 재현 및 상태', '',
              '`python3 -m scripts.audit_ungated_full_cohort prepare`', '',
              '`python3 -m scripts.audit_ungated_full_cohort run --workers 8`', '',
              '`python3 -m scripts.audit_ungated_full_cohort collect`', '',
              '고정 입력·계획·신규 배치·재시도 원본·진행 상태는 `results/audits/ungated_full_cohort_20260929_v1/`에 저장한다.',
              '전체 배치가 완료되면 모델 × FPQ/NFP × 점수 전이별 분류 집계와 각 문항의 근거를 `summary.json`·`annotations.json`에 저장한다.', '']
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC/'README.md').write_text('\n'.join(lines))
    return summary


def run_batch(index, batch):
    # Each index has only one worker. Preserve failed attempts before retrying.
    for attempt in range(3):
        folder = OUT/'batches'/f'{index:03d}'
        if folder.exists() and not (folder/'result.json').exists():
            archive = OUT/'retry_archive'/f'{index:03d}'/str(time.time_ns())
            archive.parent.mkdir(parents=True, exist_ok=True)
            folder.rename(archive)
        result = base.job(index, batch)
        if result[1] in ['complete', 'cached']:
            return result
        failure = json.loads((folder/'failure.json').read_text())
        error = failure.get('error', '').lower()
        if any(term in error for term in ['rate limit', 'rate_limit', 'usage limit', 'hit your limit', 'oauth', 'authentication', 'unauthorized', 'credit balance']):
            return index, 'service_blocked'
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['prepare', 'run', 'collect'])
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    assert 1 <= args.workers <= 8
    data = prepare()
    plan, manifest, inputs, reused, pending = data
    base.OUT = OUT
    if args.action != 'run':
        oldstatus = OUT/'status.json'
        state = json.loads(oldstatus.read_text())['state'] if oldstatus.exists() else 'prepared_not_running'
        summary = collect(*data, state=state)
        print(json.dumps({k: summary[k] for k in ['state','valid','expected','remaining']}))
        return
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect(*data, state='running')
        first = run_batch(0, pending[:3])
        print(first, flush=True)
        if first[1] not in ['complete', 'cached']:
            collect(*data, state='stopped_first_batch_failure')
            return
        jobs = iter((i//3, pending[i:i+3]) for i in range(3, len(pending), 3))
        stopped = False
        last_update = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            active = set()
            for _ in range(args.workers):
                job = next(jobs, None)
                if job: active.add(pool.submit(run_batch, *job))
            while active:
                done, active = wait(active, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    print(result, flush=True)
                    if result[1] == 'service_blocked': stopped = True
                for _ in done:
                    job = None if stopped else next(jobs, None)
                    if job: active.add(pool.submit(run_batch, *job))
                if time.time()-last_update >= 30 or stopped:
                    collect(*data, state='service_blocked_draining' if stopped else 'running')
                    last_update = time.time()
        summary = collect(*data, state='service_blocked' if stopped else 'incomplete_requires_inspection')
        print(json.dumps({k: summary[k] for k in ['state','valid','expected','remaining']}), flush=True)


if __name__ == '__main__':
    main()
