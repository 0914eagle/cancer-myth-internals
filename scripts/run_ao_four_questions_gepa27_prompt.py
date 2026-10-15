"""Transfer the frozen 27B text-GEPA prompt to Qwen3-8B and repeat CREPE50 AO."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import types

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'external/paper_baselines_b81ace2'
OUT = ROOT / 'results/ao_four_questions_gepa27_text_prompt_crepe_20261014'
PLAIN = ROOT / 'results/ao_four_questions_crepe_20261014'
SELECTION = EXPERIMENT / 'results/nla_ao_parallel_20261012/gepa_ao_semantic_feedback_cancer_myth_luna_20261014/optimization/text/selected.json'
PYTHON = '/data3/heejae/uv/fpqa_ao_20261012/bin/python'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def setup():
    native = module('qwen8_native_helpers', EXPERIMENT / 'scripts/qwen3_8b_ao_crepe.py')
    rows, baseline_protocol = native.rows_and_protocol()
    selection = json.loads(SELECTION.read_text())
    prompt = selection['candidate']['answer']
    settings = dict(baseline_protocol, protocol='qwen3_8b_transferred_gepa27_text_prompt_v1',
        source_prompt=prompt, actual_answer_audit_model=None,
        actual_answer_audit_prompt=None)
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'selected_prompt_provenance.json', dict(source_file=str(SELECTION),
        source_file_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
        selected_candidate=selection, optimized_source_model='google/gemma-3-27b-it',
        optimized_dataset='cancer_myth', applied_source_model='Qwen/Qwen3-8B',
        applied_dataset='crepe', prompt_modified=False,
        scope='Frozen prompt transfer, not new GEPA optimization or a 27B rerun.'))
    (OUT / 'source_prompt.txt').write_text(prompt + '\n')
    native.OUT = OUT / 'source_runtime'
    native.rows_and_protocol = lambda: (rows, settings)
    save(native.OUT / 'protocol.json', settings)
    return native, rows, settings


def native_stage():
    native, _, _ = setup()
    native.native_worker()


def stance(text):
    found = [s for s in ['Accept', 'Doubt', 'Reject', 'Unclear']
             if re.search(r'\b' + s + r'\b', text, re.I)]
    return found[0] if len(found) == 1 else 'Unparsed'


def compare():
    old = {r['question_id']: r for r in
           map(json.loads, (PLAIN / 'crepe50.jsonl').read_text().splitlines())}
    new = list(map(json.loads, (OUT / 'crepe50.jsonl').read_text().splitlines()))
    assert len(old) == len(new) == 50 and {r['question_id'] for r in new} == set(old)
    combined = []
    counts = {condition: Counter() for condition in ['plain', 'gepa_prompt']}
    match = {condition: 0 for condition in counts}
    transitions = Counter()
    for r in new:
        before = old[r['question_id']]
        assert r['question'] == before['question'] and r['question_type'] == before['question_type']
        assert r['source_token_ids'] == before['source_token_ids']
        assert r['source_native_id'] != before['source_native_id']
        record = dict(question_id=r['question_id'], question_type=r['question_type'],
            question=r['question'], plain_answer=before['actual_answer'],
            gepa_prompt_answer=r['actual_answer'])
        tags = {}
        for condition, item in [('plain', before), ('gepa_prompt', r)]:
            tag = stance(item['stance'])
            tags[condition] = tag
            counts[condition][(r['question_type'], tag)] += 1
            prediction = {'Accept': 'NFP', 'Doubt': 'FPQ', 'Reject': 'FPQ'}.get(tag)
            match[condition] += prediction == r['question_type']
            record[condition + '_stance_tag'] = tag
            record[condition + '_gold_match'] = prediction == r['question_type']
            for field in ['target_premise', 'stance', 'likely_behavior', 'correction_content']:
                record[condition + '_' + field] = item[field]
        transitions[(tags['plain'], tags['gepa_prompt'])] += 1
        combined.append(record)
    with (OUT / 'plain_vs_gepa_prompt.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    summary = dict(questions=50, FPQ=25, NFP=25,
        source_model='Qwen3-8B', borrowed_prompt_model='Gemma3-27B', borrowed_prompt_dataset='Cancer-Myth',
        controlled='Same model revisions, questions, queries, greedy BF16 generation, source GPU4/batch16, reader GPU0/batch64, L18/question-end20-or-fewer prefill states.',
        results={condition: dict(stance_counts={label: dict((tag, counts[condition][(label, tag)])
                     for tag in ['Accept', 'Doubt', 'Reject', 'Unclear', 'Unparsed'])
                     for label in ['FPQ', 'NFP']},
                     gold_matches=match[condition], agreement_all_items=match[condition]/50)
                 for condition in counts},
        stance_transitions=[dict(plain=a, gepa_prompt=b, n=n) for (a, b), n in transitions.items()],
        answer_changed=sum(r['plain_answer'] != r['gepa_prompt_answer'] for r in combined),
        external_model_calls=0,
        limits=['Exploratory reused train50; no independent test claim.',
                'Stance/gold match is an operational proxy, not validated source-belief accuracy.',
                'Fresh answers must be judged independently; old Luna/Well labels are not transferred.',
                'No new GEPA optimization, reader fine-tuning, Well judging, or translation calls.'])
    save(OUT / 'comparison_summary.json', summary)
    return summary


def run():
    native, rows, settings = setup()
    ids = {r['id']: native.request(r, settings['source_prompt'], settings) for r in rows}
    runtime = native.OUT
    stop = runtime / 'native/STOP'
    stop.unlink(missing_ok=True)
    log = (OUT / 'native_worker.log').open('a')
    worker = subprocess.Popen([PYTHON, '-u', str(Path(__file__).resolve()), '--stage', 'native'],
        cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HOME': '/data3/heejae/hf_cache'},
        stdout=log, stderr=log)
    started = time.monotonic()
    previous = -1
    try:
        while True:
            complete = sum((runtime / 'native/results' / (ident + '.json')).exists() for ident in ids.values())
            if complete != previous:
                print(f'New prompt native answers: {complete}/50', flush=True)
                previous = complete
            save(OUT / 'controller_status.json', dict(state='running', stage='native_generation',
                complete=complete, expected=50, source_gpu=4, worker_pid=worker.pid))
            if complete == 50:
                break
            if worker.poll() is not None:
                raise RuntimeError('Source worker exited before completing50; inspect native_worker.log')
            if time.monotonic() - started > 900:
                raise TimeoutError('Source generation exceeded900seconds')
            time.sleep(1)
    finally:
        save(stop, dict(reason='Owned50-question source generation finished; normal shutdown'))
        if worker.poll() is None:
            worker.wait(timeout=120)
        log.close()
    records = []
    for row in rows:
        record = json.loads((runtime / 'native/results' / (ids[row['id']] + '.json')).read_text())
        assert record['completed'] and record['capture_mode'] == 'actual_native_generation_prefill'
        assert record['candidate_prompt'] == settings['source_prompt']
        records.append(dict(question_id=row['id'], question_type='FPQ' if row['label'] else 'NFP',
            question=row['question'], actual_answer=record['answer'], answer_completed=record['completed'],
            native_record=record, behavior_text='', yesno_raw=''))
    inputs = OUT / 'source_answers'
    inputs.mkdir(exist_ok=True)
    (inputs / 'crepe50.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    save(inputs / 'protocol.json', settings)
    save(OUT / 'source_generation_summary.json', dict(questions=50,
        completed=sum(r['answer_completed'] for r in records),
        output_tokens=sum(r['native_record']['output_tokens'] for r in records),
        seconds=time.monotonic()-started, capture_verification=json.loads(
            (runtime / 'native/capture_verification.json').read_text())))
    print('Source GPU4 released; starting four-query AO on GPU0', flush=True)
    reader = module('four_question_reader', ROOT / 'scripts/run_ao_four_questions_crepe.py')
    reader.SOURCE = inputs
    reader.OUT = OUT
    save(OUT / 'controller_status.json', dict(state='running', stage='ao_four_queries', expected=200, reader_gpu=0))
    reader.main(types.SimpleNamespace(gpu=0, batch_size=64))
    summary = compare()
    save(OUT / 'controller_status.json', dict(state='complete', source_answers=50,
        ao_descriptions=200, output_directory=str(OUT)))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['run', 'native'], default='run')
    args = parser.parse_args()
    try:
        native_stage() if args.stage == 'native' else run()
    except Exception as error:
        save(OUT / ('native_failure.json' if args.stage == 'native' else 'controller_status.json'),
             dict(state='failed', error=repr(error)))
        raise
