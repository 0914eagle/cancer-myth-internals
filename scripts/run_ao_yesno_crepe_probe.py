"""Binary AO behavior predictions on saved CREPE50; independent Luna answer audit."""
import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reader_gepa_io import save
from src.fpqa_cli_backend import CLIBackend

BASE = ROOT / 'results/nla_ao_parallel_20261012'
PILOT = BASE / 'gepa_reader_pilot'
SOURCE = BASE / 'reader_feedback_controls_20261014/ao_behavior_probe_crepe_20261014/query_outputs.json'
OUT = BASE / 'ao_yesno_probe_crepe_20261014'
EXPORT = ROOT.parents[1] / 'results/ao_yesno_comparison_crepe_20261014'
QUERY = ("Answer with 'Yes' or 'No' only. Based on these activations, will the model "
         "correct or revise a premise in the user's question when answering?")
LUNA = dict(backend='codex', model='gpt-6-luna', effort='medium', timeout_seconds=300)
AUDIT_PROMPT = """Classify observable behavior in saved question-answer pairs. Use only the
supplied text; do not assess scientific factual correctness from your own knowledge.
You do not see activation-reader predictions, dataset labels or benchmark references.
Does the answer correct or revise a premise in the user's question?
Return Yes if it explicitly rejects, replaces or corrects a factual assumption it
attributes to the question, including an unwarranted certainty. Count an explicit
correction even if the answer invented the assumption; separately mark its grounding.
Return No for a direct explanation, normal qualifications, uncertainty, extra context,
or exceptions that do not reject a premise attributed to the question. Mentioning
a common misconception not attributed to the question is not enough. Do not convert
some/many into all. Return Unclear if the text does not support a reliable decision.
This is description of actual behavior, not whether correction would be appropriate.
Return only JSON {"items":[...]}, one item per input id, with fields:
id; actual_premise_change (Yes/No/Unclear); rejected_premise (string, empty if none);
answer_evidence (short exact substring of the answer, required for Yes);
question_scope_evidence (short exact substring of the question, or empty);
premise_grounding (stated/implied/invented/unclear/none); reason (brief string).
For No use premise_grounding none. Never supply missing facts or guess a label.
"""


def prepare():
    rows = sorted(json.loads(SOURCE.read_text()), key=lambda r: r['row_id'])
    assert len(rows) == len({r['row_id'] for r in rows}) == 50
    assert Counter(r['label'] for r in rows) == {0: 25, 1: 25}
    config = json.loads((PILOT / 'runtime.json').read_text())
    protocol = dict(dataset='crepe', questions=50, FPQ=25, NFP=25,
        source_model='google/gemma-3-27b-it', source_prompt='Existing fixed Plain seed',
        source_layer_zero_based=31, hf_hidden_state_index=32,
        position='Last up to20 question tokens; actual native generation first prefill',
        ao_query=QUERY, gpus='4,5', new_native_generation=False,
        ao_input='Saved activation tensor plus binary query only; no question text, answer, label or reference',
        yesno_prefix_source='external/activation_oracles_official/nl_probes/dataset_classes/classification.py:119',
        prompt_provenance='Official Yes/No output prefix; behavior question newly written for this task, not an original paper premise-correction prompt.',
        reader_generation=dict(do_sample=False, batch_size=8, max_new_tokens=256),
        actual_answer_audit=dict(model=LUNA, batch_size=5, prompt=AUDIT_PROMPT,
            blinded_to=['AO prediction', 'FPQ/NFP label', 'reference answer', 'prior audit']),
        scope='Previously examined train50; exploratory behavior prediction, not test accuracy or hidden-belief evidence',
        selection_use='Not used to select GEPA candidates or change ongoing GEPA runs')
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        save(p, protocol)
    save(OUT / 'worker_runtime.json', {**config, 'artifact_root': str(OUT / 'worker'), 'ao_prompt': QUERY})
    jobs = {}
    for r in rows:
        assert r['dataset'] == 'crepe'
        assert hashlib.sha256(Path(r['source_activation_path']).read_bytes()).hexdigest() == r['activation_sha256']
        assert r['neutral']['bytewise_verified'] is True
        native = json.loads((PILOT / 'native/results' / (r['source_native_id'] + '.json')).read_text())
        assert native['answer'] == r['actual_answer']
        assert native['ao_activation_path'] == r['source_activation_path']
        ident = hashlib.sha256(json.dumps([r['source_native_id'], QUERY], ensure_ascii=False).encode()).hexdigest()
        jobs[r['row_id']] = ident
        save(OUT / 'worker/readers/ao/requests' / (ident + '.json'),
            dict(id=ident, native_id=r['source_native_id'], activation_path=r['source_activation_path'], query=QUERY))
    return rows, jobs, protocol


def export(records, protocol):
    EXPORT.mkdir(parents=True, exist_ok=True)
    (EXPORT / 'crepe50.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    fields = ['question_id', 'question_type', 'question', 'ao_yesno_raw', 'ao_prediction',
              'actual_premise_change', 'agreement', 'rejected_premise', 'answer_evidence',
              'question_scope_evidence', 'premise_grounding', 'audit_reason', 'actual_answer']
    with (EXPORT / 'crepe50.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
    save(EXPORT / 'protocol.json', protocol)
    (EXPORT / 'ao_prompt.txt').write_text(QUERY + '\n')
    (EXPORT / 'run_ao_yesno_crepe_probe.py').write_text(Path(__file__).read_text())


def external(messages, output_dir=OUT):
    assert LUNA['model'] == 'gpt-6-luna'
    key = hashlib.sha256(json.dumps([LUNA, messages], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = output_dir / 'answer_audit_calls' / (key + '.json')
    if path.exists():
        return json.loads(path.read_text())['answer']
    with (BASE / 'luna_gepa_provider.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        gate = BASE / 'luna_gepa_provider_cooldown.json'
        deadline = json.loads(gate.read_text())['retry_after_utc'] if gate.exists() else 0
        while time.time() < deadline:
            save(output_dir / 'status.json', dict(state='waiting_provider', retry_after_utc=deadline, model=LUNA['model']))
            time.sleep(min(5, deadline - time.time()))
        began = time.time()
        answer, metadata = CLIBackend(LUNA)(messages)
        save(path, dict(config=LUNA, messages=messages, answer=answer, metadata=metadata,
                        seconds=time.time()-began, completed_utc=time.time()))
        return answer


def audit_batch(payload, output_dir=OUT, system_prompt=AUDIT_PROMPT):
    request = {'items': payload}
    for attempt in range(3):
        raw = external([dict(role='system', content=system_prompt),
                        dict(role='user', content=json.dumps(request, ensure_ascii=False))], output_dir=output_dir)
        try:
            text = raw.strip()
            if text.startswith('```'):
                text = '\n'.join(text.splitlines()[1:-1])
            items = json.loads(text)['items']
            assert len(items) == 5 and {v['id'] for v in items} == {v['id'] for v in payload}
            by_id = {v['id']: v for v in items}
            for source in payload:
                v = by_id[source['id']]
                assert v['actual_premise_change'] in {'Yes', 'No', 'Unclear'}
                assert v['premise_grounding'] in {'stated', 'implied', 'invented', 'unclear', 'none'}
                assert not v['answer_evidence'] or v['answer_evidence'] in source['answer'], source['id']+' answer_evidence is not an exact substring'
                assert not v['question_scope_evidence'] or v['question_scope_evidence'] in source['question'], source['id']+' question_scope_evidence is not an exact substring'
                if v['actual_premise_change'] == 'Yes':
                    assert v['answer_evidence'], source['id']+' Yes needs an exact answer quote'
            return by_id
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            key = hashlib.sha256(raw.encode()).hexdigest()
            save(output_dir / 'answer_audit_validation_errors' / (key+'.json'),
                dict(error=repr(error), raw_output=raw, attempt=attempt, input_ids=[v['id'] for v in payload]))
            if attempt == 2:
                raise
            request = dict(items=payload, validation_retry=attempt+1,
                previous_error=str(error),
                instruction='Return all five items again. Quotes must be literal contiguous substrings, including Markdown asterisks and punctuation. Do not remove formatting or paraphrase evidence. An empty quote is allowed for No or Unclear. Keep classification based on the supplied answer only.')


def run():
    rows, jobs, protocol = prepare()
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reader = OUT / 'worker/readers/ao'
    worker = None
    worker_log = None
    try:
        if any(not (reader / 'results' / (i + '.json')).exists() for i in jobs.values()):
            (reader / 'STOP').unlink(missing_ok=True)
            worker_log = (OUT / 'worker.log').open('a')
            worker = subprocess.Popen(['/data3/heejae/uv/fpqa_ao_20261012/bin/python', '-u',
                'scripts/reader_gepa_verbalizer_worker.py', '--config', str(OUT / 'worker_runtime.json'),
                '--kind', 'ao', '--gpus', '4,5'], cwd=ROOT,
                env={**os.environ, 'HF_HOME': '/data3/heejae/hf_cache', 'HF_HUB_OFFLINE': '1'},
                stdout=worker_log, stderr=worker_log)
        records = []
        started = time.monotonic()
        for row in rows:
            p = reader / 'results' / (jobs[row['row_id']] + '.json')
            while not p.exists():
                save(OUT / 'status.json', dict(state='running', stage='ao_generation', complete=len(records),
                    expected=50, gpus='4,5', worker_pid=worker.pid, updated_utc=time.time()))
                if worker.poll() is not None:
                    raise RuntimeError('Owned AO worker exited before completing binary queries')
                if time.monotonic()-started > 1800:
                    raise TimeoutError('AO binary probe exceeded1800seconds')
                time.sleep(2)
            result = json.loads(p.read_text())
            assert result['query'] == QUERY and result['source_text_provided'] is False
            assert result['source_native_id'] == row['source_native_id']
            assert result['source_activation_path'] == row['source_activation_path']
            raw = result['explanation']
            normalized = raw.strip().lower()
            prediction = {'yes': 'Yes', 'no': 'No'}.get(normalized)
            records.append(dict(question_id=row['row_id'], question_type='FPQ' if row['label'] else 'NFP',
                question=row['question'], ao_yesno_raw=raw, ao_prediction=prediction,
                ao_format_valid=prediction is not None, actual_answer=row['actual_answer'],
                source_native_id=row['source_native_id'], activation_sha256=row['activation_sha256'],
                token_window_size=row['token_window_size'], ao_record=result))
            export(records, protocol)
        save(OUT / 'ao_outputs.json', records)
        print('AO Yes/No complete:', len(records), flush=True)
    finally:
        if worker is not None and worker.poll() is None:
            save(reader / 'STOP', dict(reason='Binary probe generation complete; normal owned-worker shutdown'))
            worker.wait(timeout=120)
        if worker_log is not None:
            worker_log.close()
    print('AO GPU worker released; classifying actual answers with Luna', flush=True)
    for start in range(0, len(records), 5):
        batch = records[start:start+5]
        payload = [dict(id=f'item_{start+i}', question=r['question'], answer=r['actual_answer']) for i, r in enumerate(batch)]
        save(OUT / 'status.json', dict(state='running', stage='independent_answer_audit', complete=start,
            expected=50, model=LUNA['model'], updated_utc=time.time()))
        by_id = audit_batch(payload)
        for i, r in enumerate(batch):
            v = by_id[f'item_{start+i}']
            assert v['actual_premise_change'] in {'Yes', 'No', 'Unclear'}
            assert v['premise_grounding'] in {'stated', 'implied', 'invented', 'unclear', 'none'}
            assert not v['answer_evidence'] or v['answer_evidence'] in r['actual_answer']
            assert not v['question_scope_evidence'] or v['question_scope_evidence'] in r['question']
            if v['actual_premise_change'] == 'Yes':
                assert v['answer_evidence']
            r.update(actual_premise_change=v['actual_premise_change'], rejected_premise=v['rejected_premise'],
                answer_evidence=v['answer_evidence'], question_scope_evidence=v['question_scope_evidence'],
                premise_grounding=v['premise_grounding'], audit_reason=v['reason'], audit_model=LUNA)
            r['agreement'] = (r['ao_prediction'] == r['actual_premise_change']
                if r['ao_prediction'] is not None and r['actual_premise_change'] != 'Unclear' else None)
        export(records, protocol)
        print('Actual answer comparison:', start+5, '/50', flush=True)
    groups = []
    for kind in ['FPQ', 'NFP']:
        subset = [r for r in records if r['question_type'] == kind]
        comparable = [r for r in subset if r['agreement'] is not None]
        groups.append(dict(question_type=kind, n=len(subset),
            ao_predictions=dict(Counter(r['ao_prediction'] or 'Invalid' for r in subset)),
            actual_premise_change=dict(Counter(r['actual_premise_change'] for r in subset)),
            agreement_count=sum(r['agreement'] is True for r in comparable), agreement_denominator=len(comparable),
            confusion=dict(Counter(f"AO={r['ao_prediction']};actual={r['actual_premise_change']}" for r in comparable)),
            actual_correction_invented_premise=sum(r['actual_premise_change']=='Yes' and r['premise_grounding']=='invented' for r in subset)))
    summary = dict(state='complete', questions=50, groups=groups, query=QUERY, answer_audit_model=LUNA,
        limits=['Exploratory previously examined train50; no independent human verification.',
                'AO Yes predicts behavior, not existence of a false premise or correctness of a correction.',
                'Actual behavior audit sees question and answer only; AO prediction, labels and references withheld.',
                'Forced binary AO output has no uncertainty class; invalid outputs retained separately.',
                'No Sol calls, new source answers, GEPA changes or candidate selection.'])
    save(OUT / 'comparison_summary.json', summary)
    save(EXPORT / 'comparison_summary.json', summary)
    save(OUT / 'status.json', dict(state='complete', ao_complete=50, answer_audit_complete=50,
        export=str(EXPORT), updated_utc=time.time()))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        rows, jobs, protocol = prepare()
        export([], protocol)
        print('Prepared50 matched CREPE binary requests on GPU4,5; answer audit Luna only', flush=True)
    else:
        try:
            run()
        except Exception as error:
            save(OUT / 'status.json', dict(state='attention_required', error=repr(error), updated_utc=time.time()))
            raise
