"""Persistent native Gemma27 inference with answer-linked prefill activations."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    a = p.parse_args()
    c = json.loads(a.config.read_text())
    out = Path(c['artifact_root'])
    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpus
    locks = []
    Path('/tmp/paper-baselines-gpu-locks').mkdir(parents=True, exist_ok=True)
    for gpu in a.gpus.split(','):
        handle = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    import torch
    from src.paper_baselines import HFTask, digest
    from scripts.nla_generation_capture import decoder_layer
    baseline = json.loads(Path(c['baseline_config']).read_text())
    registry = json.loads((ROOT / baseline['registry']).read_text())
    native = out / 'native'
    requests = native / 'requests'
    results = native / 'results'
    acts = native / 'activations'
    for path in [requests, results, acts]:
        path.mkdir(parents=True, exist_ok=True)
    def save(path, data):
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
        temp.replace(path)
    def request(row, prompt):
        messages = [{'role': 'system', 'content': registry['transport_common'] + '\n\n' + prompt},
                    {'role': 'user', 'content': row['question']}]
        ident = digest([baseline['models']['gemma27'], messages, 'native_reader_gepa_v1'])
        path = requests / (ident + '.json')
        if not path.exists():
            save(path, dict(id=ident, row=row, messages=messages, candidate_prompt=prompt))
        return ident
    initial = [request(row, registry['prompts']['plain']) for row in c['cohort']]
    save(native / 'cohort_request_ids.json', initial)
    save(native / 'status.json', dict(state='loading', gpus=a.gpus, expected_initial=100))
    backend = HFTask(baseline, 'gemma27')
    modules = {k: decoder_layer(backend.model, k + 1, 62)[0] for k in [31, 41]}
    source_root = Path(c['original_artifact_root'])
    old_nla = {(r['base_id'], r['condition'], r['position_family']): r for r in
               (json.loads(s) for s in (source_root / 'gemma27/manifest.jsonl').read_text().splitlines())}
    old_ao = {(r['base_id'], r['condition']): r for r in
              (json.loads(s) for s in (source_root / 'gemma27/ao_manifest.jsonl').read_text().splitlines()) if r['layer'] == 31}
    processed = 0
    checked_layer = False
    def generate(job):
        nonlocal checked_layer
        row = job['row']; messages = job['messages']; ident = job['id']
        batch = backend.inputs(messages)
        ids = batch['input_ids'][0].tolist()
        text = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        offsets = backend.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert offsets['input_ids'] == ids
        start = text.rfind(row['question']); end = start + len(row['question'])
        assert start >= 0
        positions = [i for i, (x, y) in enumerate(offsets['offset_mapping']) if x < end and y > start and x != y]
        qend = positions[-1]; window_start = max(positions[0], qend - 19)
        observed = {}; handles = []
        for k, module in modules.items():
            def hook(_module, _inputs, result, k=k):
                hidden = result[0] if isinstance(result, tuple) else result
                if hidden.shape[1] != len(ids):
                    return
                assert k not in observed
                observed[k] = (hidden[0, qend] if k == 41 else hidden[0, window_start:qend + 1]).detach().float().cpu()
            handles.append(module.register_forward_hook(hook))
        try:
            with torch.inference_mode():
                output = backend.model.generate(**batch, do_sample=False,
                    max_new_tokens=1 if job.get('capture_only') else baseline['max_new_tokens'],
                    pad_token_id=backend.tokenizer.pad_token_id)
        finally:
            for handle in handles:
                handle.remove()
        tokens = output[0, len(ids):].detach().cpu()
        eos = backend.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        completed = len(tokens) > 0 and tokens[-1].item() in eos
        assert set(observed) == {31, 41}
        if not checked_layer:
            with torch.inference_mode():
                direct = backend.model(**batch, use_cache=False, output_hidden_states=True)
            for k in [31, 41]:
                value = direct.hidden_states[k + 1][0, qend] if k == 41 else direct.hidden_states[k + 1][0, window_start:qend + 1]
                assert torch.equal(observed[k].view(torch.int32), value.detach().float().cpu().view(torch.int32))
            del direct
            checked_layer = True
        record = dict(id=ident, row_id=row['id'], dataset=row['dataset'], question=row['question'],
            messages=messages, candidate_prompt=job['candidate_prompt'],
            answer=backend.tokenizer.decode(tokens, skip_special_tokens=True).strip(), completed=bool(completed),
            finish_reason='eos' if completed else 'length', input_tokens=len(ids), output_tokens=len(tokens),
            source_model_revision=baseline['models']['gemma27']['revision'],
            source_gpu_indices=a.gpus,
            capture_only=bool(job.get('capture_only', False)),
            capture_mode='actual_native_generation_prefill', source_question_positions=[positions[0], qend],
            timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reader_reuse={})
        for kind, k in [('nla', 41), ('ao', 31)]:
            value = observed[k]
            assert torch.isfinite(value).all() and value.shape[-1] == 5376
            path = acts / f'{ident}_{kind}.pt'; torch.save(value, path)
            record[kind + '_activation_path'] = str(path)
            record[kind + '_source_token_ids'] = [ids[qend]] if kind == 'nla' else ids[window_start:qend + 1]
            if job['candidate_prompt'] == registry['prompts']['plain']:
                old = old_nla.get((row['id'], 'plain', 'question_last_token')) if kind == 'nla' else old_ao.get((row['id'], 'plain'))
                if old:
                    prior = torch.load(old['activation_path'], map_location='cpu', weights_only=True)
                    equal = value.shape == prior.shape and torch.equal(value.view(torch.int32), prior.view(torch.int32))
                    record['reader_reuse'][kind] = dict(bitwise_equal=bool(equal), original_activation_id=old['id'],
                        original_activation_path=old['activation_path'])
        return record
    def run_one(ident):
        nonlocal processed
        path = results / (ident + '.json')
        if path.exists():
            return
        job = json.loads((requests / (ident + '.json')).read_text())
        try:
            record = generate(job)
            save(path, record)
            processed += 1
            print('Native', processed, job['row']['id'], record['finish_reason'], record['output_tokens'], flush=True)
        except Exception as error:
            save(results / (ident + '.error.json'), dict(id=ident, error=repr(error)))
            raise
        save(native / 'status.json', dict(state='running', stage='native_generation', gpus=a.gpus,
             initial_processed=sum((results / (key + '.json')).exists() for key in initial), expected_initial=100,
             processed_this_process=processed))
    for ident in initial:
        run_one(ident)
    save(native / 'cohort_status.json', dict(state='complete', processed=100,
        completed=sum(json.loads((results / (key + '.json')).read_text())['completed'] for key in initial)))
    print('Initial100 complete; serving candidate requests', flush=True)
    while not (native / 'STOP').exists():
        pending = [p for p in requests.glob('*.json') if not (results / p.name).exists()]
        pending.sort(key=lambda p: p.stat().st_mtime_ns)
        if pending:
            for path in pending:
                run_one(path.stem)
        else:
            save(native / 'status.json', dict(state='ready', gpus=a.gpus, initial_processed=100,
                expected_initial=100, processed_this_process=processed))
            time.sleep(1)
    save(native / 'status.json', dict(state='stopped', processed_this_process=processed))


if __name__ == '__main__':
    main()
