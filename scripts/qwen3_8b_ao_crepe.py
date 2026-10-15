"""Matched Qwen3-8B CREPE50 native answers and public AO; one GPU per worker."""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'external/activation_oracles_official'))
from scripts.reader_gepa_io import save
from scripts.run_ao_behavior_query_probe import BEHAVIOR_QUERY
from scripts.run_ao_yesno_crepe_probe import QUERY as YESNO_QUERY, audit_batch, AUDIT_PROMPT, LUNA

OUT = ROOT / 'results/qwen3_8b_ao_crepe_20261014'
EXPORT = ROOT.parents[1] / 'results/qwen3_8b_ao_crepe_20261014'
PILOT = ROOT / 'results/nla_ao_parallel_20261012/gepa_reader_pilot'
PYTHON = '/data3/heejae/uv/fpqa_ao_20261012/bin/python'
QUERIES = dict(neutral='Describe the semantic content represented by these activations.',
               behavior=BEHAVIOR_QUERY, yesno=YESNO_QUERY)
ANSWER_AUDIT = AUDIT_PROMPT + """
Require an observable rejection of an assumption attributed to the question.
If the question asserts A and the answer explains why A happens, classify No.
Negating a different claim B is not rejection of A. A question asking how someone
knows something can presuppose certainty: an answer explicitly saying they cannot
know for sure can count Yes. A normal statement of uncertainty is not enough.
Do not turn an explanation of a trade-off into rejection of a trade-off the answer
explicitly acknowledges. Unasked common-misconception context is not itself Yes.
"""


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def state(name, **fields):
    save(OUT / name, {**fields, 'updated_utc': time.time()})


def rows_and_protocol():
    downloads = json.loads((OUT / 'downloads.json').read_text())
    plan = json.loads((PILOT / 'plan.json').read_text())
    config = json.loads((PILOT / 'runtime.json').read_text())
    bundle = json.loads(Path(config['bundle']).read_text())
    lookup = {r['id']: r for r in bundle['rows']}
    rows = [lookup[i] for i in sorted(plan['splits']['crepe']['train_ids'])]
    assert len(rows) == 50 and Counter(r['label'] for r in rows) == {0: 25, 1: 25}
    registry = json.loads((ROOT / json.loads(Path(config['baseline_config']).read_text())['registry']).read_text())
    settings = dict(protocol='qwen3_8b_matched_ao_crepe_v1', source_checkpoint=downloads['source'],
        ao_checkpoint=downloads['ao'], dataset='crepe', train_ids=[r['id'] for r in rows],
        FPQ=25, NFP=25, source_prompt=plan['starting_prompt'], transport_common=registry['transport_common'],
        source_gpu=4, reader_gpu=5, dtype='bfloat16', source_thinking=False, reader_thinking=False,
        do_sample=False, source_max_new_tokens=8192, reader_max_new_tokens=256,
        source_batch_planned=16, reader_batch_planned=64, benchmark_batches=[16, 32, 64],
        benchmark_max_new_tokens=64, benchmark_scope='Short-generation capacity/throughput check, not proof of full8192-token source capacity.',
        source_layers=[18, 27], ao_primary_layer=18, hf_hidden_state_indices=[19, 28],
        source_positions='Last up to20 question tokens, actual answer generation first prefill',
        reader_queries=QUERIES, reader_input='Activations plus query only; no question, answer, label, annotation or reference text.',
        capture_verification='Compare first native generation prefill hook at18/27 to HF hidden_states19/28 on identical padded inputs.',
        source_reuse='No Gemma/Qwen2.5 activations or native answers reused.',
        actual_answer_audit_model=LUNA, actual_answer_audit_prompt=ANSWER_AUDIT,
        scope='Previously examined train50; exploratory, not independent test or final main table.')
    p = OUT / 'protocol.json'
    if p.exists():
        assert json.loads(p.read_text()) == settings
    else:
        save(p, settings)
    return rows, settings


def request(row, prompt, protocol, capture_only=False):
    messages = [dict(role='system', content=protocol['transport_common']+'\n\n'+prompt),
                dict(role='user', content=row['question'])]
    version = 'qwen3_8b_prefill_only_v1' if capture_only else 'qwen3_8b_prefill_v1'
    ident = sha([protocol['source_checkpoint']['revision'], messages, version])
    save(OUT / 'native/requests' / (ident+'.json'),
        dict(id=ident, row=row, messages=messages, candidate_prompt=prompt, capture_only=capture_only))
    return ident


def lock_gpu(gpu):
    Path('/tmp/paper-baselines-gpu-locks').mkdir(parents=True, exist_ok=True)
    handle = Path(f'/tmp/paper-baselines-gpu-locks/gpu-{gpu}.lock').open('a')
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def native_worker():
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    gpu_lock = lock_gpu(4)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows, protocol = rows_and_protocol()
    base = protocol['source_checkpoint']['local_path']
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa').eval()
    assert model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
    assert all(p.device.type == 'cuda' for p in model.parameters())
    initial = [request(r, protocol['source_prompt'], protocol) for r in rows]
    save(OUT / 'native/cohort_ids.json', initial)
    for d in ['native/activations', 'native/results']:
        (OUT / d).mkdir(parents=True, exist_ok=True)
    checked = False

    def generate(jobs, cap, capture):
        nonlocal checked
        rendered = [tok.apply_chat_template(j['messages'], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for j in jobs]
        data = tok(rendered, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda:0')
        width = data['input_ids'].shape[1]
        windows = []
        for i, (text, job) in enumerate(zip(rendered, jobs)):
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = enc['input_ids']
            pad = width-len(ids)
            assert data['input_ids'][i, pad:].tolist() == ids
            start = text.rfind(job['row']['question'])
            assert start >= 0
            end = start+len(job['row']['question'])
            positions = [p for p, (a, b) in enumerate(enc['offset_mapping']) if a < end and b > start and a != b]
            positions = positions[-20:]
            assert positions
            windows.append(dict(positions=[p+pad for p in positions], unpadded_positions=positions,
                                source_token_ids=[ids[p] for p in positions], left_padding=pad, input_ids=ids))
        observed = {}
        handles = []
        if capture:
            for layer in protocol['source_layers']:
                def hook(_module, _inputs, output, layer=layer):
                    value = output[0] if isinstance(output, tuple) else output
                    if value.shape[1] == width and layer not in observed:
                        observed[layer] = [value[i, w['positions']].detach().float().cpu()
                                           for i, w in enumerate(windows)]
                handles.append(model.model.layers[layer].register_forward_hook(hook))
        try:
            with torch.inference_mode():
                tokens = model.generate(**data, do_sample=False, max_new_tokens=cap, use_cache=True,
                    pad_token_id=tok.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
                    top_p=None, top_k=None, temperature=None)
        finally:
            for h in handles:
                h.remove()
        if capture and not checked:
            # HF generate assigns positions from the left-padding mask. A bare
            # Qwen forward otherwise assigns positions including the padding.
            # Match generate exactly before comparing the saved prefill.
            position_ids = data['attention_mask'].long().cumsum(-1)-1
            position_ids.masked_fill_(data['attention_mask']==0, 1)
            with torch.inference_mode():
                direct = model(**data, position_ids=position_ids, use_cache=True, output_hidden_states=True)
            for layer in protocol['source_layers']:
                for i, w in enumerate(windows):
                    expected = direct.hidden_states[layer+1][i, w['positions']].detach().float().cpu()
                    assert torch.equal(observed[layer][i], expected), f'Layer mapping mismatch at{layer}'
            checked = True
            save(OUT / 'native/capture_verification.json', dict(state='passed',
                layers=[18, 27], hidden_state_indices=[19, 28], compared_items=len(jobs), exact=True))
            del direct
        return tokens[:, width:].detach().cpu(), windows, observed

    benchmark = []
    examples = [json.loads((OUT / 'native/requests' / (i+'.json')).read_text()) for i in initial]
    generate(examples[:1], 8, False)
    for size in protocol['benchmark_batches']:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        try:
            tokens, _, _ = generate([examples[i % len(examples)] for i in range(size)], 64, False)
            torch.cuda.synchronize()
            benchmark.append(dict(batch_size=size, status='success', seconds=time.perf_counter()-began,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, output_width=tokens.shape[1]))
            del tokens
        except torch.cuda.OutOfMemoryError as error:
            benchmark.append(dict(batch_size=size, status='oom', error=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            break
        save(OUT / 'native/batch_benchmark.json', benchmark)
        print('Native benchmark', benchmark[-1], flush=True)
    save(OUT / 'native/batch_benchmark.json', benchmark)
    batch_size = 16 if any(r['batch_size']==16 and r['status']=='success' for r in benchmark) else 8
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    while not (OUT / 'native/STOP').exists():
        pending = [p for p in (OUT / 'native/requests').glob('*.json')
                   if not (OUT / 'native/results' / p.name).exists()]
        pending.sort(key=lambda p: p.stat().st_mtime_ns)
        if not pending:
            state('native/status.json', state='ready', gpu=4, initial_complete=sum((OUT/'native/results'/(i+'.json')).exists() for i in initial), initial_expected=50)
            time.sleep(1)
            continue
        capture_only = json.loads(pending[0].read_text()).get('capture_only', False)
        selected = [p for p in pending if json.loads(p.read_text()).get('capture_only', False)==capture_only][:batch_size]
        jobs = [json.loads(p.read_text()) for p in selected]
        state('native/status.json', state='running', gpu=4, batch_size=len(jobs), pending=len(pending))
        try:
            tokens, windows, observed = generate(jobs, 1 if capture_only else 8192, True)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size//2)
            gc.collect()
            torch.cuda.empty_cache()
            print('Native full-generation OOM; reducing batch to', batch_size, flush=True)
            continue
        for i, (job, w) in enumerate(zip(jobs, windows)):
            value = tokens[i].tolist()
            stop = next((j for j, t in enumerate(value) if t in eos), None)
            if stop is not None:
                value = value[:stop+1]
            paths = {}
            for layer in protocol['source_layers']:
                v = observed[layer][i]
                assert v.shape == (len(w['positions']), 4096) and torch.isfinite(v).all()
                path = OUT / 'native/activations' / f"{job['id']}_L{layer}.pt"
                temp = path.with_suffix('.tmp')
                torch.save(v, temp)
                temp.replace(path)
                paths[str(layer)] = str(path)
            save(OUT / 'native/results' / (job['id']+'.json'), dict(id=job['id'], row_id=job['row']['id'],
                dataset=job['row']['dataset'], label=job['row']['label'], question=job['row']['question'],
                messages=job['messages'], candidate_prompt=job['candidate_prompt'],
                answer=tok.decode(value, skip_special_tokens=True).strip(), completed=stop is not None,
                finish_reason='eos' if stop is not None else 'length', output_tokens=len(value), input_tokens=len(w['input_ids']),
                input_ids=w['input_ids'], source_positions=w['unpadded_positions'], source_token_ids=w['source_token_ids'],
                activation_paths=paths, ao_activation_path=paths['18'], source_model_revision=protocol['source_checkpoint']['revision'],
                capture_mode='actual_native_generation_prefill', capture_only=capture_only, source_batch_size=len(jobs), source_gpu=4,
                source_thinking=False, timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('Native saved', len(jobs), 'initial', sum((OUT/'native/results'/(i+'.json')).exists() for i in initial), '/50', flush=True)
    state('native/status.json', state='stopped', gpu=4)


def reader_worker():
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    gpu_lock = lock_gpu(5)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from safetensors import safe_open
    from nl_probes.utils.dataset_utils import create_training_datapoint
    from nl_probes.utils.eval import run_evaluation
    _, protocol = rows_and_protocol()
    base = protocol['source_checkpoint']['local_path']
    adapter = protocol['ao_checkpoint']['local_path']
    ac = json.loads((Path(adapter) / 'ao_config.json').read_text())
    assert ac['model_name'] == 'Qwen/Qwen3-8B' and [18] in ac['act_layer_combinations']
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16,
        device_map={'': 0}, attn_implementation='sdpa').eval()
    model = PeftModel.from_pretrained(model, adapter, adapter_name='ao', is_trainable=False).eval()
    assert model.active_adapter == 'ao'
    with safe_open(str(Path(adapter) / 'adapter_model.safetensors'), framework='pt') as f:
        keys = list(f.keys())
    loaded = model.state_dict()
    assert keys and all(k.replace('.lora_A.weight', '.lora_A.ao.weight').replace('.lora_B.weight', '.lora_B.ao.weight') in loaded for k in keys)
    module = model.get_base_model().model.layers[ac['hook_onto_layer']]

    def read(jobs, cap):
        points = []
        for job in jobs:
            v = torch.load(job['activation_path'], map_location='cpu', weights_only=True)
            assert v.ndim == 2 and 0 < len(v) <= 20 and v.shape[1] == 4096 and torch.isfinite(v).all()
            points.append(create_training_datapoint(datapoint_type='FPQA_preanswer', prompt=job['query'],
                target_response='N/A', layer=18, num_positions=len(v), tokenizer=tok, acts_BD=v,
                feature_idx=-1, context_input_ids=None, context_positions=None, ds_label=None,
                meta_info={'id':job['id']}))
        return run_evaluation(eval_data=points, model=model, tokenizer=tok, submodule=module,
            device=torch.device('cuda:0'), dtype=torch.bfloat16, global_step=-1, lora_path=None,
            eval_batch_size=len(jobs), steering_coefficient=ac['steering_coefficient'],
            generation_kwargs=dict(do_sample=False, max_new_tokens=cap, disable_compile=True,
                pad_token_id=tok.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
                top_p=None, top_k=None, temperature=None))
    folder = OUT / 'readers/ao'
    for part in ['requests', 'results']:
        (folder / part).mkdir(parents=True, exist_ok=True)
    benchmark = []
    state('readers/ao/status.json', state='ready', gpu=5)
    while not list((folder/'requests').glob('*.json')):
        if (folder/'STOP').exists():
            state('readers/ao/status.json', state='stopped', gpu=5)
            return
        time.sleep(1)
    seeds = [json.loads(p.read_text()) for p in list((folder/'requests').glob('*.json'))[:16]]
    for size in protocol['benchmark_batches']:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        try:
            results = read([seeds[i % len(seeds)] for i in range(size)], 64)
            torch.cuda.synchronize()
            benchmark.append(dict(batch_size=size, status='success', seconds=time.perf_counter()-began,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            del results
        except torch.cuda.OutOfMemoryError as error:
            benchmark.append(dict(batch_size=size, status='oom', error=str(error)))
            gc.collect()
            torch.cuda.empty_cache()
            break
        save(folder/'batch_benchmark.json', benchmark)
        print('AO benchmark', benchmark[-1], flush=True)
    save(folder/'batch_benchmark.json', benchmark)
    successes = [r['batch_size'] for r in benchmark if r['status']=='success']
    batch_size = max(successes) if successes else 8
    while not (folder/'STOP').exists():
        pending = [p for p in (folder/'requests').glob('*.json') if not (folder/'results'/p.name).exists()]
        pending.sort(key=lambda p: p.stat().st_mtime_ns)
        if not pending:
            state('readers/ao/status.json', state='ready', gpu=5, batch_size=batch_size, processed=len(list((folder/'results').glob('*.json'))))
            time.sleep(1)
            continue
        chunk = [json.loads(p.read_text()) for p in pending[:batch_size]]
        state('readers/ao/status.json', state='running', gpu=5, batch_size=len(chunk), pending=len(pending))
        try:
            results = read(chunk, 256)
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size//2)
            gc.collect()
            torch.cuda.empty_cache()
            print('AO full-output OOM; reducing batch to', batch_size, flush=True)
            continue
        for job, result in zip(chunk, results):
            save(folder/'results'/(job['id']+'.json'), dict(id=job['id'], explanation=result.api_response,
                rendered_prompt=result.prompt, query=job['query'], query_kind=job['query_kind'],
                source_native_id=job['native_id'], source_activation_path=job['activation_path'],
                source_text_provided=False, source_layer=18, reader_batch_size=len(chunk), reader_gpu=5,
                ao_model_revision=protocol['ao_checkpoint']['revision'], timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('AO saved', len(chunk), 'total', len(list((folder/'results').glob('*.json'))), flush=True)
    state('readers/ao/status.json', state='stopped', gpu=5)


def export(records, protocol):
    EXPORT.mkdir(parents=True, exist_ok=True)
    (EXPORT/'crepe50.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    fields = ['question_id', 'question_type', 'question', 'actual_answer', 'answer_completed',
              'neutral_text', 'behavior_text', 'yesno_raw', 'yesno_prediction', 'actual_premise_change', 'answer_evidence']
    with (EXPORT/'crepe50.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k:r.get(k) for k in fields})
    save(EXPORT/'protocol.json', protocol)
    for source, target in [('native/batch_benchmark.json','native_batch_benchmark.json'),
                           ('readers/ao/batch_benchmark.json','ao_batch_benchmark.json'),
                           ('native/capture_verification.json','capture_verification.json')]:
        p = OUT/source
        if p.exists():
            save(EXPORT/target, json.loads(p.read_text()))


def pilot():
    rows, protocol = rows_and_protocol()
    control_lock = (OUT/'controller.lock').open('a')
    fcntl.flock(control_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ids = {r['id']:request(r, protocol['source_prompt'], protocol) for r in rows}
    workers = []
    logs = []
    for path in ['native/STOP','readers/ao/STOP']:
        (OUT/path).unlink(missing_ok=True)
    try:
        for role in ['native','reader']:
            log = (OUT/(role+'_worker.log')).open('a')
            logs.append(log)
            worker = subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--role',role], cwd=ROOT,
                env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'}, stdout=log, stderr=log)
            workers.append(worker)
        records = []
        began = time.monotonic()
        queued = set()
        while len(records) < 50:
            for row in rows:
                p = OUT/'native/results'/(ids[row['id']]+'.json')
                if not p.exists() or row['id'] in queued:
                    continue
                native = json.loads(p.read_text())
                for kind, query in QUERIES.items():
                    ident = sha([native['id'], kind, query, 'qwen3_8b_ao_v1'])
                    save(OUT/'readers/ao/requests'/(ident+'.json'), dict(id=ident, native_id=native['id'],
                        activation_path=native['ao_activation_path'], query=query, query_kind=kind))
                queued.add(row['id'])
            records = []
            for row in rows:
                if row['id'] not in queued:
                    continue
                native = json.loads((OUT/'native/results'/(ids[row['id']]+'.json')).read_text())
                paths = {kind:OUT/'readers/ao/results'/(sha([native['id'],kind,query,'qwen3_8b_ao_v1'])+'.json')
                         for kind, query in QUERIES.items()}
                if not all(p.exists() for p in paths.values()):
                    continue
                readers = {kind:json.loads(p.read_text()) for kind,p in paths.items()}
                assert all(r['source_native_id']==native['id'] and r['source_text_provided'] is False for r in readers.values())
                raw = readers['yesno']['explanation']
                records.append(dict(question_id=row['id'], question_type='FPQ' if row['label'] else 'NFP',
                    question=row['question'], actual_answer=native['answer'], answer_completed=native['completed'],
                    native_record=native, neutral_text=readers['neutral']['explanation'],
                    behavior_text=readers['behavior']['explanation'], yesno_raw=raw,
                    yesno_prediction={'yes':'Yes','no':'No'}.get(raw.strip().lower()), readers=readers))
            export(records, protocol)
            state('status.json', state='running', stage='matched_native_and_ao', source_queued=len(queued),
                  complete=len(records), expected=50, native_pid=workers[0].pid, reader_pid=workers[1].pid)
            if len(records)==50:
                break
            for worker in workers:
                if worker.poll() is not None:
                    raise RuntimeError('Owned8B worker exited; inspect worker log')
            if time.monotonic()-began > 7200:
                raise TimeoutError('Initial8B matching stage exceeded7200seconds')
            time.sleep(2)
    finally:
        for name in ['native/STOP','readers/ao/STOP']:
            save(OUT/name, dict(reason='Initial8B observation complete; normal owned-worker shutdown'))
        for worker in workers:
            if worker.poll() is None:
                worker.wait(timeout=120)
        for log in logs:
            log.close()
    print('Qwen8B50 native answers and150 AO outputs complete; GPUs released', flush=True)
    for start in range(0, 50, 5):
        batch = records[start:start+5]
        payload = [dict(id=f'item_{start+i}',question=r['question'],answer=r['actual_answer']) for i,r in enumerate(batch)]
        state('status.json',state='running',stage='luna_answer_comparison',complete=start,expected=50)
        audits = audit_batch(payload, output_dir=OUT, system_prompt=ANSWER_AUDIT)
        for i,r in enumerate(batch):
            v = audits[f'item_{start+i}']
            r.update(actual_premise_change=v['actual_premise_change'],answer_evidence=v['answer_evidence'],
                     answer_audit=v,answer_audit_model=LUNA)
        export(records,protocol)
        print('Luna answer comparison',start+5,'/50',flush=True)
    summary=dict(questions=50,ao_descriptions=150,source_model='Qwen3-8B',results=[])
    for label in ['FPQ','NFP']:
        selected=[r for r in records if r['question_type']==label]
        summary['results'].append(dict(question_type=label,n=len(selected),
            ao_yesno=dict(Counter(r['yesno_prediction'] or 'Invalid' for r in selected)),
            actual_premise_change=dict(Counter(r['actual_premise_change'] for r in selected))))
    summary['limits']=['Previously examined train50; exploratory, no independent test claims.',
                       'Luna behavior labels are provisional and need content review; premise revision is not correction accuracy.',
                       'No Gemma/Qwen2.5 answer, activation, or Sol judgment reuse. All new external calls pinned Luna.']
    save(OUT/'comparison_summary.json',summary)
    save(EXPORT/'comparison_summary.json',summary)
    (EXPORT/'qwen3_8b_ao_crepe.py').write_text(Path(__file__).read_text())
    state('status.json',state='complete',questions=50,ao_descriptions=150,export=str(EXPORT))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=['native','reader','pilot'],default='pilot')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare_only:
            rows, protocol=rows_and_protocol()
            for r in rows:
                request(r,protocol['source_prompt'],protocol)
            print('PreparedQwen3-8B sourceGPU4 AO-GPU5 CREPE50,150 AO jobs; batch benchmarks16/32/64',flush=True)
        else:
            {'native':native_worker,'reader':reader_worker,'pilot':pilot}[args.role]()
    except Exception as error:
        state('status.json' if args.role=='pilot' else args.role+'_failure.json',
              state='attention_required',error=repr(error),role=args.role)
        raise
