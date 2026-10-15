"""Four independent AO questions on the same saved Qwen3-8B pre-answer states."""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'external/paper_baselines_b81ace2'
SOURCE = ROOT / 'results/qwen3_8b_ao_crepe_20261014'
OUT = ROOT / 'results/ao_four_questions_crepe_20261014'

BOUNDARY = (
    'Use only evidence in these activations, not your own factual knowledge. '
    'Do not assume the question contains an error. If the requested information '
    'is not clear from the activations, say "Unclear" rather than inventing it. '
    'Answer briefly in English.'
)
QUERIES = {
    'target_premise': (
        'What key factual assumption in the user\'s question is represented in '
        'these activations? State the assumption as a specific claim; do not '
        'judge whether it is true or false. '
    ) + BOUNDARY,
    'stance': (
        'How does the model appear to treat the key factual assumption in the '
        'user\'s question: Accept, Doubt, Reject, or Unclear? Briefly name the '
        'assumption and explain the apparent stance. Describe the model\'s '
        'represented stance, not the stance it should take. '
    ) + BOUNDARY,
    'likely_behavior': (
        'When answering the user\'s question, what does the model appear likely '
        'to do with its key factual assumption: answer using it, qualify it, '
        'or explicitly correct it? Name the assumption briefly and describe '
        'the likely response behavior, not what the model should do. '
    ) + BOUNDARY,
    'correction_content': (
        'If these activations indicate that the model is preparing to correct '
        'or qualify an assumption in the user\'s question, what exact claim '
        'will it address and what replacement or qualification will it give? '
        'Describe the apparent planned correction, not a correction you think '
        'is appropriate. If no correction is indicated, say "No clear '
        'correction indicated". '
    ) + BOUNDARY,
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def export(rows, protocol):
    records = []
    for row in rows:
        native = row['native_record']
        paths = {kind: OUT / 'outputs' / (digest([native['id'], kind, query,
                 protocol['ao_checkpoint']['revision']]) + '.json')
                 for kind, query in QUERIES.items()}
        if not all(path.exists() for path in paths.values()):
            continue
        readers = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
        records.append(dict(question_id=row['question_id'], question_type=row['question_type'],
            question=row['question'], actual_answer=row['actual_answer'],
            answer_completed=row['answer_completed'], source_native_id=native['id'],
            source_activation_path=native['ao_activation_path'],
            source_positions=native['source_positions'], source_token_ids=native['source_token_ids'],
            source_layer=18, source_capture_mode=native['capture_mode'],
            **{kind: readers[kind]['explanation'] for kind in QUERIES}, readers=readers,
            previous_behavior_description=row['behavior_text'],
            previous_yesno_output=row['yesno_raw'],
            prior_luna_behavior_label=row.get('actual_premise_change'),
            prior_luna_answer_evidence=row.get('answer_evidence')))
    (OUT / 'crepe50.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    fields = ['question_id', 'question_type', 'question', 'actual_answer', *QUERIES,
              'previous_behavior_description', 'previous_yesno_output',
              'prior_luna_behavior_label', 'prior_luna_answer_evidence']
    with (OUT / 'crepe50.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) for k in fields} for r in records)
    return records


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    lockdir = Path('/tmp/paper-baselines-gpu-locks')
    lockdir.mkdir(parents=True, exist_ok=True)
    gpu_lock = (lockdir / f'gpu-{args.gpu}.lock').open('a')
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, str(EXPERIMENT / 'external/activation_oracles_official'))
    from nl_probes.utils.dataset_utils import create_training_datapoint
    from nl_probes.utils.eval import run_evaluation

    rows = [json.loads(line) for line in (SOURCE / 'crepe50.jsonl').read_text().splitlines()]
    source_protocol = json.loads((SOURCE / 'protocol.json').read_text())
    assert len(rows) == 50 and Counter(r['question_type'] for r in rows) == {'FPQ': 25, 'NFP': 25}
    protocol = dict(experiment='four_independent_ao_questions_v1',
        source_checkpoint=source_protocol['source_checkpoint'],
        ao_checkpoint=source_protocol['ao_checkpoint'],
        source_prompt=source_protocol['source_prompt'], source_layer_zero_based=18,
        source_positions=source_protocol['source_positions'], source_thinking=False,
        reader_thinking=False, reader_gpu=args.gpu, reader_batch_size=args.batch_size,
        max_new_tokens=256, do_sample=False, dtype='bfloat16', queries=QUERIES,
        independent_queries=True, other_query_outputs_provided=False,
        reader_input='Saved activations and the single query only; no source text, answer, labels or references.',
        source_questions=50, source_FPQs=25, source_NFPs=25, expected_descriptions=200,
        source_reuse='Existing native-generation first-prefill states and the exact corresponding answers.',
        external_model_calls=0, reader_family='Original Adam Karvonen AO; not Bauer improved AO.',
        limits=['Exploratory previously examined CREPE train50; not independent test.',
                'Descriptions are hypotheses about the source, not verified beliefs or causes.',
                'Prior Luna behavior labels are provisional; no new automatic correctness score.'])
    OUT.mkdir(parents=True, exist_ok=True)
    pp = OUT / 'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol
    else:
        save(pp, protocol)
    save(OUT / 'status.json', dict(state='loading', gpu=args.gpu, expected=200))
    base = source_protocol['source_checkpoint']['local_path']
    adapter = source_protocol['ao_checkpoint']['local_path']
    ac = json.loads((Path(adapter) / 'ao_config.json').read_text())
    assert ac['model_name'] == 'Qwen/Qwen3-8B' and [18] in ac['act_layer_combinations']
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={'': 0}, attn_implementation='sdpa').eval()
    model = PeftModel.from_pretrained(model, adapter, adapter_name='ao', is_trainable=False).eval()
    assert model.active_adapter == 'ao' and all(p.device.type == 'cuda' for p in model.parameters())
    jobs = []
    for row in rows:
        native = row['native_record']
        assert native['completed'] and native['capture_mode'] == 'actual_native_generation_prefill'
        assert native['source_model_revision'] == source_protocol['source_checkpoint']['revision']
        v = torch.load(native['ao_activation_path'], map_location='cpu', weights_only=True)
        assert v.shape == (len(native['source_token_ids']), 4096) and 0 < len(v) <= 20
        assert torch.isfinite(v).all()
        for kind, query in QUERIES.items():
            ident = digest([native['id'], kind, query, source_protocol['ao_checkpoint']['revision']])
            if not (OUT / 'outputs' / (ident + '.json')).exists():
                jobs.append((ident, row, kind, query, v))
    started = time.monotonic()
    for start in range(0, len(jobs), args.batch_size):
        chunk = jobs[start:start + args.batch_size]
        points = [create_training_datapoint(datapoint_type='FPQA_four_queries', prompt=query,
            target_response='N/A', layer=18, num_positions=len(v), tokenizer=tok, acts_BD=v,
            feature_idx=-1, meta_info={'id': ident}) for ident, row, kind, query, v in chunk]
        captured = []
        original_generate = model.generate

        def tracked_generate(*positional, **kwargs):
            tokens = original_generate(*positional, **kwargs)
            captured.append(tokens[:, kwargs['input_ids'].shape[1]:].detach().cpu())
            return tokens

        model.generate = tracked_generate
        try:
            results = run_evaluation(eval_data=points, model=model, tokenizer=tok,
                submodule=model.get_base_model().model.layers[ac['hook_onto_layer']],
                device=torch.device('cuda:0'), dtype=torch.bfloat16, global_step=-1,
                lora_path=None, eval_batch_size=len(chunk), steering_coefficient=ac['steering_coefficient'],
                generation_kwargs=dict(do_sample=False, max_new_tokens=256, disable_compile=True,
                    pad_token_id=tok.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
                    top_p=None, top_k=None, temperature=None))
        finally:
            model.generate = original_generate
        assert len(captured) == 1 and len(results) == len(chunk)
        eos = model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos)
        for i, ((ident, row, kind, query, v), result) in enumerate(zip(chunk, results)):
            value = captured[0][i].tolist()
            stop = next((j for j, token in enumerate(value) if token in eos), None)
            save(OUT / 'outputs' / (ident + '.json'), dict(id=ident, question_id=row['question_id'],
                source_native_id=row['native_record']['id'],
                source_activation_path=row['native_record']['ao_activation_path'],
                source_activation_shape=list(v.shape), query_kind=kind, query=query,
                explanation=result.api_response.strip(), rendered_prompt=result.prompt,
                completed=stop is not None, finish_reason='eos' if stop is not None else 'length',
                output_tokens=stop + 1 if stop is not None else len(value), reader_gpu=args.gpu,
                reader_batch_size=len(chunk), source_text_provided=False,
                other_query_outputs_provided=False,
                ao_model_revision=source_protocol['ao_checkpoint']['revision'],
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        records = export(rows, protocol)
        complete = len(list((OUT / 'outputs').glob('*.json')))
        save(OUT / 'status.json', dict(state='running', gpu=args.gpu, descriptions=complete,
            complete_questions=len(records), expected=200, elapsed_seconds=time.monotonic() - started))
        print(f'Saved {complete}/200 descriptions; {len(records)}/50 paired questions', flush=True)
    records = export(rows, protocol)
    assert len(records) == 50
    outputs = [v for r in records for v in r['readers'].values()]
    save(OUT / 'generation_summary.json', dict(questions=50, descriptions=len(outputs),
        completed=sum(r['completed'] for r in outputs), truncated=sum(not r['completed'] for r in outputs),
        output_tokens=sum(r['output_tokens'] for r in outputs),
        generation_seconds=time.monotonic() - started, external_model_calls=0,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30))
    save(OUT / 'status.json', dict(state='complete', gpu=args.gpu, descriptions=200,
        complete_questions=50, expected=200, gpu_release='Process exits normally after this write.'))
    print('Four-question CREPE50 extraction complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        save(OUT / 'status.json', dict(state='failed', error=repr(error), gpu=args.gpu))
        raise
