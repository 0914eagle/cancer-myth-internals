"""Resumable Gemma 4 question gates, answer baselines and premise diagnostics.

Inference receives only question text, generated intermediates, and (in a
separately named diagnostic arm) verbatim source premises. No training occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frontier_claude as io
from scripts import run_luna_direct_cot_gate as gate
from scripts import run_luna_premise_gate as premise
from src.baseline_generation import COT_REASON, FP_ANSWER
from src.pilot import COT_ANSWER, output_lock

MODEL = 'google/gemma-4-12B-it'
REVISION = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
SNAPSHOT = Path('/data1/heejae/hf_cache/hub/models--google--gemma-4-12B-it/snapshots') / REVISION
SOURCE = ROOT / 'results/frontier_cli/sonnet_plain_20260922_v1/evaluation_questions.json'
ANNOTATIONS = ROOT / 'docs/reviews/annotated_premises_2026-09-22/run_v1/source_annotations.jsonl'
ANSWER_STAGES = ('plain', 'zero_shot_cot_one_step', 'premise_review_answer', 'fp_unconditional')
BASE_STAGES = ('direct', 'plain', 'review', 'cot', 'zero_shot_cot_one_step',
               'premise_review_answer', 'fp_unconditional')
DIAGNOSTIC_STAGES = ('extract', 'verify_extracted', 'verify_annotated')
CAPS = {'direct': 256, 'cot': 256, 'review': 1024, 'plain': 2048,
        'zero_shot_cot_one_step': 4096, 'premise_review_answer': 2048,
        'fp_unconditional': 2048, 'extract': 1024,
        'verify_extracted': 2048, 'verify_annotated': 2048}


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
    temp = path.with_suffix('.jsonl.tmp')
    temp.write_text(payload)
    temp.replace(path)


def prepare(out, diagnostics):
    rows = json.loads(SOURCE.read_text())
    assert len(rows) == len({r['id'] for r in rows}) == 732
    assert Counter(r['set'] for r in rows) == {'fpq': 583, 'nfp': 149}
    references = {}
    if diagnostics:
        references = {r['id']: r for r in io.read_rows(ANNOTATIONS)}
        assert all(references[r['id']]['question'] == r['question'] for r in rows if r['id'] in references)
    plan = {'model': MODEL, 'revision': REVISION, 'dtype': 'bfloat16',
            'quantization': None, 'thinking_enabled': False, 'do_sample': False,
            'attention': 'sdpa', 'max_new_tokens': CAPS, 'truncation_policy':
            'Save truncated attempt; extend once to twice the stage cap. Still truncated is invalid.',
            'stages': list(BASE_STAGES + (DIAGNOSTIC_STAGES if diagnostics else ())),
            'answer_stages': list(ANSWER_STAGES), 'expected': {'fpq': 583, 'nfp': 149},
            'systems': {'gate': gate.SYSTEM, 'answer': io.SYSTEM_PROMPT, 'premise': premise.SYSTEM},
            'prompts': {'direct': gate.DIRECT, 'review': gate.REVIEW, 'cot': gate.COT,
                        'plain': '{question}', 'zero_shot_cot_one_step': COT_REASON,
                        'premise_review_answer': COT_ANSWER, 'fp_unconditional': FP_ANSWER,
                        'extract': premise.EXTRACT, 'verify': premise.VERIFY},
            'gate_rule': 'Use the generated JSON boolean; no calibrated operating point; AUROC not reported.',
            'format_policy': 'Accept one complete JSON object with surrounding explanation, or exactly two gate key/value lines; reject multiple objects and inconsistent decisions. Mark recoveries; never regenerate for formatting.',
            'verification_rule': 'Any supplied premise with reported false_probability >= 0.5 enables gate.',
            'limitations': ['Greedy, non-thinking Gemma differs from Luna medium reasoning.',
                           'One-step CoT is a single question-plus-Lets-think-step-by-step call; old Qwen zero-shot CoT had separate reasoning/final calls.',
                           'Annotation arm is privileged-input diagnosis, not a deployable gate or label oracle.',
                           'Extraction and verification outputs are not expert quality labels.',
                           'Behavioral disagreement alone does not establish internal suppression.'],
            'source_sha256': {str(p): io.sha(p) for p in
                             [SOURCE, Path(__file__), Path(gate.__file__), Path(premise.__file__)]},
            'annotation_sha256': io.sha(ANNOTATIONS) if diagnostics else None,
            'missing_annotation_ids': [r['id'] for r in rows if r['id'] not in references] if diagnostics else []}
    io.write(out / 'plan.json', plan, frozen=True)
    io.write(out / 'evaluation_questions.json', rows, frozen=True)
    io.write(out / 'generation_inputs.json', [{'id': r['id'], 'question': r['question']} for r in rows], frozen=True)
    jsonl(out / 'evaluation_questions.jsonl', rows)
    if diagnostics:
        io.write(out / 'annotation_inputs.json',
                 [{'id': r['id'], 'question': r['question'], 'premises': references[r['id']]['premises']}
                  for r in rows if r['id'] in references], frozen=True)
    return plan, rows, references


def prompt_for(stage, row, records, refs):
    qid, question = row['id'], row['question']
    if stage in ('direct', 'plain', 'review', 'zero_shot_cot_one_step', 'fp_unconditional', 'extract'):
        template = {'direct': gate.DIRECT, 'plain': '{question}', 'review': gate.REVIEW,
                    'zero_shot_cot_one_step': COT_REASON, 'fp_unconditional': FP_ANSWER,
                    'extract': premise.EXTRACT}[stage]
        prompt = template.format(question=question)
        extra = {}
    elif stage in ('cot', 'premise_review_answer'):
        previous = records.get(('review', qid))
        if not previous or previous['status'] != 'complete':
            return None
        review = previous['response']
        template = gate.COT if stage == 'cot' else COT_ANSWER
        prompt = template.format(question=question, review=review)
        extra = {'review_sha256': io.digest(review)}
    else:
        if stage == 'verify_annotated':
            if qid not in refs:
                return None
            pp = refs[qid]['premises']
        else:
            previous = records.get(('extract', qid))
            if not previous or previous['status'] != 'complete':
                return None
            pp = previous['result']['premises']
        prompt = premise.VERIFY.format(question=question, premises=json.dumps(pp, ensure_ascii=False))
        extra = {'premises': pp}
    system = (io.SYSTEM_PROMPT if stage in ANSWER_STAGES else
              premise.SYSTEM if stage in DIAGNOSTIC_STAGES else gate.SYSTEM)
    return {'id': qid, 'stage': stage, 'system': system, 'prompt': prompt, **extra}


def structured(response, stage):
    try:
        return premise.decode(response), False
    except ValueError:
        pass
    # Parse only explicit structured output, never infer a verdict from prose.
    decoder = json.JSONDecoder()
    values, cursor = [], 0
    while cursor < len(response):
        start = response.find('{', cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(response[start:])
        except ValueError:
            cursor = start + 1
            continue
        values.append(value)
        cursor = start + end
    if len(values) == 1 and isinstance(values[0], dict):
        return values[0], True
    if not values and stage in ('direct', 'cot'):
        match = re.fullmatch(r'\s*-?\s*has_false_premise:\s*(true|false)\s*\n'
                             r'\s*-?\s*false_probability:\s*([0-9.]+)\s*', response, re.I)
        if match:
            return {'has_false_premise': match[1].lower() == 'true',
                    'false_probability': float(match[2])}, True
    raise ValueError('No single unambiguous structured judgment')


def parse(stage, response, task):
    if not response.strip():
        raise ValueError('Empty response')
    if stage in ('direct', 'cot'):
        value, recovered = structured(response, stage)
        result = gate.decision(json.dumps(value))
        result['format_recovered'] = recovered
        return result
    if stage == 'extract' or stage.startswith('verify_'):
        value, recovered = structured(response, stage)
        result = premise.validate(stage, value, premises=task.get('premises', []))
        if stage.startswith('verify_'):
            result['has_false_premise'] = any(r['false_probability'] >= .5 for r in result['judgments'])
        result['format_recovered'] = recovered
        return result
    return {}


class Runtime:
    def __init__(self, out):
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForMultimodalLM
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1', 'Only GPUs 0,1 authorized'
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, padding_side='left')
        self.model = AutoModelForMultimodalLM.from_pretrained(
            SNAPSHOT, local_files_only=True, dtype=torch.bfloat16, device_map='balanced',
            max_memory={0: '20GiB', 1: '20GiB'}, attn_implementation='sdpa').eval()
        assert all(str(d) not in ('cpu', 'disk', 'meta') for d in self.model.hf_device_map.values())
        self.device = self.model.get_input_embeddings().weight.device
        self.eos = self.model.generation_config.eos_token_id
        self.eos = set(self.eos if isinstance(self.eos, list) else [self.eos])
        io.write(out / 'runtime.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
                 'device_map': self.model.hf_device_map, 'pid': os.getpid(), 'cuda_visible_devices': '0,1',
                 'generation_config': self.model.generation_config.to_dict()})

    def generate(self, tasks, cap):
        texts = [self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': t['system']}, {'role': 'user', 'content': t['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for t in tasks]
        inputs = self.tokenizer(texts, add_special_tokens=False, padding=True,
                                return_tensors='pt').to(self.device)
        width = inputs['input_ids'].shape[1]
        start = time.time()
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False,
                                            use_cache=True, logits_to_keep=1)
        output = []
        for i, sequence in enumerate(generated[:, width:].tolist()):
            end = next((j + 1 for j, token in enumerate(sequence) if token in self.eos), len(sequence))
            ids = sequence[:end]
            raw = self.tokenizer.decode(ids, skip_special_tokens=False)
            response = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            output.append({'response': response, 'raw_response': raw, 'generated_token_ids': ids,
                           'input_tokens': int(inputs['attention_mask'][i].sum()),
                           'output_tokens': len(ids), 'truncated': not any(t in self.eos for t in ids),
                           'max_new_tokens': cap, 'batch_wall_seconds': time.time() - start,
                           'batch_size': len(tasks)})
        return output


def report(out, plan, rows, records, state, stage=None):
    metrics = {}
    for method in ('direct', 'cot', 'verify_extracted', 'verify_annotated'):
        if method not in plan['stages']:
            continue
        metric = {'auroc': None, 'score_kind': 'generated judgment, not token probabilities'}
        for kind in ('fpq', 'nfp'):
            selected = [records.get((method, q['id'])) for q in rows if q['set'] == kind]
            valid = [r for r in selected if r and r['status'] == 'complete']
            positive = sum(r['result']['has_false_premise'] for r in valid)
            metric[kind] = {'valid': len(valid), 'positive': positive,
                            'rate': positive / len(valid) if valid else None}
        metrics[method] = metric
    counts = {s: dict(Counter(r['status'] for (ss, _), r in records.items() if ss == s)) for s in plan['stages']}
    value = {'state': state, 'stage': stage, 'pid': os.getpid(), 'updated_unix': time.time(),
             'expected_per_stage': len(rows), 'counts': counts, 'gate_metrics': metrics}
    io.write(out / 'status.json', value)
    return value


def run(args):
    out = args.out_dir.resolve()
    plan, all_rows, refs = prepare(out, args.diagnostics)
    # Small smoke inventory includes both kinds. Full processing order uses IDs, not labels.
    rows = all_rows
    if args.limit:
        groups = [[r for r in rows if r['set'] == k] for k in ('fpq', 'nfp')]
        rows = [r for pair in zip(*groups) for r in pair][:args.limit]
    records = {}
    plan_hash = io.sha(out / 'plan.json')
    for path in (out / 'records').glob('*/*.json'):
        record = json.loads(path.read_text())
        assert record['plan_hash'] == plan_hash, 'Saved run differs from frozen plan'
        records[record['stage'], record['id']] = record
    with output_lock(out / 'generation'):
        if (out / 'STOP').exists():
            raise ValueError('STOP marker exists')
        report(out, plan, rows, records, 'loading')
        runtime = Runtime(out)
        for stage in plan['stages']:
            tasks = []
            for row in rows:
                if (stage, row['id']) in records:
                    continue
                task = prompt_for(stage, row, records, refs)
                if task is not None:
                    tasks.append(task)
            tasks.sort(key=lambda t: (len(t['prompt']), t['id']))
            batch_size = args.batch_size
            cursor = 0
            while cursor < len(tasks):
                if (out / 'STOP').exists():
                    report(out, plan, rows, records, 'stopped', stage)
                    return
                batch = tasks[cursor:cursor + batch_size]
                report(out, plan, rows, records, 'running', stage)
                try:
                    outputs = runtime.generate(batch, CAPS[stage])
                except runtime.torch.cuda.OutOfMemoryError:
                    runtime.torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    continue
                for task, result in zip(batch, outputs):
                    attempts = [result]
                    if result['truncated']:
                        result = runtime.generate([task], 2 * CAPS[stage])[0]
                        attempts.append(result)
                    record = {**task, **result, 'attempts': attempts, 'plan_hash': plan_hash,
                              'finished_unix': time.time(), 'status': 'complete'}
                    try:
                        if result['truncated']:
                            raise ValueError('Output still truncated after extended generation')
                        record['result'] = parse(stage, result['response'], task)
                    except ValueError as exc:
                        record.update(status='invalid', error=str(exc))
                    io.write(out / 'records' / stage / (task['id'] + '.json'), record)
                    records[stage, task['id']] = record
                cursor += len(batch)
                current = report(out, plan, rows, records, 'running', stage)
                print(json.dumps({'stage': stage, 'counts': current['counts'][stage],
                                  'last_batch_seconds': outputs[0]['batch_wall_seconds']}), flush=True)
            if stage in ANSWER_STAGES:
                answers = [{'id': r['id'], 'question': r['question'], 'method': stage,
                            'response': records[stage, r['id']]['response'], 'status': 'complete'}
                           for r in rows if records.get((stage, r['id']), {}).get('status') == 'complete']
                jsonl(out / 'answers' / (stage + '.jsonl'), answers)
                io.write(out / 'answers' / (stage + '.ready.json'),
                         {'valid': len(answers), 'expected': len(rows), 'generation_finished': True,
                          'answer_sha256': io.sha(out / 'answers' / (stage + '.jsonl'))})
        current = report(out, plan, rows, records, 'generation_finished', None)
        print(json.dumps(current), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--diagnostics', action='store_true')
    args = ap.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        raise ValueError('Invalid size')
    try:
        run(args)
    except Exception as exc:
        io.write(args.out_dir / 'error.json', {'type': type(exc).__name__, 'error': str(exc), 'at': time.time()})
        raise


if __name__ == '__main__':
    main()
