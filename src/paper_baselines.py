"""Matched baseline pipelines, resumable stage cache, and native QA scoring.

Task backends receive only rendered messages, never evaluation annotations.
Heavy libraries are imported lazily so preflight and tests work without a GPU.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FIXED = ('plain', 'balanced', 'cot', 'prewome', 'extract_verify', 'direct_gate', 'cot_gate')
TRAINED = ('tfidf_gate', 'probe_gate', 'gepa_gate', 'gepa', 'gepa_both')
DETECTORS = ('tfidf_gate', 'probe_gate', 'direct_gate', 'cot_gate', 'gepa_gate', 'gepa_both')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().split('\n') if s.strip()]


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def norm(s):
    import unicodedata
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())


def parse_direct(text):
    match = re.fullmatch(r'\s*(Yes|No)[.!]?\s*', text, re.I)
    return None if not match else int(match[1].lower() == 'yes')


def parse_review(text):
    matches = list(re.finditer(r'^Verdict:\s*(Yes|No)\s*$', text, re.M | re.I))
    if len(matches) != 1 or text[matches[0].end():].strip() or not text.lower().startswith('review:'):
        return None, ''
    review = text[len('Review:'):matches[0].start()].strip()
    return (int(matches[0][1].lower() == 'yes'), review) if review else (None, '')


def final_answer(text):
    parts = re.split(r'Final answer:', text, flags=re.I)
    return (parts[1].strip(), False) if len(parts) == 2 and parts[1].strip() else (text, True)


def qa_question(row):
    q = row['question']
    if row['metric'] == 'choice':
        q += '\n\nOptions:\n' + '\n'.join(f'{k}. {v}' for k, v in row['choices'].items())
        q += '\nEnd your response with Answer: <option letter>.'
    elif row['metric'] == 'numeric':
        q += '\nEnd your response with Answer: <final numeric answer>.'
    elif row['metric'] == 'label':
        q += '\nEnd your response with Answer: <' + ' / '.join(row['allowed_labels']) + '>.'
    else:
        raise ValueError('Unknown QA metric')
    return q


def qa_score(row, answer):
    # Strict shared answer extraction; no extra repair/generation calls.
    values = re.findall(r'^\s*Answer:\s*(.+?)\s*$', answer, flags=re.M | re.I)
    if not values:
        return {'correct': False, 'prediction': None, 'format_invalid': True}
    value = values[-1].strip()
    if row['metric'] == 'choice':
        match = re.fullmatch(r'\(?([A-Za-z])\)?[.]?', value)
        pred = match[1].upper() if match else None
    elif row['metric'] == 'label':
        pred = value.lower().rstrip('.')
        if pred not in row['allowed_labels']:
            pred = None
    else:
        from decimal import Decimal, InvalidOperation
        try:
            pred = str(Decimal(value.replace(',', '').strip()).normalize())
            if not Decimal(pred).is_finite():
                pred = None
        except InvalidOperation:
            pred = None
    gold = str(row['gold'])
    if row['metric'] == 'numeric':
        from decimal import Decimal
        gold = str(Decimal(gold.replace(',', '')).normalize())
    return {'correct': pred == gold, 'prediction': pred, 'format_invalid': pred is None}


class WellBridge:
    def __init__(self, config):
        self.config = config

    def __call__(self, **req):
        result = subprocess.run([
            str(ROOT / self.config['template_python']), str(ROOT / 'scripts/paper_well_bridge.py'),
            '--registry', str(ROOT / self.config['registry']),
            '--well-root', str(ROOT / self.config['well_root'])],
            input=json.dumps(req), text=True, capture_output=True, check=True, timeout=60)
        return json.loads(result.stdout)


class CachedTask:
    def __init__(self, backend, out, identity, common):
        self.backend, self.out, self.identity, self.common = backend, Path(out), identity, common
        self.calls = 0

    def __call__(self, messages):
        msgs = [dict(m) for m in messages]
        if [m['role'] for m in msgs] != ['system', 'user']:
            raise ValueError('Expected pinned system/user format')
        msgs[0]['content'] = self.common + '\n\n' + msgs[0]['content']
        request = {'identity': self.identity, 'messages': msgs}
        key = digest(request)
        path = self.out / 'calls' / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['request'] != request:
                raise ValueError('Cache identity mismatch')
            return saved['text']
        if (self.out / 'STOP').exists():
            raise RuntimeError('STOP requested')
        started = time.time()
        text, metadata = self.backend(msgs)
        if not text.strip():
            raise ValueError('Empty generation')
        write_json(path, {'request': request, 'text': text, 'metadata': metadata,
                          'elapsed_seconds': time.time() - started}, immutable=True)
        self.calls += 1
        return text


class Pipeline:
    def __init__(self, registry, call, bridge):
        self.registry, self.call, self.bridge = registry, call, bridge
        self.prompts = registry['prompts']

    def simple(self, prompt, question):
        return self.call([{'role': 'system', 'content': prompt}, {'role': 'user', 'content': question}])

    def well_stage(self, family, template, **kwargs):
        stage = self.bridge(op='render', family=family, template=template, kwargs=kwargs)
        raw = self.call(stage['messages'])
        return raw, self.bridge(op='parse', parser=stage['parser'], text=raw)

    def run(self, method, question, source, candidate=None, gate=None):
        candidate = candidate or {}
        info = {}
        if method in ('plain', 'balanced', 'cot', 'correct_route', 'gepa'):
            prompt = candidate['answer'] if method == 'gepa' else self.prompts[method]
            raw = self.simple(prompt, question)
            answer, bad = final_answer(raw) if method == 'cot' else (raw, False)
            return {'answer': answer, 'raw_answer': raw, 'format_invalid': bad}
        if method in ('prewome', 'extract_verify'):
            few = self.registry['fixed_examples'][source]['examples']
            raw, claims = self.well_stage('prewome', 'PresuppositionExtractionTemplate',
                                         question=question, few_shot_data=few, passages=[])
            info = {'claims': claims, 'extraction_raw': raw}
            if method == 'prewome':
                raw, feedback = self.well_stage('prewome', 'FeedbackActionTemplate',
                    question=question, model_detected_presuppositions=claims,
                    passages=[], few_shot_data=few)
                info.update(review_raw=raw, feedback_action=feedback)
                raw, answer = self.well_stage('prewome', 'FinalAnswerTemplate',
                    question=question, model_feedback_action=feedback, few_shot_data=few)
            else:
                checks = [self.well_stage('atomic', 'LLMCheckTemplate',
                    model_detected_presupposition=c, passages=[], few_shot_data=few) for c in claims]
                info.update(check_raw=[r for r, _ in checks], checks=[v for _, v in checks],
                    invalid_check_indices=[i for i,(r,_) in enumerate(checks)
                                           if not re.search(r'\b(yes|true|no|false)\b',r.lower())])
                raw, answer = self.well_stage('atomic', 'FactCheckFinalAnswerTemplate',
                    question=question, model_detected_presuppositions=claims,
                    factcheck_results=info['checks'], few_shot_data=few)
            return {**info, 'answer': answer, 'raw_answer': raw, 'format_invalid': False}
        review, raw = '', ''
        if method == 'direct_gate':
            raw = self.simple(self.prompts['direct_detector'], question)
            verdict = parse_direct(raw)
        elif method in ('cot_gate', 'gepa_gate', 'gepa_both'):
            raw = self.simple(candidate.get('detector', self.prompts['cot_detector']), question)
            verdict, review = parse_review(raw)
        elif method in ('tfidf_gate', 'probe_gate'):
            if gate is None:
                raise ValueError('Trained gate required')
            probability, threshold = gate(question)
            verdict = int(probability >= threshold)
            info.update(probability=probability, threshold=threshold)
        else:
            raise ValueError('Unknown method: ' + method)
        prompt = self.prompts['correct_route' if verdict == 1 else 'plain']
        if candidate.get('supplement'):
            prompt += '\n\n' + candidate['supplement']
        user = self.registry['stage_inputs']['gate_yes_with_review_template'].format(
            question=question, review=review) if verdict == 1 and review else question
        answer = self.simple(prompt, user)
        return {**info, 'answer': answer, 'raw_answer': answer, 'verdict': verdict,
                'detector_raw': raw, 'review': review, 'invalid_detection': verdict is None,
                'format_invalid': False}


class HFTask:
    def __init__(self, config, model_key):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration
        self.torch, self.config = torch, config
        spec = config['models'][model_key]
        path = Path(config['cache_root']) / ('models--' + spec['model_id'].replace('/', '--')) / 'snapshots' / spec['revision']
        if not path.is_dir():
            raise FileNotFoundError(f'Pinned local checkpoint missing: {path}')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls = Gemma3ForConditionalGeneration if 'gemma-3-' in spec['model_id'] else AutoModelForCausalLM
        self.model = cls.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map='auto', max_memory={i: config['max_memory_per_gpu'] for i in range(torch.cuda.device_count())},
            attn_implementation='sdpa').eval()
        if any(str(v) in ('cpu', 'disk') for v in self.model.hf_device_map.values()):
            raise RuntimeError('CPU/disk offload disallowed; assign more GPUs')
        torch.manual_seed(config['seed'])
        self.layers = sorted(set(math.ceil(spec['n_layers'] * f) for f in (.25, .5, .75, 1.)))

    def inputs(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch = self.tokenizer(text, add_special_tokens=False, return_tensors='pt')
        if batch['input_ids'].shape[1] > self.config['max_input_tokens']:
            raise ValueError('Input exceeds pinned limit; no silent truncation')
        return {k: v.to(self.model.get_input_embeddings().weight.device) for k,v in batch.items()}

    def __call__(self, messages):
        batch = self.inputs(messages)
        n = batch['input_ids'].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(**batch, do_sample=False, max_new_tokens=self.config['max_new_tokens'],
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        tokens = output[0, n:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        ended = len(tokens) > 0 and tokens[-1].item() in eos
        if not ended:
            raise RuntimeError('Generation truncated; retained task has no completed answer')
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip(), {
            'input_tokens': n, 'output_tokens': len(tokens), 'finish_reason': 'eos',
            'device_map': {k: str(v) for k,v in self.model.hf_device_map.items()}}

    def features(self, messages):
        with self.torch.inference_mode():
            output = self.model(**self.inputs(messages), output_hidden_states=True, use_cache=False)
        return {str(layer): output.hidden_states[layer][0, -1].float().cpu().numpy() for layer in self.layers}


def fit_gate(train, dev, kind, feature_fn=None):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    y = np.array([r['label'] for r in train]); yd = np.array([r['label'] for r in dev])
    if set(y) != {0, 1} or set(yd) != {0, 1}:
        raise ValueError('Both labels required in train/dev')
    if kind == 'tfidf_gate':
        vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=50000)
        x = vectorizer.fit_transform([r['question'] for r in train])
        xd = vectorizer.transform([r['question'] for r in dev])
        inputs = {'0': (x, xd, vectorizer)}
    else:
        a = [feature_fn(r['question']) for r in train]; b = [feature_fn(r['question']) for r in dev]
        inputs = {}
        for layer in a[0]:
            scaler = StandardScaler()
            x = scaler.fit_transform(np.array([r[layer] for r in a]))
            xd = scaler.transform(np.array([r[layer] for r in b]))
            inputs[layer] = (x, xd, scaler)
    best = None
    for layer, (x, xd, transform) in inputs.items():
        for c in (.01, .1, 1., 10.):
            clf = LogisticRegression(C=c, class_weight='balanced', max_iter=2000, solver='lbfgs')
            clf.fit(x, y)
            prob = clf.predict_proba(xd)[:,1]
            for threshold in np.arange(.05, 1., .05):
                pred = prob >= threshold
                fpr = float(pred[yd == 0].mean())
                key = (balanced_accuracy_score(yd,pred), -fpr, -c, -int(layer), float(threshold))
                if best is None or key > best[0]:
                    best = (key, {'kind':kind,'layer':layer,'transform':transform,'classifier':clf,
                                  'threshold':float(threshold),'C':c,'dev_balanced_accuracy':float(key[0]),
                                  'train_ids':[r['id'] for r in train],'dev_ids':[r['id'] for r in dev]})
    return best[1]


def gate_predict(artifact, question, feature_fn=None):
    if artifact['kind'] == 'tfidf_gate':
        x = artifact['transform'].transform([question])
    else:
        x = artifact['transform'].transform([feature_fn(question)[artifact['layer']]])
    return float(artifact['classifier'].predict_proba(x)[0,1]), artifact['threshold']


def summarize(records):
    result = {'n': len(records)}
    for label, name in ((1, 'fpq'), (0, 'nfp')):
        rows = [r for r in records if r.get('label') == label]
        scored = [r for r in rows if r.get('rating') is not None]
        result[name] = {'n':len(rows), 'scored':len(scored),
            'well_ge4':sum(r['rating'] >= 4 for r in scored)/len(scored) if scored else None,
            'mean_rating':sum(r['rating'] for r in scored)/len(scored) if scored else None}
        detected = [r for r in rows if 'verdict' in r]
        if detected:
            valid = [r for r in detected if r['verdict'] is not None]
            result[name].update(detection_n=len(detected), invalid=sum(r['verdict'] is None for r in detected),
                yes_rate_valid=sum(r['verdict'] == 1 for r in valid)/len(valid) if valid else None)
    return result
