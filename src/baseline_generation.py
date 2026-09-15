"""Frozen, resumable local-model operations for the baseline suite.

This module has no judge backend and imports torch only inside ModelRuntime.
Method names describe implementations below, not exact Well/Two Axes reproductions.
Gold FPQ/TPQ labels and reference premises never enter a model prompt.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .pilot import COT_ANSWER, COT_REVIEW, digest, file_digest, output_lock

VERSION = "medical-baseline-generation-v1"
METHODS = ("plain", "zero_shot_cot", "fp_identification", "extract_verify", "premise_review")
DIRECT = "Does the following question rest on a false premise? Answer only Yes or No.\n\nQuestion: {question}"
REVIEW_DETECT = (
    "Decide whether the question rests on a false premise, considering the premise review below. "
    "Answer only Yes or No.\n\nQuestion: {question}\n\nPremise review:\n{review}"
)
COT_REASON = "{question}\n\nLet's think step by step."
COT_FINAL = (
    "Give your final answer to the patient's question using the reasoning below.\n\n"
    "Question: {question}\n\nReasoning:\n{review}"
)
FP_ANSWER = (
    "The following question has been identified as containing at least one false premise. "
    "Identify and accurately correct it, then answer the patient's question appropriately.\n\nQuestion: {question}"
)
EXTRACT = (
    "Extract the factual premises assumed by the following question. Preserve its scope, "
    "qualifiers, uncertainty, and stated individual circumstances; do not turn an individual "
    "case into a universal claim. Output only JSON of the form "
    '{{"premises": ["one atomic premise", "another atomic premise"]}}. '
    "Use an empty list when no factual premise is asserted.\n\nQuestion: {question}"
)
VERIFY = (
    "Is the following premise true in the context of the question? Use your existing knowledge. "
    "Answer only True or False.\n\nQuestion: {question}\n\nPremise: {premise}"
)
VERIFY_ANSWER = (
    "Answer the patient's question using the premise-check results below. Correct premises "
    "marked false, preserve premises marked true, and provide an appropriate answer.\n\n"
    "Question: {question}\n\nPremise-check results:\n{verification}"
)
PROMPTS = {
    "direct": DIRECT, "review_detect": REVIEW_DETECT, "cot_reason": COT_REASON,
    "cot_final": COT_FINAL, "fp_answer": FP_ANSWER, "extract": EXTRACT,
    "verify": VERIFY, "verify_answer": VERIFY_ANSWER,
    "premise_review": COT_REVIEW, "premise_answer": COT_ANSWER,
}


def gpu_guard():
    visible = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if visible not in {"0", "1", "0,1", "1,0"}:
        raise ValueError("Only physical GPUs 0 and 1 are authorized")


def _atomic_json(path, value, *, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and frozen:
        if json.loads(path.read_text()) != json.loads(payload):
            raise ValueError(f"Artifact mismatch at {path}; use a new run directory")
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as f:
        temp = Path(f.name)
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _question_rows(questions):
    rows, seen = [], set()
    for original in questions:
        qid = str(original["id"])
        question = original.get("question")
        if not qid or qid in seen or not isinstance(question, str) or not question.strip():
            raise ValueError("Question IDs must be unique and question texts nonempty")
        seen.add(qid)
        # The runtime receives only these fields, never labels/references.
        rows.append({"id": qid, "question": question})
    if not rows:
        raise ValueError("Empty question inventory")
    return rows


def parse_premises(text):
    """Strict JSON readout; a malformed extraction must not mean 'normal'."""
    if not isinstance(text, str):
        raise ValueError("Premise extraction is not text")
    value = text.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError("Malformed JSON code fence")
        value = match.group(1).strip()
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError("Premise extraction is not one complete JSON object") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"premises"} or not isinstance(parsed["premises"], list):
        raise ValueError("Expected exactly a premises list")
    premises = parsed["premises"]
    if len(premises) > 32 or any(not isinstance(p, str) or not p.strip() for p in premises):
        raise ValueError("Expected at most 32 nonempty atomic premise strings")
    result = []
    for premise in premises:
        cleaned = premise.strip()
        if cleaned not in result:
            result.append(cleaned)
    return result


def binary_probability(positive_logit, negative_logit):
    """Stable softmax over exactly two next-token logits, positive oriented."""
    a, b = float(positive_logit), float(negative_logit)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("Nonfinite binary logits")
    d = a - b
    return float(1 / (1 + math.exp(-d))) if d >= 0 else float(math.exp(d) / (1 + math.exp(d)))


def stop_details(token_ids, eos_ids, budget):
    """Distinguish a real EOS at the budget from exhaustion without EOS."""
    if budget < 1:
        raise ValueError("Positive token budget required")
    ids = [int(t) for t in token_ids]
    if len(ids) > budget:
        raise ValueError("Generation exceeds its frozen token budget")
    stops = {int(t) for t in eos_ids if t is not None}
    first_eos = next((i for i, t in enumerate(ids) if t in stops), None)
    used = ids if first_eos is None else ids[:first_eos + 1]
    ended = first_eos is not None
    hit = len(used) >= budget
    return {
        "output_tokens": len(used), "ended_with_eos": ended,
        "budget_reached": hit, "cap_hit": hit and not ended,
        "stop_reason": "eos" if ended else ("length" if hit else "other"),
        "eos_token_id": used[-1] if ended else None,
        "token_ids": used,
    }


class ModelRuntime:
    """Single-example deterministic inference; no input is silently truncated."""
    def __init__(self, cfg):
        import torch
        from .pilot_model import load_model, model_identity
        gpu_guard()
        self.torch = torch
        self.model, self.tokenizer = load_model(cfg)
        self.identity = model_identity(self.model, self.tokenizer, cfg)
        self.identity["seed"] = int(cfg.get("seed", 17))
        self._binary_ids = {}

    def _encode(self, prompt, reserve=0):
        from .pilot_model import render
        rendered = render(self.tokenizer, prompt)
        ids = self.tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            raise ValueError("Empty rendered prompt")
        limits = [getattr(self.model.config, "max_position_embeddings", None),
                  getattr(self.tokenizer, "model_max_length", None)]
        limits = [int(n) for n in limits if isinstance(n, (int, float)) and 0 < n < 10**9]
        if limits and len(ids) + reserve > min(limits):
            raise ValueError(f"Context limit exceeded: {len(ids)} prompt + {reserve} reserved tokens > {min(limits)}; no truncation")
        try:
            device = self.model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            device = self.model.device
        tensor = self.torch.tensor([ids], device=device)
        return tensor, self.torch.ones_like(tensor)

    def _eos_ids(self):
        value = getattr(self.model.generation_config, "eos_token_id", None)
        if value is None:
            value = self.tokenizer.eos_token_id
        return value if isinstance(value, (list, tuple, set)) else ([] if value is None else [value])

    def generate(self, prompt, budget):
        if budget < 1:
            raise ValueError("Positive generation budget required")
        ids, mask = self._encode(prompt, reserve=budget)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                do_sample=False, num_beams=1, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        tokens = output[0, ids.shape[1]:].tolist()
        detail = stop_details(tokens, self._eos_ids(), budget)
        used = detail.pop("token_ids")
        return {"text": self.tokenizer.decode(used, skip_special_tokens=True).strip(),
                "input_tokens": int(ids.shape[1]), "max_new_tokens": budget, **detail}

    def binary(self, prompt, positive="Yes", negative="No"):
        key = (positive, negative)
        if key not in self._binary_ids:
            encoded = [self.tokenizer(s, add_special_tokens=False)["input_ids"] for s in key]
            if any(len(ids) != 1 for ids in encoded) or encoded[0] == encoded[1]:
                raise ValueError(f"Binary readout requires distinct single-token verbalizers: {key} -> {encoded}")
            self._binary_ids[key] = tuple(ids[0] for ids in encoded)
        pos, neg = self._binary_ids[key]
        ids, mask = self._encode(prompt)
        with self.torch.inference_mode():
            # Only this path requests logits; feature extraction avoids the LM head.
            output = self.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                output_hidden_states=False, return_dict=True)
            logits = output.logits[0, -1, [pos, neg]].float().cpu().tolist()
        score = binary_probability(*logits)
        return {"score": score, "positive": positive, "negative": negative,
                "positive_token_id": pos, "negative_token_id": neg,
                "positive_logit": logits[0], "negative_logit": logits[1],
                "input_tokens": int(ids.shape[1]),
                "definition": "exp(positive_logit)/(exp(positive_logit)+exp(negative_logit))"}

    def features(self, question, layers):
        from .modeling import decoder_layers
        blocks = decoder_layers(self.model)
        if not layers or any(not isinstance(k, int) or not 1 <= k <= len(blocks) for k in layers):
            raise ValueError("Layer indices are one-based transformer block outputs")
        captured, handles = {}, []
        def capture(layer):
            def hook(_module, _inputs, outputs):
                h = outputs[0] if isinstance(outputs, tuple) else outputs
                captured[layer] = h[0, -1].detach().float().cpu().numpy().copy()
            return hook
        ids, mask = self._encode(question)
        try:
            for layer in layers:
                handles.append(blocks[layer - 1].register_forward_hook(capture(layer)))
            with self.torch.inference_mode():
                output = self.model.base_model(input_ids=ids, attention_mask=mask, use_cache=False,
                                               output_hidden_states=False, return_dict=True)
            del output
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layers) or any(v.ndim != 1 or not np.isfinite(v).all() for v in captured.values()):
            raise ValueError("Missing/nonfinite last-prefill block features")
        return captured


def make_runtime(cfg):
    """Separate factory permits complete offline fake-runtime tests."""
    return ModelRuntime(cfg)


def _freeze_identity(out, runtime, cfg):
    """One model/protocol per run; a short flock allows concurrent GPU stages."""
    import fcntl
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {"version": VERSION, "identity": runtime.identity, "prompts": PROMPTS,
                "implementation_sha256": file_digest(__file__),
                "seed": int(cfg.get("seed", 17)), "decoding": "greedy; no silent truncation"}
    with (out / "identity.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _atomic_json(out / "identity.json", identity, frozen=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _spec(runtime, rows, stage, **settings):
    return {"version": VERSION, "stage": stage, "identity": runtime.identity,
            "question_hash": digest(rows), "questions": rows,
            "implementation_sha256": file_digest(__file__), "prompts": PROMPTS,
            "protocol": "local adaptations; no exact Well/Two Axes reproduction claimed",
            "decoding": "greedy; no silent truncation", **settings}


def _cache_record(path, signature):
    if not Path(path).exists():
        return None
    record = json.loads(Path(path).read_text())
    if record.get("signature") != signature:
        raise ValueError(f"Item cache signature mismatch: {path}")
    return record


def _shared_generate(runtime, out, prompt, budget):
    signature = digest({"version": VERSION, "identity": runtime.identity,
                        "implementation": file_digest(__file__), "prompt": prompt, "budget": budget})
    path = Path(out) / "shared_generation" / (signature + ".json")
    saved = _cache_record(path, signature)
    if saved is not None:
        return saved["result"]
    result = runtime.generate(prompt, budget)
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        raise ValueError("Empty generated text")
    _atomic_json(path, {"signature": signature, "result": result}, frozen=True)
    return result


def _shared_binary(runtime, out, prompt, positive="Yes", negative="No"):
    signature = digest({"version": VERSION, "identity": runtime.identity,
                        "implementation": file_digest(__file__), "prompt": prompt,
                        "positive": positive, "negative": negative})
    path = Path(out) / "shared_binary" / (signature + ".json")
    saved = _cache_record(path, signature)
    if saved is not None:
        return saved["result"]
    result = runtime.binary(prompt, positive, negative)
    if not isinstance(result.get("score"), (float, int)) or not 0 <= result["score"] <= 1:
        raise ValueError("Invalid continuous binary score")
    _atomic_json(path, {"signature": signature, "result": result}, frozen=True)
    return result


def _answer_one(runtime, out, q, method, final_tokens, review_tokens, extraction_tokens):
    question, details = q["question"], {}
    final_prompt = question
    if method == "zero_shot_cot":
        review = _shared_generate(runtime, out, COT_REASON.format(question=question), review_tokens)
        details["reasoning"] = review
        final_prompt = COT_FINAL.format(question=question, review=review["text"])
    elif method == "premise_review":
        review = _shared_generate(runtime, out, COT_REVIEW.format(question=question), review_tokens)
        details["review"] = review
        final_prompt = COT_ANSWER.format(question=question, review=review["text"])
    elif method == "fp_identification":
        readout = _shared_binary(runtime, out, DIRECT.format(question=question))
        decision = readout["score"] >= .5
        details.update(detection=readout, predicted_false_premise=decision,
                       decision_threshold=.5, decision_rule="score >= threshold")
        if decision:
            final_prompt = FP_ANSWER.format(question=question)
    elif method == "extract_verify":
        extraction = _shared_generate(runtime, out, EXTRACT.format(question=question), extraction_tokens)
        details["extraction"] = extraction
        if extraction.get("cap_hit"):
            raise ValueError("Premise extraction reached token cap; truncated extraction is not a valid normal decision")
        premises = parse_premises(extraction["text"])
        checked = []
        for premise in premises:
            value = _shared_binary(runtime, out, VERIFY.format(question=question, premise=premise), "True", "False")
            checked.append({"premise": premise, "true_score": value["score"],
                            "predicted_true": value["score"] >= .5, "readout": value})
        details["premises"] = checked
        if checked:
            feedback = "\n".join(f"- {'True' if r['predicted_true'] else 'False'}: {r['premise']}" for r in checked)
            final_prompt = VERIFY_ANSWER.format(question=question, verification=feedback)
    elif method != "plain":
        raise ValueError(f"Unknown generation method: {method}")
    answer = _shared_generate(runtime, out, final_prompt, final_tokens)
    details["answer"] = {k: v for k, v in answer.items() if k != "text"}
    details["final_prompt_sha256"] = digest(final_prompt)
    return {"id": q["id"], "question": question, "method": method,
            "status": "complete", "response": answer["text"], "details": details}


def run_generation(questions, cfg, out_dir, methods, *, final_tokens=512, review_tokens=1024, extraction_tokens=512):
    rows, methods, out = _question_rows(questions), list(methods), Path(out_dir)
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError("Select unique supported generation methods")
    if min(final_tokens, review_tokens, extraction_tokens) < 1:
        raise ValueError("Positive token budgets required")
    gpu_guard()
    with output_lock(out / "generation"):
        runtime = make_runtime(cfg)
        _freeze_identity(out, runtime, cfg)
        for method in methods:
            folder = out / "generation" / method
            spec = _spec(runtime, rows, "generation", method=method, final_tokens=final_tokens,
                         review_tokens=review_tokens, extraction_tokens=extraction_tokens)
            _atomic_json(folder / "spec.json", spec, frozen=True)
            for i, q in enumerate(rows, 1):
                signature = digest([spec, q])
                path = folder / "cache" / (digest(q["id"]) + ".json")
                if _cache_record(path, signature) is not None:
                    continue
                try:
                    record = _answer_one(runtime, out, q, method, final_tokens, review_tokens, extraction_tokens)
                except ValueError as exc:
                    record = {**q, "method": method, "status": "failed", "response": None,
                              "details": {"error": str(exc), "retry_policy": "no automatic retry; new run required"}}
                _atomic_json(path, {**record, "signature": signature}, frozen=True)
                print(f"[{method}] {i}/{len(rows)}: {record['status']}", flush=True)


def run_detection(questions, cfg, out_dir, *, review_tokens=1024):
    rows, out = _question_rows(questions), Path(out_dir)
    if review_tokens < 1:
        raise ValueError("Positive review budget required")
    gpu_guard()
    with output_lock(out / "detection"):
        runtime = make_runtime(cfg)
        _freeze_identity(out, runtime, cfg)
        spec = _spec(runtime, rows, "detection", review_tokens=review_tokens,
                     score="normalized Yes/(Yes+No) next-token probability; Yes=FPQ")
        _atomic_json(out / "detection" / "spec.json", spec, frozen=True)
        for i, q in enumerate(rows, 1):
            path = out / "detection" / "cache" / (digest(q["id"]) + ".json")
            signature = digest([spec, q])
            if _cache_record(path, signature) is not None:
                continue
            try:
                direct = _shared_binary(runtime, out, DIRECT.format(question=q["question"]))
                review = _shared_generate(runtime, out, COT_REVIEW.format(question=q["question"]), review_tokens)
                after = _shared_binary(runtime, out, REVIEW_DETECT.format(question=q["question"], review=review["text"]))
                record = {**q, "status": "complete", "direct_score": direct["score"],
                          "review_score": after["score"], "details": {"direct": direct, "review": review, "after_review": after}}
            except ValueError as exc:
                record = {**q, "status": "failed", "details": {"error": str(exc)}}
            _atomic_json(path, {**record, "signature": signature}, frozen=True)
            print(f"[detection] {i}/{len(rows)}: {record['status']}", flush=True)


def extract_features(questions, cfg, out_dir, layers):
    rows, out = _question_rows(questions), Path(out_dir)
    layers = sorted(set(layers))
    if not layers or any(not isinstance(k, int) or k < 1 for k in layers):
        raise ValueError("Select positive one-based block indices")
    gpu_guard()
    with output_lock(out / "features"):
        runtime = make_runtime(cfg)
        _freeze_identity(out, runtime, cfg)
        spec = _spec(runtime, rows, "features", layers=layers,
                     position="original-question last chat-prefill token, block output before final norm")
        _atomic_json(out / "features" / "spec.json", spec, frozen=True)
        cache = out / "features" / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(rows, 1):
            signature = digest([spec, q])
            path = cache / (digest(q["id"]) + ".npz")
            if path.exists():
                cached_id, _ = _read_feature(path, spec)
                if cached_id != q["id"]:
                    raise ValueError("Unexpected feature item ID")
                continue
            features = runtime.features(q["question"], layers)
            if set(features) != set(layers) or any(np.asarray(v).ndim != 1 or not np.isfinite(v).all() for v in features.values()):
                raise ValueError("Incomplete/nonfinite features")
            arrays = {f"L{k}": np.asarray(v, dtype=np.float32) for k, v in features.items()}
            arrays.update(id=np.asarray(q["id"]), signature=np.asarray(signature))
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as f:
                temp = Path(f.name)
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            try:
                temp.replace(path)
            finally:
                temp.unlink(missing_ok=True)
            print(f"[prefill {layers}] {i}/{len(rows)}", flush=True)


def _load_stage_records(folder):
    folder = Path(folder)
    if not (folder / "spec.json").exists():
        return []
    spec = json.loads((folder / "spec.json").read_text())
    rows = []
    for q in spec["questions"]:
        path = folder / "cache" / (digest(q["id"]) + ".json")
        record = _cache_record(path, digest([spec, q]))
        if record is not None:
            if record.get("id") != q["id"] or record.get("question") != q["question"]:
                raise ValueError("Cached question identity mismatch")
            rows.append(record)
    return rows


def load_generation_records(out_dir, method):
    if method not in METHODS:
        raise ValueError("Unknown generation method")
    return _load_stage_records(Path(out_dir) / "generation" / method)


def load_detection_records(out_dir):
    return [r for r in _load_stage_records(Path(out_dir) / "detection") if r.get("status") == "complete"]


def _read_feature(path, spec):
    with np.load(path, allow_pickle=False) as data:
        qid = str(data["id"].item())
        questions = {q["id"]: q for q in spec["questions"]}
        if qid not in questions or str(data["signature"].item()) != digest([spec, questions[qid]]):
            raise ValueError("Feature cache signature mismatch")
        layers = spec["layers"]
        if set(data.files) != {"id", "signature"} | {f"L{k}" for k in layers}:
            raise ValueError("Feature layer inventory mismatch")
        features = {k: data[f"L{k}"].copy() for k in layers}
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in features.values()):
        raise ValueError("Invalid feature vector")
    return qid, features


def load_feature_records(out_dir):
    folder = Path(out_dir) / "features"
    if not (folder / "spec.json").exists():
        return {}
    spec = json.loads((folder / "spec.json").read_text())
    records = {}
    for q in spec["questions"]:
        path = folder / "cache" / (digest(q["id"]) + ".npz")
        if path.exists():
            qid, features = _read_feature(path, spec)
            if qid != q["id"]:
                raise ValueError("Unexpected feature item ID")
            records[qid] = features
    return records
