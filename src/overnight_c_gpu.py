"""Bounded GPU diagnostics for a fixed residual direction; never calls a judge.

Layer k names decoder block k-1's output (interior hidden-state indices).
Teacher-forced likelihood scores response tokens using their *preceding*
positions, including the first response token predicted by the final prompt.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
import time

import numpy as np

from .pilot import digest, file_digest
from .pilot_model import answer_prefix, render


class _Intervention:
    """One sequence, with monotonic KV-cache positions on generation calls."""

    def __init__(self, vector, alpha_abs, first, last, policy, first_k=32):
        if policy not in {"all", "first_k", "prefill", "ablate"}:
            raise ValueError(f"Unknown intervention policy: {policy}")
        if first_k < 1 or first < 0 or last < first:
            raise ValueError("Invalid predictor-position interval")
        self.vector, self.alpha_abs = vector, float(alpha_abs)
        self.first, self.last, self.policy, self.first_k = first, last, policy, first_k
        self.seen = 0
        self.count = 0
        self.delta_sum = self.hnorm_sum = self.ratio_sum = 0.0
        self.ratio_max = 0.0

    def __call__(self, module, inputs, output):
        import torch
        original = output[0] if isinstance(output, tuple) else output
        positions = torch.arange(original.shape[1], device=original.device) + self.seen
        self.seen += original.shape[1]
        mask = (positions >= self.first) & (positions <= self.last)
        if self.policy == "prefill":
            mask &= positions == self.first
        elif self.policy == "first_k":
            mask &= positions < self.first + self.first_k
        if not mask.any() or (self.alpha_abs == 0 and self.policy != "ablate"):
            return output
        if original.shape[0] != 1 or original.shape[-1] != self.vector.numel():
            raise ValueError("Direction dimension/batch does not match residual")
        hidden = original.clone()
        selected = original[:, mask, :]
        unit = self.vector.to(device=original.device, dtype=torch.float32)
        if self.policy == "ablate":
            intended = -(selected.float() @ unit).unsqueeze(-1) * unit
        else:
            intended = self.alpha_abs * unit.expand_as(selected)
        hidden[:, mask, :] = (selected.float() + intended).to(original.dtype)
        # Actual applied delta records rounding in bf16, rather than intent.
        actual = hidden[:, mask, :].float() - selected.float()
        dn, hn = actual.norm(dim=-1), selected.float().norm(dim=-1)
        ratio = dn / hn.clamp_min(1e-12)
        self.count += dn.numel()
        self.delta_sum += float(dn.sum())
        self.hnorm_sum += float(hn.sum())
        self.ratio_sum += float(ratio.sum())
        self.ratio_max = max(self.ratio_max, float(ratio.max()))
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    def audit(self):
        n = self.count
        return {
            "modified_predictor_positions": n,
            "mean_actual_delta_norm": self.delta_sum / n if n else 0.0,
            "mean_original_hidden_norm": self.hnorm_sum / n if n else None,
            "mean_actual_relative_delta": self.ratio_sum / n if n else 0.0,
            "max_actual_relative_delta": self.ratio_max,
            "policy": self.policy,
            "alpha_abs": None if self.policy == "ablate" else self.alpha_abs,
        }


class DiagnosticsRuntime:
    """A loaded causal LM reused for many independent batch-size-one tasks."""

    def __init__(self, model, tokenizer, cfg):
        from .modeling import decoder_layers
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.layers = decoder_layers(model)
        self.model.eval()
        self._directions = {}
        self.deadline = None  # Optional monotonic absolute deadline.

    def _block(self, layer):
        layer = int(layer)
        if not 1 <= layer < len(self.layers):
            raise ValueError("Use interior hidden-state indices 1..n_layers-1")
        return self.layers[layer - 1]

    def _ids(self, ids, task):
        import torch
        limit = int(task.get("max_sequence_tokens", 4096))
        if not ids or len(ids) > limit:
            raise ValueError(f"Sequence has {len(ids)} tokens; allowed 1..{limit}; no question truncation")
        return torch.tensor([ids], device=self.model.device, dtype=torch.long)

    def _answer(self, question, response, task):
        cap = int(task.get("max_answer_tokens", 512))
        if cap < 1:
            raise ValueError("max_answer_tokens must be positive")
        ids, (start, end) = answer_prefix(self.tokenizer, question, response, cap)
        if start < 1 or end <= start:
            raise ValueError("No aligned answer predictor tokens")
        return self._ids(ids, task), start, end

    def _direction(self, spec):
        import torch
        if spec is None:
            raise ValueError("This diagnostic requires a direction descriptor")
        path = Path(spec["path"]).resolve()
        scale_path = Path(spec.get("scale_path", path)).resolve()
        cache_key = digest([spec, file_digest(path), file_digest(scale_path)])
        if cache_key not in self._directions:
            with np.load(path, allow_pickle=False) as data:
                vector = np.asarray(data[spec["key"]], dtype=np.float32)
            with np.load(scale_path, allow_pickle=False) as data:
                scale_arr = np.asarray(data[spec["scale_key"]])
            norm = np.linalg.norm(vector)
            if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(norm, 1, atol=1e-4):
                raise ValueError("Direction must be a finite unit vector")
            config = getattr(self.model, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
            if hidden_size is not None and len(vector) != hidden_size:
                raise ValueError("Direction dimension differs from model hidden size")
            if scale_arr.size != 1:
                raise ValueError("Residual scale must be scalar")
            scale = float(scale_arr.item())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Residual scale must be finite and positive")
            if spec.get("random_seed") is not None:
                vector = np.random.default_rng(int(spec["random_seed"])).standard_normal(vector.shape).astype(np.float32)
                vector /= np.linalg.norm(vector)
            self._block(spec["layer"])
            self._directions[cache_key] = (torch.from_numpy(vector.copy()), scale)
        return self._directions[cache_key]

    @contextmanager
    def _hook(self, spec, task, first, last):
        vec, scale = self._direction(spec)
        alpha = float(task.get("alpha", 0.0))
        if not np.isfinite(alpha) or not np.isfinite(alpha * scale):
            raise ValueError("alpha must be finite")
        hook = _Intervention(vec, alpha * scale, first, last,
                             task.get("policy", "all"), int(task.get("first_k", 32)))
        handle = self._block(spec["layer"]).register_forward_hook(hook)
        try:
            yield hook
        finally:
            handle.remove()

    @staticmethod
    def _summary(logp):
        if not len(logp):
            return {"mean_logp": None, "sum_logp": 0.0, "token_count": 0}
        return {"mean_logp": float(np.mean(logp)), "sum_logp": float(np.sum(logp)), "token_count": len(logp)}

    def _likelihood(self, question, response, task):
        import torch
        ids, start, end = self._answer(question, response, task)
        with self._hook(task["direction"], task, start - 1, end - 2) as intervention:
            with torch.inference_mode():
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                logp = []
                # Only 32 x vocabulary entries are upcast to fp32 at once.
                for a in range(start, end, 32):
                    b = min(end, a + 32)
                    logits = output.logits[0, a - 1:b - 1, :].float()
                    target = ids[0, a:b]
                    selected = logits.gather(1, target[:, None]).squeeze(1)
                    values = selected - torch.logsumexp(logits, dim=-1)
                    logp.extend(values.cpu().tolist())
                del output
        if not np.isfinite(logp).all():
            raise ValueError("Non-finite teacher-forced likelihood")
        result = self._summary(logp)
        result.update(prefix32=self._summary(logp[:32]), rest=self._summary(logp[32:]))
        return result, intervention.audit()

    def _pooled(self, question, response, layers, task):
        import torch
        ids, start, end = self._answer(question, response, task)
        captured, handles = {}, []
        def capture(layer):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                answer = h[0, start:end].float()
                captured[layer] = {
                    "prefix32": answer[:32].mean(0).cpu().numpy(),
                    "full": answer.mean(0).cpu().numpy(),
                }
            return hook
        try:
            for layer in layers:
                handles.append(self._block(layer).register_forward_hook(capture(layer)))
            with torch.inference_mode():
                # Model output is immediately released; only pooled hooks survive.
                out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                del out
        finally:
            for h in handles:
                h.remove()
        if set(captured) != set(layers):
            raise ValueError("One or more requested layers never executed")
        return captured, end - start

    def _project(self, task):
        vector, _ = self._direction(task["direction"])
        vec = vector.numpy()
        layer = int(task["direction"]["layer"])
        pooled, n = self._pooled(task["question"], task["response"], [layer], task)
        result = {"answer_tokens": n}
        for name in ("prefix32", "full"):
            h = pooled[layer][name]
            if h.shape != vec.shape or not np.isfinite(h).all():
                raise ValueError("Invalid pooled hidden state")
            norm, dot = float(np.linalg.norm(h)), float(h @ vec)
            result[name] = {"dot": dot, "cosine": dot / norm if norm > 0 else None,
                            "norm": norm, "token_count": min(32, n) if name == "prefix32" else n}
        return result

    def _generate(self, task):
        import torch
        prompt = task.get("prompt", task["question"])
        rendered = render(self.tokenizer, prompt)
        raw_ids = self.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        ids = self._ids(raw_ids, task)
        budget = int(task.get("max_new_tokens", 512))
        if budget < 1 or len(raw_ids) + budget > int(task.get("max_sequence_tokens", 4096)):
            raise ValueError("Generation exceeds sequence budget; prompt is never truncated")
        kwargs = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
                      pad_token_id=self.tokenizer.pad_token_id)
        deadline = self.deadline
        if task.get("remaining_seconds") is not None:
            local_deadline = time.monotonic() + float(task["remaining_seconds"])
            deadline = min(deadline, local_deadline) if deadline is not None else local_deadline
        if deadline is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class Deadline(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return time.monotonic() >= deadline
            kwargs["stopping_criteria"] = StoppingCriteriaList([Deadline()])
        ctx = self._hook(task["direction"], task, len(raw_ids) - 1, len(raw_ids) + budget - 2) if task.get("direction") else nullcontext(None)
        with ctx as hook:
            with torch.inference_mode():
                output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
        answer_ids = output[0, len(raw_ids):].tolist()
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        stops = {int(v) for v in np.atleast_1d(eos).tolist() if v is not None}
        count = next((i + 1 for i, value in enumerate(answer_ids) if value in stops), len(answer_ids))
        answer_ids = answer_ids[:count]
        response = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        return {"response": response, "output_tokens": count,
                "hit_token_cap": count >= budget and (not answer_ids or answer_ids[-1] not in stops),
                "stopped_by_deadline": deadline is not None and time.monotonic() >= deadline,
                "prompt_tokens": len(raw_ids), "steering_audit": hook.audit() if hook else None}

    def _extract(self, task):
        layers = sorted({int(k) for k in task["layers"]})
        if not layers or not task["pairs"]:
            raise ValueError("Extraction requires layers and at least one pair")
        for layer in layers:
            self._block(layer)
        pairs = task["pairs"]
        if len({p["id"] for p in pairs}) != len(pairs):
            raise ValueError("Duplicate extraction pair IDs")
        cache = Path(task["cache_dir"]) if task.get("cache_dir") else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)
        identity = {"source_model": self.cfg.get("source_model"),
                    "revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
                    "chat_template": getattr(self.tokenizer, "chat_template", None),
                    "implementation": file_digest(Path(__file__))}
        all_diffs, token_counts = [], []
        for index, pair in enumerate(pairs):
            fingerprint = digest([pair, layers, identity, int(task.get("max_answer_tokens", 512)), int(task.get("max_sequence_tokens", 4096))])
            path = cache / f"{fingerprint}.npz" if cache else None
            if path and path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    diff = saved["difference"]
                    counts = saved["token_counts"].tolist()
            else:
                a, na = self._pooled(pair["question"], pair["positive"], layers, task)
                b, nb = self._pooled(pair["question"], pair["negative"], layers, task)
                diff = np.stack([[a[k][p] - b[k][p] for p in ("prefix32", "full")] for k in layers])
                counts = [na, nb]
                if path:
                    self._atomic_npz(path, {"difference": diff, "token_counts": np.asarray(counts)})
            if diff.ndim != 3 or diff.shape[:2] != (len(layers), 2) or not np.isfinite(diff).all():
                raise ValueError("Invalid paired extraction/cache")
            all_diffs.append(diff)
            token_counts.append(counts)
            print(f"[overnight extract] {index + 1}/{len(pairs)}", flush=True)
        means = np.mean(all_diffs, axis=0)
        arrays, norms = {}, {}
        for i, layer in enumerate(layers):
            for j, pooling in enumerate(("prefix32", "full")):
                key = f"L{layer}_{pooling}"
                norm = float(np.linalg.norm(means[i, j]))
                if not np.isfinite(norm) or norm <= 1e-8:
                    raise ValueError(f"Degenerate paired difference: {key}")
                arrays[key] = (means[i, j] / norm).astype(np.float32)
                norms[key] = norm
        path = Path(task["output_path"])
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                    raise ValueError("Existing extracted direction differs; use a new output path")
        else:
            self._atomic_npz(path, arrays)
        return {"output_path": str(path), "sha256": file_digest(path), "n_pairs": len(pairs),
                "pair_ids": [p["id"] for p in pairs], "layers": layers,
                "unnormalized_direction_norms": norms, "answer_token_counts": token_counts,
                "pooling": "answer hidden states: first min(32,n) and all capped answer tokens",
                "scale": "external fit residual scales; this file contains unit directions only"}

    @staticmethod
    def _atomic_npz(path, arrays):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
        tmp.replace(path)

    def execute(self, task):
        kind = task["kind"]
        if kind == "dose":
            positive, pa = self._likelihood(task["question"], task["positive"], task)
            negative, na = self._likelihood(task["question"], task["negative"], task)
            result = {"positive": positive, "negative": negative,
                      "mean_logp_margin": positive["mean_logp"] - negative["mean_logp"],
                      "sum_logp_margin": positive["sum_logp"] - negative["sum_logp"],
                      "steering_audit": {"positive": pa, "negative": na}}
        elif kind == "project":
            result = self._project(task)
        elif kind == "generate":
            result = self._generate(task)
        elif kind == "extract":
            return self._extract(task)
        else:
            raise ValueError(f"Unknown diagnostic kind: {kind}")
        result.update(kind=kind, direction=task.get("direction"), alpha=task.get("alpha", 0.0), policy=task.get("policy", "all"))
        return result
