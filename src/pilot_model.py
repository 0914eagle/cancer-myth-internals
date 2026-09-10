"""Model operations for the pilot; imports torch only on GPU execution paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .pilot import (
    COT_ANSWER,
    COT_REVIEW,
    FP_CORRECT,
    FP_DETECT,
    digest,
    file_digest,
    frozen_json,
    parse_identification,
)


def model_identity(model, tokenizer, cfg):
    import torch
    import transformers

    identity = {
        "source_model": cfg["source_model"],
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "chat_template_hash": digest(tokenizer.chat_template),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "implementation_hash": digest(
            {
                name: file_digest(Path(__file__).parent / name)
                for name in ("pilot.py", "pilot_model.py", "steering.py")
            }
        ),
    }
    return json.loads(json.dumps(identity))


def load_model(cfg):
    import torch
    from .modeling import load_causal_lm, load_tokenizer

    torch.manual_seed(int(cfg.get("seed", 17)))
    m = cfg["source_model"]
    tok = load_tokenizer(
        m["model_id"],
        cache_dir=cfg["paths"].get("cache_dir"),
        trust_remote_code=m.get("trust_remote_code", False),
        revision=m.get("revision"),
    )
    tok.padding_side = "left"
    model = load_causal_lm(m, cache_dir=cfg["paths"].get("cache_dir"))
    model.eval()
    return model, tok


def render(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def answer_prefix(tokenizer, question, answer, n):
    """Tokenize a true assistant turn; exclude end-of-turn tokens from pooling."""
    from .extract_activations import token_span_for_char_span

    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = render(tokenizer, question)
    if not text.startswith(prefix) or not text[len(prefix) :].startswith(answer):
        raise ValueError(
            "Assistant template is not prefix-compatible; cannot align response safely"
        )
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = token_span_for_char_span(
        enc["offset_mapping"], len(prefix), len(prefix) + len(answer)
    )
    end = min(end, start + n)
    return enc["input_ids"][:end], (start, end)


def learn_direction(model, tokenizer, cfg, manifest, references, *, layers, prefix_tokens, out_dir):
    """Fit-only paired mean differences and fit-only mean last-prompt norm.

    Cache per-question sufficient statistics to resume extraction. Author
    balancing is done *after* restricting references to fit questions.
    """
    import torch
    from scripts.make_paired_rows import pick_references
    from .modeling import decoder_layers

    out_dir = Path(out_dir)
    layers = sorted(set(layers))
    if (
        prefix_tokens < 1
        or not layers
        or any(not 1 <= k < len(decoder_layers(model)) for k in layers)
    ):
        raise ValueError(
            "Use positive prefix length and interior hidden-state indices; final norm is not a block output"
        )
    fit = [q for q in manifest["questions"] if q["partition"] == "fit"]
    fit_text = {q["question"].strip() for q in fit if q["set"] == "fpq"}
    selected = [r for r in references if r["example_question"].strip() in fit_text]
    chosen, author_counts = pick_references(selected)
    pairs = [
        (q, chosen[q["question"].strip()])
        for q in fit
        if q["set"] == "fpq" and len(chosen.get(q["question"].strip(), {})) == 2
    ]
    if len(pairs) < 2:
        raise ValueError("Fewer than two fit-only correct/follow pairs; inspect reference mapping")
    spec = {
        "manifest_hash": manifest["manifest_hash"],
        "identity": model_identity(model, tokenizer, cfg),
        "layers": layers,
        "prefix_tokens": prefix_tokens,
        "n_pairs": len(pairs),
        "fit_ids": [q["id"] for q in fit],
        "pair_ids": [q["id"] for q, _ in pairs],
        "author_counts": dict(author_counts),
        "scale_definition": "mean fit question last-prompt L2 norm",
    }
    frozen_json(out_dir / "fit.json", spec)
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    def hidden(ids):
        ids = torch.tensor([ids], device=model.device)
        with torch.inference_mode():
            return model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                output_hidden_states=True,
            ).hidden_states

    diffs = []
    for i, (q, pair) in enumerate(pairs):
        path = cache / (digest(["pair", q["id"]]) + ".npz")
        if path.exists():
            with np.load(path) as saved:
                diff = saved["difference"]
        else:
            pooled = []
            for role in ("corr", "follow"):
                ids, (start, end) = answer_prefix(
                    tokenizer, q["question"], pair[role][1], prefix_tokens
                )
                states = hidden(ids)
                pooled.append(
                    np.stack(
                        [states[k][0, start:end].float().mean(0).cpu().numpy() for k in layers]
                    )
                )
                del states
            diff = pooled[0] - pooled[1]
            # Atomic cache replacement: interrupted writes are never reused.
            with path.with_suffix(".tmp").open("wb") as f:
                np.savez(f, difference=diff)
            path.with_suffix(".tmp").replace(path)
        diffs.append(diff)
        print(f"[fit pairs] {i + 1}/{len(pairs)}", flush=True)
    norms = []
    for i, q in enumerate(fit):
        path = cache / (digest(["scale", q["id"]]) + ".json")
        if path.exists():
            value = json.loads(path.read_text())
        else:
            ids = tokenizer(render(tokenizer, q["question"]), add_special_tokens=False)["input_ids"]
            states = hidden(ids)
            value = [float(states[k][0, -1].float().norm()) for k in layers]
            del states
            frozen_json(path, value)
        norms.append(value)
        if i % 25 == 0 or i + 1 == len(fit):
            print(f"[fit scale] {i + 1}/{len(fit)}", flush=True)
    mean = np.mean(diffs, axis=0)
    length = np.linalg.norm(mean, axis=1)
    scales = np.mean(norms, axis=0)
    if (
        not np.isfinite(mean).all()
        or (length <= 1e-8).any()
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
    ):
        raise ValueError("Degenerate direction or residual scale")
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"L{layer}_c_pair{prefix_tokens}"] = (mean[i] / length[i]).astype(np.float32)
        arrays[f"L{layer}_norm_scale"] = np.asarray(scales[i])
    path = out_dir / "directions.npz"
    if path.exists():
        with np.load(path) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[k], v) for k, v in arrays.items()
            ):
                raise ValueError("Existing fitted artifact differs; use a new fit directory")
    else:
        with path.with_suffix(".tmp").open("wb") as f:
            np.savez(f, **arrays)
        path.with_suffix(".tmp").replace(path)
    return spec


def generate_batch(
    model, tokenizer, questions, *, method, max_new_tokens, review_tokens, spec=None
):
    import torch
    from .steering import Steerer

    def generate(prompts, budget, steer=None):
        chats = [render(tokenizer, p) for p in prompts]
        enc = tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        kwargs = dict(
            max_new_tokens=budget,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            if steer is not None and steer.alpha != 0:
                with Steerer(model, steer, from_position=enc["input_ids"].shape[1] - 1):
                    generated = model.generate(**enc, **kwargs)
            else:
                generated = model.generate(**enc, **kwargs)
        generated = generated[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Includes EOS, excludes batch padding; EOS may itself be the pad ID.
        stops = set(np.atleast_1d(model.generation_config.eos_token_id).tolist())
        counts = []
        for row in generated.tolist():
            counts.append(next((i + 1 for i, t in enumerate(row) if t in stops), len(row)))
        return [s.strip() for s in texts], enc["attention_mask"].sum(1).tolist(), counts

    extra = [{} for _ in questions]
    prompts = [q["question"] for q in questions]
    if method in {"fp_identification", "premise_cot"}:
        template = FP_DETECT if method == "fp_identification" else COT_REVIEW
        reviews, in_tokens, out_tokens = generate(
            [template.format(question=p) for p in prompts],
            16 if method == "fp_identification" else review_tokens,
        )
        for i, (q, review) in enumerate(zip(questions, reviews)):
            extra[i] = {
                "review": review,
                "review_input_tokens": in_tokens[i],
                "review_output_tokens": out_tokens[i],
            }
            if method == "fp_identification":
                decision = parse_identification(review)
                extra[i]["identified_false_premise"] = decision
                prompts[i] = (
                    FP_CORRECT.format(question=q["question"]) if decision else q["question"]
                )
            else:
                if not review:
                    raise ValueError("Empty premise review")
                prompts[i] = COT_ANSWER.format(question=q["question"], review=review)
    texts, in_tokens, out_tokens = generate(prompts, max_new_tokens, spec)
    for i in range(len(texts)):
        extra[i].update(answer_input_tokens=in_tokens[i], answer_output_tokens=out_tokens[i])
    return texts, extra
