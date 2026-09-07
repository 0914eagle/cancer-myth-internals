"""Backbone loading, copied from medical_nla/src/modeling.py.

The two guards it carries are the ones that cost hours there: a model that
device_map="auto" silently put on CPU, and a model a fifth of which landed on
meta while every card was empty. Placement is printed and a partial offload is
refused at load time.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import torch_dtype


def load_tokenizer(model_id: str, *, cache_dir: str | None, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def describe_placement(model) -> str:
    counts: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.device)
        counts[key] = counts.get(key, 0) + param.numel()
    total = sum(counts.values()) or 1
    parts = [
        f"{device}: {numel / 1e9:.1f}B params ({100 * numel / total:.0f}%)"
        for device, numel in sorted(counts.items())
    ]
    return " | ".join(parts)


def free_memory_report() -> str:
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device visible"
    parts = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        parts.append(f"cuda:{index} {free / 1e9:.1f}/{total / 1e9:.1f} GB free")
    return " | ".join(parts)


def load_causal_lm(
    model_cfg: dict[str, Any], *, cache_dir: str | None, allow_offload: bool = False
):
    kwargs: dict[str, Any] = {}
    max_memory = model_cfg.get("max_memory")
    if max_memory:
        kwargs["max_memory"] = {
            (int(k) if str(k).isdigit() else k): v for k, v in dict(max_memory).items()
        }
        print(f"[model] max_memory -> {kwargs['max_memory']}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "cuda"),
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        **kwargs,
    )
    placement = describe_placement(model)
    print(f"[model] {model_cfg['model_id']} placement -> {placement}", flush=True)

    offloaded = [
        part for part in placement.split(" | ") if part.startswith(("cpu", "meta", "disk"))
    ]
    if offloaded and not allow_offload:
        raise RuntimeError(
            f"{model_cfg['model_id']} was partly offloaded ({', '.join(offloaded)}). "
            f"Free memory: {free_memory_report()}. Free the GPUs, reduce the visible "
            "set, or pass allow_offload=True to accept a very slow run."
        )
    return model


def decoder_layers(model):
    """The list of transformer blocks, whatever the architecture calls it.

    hidden_states[k] for k >= 1 is the output of decoder_layers(model)[k-1];
    hidden_states[0] is the embedding output. Steering hooks are registered
    on the block whose output a hidden-state index names.
    """
    for attr in ("model", "transformer", "language_model"):
        inner = getattr(model, attr, None)
        if inner is None:
            continue
        for layers_attr in ("layers", "h", "blocks"):
            layers = getattr(inner, layers_attr, None)
            if layers is not None:
                return layers
        # Gemma-3 multimodal wraps the text model one level deeper.
        nested = getattr(inner, "language_model", None)
        if nested is not None and getattr(nested, "layers", None) is not None:
            return nested.layers
    raise AttributeError(f"Cannot find decoder layers on {type(model).__name__}")
