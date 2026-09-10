"""Residual-stream steering with an optional probe gate.

Layer semantics match `src.extract_activations`: hidden-state index k (k >= 1)
is the output of decoder block k-1, so a direction learned at hidden-state
index k is added by a forward hook on `decoder_layers(model)[k-1]`.

Two policies:

    unconditional   add alpha * v at every generated position (CAA-style)
    gated           run the unsteered prefill, read the gate probe at the gate
                    position, and add alpha * v only when the probe says
                    "false premise" (score > threshold)

The gate has a separate unsteered prefill. The generation pass is steered
from the last prompt token onward, including the first answer token's logits.
Interior hidden-state indices are required: the final hidden state includes
the final normalization and is not the raw final decoder block output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from .modeling import decoder_layers


@dataclass
class SteerSpec:
    hidden_state_index: int
    direction: torch.Tensor  # (d_model,), unit norm
    alpha: float


class _Adder:
    def __init__(self, direction: torch.Tensor, alpha: float, from_position: int):
        self.direction = direction
        self.alpha = alpha
        self.from_position = from_position
        self.active = True
        self.seen = 0

    def __call__(self, module, inputs, output):
        if not self.active or self.alpha == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        seq = hidden.shape[1]
        # During generation with a KV cache each call sees one new token;
        # the first call is the prefill and sees the whole prompt.
        start = max(0, self.from_position - self.seen)
        self.seen += seq
        if start >= seq:
            return output
        vec = (self.alpha * self.direction).to(hidden.dtype).to(hidden.device)
        hidden = hidden.clone()
        hidden[:, start:, :] += vec
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden


class Steerer:
    """Context manager that installs one adder hook on one block."""

    def __init__(self, model, spec: SteerSpec, from_position: int):
        layers = decoder_layers(model)
        block_index = spec.hidden_state_index - 1
        if not 0 <= block_index < len(layers) - 1:
            raise IndexError(
                f"hidden_state_index {spec.hidden_state_index} has no block "
                f"(use 1..{len(layers) - 1}; 0 is embeddings, final index is post-final-norm)"
            )
        self.block = layers[block_index]
        self.adder = _Adder(spec.direction, spec.alpha, from_position)
        self.handle = None

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self.adder)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()


@torch.inference_mode()
def prefill_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, index: int):
    """Hidden state at one hidden-state index for a (batch, seq) prompt."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[index]


def gate_position_from_row(row: dict[str, Any], prompt_len: int) -> int:
    """Where the gate probe reads: the row's resolved token span end, else the
    last prompt token. Rows come from the extraction manifest, whose spans are
    in the unpadded (right-padded) tokenization; generation uses left padding,
    so the caller converts with `prompt_len`."""
    span = row.get("target_token_span")
    if span:
        return int(span[1]) - 1
    return prompt_len - 1


@torch.inference_mode()
def generate_steered(
    model,
    tokenizer,
    chat_texts: list[str],
    *,
    spec: SteerSpec | None,
    gate: Callable[[np.ndarray], bool] | None,
    gate_index: int | None,
    gate_positions: list[int] | None,
    max_new_tokens: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Generate for a batch; returns texts and per-item records (gate score, on/off).

    Batching with a gate: items whose gate is off are generated unsteered,
    items whose gate is on are generated with the hook, as two sub-batches.
    """
    tokenizer.padding_side = "left"
    enc = tokenizer(chat_texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    records: list[dict[str, Any]] = [{} for _ in chat_texts]

    on = [True] * len(chat_texts)
    if gate is not None:
        assert gate_index is not None and gate_positions is not None
        hidden = prefill_hidden(model, enc["input_ids"], enc["attention_mask"], gate_index)
        for i, pos in enumerate(gate_positions):
            # `pos` is in the unpadded sequence; left padding shifts it right.
            pad = int((enc["attention_mask"][i] == 0).sum().item())
            vec = hidden[i, pad + pos].to(torch.float32).cpu().numpy()
            decision = bool(gate(vec))
            on[i] = decision
            records[i]["gate_on"] = decision
        del hidden

    texts = [""] * len(chat_texts)
    for steer_on in (False, True):
        idx = [i for i, flag in enumerate(on) if flag == steer_on]
        if not idx:
            continue
        sub = {k: v[idx] for k, v in enc.items()}
        sub_prompt_len = sub["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        if steer_on and spec is not None and spec.alpha != 0.0:
            # From the last prompt token onward: its residual produces the
            # first generated token's logits, so steering from prompt_len
            # would leave the opening word unsteered. The gate has already
            # been read from a separate, unsteered prefill.
            with Steerer(model, spec, from_position=sub_prompt_len - 1):
                out = model.generate(**sub, **gen_kwargs)
        else:
            out = model.generate(**sub, **gen_kwargs)
        decoded = tokenizer.batch_decode(out[:, sub_prompt_len:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            texts[i] = decoded[j].strip()
            records[i]["steered"] = bool(steer_on and spec is not None and spec.alpha != 0.0)
    _ = prompt_len
    return texts, records
