"""Read hidden states out of the backbone at chosen layers and token positions.

Adapted from medical_nla/src/extract_activations.py and kept layout-compatible
with it, so medical_nla's `src.run_nla` and `score_reconstruction_mse.py` can
be pointed at a manifest written here:

    {activation_dir}/{run_name}/
        config.yaml
        run.json
        layer24/
            last_token/manifest.jsonl, shard_037/{id}.pt
            last_subtoken/manifest.jsonl, ...
            span_mean/manifest.jsonl, ...

What it keeps from there: one forward pass per prompt (rows sharing a prompt
share a pass), one pass for every layer (`output_hidden_states=True`), float32
storage, right padding so resolved spans need no shifting, resume by id.

What it adds:

* `--layers all` -- every hidden-state index 0..n_layers, from the config's
  `source_model.n_layers`. Index 0 is the embedding output; index n_layers is
  the post-final-norm state and lives on a different scale.
* position_mode `assistant_prefix` -- the first `prefix_tokens` tokens of a
  teacher-forced assistant turn (position E, the response's opening).
"""

from __future__ import annotations

import argparse
import json
import shutil
import zlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

from .config import ensure_dir, load_config
from .jsonl import append_jsonl, read_jsonl

PASSTHROUGH_FIELDS = [
    "set",
    "label_false_premise",
    "category",
    "cancer",
    "from_model",
    "premise_text",
    "premise_span",
    "align_score",
    "align_method",
    "tpq_split",
    "position_family",
    "prefix_tokens",
    "pcr",
    "nfp_score",
    "judge_parsed",
    "pair_role",
    "pair_author",
    "pair_id",
]

SPAN_SELECTION = "span"


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _encode_text(tokenizer, text: str) -> dict[str, Any]:
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return {
        "text": text,
        "input_ids": list(encoded["input_ids"]),
        "offset_mapping": [tuple(pair) for pair in encoded["offset_mapping"]],
        "n_tokens": len(encoded["input_ids"]),
    }


def encode_chat(tokenizer, prompt: str) -> dict[str, Any]:
    """Always this path, never apply_chat_template(tokenize=True): rows sharing
    a prompt must share one tokenization. The template emits its own BOS, so
    add_special_tokens=False avoids a second one."""
    return _encode_text(tokenizer, chat_text(tokenizer, prompt))


def encode_messages(tokenizer, messages: list[dict[str, str]]) -> dict[str, Any]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = _encode_text(tokenizer, text)
    # Where the assistant content starts: the prompt-only rendering is a prefix
    # of the full transcript rendering for every template we use.
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    if not text.startswith(prefix):
        # Some templates differ by trailing whitespace; fall back to locating
        # the assistant content itself.
        idx = text.rfind(messages[-1]["content"][:40])
        prefix_len = idx if idx >= 0 else len(prefix)
    else:
        prefix_len = len(prefix)
    encoded["assistant_char_start"] = prefix_len
    return encoded


def encode_row(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("chat_messages")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {row.get('id')} has invalid chat_messages.")
        return encode_messages(tokenizer, messages)
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError(f"Row {row.get('id')} has no prompt.")
    return encode_chat(tokenizer, str(prompt))


def substring_char_span(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    if not needle:
        raise ValueError("target_text must be non-empty.")
    text_l, needle_l = text.lower(), needle.lower()
    if occurrence == -1:
        start = text_l.rfind(needle_l)
        if start < 0:
            raise ValueError(f"target_text {needle!r} not found in chat text.")
    elif occurrence >= 0:
        start, search_from = -1, 0
        for _ in range(occurrence + 1):
            start = text_l.find(needle_l, search_from)
            if start < 0:
                raise ValueError(f"target_text {needle!r} not found in chat text.")
            search_from = start + len(needle_l)
    else:
        raise ValueError("target_text_occurrence must be -1 or non-negative.")
    return start, start + len(needle)


def token_span_for_char_span(
    offset_mapping: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    positions = [
        idx
        for idx, (tok_start, tok_end) in enumerate(offset_mapping)
        if tok_start != tok_end and tok_start < end and tok_end > start
    ]
    if not positions:
        raise ValueError(f"No tokens overlap char span {start}:{end}.")
    return positions[0], positions[-1] + 1


def resolve_positions(row: dict[str, Any], encoded: dict[str, Any], activation_cfg: dict[str, Any]):
    """(span, selections): span is [start, end) in the unpadded sequence;
    selections is the list of reductions this row wants at every layer
    (empty means 'use the row's or config's target_text_strategy')."""
    mode = row.get("position_mode") or activation_cfg.get("position_mode", "last_token")
    n_tokens = int(encoded["n_tokens"])

    if mode == "last_token":
        return (n_tokens - 1, n_tokens), ["last_token"]

    if mode == "token_index":
        pos = int(row["target_token_position"])
        if not 0 <= pos < n_tokens:
            raise ValueError(f"Row {row.get('id')} token index {pos} outside 0..{n_tokens - 1}.")
        return (pos, pos + 1), ["token_index"]

    if mode == "token_span":
        start, end = (int(v) for v in row["target_token_span"])
        if not 0 <= start < end <= n_tokens:
            raise ValueError(f"Row {row.get('id')} span outside 0..{n_tokens}.")
        return (start, end), ["token_span"]

    if mode == "target_text":
        target_text = row.get("target_text")
        if target_text is None:
            raise ValueError(f"Row {row.get('id')} uses target_text mode but has no target_text.")
        occurrence = int(
            row.get("target_text_occurrence", activation_cfg.get("target_text_occurrence", 0))
        )
        char_start, char_end = substring_char_span(encoded["text"], str(target_text), occurrence)
        start, end = token_span_for_char_span(encoded["offset_mapping"], char_start, char_end)
        row["target_char_span"] = [char_start, char_end]
        return (start, end), []

    if mode == "assistant_prefix":
        if "assistant_char_start" not in encoded:
            raise ValueError(f"Row {row.get('id')} uses assistant_prefix but has no chat_messages.")
        n = int(row.get("prefix_tokens", 5))
        char_start = int(encoded["assistant_char_start"])
        first = next(
            (
                idx
                for idx, (s, e) in enumerate(encoded["offset_mapping"])
                if s != e and s >= char_start
            ),
            None,
        )
        if first is None:
            raise ValueError(f"Row {row.get('id')}: no tokens after the assistant start.")
        end = min(first + n, n_tokens)
        row["target_char_span"] = [char_start, encoded["offset_mapping"][end - 1][1]]
        return (first, end), []

    raise ValueError(f"Unsupported position_mode: {mode}")


def select_from_span(seq_hidden: "torch.Tensor", span: tuple[int, int], selection: str):
    start, end = span
    if selection in {"last_token", "token_index", "last_subtoken"}:
        return seq_hidden[end - 1], str(end - 1)
    if selection == "first_subtoken":
        return seq_hidden[start], str(start)
    if selection == "span_mean":
        return seq_hidden[start:end].mean(dim=0), f"{start}:{end}:mean"
    if selection in {"span", "token_span"}:
        return seq_hidden[start:end], f"{start}:{end}"
    raise ValueError(f"Unsupported selection: {selection}")


def shard_dir_name(row_id: str, n_shards: int) -> str:
    return f"shard_{zlib.crc32(row_id.encode('utf-8')) % n_shards:03d}"


class SelectionWriter:
    def __init__(self, root: Path, layer: int, selection: str, *, n_shards: int, resume: bool):
        self.dir = root / f"layer{layer:02d}" / selection
        self.manifest = self.dir / "manifest.jsonl"
        self.layer, self.selection, self.n_shards = layer, selection, n_shards
        self.seen: set[str] = set()
        if resume and self.manifest.exists():
            self.seen = {str(row["id"]) for row in read_jsonl(self.manifest)}
        elif self.dir.exists():
            shutil.rmtree(self.dir)
        ensure_dir(self.dir)

    def has(self, row_id: str) -> bool:
        return row_id in self.seen

    def write(self, row_id: str, tensor: "torch.Tensor", manifest_row: dict[str, Any]) -> None:
        import torch

        shard = ensure_dir(self.dir / shard_dir_name(row_id, self.n_shards))
        path = shard / f"{row_id}.pt"
        torch.save(tensor, path)
        manifest_row = dict(manifest_row)
        manifest_row.update(
            activation_path=str(path),
            layer=self.layer,
            hidden_state_index=self.layer,
            selection=self.selection,
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
        )
        append_jsonl(self.manifest, manifest_row)
        self.seen.add(row_id)


def group_by_prompt(rows: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        messages = row.get("chat_messages")
        if messages is not None:
            key = "messages:" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        else:
            prompt = row.get("prompt")
            if not prompt:
                raise ValueError(f"Row {row.get('id')} has no prompt.")
            key = str(prompt)
        groups.setdefault(key, []).append(row)
    return groups


def resolve_layers(spec: list[str] | None, cfg: dict[str, Any]) -> list[int]:
    n_layers = int(cfg["source_model"]["n_layers"])
    if spec is None:
        spec = [str(cfg.get("activation", {}).get("layers", "all"))]
    if len(spec) == 1 and str(spec[0]).lower() == "all":
        return list(range(0, n_layers + 1))
    layers = sorted({int(v) for v in spec})
    if layers and layers[-1] > n_layers:
        raise SystemExit(f"layer {layers[-1]} > n_layers {n_layers} (hidden_states has n_layers+1)")
    return layers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Hidden-state indices, or 'all' (0..n_layers from the config). Default: config.",
    )
    parser.add_argument("--span-layers", nargs="+", type=int, default=[])
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=["first_subtoken", "last_subtoken", "span_mean"],
        help="Reductions stored for every target_text/assistant_prefix row.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import torch

    from .modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    activation_cfg = cfg.get("activation") or {}
    run_name = args.run_name or cfg.get("run_name", "e1")
    layers = resolve_layers(args.layers, cfg)
    span_layers = sorted(set(args.span_layers) & set(layers))
    batch_size = int(args.batch_size or activation_cfg.get("batch_size") or 8)

    out_dir = ensure_dir(args.output_dir or (Path(cfg["paths"]["activation_dir"]) / run_name))
    shutil.copy2(args.config, out_dir / "config.yaml")

    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    counts = Counter(str(row.get("id")) for row in rows)
    repeated = [row_id for row_id, n in counts.most_common(5) if n > 1]
    if repeated:
        raise SystemExit(f"duplicate row ids in input, e.g. {repeated}")
    groups = group_by_prompt(rows)
    if args.limit_prompts:
        groups = OrderedDict(list(groups.items())[: args.limit_prompts])
        rows = [row for group in groups.values() for row in group]
    print(
        f"[input] {len(rows):,} rows over {len(groups):,} distinct prompts "
        f"({len(rows) / max(len(groups), 1):.1f} rows per forward pass)",
        flush=True,
    )

    torch.manual_seed(int(cfg.get("seed", 17)))
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    tokenizer.padding_side = "right"

    encodings: dict[str, dict[str, Any]] = {}
    plans: list[tuple[str, list[dict[str, Any]]]] = []
    for input_key, group in groups.items():
        encoded = encode_row(tokenizer, group[0])
        encodings[input_key] = encoded
        planned = []
        for row in group:
            row = dict(row)
            span, selections = resolve_positions(row, encoded, activation_cfg)
            if not selections:
                selections = list(
                    args.strategies
                    or [
                        row.get("target_text_strategy")
                        or activation_cfg.get("target_text_strategy", "last_subtoken")
                    ]
                )
            row["target_token_span"] = [int(span[0]), int(span[1])]
            planned.append({"row": row, "span": span, "selections": selections})
        plans.append((input_key, planned))

    token_counts = [int(e["n_tokens"]) for e in encodings.values()]
    print(
        f"[tokens] prompt length min {min(token_counts)} / mean "
        f"{sum(token_counts) / len(token_counts):.0f} / max {max(token_counts)}",
        flush=True,
    )

    selection_names = sorted({s for _, planned in plans for p in planned for s in p["selections"]})
    writers: dict[tuple[int, str], SelectionWriter] = {}
    for layer in layers:
        wanted = list(selection_names)
        if layer in span_layers:
            wanted.append(SPAN_SELECTION)
        for selection in wanted:
            writers[(layer, selection)] = SelectionWriter(
                out_dir, layer, selection, n_shards=args.shards, resume=args.resume
            )
    print(f"[layers] {layers[0]}..{layers[-1]} ({len(layers)}) x selections {selection_names}", flush=True)

    def outputs_for(planned: dict[str, Any]) -> list[tuple[int, str]]:
        keys = [(layer, sel) for layer in layers for sel in planned["selections"]]
        if planned["row"].get("position_mode") in {"target_text", "assistant_prefix"}:
            keys += [(layer, SPAN_SELECTION) for layer in span_layers]
        return keys

    if args.resume:
        pending = [
            (prompt, planned)
            for prompt, planned in plans
            if any(
                not writers[key].has(str(p["row"]["id"])) for p in planned for key in outputs_for(p)
            )
        ]
        if len(pending) != len(plans):
            print(f"[resume] {len(plans) - len(pending):,} prompts complete, {len(pending):,} to run")
        plans = pending
    if not plans:
        print("[done] nothing to do; every requested output already exists")
        return

    plans.sort(key=lambda item: encodings[item[0]]["n_tokens"])

    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    n_written = 0

    for batch_start in range(0, len(plans), batch_size):
        batch = plans[batch_start : batch_start + batch_size]
        padded = tokenizer.pad(
            [{"input_ids": encodings[prompt]["input_ids"]} for prompt, _ in batch],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in padded.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if max(layers) >= len(hidden_states):
            raise IndexError(f"layer {max(layers)} out of range for {len(hidden_states)} hidden states")

        for layer in layers:
            layer_hidden = hidden_states[layer].to("cpu", torch.float32)
            for index, (input_key, planned_rows) in enumerate(batch):
                seq_hidden = layer_hidden[index]
                for planned in planned_rows:
                    row, span = planned["row"], planned["span"]
                    row_id = str(row["id"])
                    selections = list(planned["selections"])
                    if layer in span_layers and row.get("position_mode") in {
                        "target_text",
                        "assistant_prefix",
                    }:
                        selections.append(SPAN_SELECTION)
                    for selection in selections:
                        writer = writers[(layer, selection)]
                        if writer.has(row_id):
                            continue
                        tensor, position = select_from_span(seq_hidden, span, selection)
                        manifest_row = {
                            "id": row_id,
                            "base_id": row.get("base_id", row_id),
                            "prompt": row.get("prompt"),
                            "chat_text": encodings[input_key]["text"],
                            "model_id": model_cfg["model_id"],
                            "position": position,
                            "position_mode": row.get("position_mode")
                            or activation_cfg.get("position_mode"),
                            "target_text": row.get("target_text"),
                            "target_text_strategy": selection,
                            "target_token_span": row.get("target_token_span"),
                            "target_char_span": row.get("target_char_span"),
                            "prompt_token_count": int(encodings[input_key]["n_tokens"]),
                        }
                        for field in PASSTHROUGH_FIELDS:
                            if field in row:
                                manifest_row[field] = row.get(field)
                        writer.write(row_id, tensor.clone(), manifest_row)
                        n_written += 1
            del layer_hidden
        del outputs, hidden_states

        done = min(batch_start + batch_size, len(plans))
        if done % (batch_size * 25) == 0 or done == len(plans):
            print(f"[extract] {done:,}/{len(plans):,} prompts | {n_written:,} tensors written", flush=True)

    run_record = {
        "run_name": run_name,
        "inputs": list(args.input),
        "model_id": model_cfg["model_id"],
        "layers": layers,
        "span_layers": span_layers,
        "selections": selection_names,
        "n_rows": len(rows),
        "n_prompts": len(groups),
        "batch_size": batch_size,
        "store_dtype": "float32",
        "tensors_written": n_written,
        "manifests": {
            f"layer{layer:02d}/{selection}": str(writer.manifest)
            for (layer, selection), writer in writers.items()
        },
    }
    (out_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(f"[done] {n_written:,} tensors under {out_dir}")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
