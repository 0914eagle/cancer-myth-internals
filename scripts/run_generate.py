"""Plain responses: the question as the only user turn, no system prompt.

This is Cancer-Myth's "Plain" condition (evaluate.py: `lm(question)`), the
Table 3 protocol for open models. Greedy by default so the PCR labels are a
property of the model, not of a sampling seed; --paper-protocol samples at
0.7 as the paper did.

Output: {result_dir}/{run_name}/{model}/plain_responses.jsonl with
{id, set, question, response, model_id, decoding}. Resumable by id.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config, model_short_name
from src.jsonl import append_jsonl, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--paper-protocol", action="store_true", help="sample at T=0.7 like evaluate.py")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sets", nargs="+", default=None, help="restrict to these sets (fpq nfp tpq)")
    args = parser.parse_args()

    import torch

    from src.modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    gen_cfg = dict(cfg["generation"])
    run_name = args.run_name or cfg.get("run_name", "e1")
    short = model_short_name(cfg)
    out_path = Path(args.output or Path(cfg["paths"]["result_dir"]) / run_name / short / "plain_responses.jsonl")
    ensure_dir(out_path.parent)

    rows = list(read_jsonl(args.questions))
    if args.sets:
        rows = [r for r in rows if r["set"] in set(args.sets)]
    # TPQ questions are NFP questions with an annotation; generate once per text.
    seen_text: set[str] = set()
    unique = []
    for r in rows:
        if r["question"] in seen_text:
            continue
        seen_text.add(r["question"])
        unique.append(r)
    rows = unique
    if args.limit:
        rows = rows[: args.limit]
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"[generate] {len(rows)} questions, {len(done)} done, {len(todo)} to run -> {out_path}", flush=True)
    if not todo:
        return

    torch.manual_seed(int(cfg.get("seed", 17)))
    model_cfg = cfg["source_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    tokenizer = load_tokenizer(
        model_cfg["model_id"], cache_dir=cache_dir, trust_remote_code=model_cfg.get("trust_remote_code", False)
    )
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()

    max_new = int(args.max_new_tokens or gen_cfg.get("max_new_tokens", 512))
    batch_size = int(args.batch_size or gen_cfg.get("batch_size", 8))
    if args.paper_protocol:
        decoding = {"do_sample": True, "temperature": 0.7}
    else:
        decoding = {"do_sample": False}
    gen_kwargs = dict(max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id, **decoding)

    def chat(q: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )

    todo.sort(key=lambda r: len(r["question"]))
    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        enc = tokenizer([chat(r["question"]) for r in batch], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(**enc, **gen_kwargs)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for r, text in zip(batch, texts):
            append_jsonl(
                out_path,
                {
                    "id": r["id"],
                    "set": r["set"],
                    "question": r["question"],
                    "response": text.strip(),
                    "model_id": model_cfg["model_id"],
                    "decoding": json.dumps(decoding, sort_keys=True),
                    "max_new_tokens": max_new,
                },
            )
        done_n = min(start + batch_size, len(todo))
        if done_n % (batch_size * 10) == 0 or done_n == len(todo):
            print(f"[generate] {done_n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
