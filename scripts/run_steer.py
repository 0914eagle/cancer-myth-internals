"""E1 step 3 / E2: steer with the C direction, unconditionally or behind the A gate.

    python scripts/run_steer.py --config configs/llama31_8b.yaml \
        --questions $DATA/e1_rows_v1/questions.jsonl \
        --directions $ART/results/e1/llama31_8b/direction_c/directions.npz \
        --layer 16 --alpha 4 --policy gated --gate-layer 16 --gate-threshold 0.0 \
        --output $ART/results/e2/llama31_8b/gated_L16_a4.jsonl

Then score with run_judge.py and summarize with summarize_judge.py; the
comparison that matters is PCR on fpq against NFP on nfp for
policy=unconditional vs policy=gated at the same alpha.

The gate reads the A logistic probe stored in directions.npz at the last
prompt token (position D) of the unsteered prefill; alpha scales the unit C
direction in units of that layer's typical residual norm (--alpha-mode norm)
or absolutely (--alpha-mode abs).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config
from src.jsonl import append_jsonl, read_jsonl
from src.probes import probe_score
from src.steering import SteerSpec, generate_steered, prefill_hidden


def subset(rows: list[dict], n_fpq: int, n_nfp: int, n_tpq: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for s, n in (("fpq", n_fpq), ("nfp", n_nfp), ("tpq", n_tpq)):
        pool = [r for r in rows if r["set"] == s]
        rng.shuffle(pool)
        out += pool[:n] if n >= 0 else pool
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--direction-key", default="c_e", choices=["c_e", "c_d"])
    parser.add_argument("--layer", type=int, required=True, help="hidden-state index to steer at")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--alpha-mode", choices=["norm", "abs"], default="norm")
    parser.add_argument("--policy", choices=["none", "unconditional", "gated"], required=True)
    parser.add_argument("--gate-layer", type=int, default=None)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--n-fpq", type=int, default=100)
    parser.add_argument("--n-nfp", type=int, default=100)
    parser.add_argument("--n-tpq", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch

    from src.modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    model_cfg = cfg["source_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    out_path = Path(args.output)
    ensure_dir(out_path.parent)

    rows = subset(list(read_jsonl(args.questions)), args.n_fpq, args.n_nfp, args.n_tpq, args.seed)
    # nfp/tpq share text; generate each text once and judge under both.
    seen, uniq = set(), []
    for r in rows:
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        uniq.append(r)
    rows = uniq
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    print(f"[steer] {len(rows)} questions to run ({len(done)} done) policy={args.policy} L{args.layer} a={args.alpha}", flush=True)
    if not rows:
        return

    npz = np.load(args.directions)
    direction = torch.tensor(npz[f"L{args.layer}_{args.direction_key}"], dtype=torch.float32)
    direction = direction / direction.norm()
    gate_layer = args.gate_layer if args.gate_layer is not None else args.layer
    probe = None
    if args.policy == "gated":
        probe = {k: npz[f"L{gate_layer}_a_probe_D_{k}"] for k in ("mu", "sd", "w", "b")}

    tokenizer = load_tokenizer(model_cfg["model_id"], cache_dir=cache_dir, trust_remote_code=model_cfg.get("trust_remote_code", False))
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    max_new = int(args.max_new_tokens or cfg["generation"].get("max_new_tokens", 512))

    def chat(q: str) -> str:
        return tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

    # alpha in units of the typical residual norm at this layer, measured on
    # the first batch's last prompt tokens, so one alpha means the same
    # relative push across layers and models.
    scale = 1.0
    if args.alpha_mode == "norm" and args.policy != "none":
        tokenizer.padding_side = "left"
        enc = tokenizer([chat(r["question"]) for r in rows[: args.batch_size]], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        h = prefill_hidden(model, enc["input_ids"], enc["attention_mask"], args.layer)[:, -1].float()
        scale = float(h.norm(dim=-1).mean().item())
        print(f"[steer] residual norm at L{args.layer} last token ~ {scale:.1f}", flush=True)
    alpha_abs = args.alpha * scale if args.alpha_mode == "norm" else args.alpha
    spec = None if args.policy == "none" else SteerSpec(args.layer, direction, alpha_abs)
    gate = (lambda vec: probe_score(probe, vec) > args.gate_threshold) if probe is not None else None

    rows.sort(key=lambda r: len(r["question"]))
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        chats = [chat(r["question"]) for r in batch]
        # Gate at the last prompt token: in the unpadded sequence that is index
        # (n_tokens - 1); generate_steered adds the left-pad offset itself.
        positions = None
        if gate is not None:
            positions = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) - 1 for c in chats]
        texts, records = generate_steered(
            model, tokenizer, chats, spec=spec, gate=gate, gate_index=gate_layer, gate_positions=positions, max_new_tokens=max_new
        )
        for r, text, rec in zip(batch, texts, records):
            append_jsonl(
                out_path,
                {
                    "id": r["id"],
                    "set": r["set"],
                    "question": r["question"],
                    "response": text,
                    "model_id": model_cfg["model_id"],
                    "policy": args.policy,
                    "layer": args.layer,
                    "alpha": args.alpha,
                    "alpha_abs": alpha_abs,
                    "alpha_mode": args.alpha_mode,
                    "direction_key": args.direction_key,
                    "gate_layer": gate_layer if gate is not None else None,
                    "gate_threshold": args.gate_threshold if gate is not None else None,
                    **rec,
                },
            )
        print(f"[steer] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    print(f"[done] {out_path}")
    print(json.dumps({"gate_on_rate": None if gate is None else float(np.mean([r.get("gate_on", False) for r in read_jsonl(out_path)]))}))


if __name__ == "__main__":
    main()
