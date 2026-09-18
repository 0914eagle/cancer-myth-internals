"""Twin LoRA pipeline (method B, chat 9/18). Stages: data -> train -> generate -> (judge) -> compare.

    export OUT=$SUITE_DIR/lora/twin_v1
    # 1) data: training rows + held-out evaluation rows. Generates the model's OWN Plain
    #    answers to the training twins (GPU, ~200 generations); everything else is offline.
    CUDA_VISIBLE_DEVICES=0 python scripts/lora_twin.py data --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml \\
        --twins $ROWS/questions_twins.jsonl --e1-questions $ROWS/questions.jsonl \\
        --well-dir $SUITE_DIR/well_judge_claude_fpu --out $OUT --negatives twins
    # 2) train: LoRA on train.jsonl (GPU, minutes)
    CUDA_VISIBLE_DEVICES=0 python scripts/lora_twin.py train --config configs/qwen25_7b.yaml --out $OUT
    # 3) generate: Plain answers of base+adapter on the held-out rows, and of the BASE on held-out twins
    CUDA_VISIBLE_DEVICES=0 python scripts/lora_twin.py generate --config configs/qwen25_7b.yaml --out $OUT --which adapter
    CUDA_VISIBLE_DEVICES=0 python scripts/lora_twin.py generate --config configs/qwen25_7b.yaml --out $OUT --which base
    # 4) judge with the usual Well pipeline (claude backend), then
    python scripts/lora_twin.py compare --out $OUT --lora-well $OUT/well_judge_claude \\
        --baseline-well $SUITE_DIR/well_judge_claude $SUITE_DIR/well_judge_claude_fpu

The control adapter (Well's FPQ-only LoRA) is the same pipeline with
--negatives none and another --out. Judge calls: none in this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import lora_twin as lt
from src.config import load_config
from src.jsonl import read_jsonl
from src.pilot import digest, file_digest, frozen_json, output_lock

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def suite_budget(suite_dir):
    path = Path(suite_dir) / "generation_defaults.json"
    return int(json.loads(path.read_text())["final_tokens"]) if path.exists() else 512


def stage_data(args):
    from src import baseline_generation as bg
    from src.baseline_suite import load_suite
    from src.style_controls import split_twin_file
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    suite = load_suite(args.suite_dir)
    questions = suite["questions"]
    twin_rows = list(read_jsonl(args.twins))
    split_twin_file(twin_rows)  # validates the file shape
    twins, fparas = lt.map_twin_rows(twin_rows, list(read_jsonl(args.e1_questions)), questions)
    print(f"[data] twins {len(twins)}, false paraphrases {len(fparas)} mapped to suite FPQ", flush=True)
    fp_answers = lt.read_answers(Path(args.suite_dir) / "answers" / "fp_unconditional.jsonl")
    fp_scores = lt.judge_scores(args.well_dir, "fp_unconditional")
    fpq_train, _ = lt.split_origins(questions, args.train_partitions)
    twin_answers = {}
    if args.negatives == "twins":
        rows = [{"id": twins[o]["id"], "question": twins[o]["question"]} for o in sorted(fpq_train) if o in twins]
        budget = suite_budget(args.suite_dir)
        gen_dir = out / "twin_plain"
        if not args.skip_generation:
            cfg = load_config(args.config)
            print(f"[data] generating Plain answers for {len(rows)} training twins (budget {budget})", flush=True)
            bg.run_generation(rows, cfg, gen_dir, ["plain"], final_tokens=budget)
        twin_answers = {r["id"]: r["response"] for r in bg.load_generation_records(gen_dir, "plain")
                        if r.get("status") == "complete" and isinstance(r.get("response"), str)}
        print(f"[data] twin Plain answers available: {len(twin_answers)}/{len(rows)}", flush=True)
    train_rows, summary = lt.assemble_training(
        questions, twins, fparas, fp_answers, fp_scores, twin_answers,
        train_partitions=args.train_partitions, negatives=args.negatives,
        min_score=args.min_score, fpara_positives=not args.no_fpara)
    eval_rows = lt.evaluation_rows(questions, twins, args.train_partitions)
    with output_lock(out / "data"):
        lt.write_jsonl(out / "train.jsonl", train_rows)
        lt.write_jsonl(out / "eval_questions.jsonl", eval_rows)
        summary["eval_rows"] = {"total": len(eval_rows),
                                "by_kind": {k: sum((r.get("kind") or r["set"]) == k for r in eval_rows) for k in ("fpq", "nfp", "twin")}}
        summary["sources"] = {"twins": str(args.twins), "well_dir": str(args.well_dir), "suite_dir": str(Path(args.suite_dir).resolve())}
        (out / "data_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Written to {out}. Judge calls: 0.")


def stage_train(args):
    import torch
    from src.modeling import load_causal_lm, load_tokenizer
    from src.pilot_model import render
    out = Path(args.out).resolve()
    rows = list(read_jsonl(out / "train.jsonl"))
    summary = json.loads((out / "data_summary.json").read_text())
    if digest(rows) != summary["data_hash"]:
        raise ValueError("train.jsonl differs from data_summary.json; rerun the data stage")
    adapter_dir = out / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        raise ValueError(f"Adapter already exists at {adapter_dir}; use another --out")
    from peft import LoraConfig, get_peft_model
    import peft
    cfg = load_config(args.config)
    m = cfg["source_model"]
    torch.manual_seed(args.seed)
    tok = load_tokenizer(m["model_id"], cache_dir=cfg["paths"].get("cache_dir"),
                         trust_remote_code=m.get("trust_remote_code", False), revision=m.get("revision"))
    model = load_causal_lm(m, cache_dir=cfg["paths"].get("cache_dir"))
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=args.targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    examples, truncated = [], 0
    for r in rows:
        ids, labels, cut = lt.encode_example(tok, render, r["question"], r["target"], args.max_len)
        truncated += cut
        examples.append((ids, labels))
    print(f"[train] {len(examples)} examples; {truncated} truncated to {args.max_len}; "
          f"max length {max(len(i) for i, _ in examples)}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    per_epoch = math.ceil(len(examples) / (args.batch * args.grad_accum))
    total = per_epoch * args.epochs
    from transformers import get_cosine_schedule_with_warmup
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(args.warmup * total)), total)
    device = model.get_input_embeddings().weight.device
    identity = {"version": lt.VERSION, "base": m, "data_hash": summary["data_hash"], "data_summary": summary,
                "hyperparams": {k: getattr(args, k) for k in ("r", "alpha", "dropout", "targets", "lr", "epochs", "batch",
                                                              "grad_accum", "max_len", "seed", "warmup", "gradient_checkpointing")},
                "steps": total, "peft_version": peft.__version__, "torch_version": torch.__version__}
    frozen_json(out / "train_identity.json", identity)
    log = (out / "train_log.jsonl").open("w", encoding="utf-8")
    model.train()
    step, t0 = 0, time.time()
    with output_lock(out / "train"):
        for epoch in range(args.epochs):
            groups = lt.batches(len(examples), args.batch, args.seed, epoch)
            running, count = 0.0, 0
            for gi, group in enumerate(groups, 1):
                batch = lt.pad_batch([examples[i] for i in group], tok.pad_token_id)
                tensors = {k: torch.tensor(v, device=device) for k, v in batch.items()}
                loss = model(**tensors).loss / args.grad_accum
                loss.backward()
                running += float(loss) * args.grad_accum; count += 1
                if gi % args.grad_accum == 0 or gi == len(groups):
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % args.log_every == 0 or step == total:
                        entry = {"epoch": epoch, "step": step, "of": total, "loss": running / count,
                                 "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                        log.write(json.dumps(entry) + "\n"); log.flush()
                        print(f"[train] {json.dumps(entry)}", flush=True)
                        running, count = 0.0, 0
        model.save_pretrained(adapter_dir)
    log.close()
    print(f"Adapter saved to {adapter_dir} ({total} steps). Judge calls: 0.")


def _adapted_runtime(cfg, adapter_dir):
    from src import baseline_generation as bg
    from peft import PeftModel
    runtime = bg.make_runtime(cfg)
    merged = PeftModel.from_pretrained(runtime.model, str(adapter_dir)).merge_and_unload()
    merged.eval()
    runtime.model = merged
    runtime.identity = {**runtime.identity,
                        "adapter": {"path": str(adapter_dir),
                                    "weights_sha256": file_digest(adapter_dir / "adapter_model.safetensors"),
                                    "train_identity": json.loads((adapter_dir.parent / "train_identity.json").read_text())}}
    return runtime


def stage_generate(args):
    from src import baseline_generation as bg
    out = Path(args.out).resolve()
    eval_rows = list(read_jsonl(out / "eval_questions.jsonl"))
    which_rows = args.rows or ("all" if args.which == "adapter" else "twins")
    rows = [r for r in eval_rows if which_rows == "all" or r.get("kind") == "twin"]
    if not rows:
        raise ValueError("No evaluation rows selected")
    budget = suite_budget(args.suite_dir) if args.suite_dir else args.final_tokens
    cfg = load_config(args.config)
    method = f"lora_{out.name}" if args.which == "adapter" else "plain"
    gen_dir = out / f"eval_{args.which}"
    if args.which == "adapter":
        adapter_dir = out / "adapter"
        if not (adapter_dir / "adapter_config.json").exists():
            raise ValueError(f"No adapter at {adapter_dir}; run the train stage")
        bg.make_runtime = lambda c, _cfg=cfg: _adapted_runtime(_cfg, adapter_dir)  # the generation loop builds its runtime here
    print(f"[generate] {args.which}: {len(rows)} rows ({which_rows}), budget {budget}, method {method}", flush=True)
    bg.run_generation([{"id": r["id"], "question": r["question"]} for r in rows], cfg, gen_dir / "model", ["plain"], final_tokens=budget)
    lookup = {r["id"]: r for r in rows}
    records = bg.load_generation_records(gen_dir / "model", "plain")
    exported = [{**rec, "method": method, "set": lookup[rec["id"]]["set"], "partition": lookup[rec["id"]]["partition"],
                 "kind": lookup[rec["id"]].get("kind") or lookup[rec["id"]]["set"]} for rec in records]
    with output_lock(gen_dir / "export"):
        lt.write_jsonl(gen_dir / "answers" / f"{method}.jsonl", exported)
    done = sum(r.get("status") == "complete" for r in exported)
    print(f"[generate] {done}/{len(rows)} complete -> {gen_dir / 'answers' / (method + '.jsonl')}. Judge calls: 0.")
    print("Next: python scripts/evaluate_well.py prepare --backend claude "
          f"--questions {out / 'eval_questions.jsonl'} --answers <answer files> --out-dir {out / 'well_judge_claude'}")


def stage_compare(args):
    out = Path(args.out).resolve()
    eval_rows = list(read_jsonl(out / "eval_questions.jsonl"))
    ids = {r["id"] for r in eval_rows}
    score_sets = {}
    for well in args.baseline_well:  # plain and fp_unconditional may live in different judge dirs
        for method in ("plain", "fp_unconditional"):
            try:
                scores = lt.judge_scores(well, method)
            except ValueError:
                continue
            merged = score_sets.setdefault(method, {})
            merged.update({q: s for q, s in scores.items() if q in ids and s is not None})
    for well in args.lora_well:
        from src.well_eval import load_plan
        plan, _ = load_plan(well)
        for method in plan["mapping"]:
            scores = {q: s for q, s in lt.judge_scores(well, method).items() if q in ids}
            merged = score_sets.setdefault(method, {})
            for q, s in scores.items():
                if s is not None:
                    merged[q] = s  # base Plain on twins comes from the lora judge dir
    lines, table = lt.s5_table(eval_rows, score_sets)
    header = [f"# Twin LoRA comparison ({out.name})", "",
              f"Evaluation rows {len(eval_rows)}: held-out suite FPQ/NFP plus true twins of held-out FPQ origins (judged with the NFP/TPQ template).",
              "S5/all counts a score of 5 over ALL rows of the group (missing = not 5). Rescue/harm: rows that became 5 / stopped being 5 relative to plain.", ""]
    (out / "compare.md").write_text("\n".join(header + lines) + "\n")
    (out / "compare.json").write_text(json.dumps(table, indent=2) + "\n")
    print("\n".join(header + lines))
    print(f"Written to {out / 'compare.md'}. Judge calls: 0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("data")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True)
    p.add_argument("--twins", required=True, help="questions_twins.jsonl (true twins + false paraphrases)")
    p.add_argument("--e1-questions", required=True, help="questions.jsonl the twins were built from (pair_id scheme)")
    p.add_argument("--well-dir", required=True, help="Well judge dir holding fp_unconditional scores")
    p.add_argument("--out", required=True)
    p.add_argument("--negatives", choices=lt.NEGATIVES, default="twins")
    p.add_argument("--min-score", type=int, default=5, help="keep FPQ targets the judge scored at least this")
    p.add_argument("--no-fpara", action="store_true", help="drop the false-paraphrase positives")
    p.add_argument("--train-partitions", nargs="+", default=["fit"])
    p.add_argument("--skip-generation", action="store_true", help="use cached twin Plain answers only")
    p = sub.add_parser("train")
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--r", type=int, default=16); p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05); p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1); p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=1536); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--warmup", type=float, default=0.05); p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p = sub.add_parser("generate")
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--which", choices=("adapter", "base"), required=True)
    p.add_argument("--rows", choices=("all", "twins"), default=None, help="default: all for adapter, twins for base")
    p.add_argument("--suite-dir", help="read the frozen final token budget from here")
    p.add_argument("--final-tokens", type=int, default=1024)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--baseline-well", nargs="+", required=True, help="judge dirs holding plain / fp_unconditional scores")
    p.add_argument("--lora-well", nargs="+", default=[])
    args = ap.parse_args()
    {"data": stage_data, "train": stage_train, "generate": stage_generate, "compare": stage_compare}[args.stage](args)


if __name__ == "__main__":
    main()
