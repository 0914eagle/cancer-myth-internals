"""Portable Qwen2.5 experiment. Downloads/training run only via explicit subcommands.

Raw question is the sole user turn. Assistant targets carry all supervision.
No judge calls, no routing of answers, no model-generated premise labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "configs/premise_sft"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def normalized(text):
    return " ".join(text.lower().split())


def premise_usable(text):
    return isinstance(text, str) and bool(text.strip()) and normalized(text).rstrip(".") != "from physicians"


def audit_splits(rows):
    ids, questions, groups, myths = {}, {}, {}, {}
    for r in rows:
        part = r["partition"]
        if part not in {"fit", "dev", "test"} or type(r["label"]) is not bool:
            raise ValueError(f"Bad partition/label: {r['id']}")
        if r["id"] in ids:
            raise ValueError(f"Duplicate ID: {r['id']}")
        ids[r["id"]] = part
        q = normalized(r["question"])
        if not q or q in questions:
            raise ValueError(f"Empty/duplicate question: {r['id']}")
        questions[q] = part
        if groups.setdefault(r["group_id"], part) != part:
            raise ValueError(f"Group crosses splits: {r['id']}")
        for p in r["false_premises"]:
            if myths.setdefault(normalized(p), part) != part:
                # Exact myth grouping is guaranteed for the Cancer-Myth split.
                if r["dataset"] == "cancer":
                    raise ValueError(f"Source myth crosses splits: {r['id']}")
        if not r["label"] and r["false_premises"]:
            raise ValueError(f"Normal row has false-premise targets: {r['id']}")
    return dict(Counter(r["partition"] + "_" + ("fpq" if r["label"] else "nfp") for r in rows))


def cancer_rows(bundle=BUNDLE):
    manifest = json.loads((bundle / "bundle.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(bundle / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    answers = {r["id"]: r for r in read_rows(bundle / "cancer_fit_answers.jsonl")}
    rows = []
    for r in read_rows(bundle / "cancer_questions.jsonl"):
        label = r["set"] == "fpq"
        valid = not label or premise_usable(r["premise_text"])
        a = answers.get(r["id"])
        if a and (r["partition"] != "fit" or a["question"] != r["question"]):
            raise ValueError("Answer-target join mismatch")
        rows.append({"id": r["id"], "dataset": "cancer", "partition": r["partition"],
                     "group_id": r["group_id"], "question": r["question"], "label": label,
                     "false_premises": [r["premise_text"]] if label and valid else [],
                     "target_usable": valid, "reference_correction": r["correction"],
                     "answer": a["answer"] if a else None,
                     "answer_provenance": a if a else None})
    return rows


def crepe_rows(raw_dir):
    rows, excluded, seen = [], [], {}
    # Test/dev take precedence when a duplicate question appears in an earlier split.
    for split, part in [("test", "test"), ("validation", "dev"), ("train", "fit")]:
        for r in read_rows(Path(raw_dir) / f"{split}.jsonl"):
            labels = {str(x).replace("_", " ").strip().lower() for x in r["labels"]}
            rid = "crepe_" + str(r["id"])
            if labels not in ({"normal"}, {"false presupposition"}):
                excluded.append({"id": rid, "reason": "ambiguous_labels", "split": split})
                continue
            key = normalized(r["question"])
            if key in seen:
                excluded.append({"id": rid, "reason": "duplicate_question", "kept_split": seen[key]})
                continue
            seen[key] = split
            label = labels == {"false presupposition"}
            premises = list(dict.fromkeys(p.strip() for p in r.get("presuppositions", []) if premise_usable(p)))
            rows.append({"id": rid, "dataset": "crepe", "partition": part, "group_id": rid,
                         "question": r["question"], "label": label,
                         "false_premises": premises if label else [],
                         "target_usable": not label or bool(premises),
                         "reference_correction": r.get("corrections", []), "answer": None})
    # Preserve chronological split; conservatively exclude train/dev exact-premise overlaps
    # with later splits, rather than silently claiming unseen-premise transfer.
    priority = {"test": 2, "dev": 1, "fit": 0}
    owners = {}
    for r in rows:
        for p in r["false_premises"]:
            k = normalized(p)
            if k not in owners or priority[r["partition"]] > priority[owners[k]]:
                owners[k] = r["partition"]
    kept = []
    for r in rows:
        if any(owners[normalized(p)] != r["partition"] for p in r["false_premises"]):
            excluded.append({"id": r["id"], "reason": "exact_premise_in_later_split"})
        else:
            kept.append(r)
    return sorted(kept, key=lambda r: r["id"]), excluded


def target_text(row, objective):
    obj = {"has_false_premise": row["label"]}
    if objective in {"premise", "joint"}:
        obj["false_premises"] = row["false_premises"]
    if objective == "answer":
        return row["answer"]
    text = json.dumps(obj, ensure_ascii=False)
    return text + "\n\nAnswer:\n" + row["answer"] if objective == "joint" else text


def select_train(rows, objective, cohort):
    if objective in {"answer", "joint"} and cohort != "answer_matched":
        raise ValueError("answer/joint require --cohort answer_matched for matched controls")
    selected = [r for r in rows if r["partition"] == "fit" and r["target_usable"]
                and (cohort != "answer_matched" or r.get("answer"))]
    if not selected or {r["label"] for r in selected} != {True, False}:
        raise ValueError("Training requires both labels and non-empty targets")
    return selected


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("Prepared directory already exists; use a fresh path")
    rows = cancer_rows()
    sets = {"cancer": (rows, [])}
    if args.crepe:
        cr, excluded = crepe_rows(args.crepe)
        cm_eval = {normalized(r["question"]) for r in rows if r["partition"] != "fit"}
        clean = []
        for r in cr:
            if r["partition"] == "fit" and normalized(r["question"]) in cm_eval:
                excluded.append({"id": r["id"], "reason": "exact_question_in_cancer_eval"})
            else:
                clean.append(r)
        sets["crepe"] = clean, excluded
    for name, (records, excluded) in sets.items():
        counts = audit_splits(records)
        dest = out / name
        for part in ("fit", "dev", "test"):
            write_rows(dest / f"{part}.jsonl", [r for r in records if r["partition"] == part])
        write_rows(dest / "excluded.jsonl", excluded)
        write_json(dest / "manifest.json", {"counts": counts, "excluded": len(excluded),
                   "unusable_premise_targets": [r["id"] for r in records if not r["target_usable"]],
                   "files": {p.name: sha(p) for p in dest.glob("*.jsonl")},
                   "warning": "Reference premise != clinically revalidated question-specific premise. Existing Cancer splits previously inspected."})
        print(name, counts, "excluded", len(excluded), flush=True)


def download(args):
    from huggingface_hub import HfApi, snapshot_download
    from datasets import load_dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out / "downloads.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
    else:
        api = HfApi()
        lock = {"model_id": MODEL, "model_revision": MODEL_REVISION, "datasets": {}}
        for name, repo in [("cancer", "Cancer-Myth/Cancer-Myth"), ("crepe", "tasksource/CREPE")]:
            lock["datasets"][name] = {"repo": repo, "revision": api.dataset_info(repo).sha}
        # Resolve once before downloading; subsequent invocations never silently move revisions.
        write_json(lock_path, lock)
    model_path = snapshot_download(MODEL, revision=lock["model_revision"],
                                   allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
    lock["model_path"] = model_path
    write_json(lock_path, lock)
    for name, spec in lock["datasets"].items():
        data = load_dataset(spec["repo"], revision=spec["revision"], trust_remote_code=False)
        dest = out / "raw" / name
        hashes = {}
        for split, records in data.items():
            split = "validation" if split == "dev" else split
            path = dest / f"{split}.jsonl"
            write_rows(path, records)
            hashes[path.name] = sha(path)
        spec["raw_hashes"] = hashes
        write_json(lock_path, lock)
    print(json.dumps(lock, indent=2))


def runtime():
    names = ["torch", "transformers", "peft", "accelerate", "huggingface-hub", "tokenizers"]
    return {name: importlib.metadata.version(name) for name in names}


def check(args):
    import torch
    versions = runtime()
    if versions["torch"] != "2.5.1+cu121" or torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected torch 2.5.1+cu121 / CUDA 12.1, got {versions}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError("CUDA unavailable or insufficient visible GPUs; inspect nvidia-smi/driver")
    devices = []
    for i in range(args.gpus):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"GPU {i} does not support BF16")
            x = torch.ones((512, 512), device=f"cuda:{i}", dtype=torch.bfloat16)
            _ = x @ x
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            devices.append({"index": i, "name": torch.cuda.get_device_name(i), "free": free, "total": total})
    write_json(args.out, {"versions": versions, "cuda": torch.version.cuda, "devices": devices,
                         "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True)})
    print(devices)


def encode(tokenizer, question, target, max_length):
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                           tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template([{"role": "user", "content": question},
                                         {"role": "assistant", "content": target}], tokenize=True)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template prefix mismatch; cannot safely mask user tokens")
    if len(full) > max_length:
        raise ValueError(f"Example length {len(full)} > {max_length}; no silent truncation")
    labels = [-100] * len(prompt) + full[len(prompt):]
    if not any(v != -100 for v in labels[1:]):
        raise ValueError("No supervised assistant tokens")
    return {"input_ids": full, "labels": labels}


def collate(examples, pad_id):
    import torch
    length = ((max(len(r["input_ids"]) for r in examples) + 7) // 8) * 8
    return {"input_ids": torch.tensor([r["input_ids"] + [pad_id] * (length-len(r["input_ids"])) for r in examples]),
            "attention_mask": torch.tensor([[1]*len(r["input_ids"]) + [0]*(length-len(r["input_ids"])) for r in examples]),
            "labels": torch.tensor([r["labels"] + [-100]*(length-len(r["labels"])) for r in examples])}


def model_location(lock_path):
    lock = json.loads(Path(lock_path).read_text())
    if lock["model_id"] != MODEL or lock["model_revision"] != MODEL_REVISION:
        raise ValueError("This experiment is pinned to the existing Qwen2.5-7B revision")
    return lock["model_path"]


def train(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    world, rank, local = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; no CPU fallback")
    if torch.__version__ != "2.5.1+cu121":
        raise RuntimeError("Use the pinned torch 2.5.1+cu121 environment for this experiment")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local)
    # Fail fast if NCCL cannot communicate, before loading 15 GB per process.
    ping = torch.tensor([1], device=device)
    if world > 1:
        dist.all_reduce(ping)
    assert ping.item() == world
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    exists = torch.tensor([int(out.exists()) if rank == 0 else 0], device=device)
    if world > 1:
        dist.broadcast(exists, src=0)
    if exists.item():
        raise ValueError("Run directory already exists; choose a new run name (no silent resume/overwrite)")
    data_path = Path(args.data) / "fit.jsonl"
    rows = select_train(read_rows(data_path), args.objective, args.cohort)
    path = model_location(args.model_lock)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    examples = []
    for row in rows:
        try:
            examples.append(encode(tokenizer, row["question"], target_text(row, args.objective), args.max_length))
        except ValueError as e:
            raise ValueError(f"{row['id']}: {e}") from e
    if rank == 0:
        out.mkdir(parents=True)
        write_json(out / "train_manifest.json", {"args": vars(args), "versions": runtime(),
                   "world_size": world, "effective_batch": world*args.batch_size*args.grad_accum,
                   "ids": [r["id"] for r in rows], "data_sha256": sha(data_path),
                   "model_revision": MODEL_REVISION, "model_lock_sha256": sha(args.model_lock),
                   "implementation_sha256": sha(__file__),
                   "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                   "counts": dict(Counter("fpq" if r["label"] else "nfp" for r in rows)),
                   "max_encoded_length": max(len(r["input_ids"]) for r in examples),
                   "loss": "assistant tokens only, globally token-normalized per optimizer update",
                   "selection": "fixed epochs; no test-based checkpoint selection",
                   "smoke_only": args.max_steps > 0})
    if world > 1:
        dist.barrier()
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None).to(device)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=TARGETS, bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if rank == 0:
        model.print_trainable_parameters()
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
    sampler = DistributedSampler(examples, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(examples, sampler=sampler, batch_size=args.batch_size,
                        collate_fn=lambda x: collate(x, tokenizer.pad_token_id), num_workers=0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    steps_per_epoch = math.ceil(len(loader)/args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, math.ceil(total_steps*0.05))
    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        return 0.5*(1+math.cos(math.pi*min(1, (step-warmup)/max(1,total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step = 0
    start = time.time()
    wrapped.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        for _ in range(steps_per_epoch):
            batches = []
            for _ in range(args.grad_accum):
                batch = next(iterator, None)
                if batch is None:
                    break
                batches.append({k: v.to(device) for k, v in batch.items()})
            if not batches:
                break
            denominator = torch.tensor(sum(int((b["labels"][:, 1:] != -100).sum()) for b in batches), device=device)
            if world > 1:
                dist.all_reduce(denominator)
            optimizer.zero_grad(set_to_none=True)
            numerator = torch.zeros((), device=device)
            for i, batch in enumerate(batches):
                sync = wrapped.no_sync() if world > 1 and i < len(batches)-1 else contextlib.nullcontext()
                with sync:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = wrapped(**batch)
                    n = (batch["labels"][:, 1:] != -100).sum()
                    loss = result.loss * n * world / denominator
                    loss.backward()
                    numerator += result.loss.detach() * n
                    del result, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError("Non-finite gradient; no optimizer step taken")
            optimizer.step()
            scheduler.step()
            step += 1
            if world > 1:
                dist.all_reduce(numerator)
            stats = {"epoch": epoch+1, "step": step, "loss": float(numerator/denominator),
                     "lr": scheduler.get_last_lr()[0], "seconds": time.time()-start,
                     "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                     "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
            if world > 1:
                memory = torch.tensor([stats["peak_allocated_gib"], stats["peak_reserved_gib"]], device=device)
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
                stats["peak_allocated_gib"], stats["peak_reserved_gib"] = memory.tolist()
            if rank == 0:
                print(json.dumps(stats), flush=True)
                with (out / "train_log.jsonl").open("a") as f:
                    f.write(json.dumps(stats)+"\n")
            if step >= total_steps:
                break
        if rank == 0:
            model.save_pretrained(out / f"epoch_{epoch+1}", safe_serialization=True)
            tokenizer.save_pretrained(out / f"epoch_{epoch+1}")
        if world > 1:
            dist.barrier()
        if step >= total_steps:
            break
    if rank == 0:
        model.save_pretrained(out / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(out / "adapter")
        write_json(out / "complete.json", {"steps": step, "seconds": time.time()-start,
                    "smoke_only": args.max_steps > 0, "last_epoch": epoch+1})
    if world > 1:
        dist.destroy_process_group()


def evaluation_messages(question, mode):
    if mode == "diagnostic":
        question += '\n\nReturn JSON with "has_false_premise" (boolean) and "false_premises" (list of false assumptions; empty if none).'
    messages = [{"role": "user", "content": question}]
    if mode == "answer":
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant. Respond to the user's question in English. Return only your final answer."})
    return messages


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = model_location(args.model_lock)
    out = Path(args.out)
    if out.exists():
        raise ValueError("Generation directory exists; use a new path")
    rows = read_rows(args.questions)
    if args.limit:
        rows = rows[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None).to("cuda:0")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    out.mkdir(parents=True)
    write_json(out / "generation_manifest.json", {"args": vars(args), "versions": runtime(),
               "questions_sha256": sha(args.questions), "mode": args.mode,
               "instruction_added": args.mode != "raw", "model_revision": MODEL_REVISION,
               "implementation_sha256": sha(__file__),
               "messages_example": evaluation_messages(rows[0]["question"], args.mode) if rows else [],
               "adapter_config_sha256": sha(Path(args.adapter)/"adapter_config.json") if args.adapter else None,
               "adapter_weights_sha256": sha(Path(args.adapter)/"adapter_model.safetensors") if args.adapter else None})
    for r in rows:
        ids = tokenizer.apply_chat_template(evaluation_messages(r["question"], args.mode),
                     tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            seq = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        new = seq[0, ids.shape[1]:]
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = len(new) == args.max_new_tokens and int(new[-1]) not in eos
        text = tokenizer.decode(new, skip_special_tokens=True)
        record = {"id": r["id"], "question": r["question"], "response": text,
                  "truncated": truncated, "generated_tokens": len(new)}
        with (out / "answers.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False)+"\n")
        print(r["id"], "truncated" if truncated else "complete", flush=True)


def parse_prediction(text):
    # Strict output parsing: never reinterpret free-form prose as a successful gate.
    text = text.strip()
    obj, end = json.JSONDecoder().raw_decode(text)
    tail = text[end:].strip()
    if tail and not tail.startswith("Answer:"):
        raise ValueError("Unexpected content after diagnostic JSON")
    if not isinstance(obj, dict) or type(obj.get("has_false_premise")) is not bool:
        raise ValueError("Missing boolean verdict")
    if "false_premises" in obj:
        p = obj["false_premises"]
        if not isinstance(p, list) or any(not isinstance(x, str) or not x.strip() for x in p):
            raise ValueError("Invalid premise list")
        if bool(p) != obj["has_false_premise"]:
            raise ValueError("Verdict and premise list conflict")
    return obj


def score(args):
    refs = {r["id"]: r for r in read_rows(args.questions)}
    pred = read_rows(args.answers)
    if len({r["id"] for r in pred}) != len(pred):
        raise ValueError("Duplicate predictions")
    seen, counts, cases = set(), Counter(), []
    for p in pred:
        if p["id"] not in refs:
            raise ValueError("Unknown prediction ID")
        r = refs[p["id"]]
        seen.add(p["id"])
        label = "fpq" if r["label"] else "nfp"
        try:
            if p.get("truncated"):
                raise ValueError("Truncated")
            obj = parse_prediction(p["response"])
        except (ValueError, TypeError) as e:
            counts[label+"_invalid"] += 1
            cases.append({"id": p["id"], "invalid": str(e), "response": p["response"]})
            continue
        counts[label+"_yes" if obj["has_false_premise"] else label+"_no"] += 1
        cases.append({"id": p["id"], "predicted": obj, "reference_label": r["label"],
                      "reference_premises": r["false_premises"], "target_usable": r.get("target_usable", True),
                      "question": r["question"]})
    for rid in refs.keys()-seen:
        counts[("fpq" if refs[rid]["label"] else "nfp")+"_missing"] += 1
    totals = Counter("fpq" if r["label"] else "nfp" for r in refs.values())
    summary = {"counts": dict(counts), "total": dict(totals),
               "fpq_yes_over_all": counts["fpq_yes"]/totals["fpq"] if totals["fpq"] else None,
               "nfp_yes_over_all": counts["nfp_yes"]/totals["nfp"] if totals["nfp"] else None,
               "complete_valid": not any(counts[k] for k in counts if k.endswith(("invalid", "missing"))),
               "note": "Invalid/missing shown separately, never counted as No. No AUROC from hard labels. Premise semantics and final-answer Well scores require separate assessment."}
    write_json(args.out, summary)
    write_rows(Path(args.out).with_suffix(".cases.jsonl"), cases)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--crepe")
    p = sub.add_parser("check")
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", type=int, default=2)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--model-lock", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--objective", choices=["binary", "premise", "answer", "joint"], required=True)
    p.add_argument("--cohort", choices=["all", "answer_matched"], default="all")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=0, help=">0: smoke only; not a main result")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=17)
    p = sub.add_parser("generate")
    p.add_argument("--model-lock", required=True)
    p.add_argument("--adapter")
    p.add_argument("--questions", required=True)
    p.add_argument("--mode", choices=["raw", "answer", "diagnostic"], default="raw")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p = sub.add_parser("score")
    p.add_argument("--questions", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True)
    args = ap.parse_args()
    for key in ("batch_size", "grad_accum", "epochs", "max_length", "gpus", "max_new_tokens"):
        if hasattr(args, key) and getattr(args, key) < 1:
            ap.error(f"{key} must be positive")
    for key in ("max_steps", "limit"):
        if hasattr(args, key) and getattr(args, key) < 0:
            ap.error(f"{key} cannot be negative")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
