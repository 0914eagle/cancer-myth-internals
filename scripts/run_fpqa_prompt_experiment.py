"""Preflight, optimize, or evaluate question-only prompts. Preflight makes no API calls."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import (
    METHODS, Evaluator, RecordedCaller, detection_metrics, digest, freeze, load_rows,
    response_metrics, seed_prompt, validate_splits, well_crepe_judge,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimize(evaluator, prompt, train, dev, config, out, reflection):
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
    if config["max_metric_calls"] <= len(dev):
        raise ValueError("Metric budget must exceed initial full validation cost")
    task = evaluator.task
    result = optimize_anything(
        seed_candidate={"system_prompt": prompt}, evaluator=evaluator, dataset=train, valset=dev,
        objective=("Classify false-premise questions accurately while preserving normal questions. Output exactly Yes or No."
                   if task == "detection" else "Correct false premises accurately and answer normal questions without inventing errors."),
        background="Only the user question is available at inference. Do not use tools or request gold annotations. Labels/references in evaluator feedback are training feedback, not inference inputs.",
        config=GEPAConfig(engine=EngineConfig(run_dir=str(out / "gepa"), max_metric_calls=config["max_metric_calls"],
                                             parallel=False, cache_evaluation=True, seed=config["seed"]),
                          reflection=ReflectionConfig(reflection_lm=reflection)))
    best = result.best_candidate["system_prompt"]
    freeze(out / "optimized_prompt.json", {"system_prompt": best, "task": task,
                                           "best_idx": result.best_idx,
                                           "validation_score": result.val_aggregate_scores[result.best_idx]})
    (out / "best_system_prompt.txt").write_text(best + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "optimize", "evaluate"))
    parser.add_argument("--data", type=Path, required=True, help="Prepared root containing manifest.json")
    parser.add_argument("--dataset", choices=("crepe", "falseqa"), required=True)
    parser.add_argument("--task", choices=("detection", "response"), required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fpqa_prompting/luna6_sonnet55.json")
    parser.add_argument("--well-root", type=Path, default=ROOT / "external/fpqa_20261002/Well")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", choices=("balanced", "fpq_only"), default="balanced")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--prompt", type=Path, help="optimized_prompt.json from a completed optimization")
    parser.add_argument("--limit", type=int, default=0, help="Evaluation-only smoke limit; mark outputs as partial")
    args = parser.parse_args()
    if args.method not in METHODS[args.task]:
        parser.error("Method does not match task")
    if args.task == "response" and args.dataset != "crepe":
        parser.error("FalseQA response evaluation needs an independently specified reference-rebuttal rubric; detection is ready")
    if args.task == "detection" and args.objective != "balanced":
        parser.error("Detection optimization must include both labels")
    if args.command == "optimize" and args.method != "gepa":
        parser.error("Only GEPA is an optimization method")
    if args.limit < 0 or (args.command == "optimize" and args.limit):
        parser.error("--limit is for evaluation smoke tests only")
    config = json.loads(args.config.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    # Verify only the requested dataset files; all hashes were frozen at preparation.
    for rel, expected in manifest["files"].items():
        if rel.startswith(args.dataset + "/") and sha(args.data / rel) != expected:
            raise ValueError(f"Prepared data changed: {rel}")
    base = args.data / args.dataset
    validate_splits(*(load_rows(base / f"{s}.jsonl") for s in ("train", "dev", "test")))
    suffix = "_fpq" if args.objective == "fpq_only" else ""
    train = load_rows(base / f"opt_train{suffix}.jsonl")
    dev = load_rows(base / f"opt_dev{suffix}.jsonl")
    prompt = seed_prompt(args.task, args.method)
    if args.prompt:
        saved = json.loads(args.prompt.read_text())
        if saved["task"] != args.task:
            raise ValueError("Cannot use a response-optimized prompt as a detection prompt or vice versa")
        prompt = saved["system_prompt"]
    elif args.command == "evaluate" and args.method == "gepa":
        parser.error("GEPA evaluation requires --prompt from a completed optimization")
    if args.command == "optimize" and args.prompt:
        parser.error("Start from the registered seed; use a separately named protocol for warm starts")
    template_hashes = {}
    if args.task == "response":
        for row in [next(r for r in dev if r["label"] == y) for y in ({1} if args.objective == "fpq_only" else {0, 1})]:
            well_crepe_judge(args.well_root, row, "Preflight placeholder; not a generated answer.")
        for kind in ("FPQ", "TPQ"):
            path = args.well_root / f"data_gen/template/CREPE{kind}_template.py"
            template_hashes[path.name] = sha(path)
    plan = {"task": args.task, "dataset": args.dataset, "method": args.method,
            "objective": args.objective, "config": config, "prompt": prompt,
            "optimization_n_train": len(train), "optimization_n_dev": len(dev),
            "dataset_manifest_sha256": sha(args.data / "manifest.json"),
            "judge_template_sha256": template_hashes,
            "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT / "src/fpqa_prompting.py", ROOT / "src/fpqa_cli_backend.py")},
            "versions": {name: importlib.metadata.version(name) for name in ("gepa",)},
            "cli_versions": {backend: subprocess.run([backend, "--version"], capture_output=True, text=True, check=True).stdout.strip()
                             if shutil.which(backend) else None
                             for backend in {config[r]["backend"] for r in ("task_model", "judge_model", "reflection_model")}
                             if backend != "local_http"},
            "split": args.split, "limit": args.limit, "generated_binary_auroc": None,
            "adaptation": "Well rubric via CLI; shared text envelope, provider context/defaults differ; not exact Well numerical reproduction"}
    if args.command == "preflight":
        plan["transport_present_not_auth_verified"] = {
            role: (bool(os.getenv(config[role]["base_url_env"])) if config[role]["backend"] == "local_http"
                   else bool(shutil.which(config[role]["backend"])))
            for role in ("task_model", "judge_model", "reflection_model")}
        freeze(args.out / "preflight.json", plan)
        print(json.dumps(plan, indent=2))
        return
    if args.command == "optimize" and (args.out / "optimized_prompt.json").exists():
        raise ValueError("Optimization already complete; evaluate the saved prompt")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze(args.out / "run.json", {**plan, "command": args.command})
    caller = RecordedCaller(config, args.out)
    evaluator = Evaluator(args.task, args.method, caller, args.well_root)
    if args.command == "optimize":
        def reflection(messages):
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            return caller("reflection", messages)
        optimize(evaluator, prompt, train, dev, config, args.out, reflection)
        print(f"Optimized prompt saved under {args.out}")
        return
    rows = load_rows(base / f"{args.split}.jsonl")
    if args.limit:
        # Deterministic evaluation smoke; not a stratified or representative result.
        rows = sorted(rows, key=lambda r: digest(r["id"]))[:args.limit]
    def run(row):
        path = args.out / "records" / f"{digest(row['id'])}.json"
        if path.exists():
            return json.loads(path.read_text())
        _, record = evaluator({"system_prompt": prompt}, row)
        freeze(path, record)
        return record
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        records = []
        for record in pool.map(run, rows):
            records.append(record)
            if len(records) % 25 == 0 or len(records) == len(rows):
                print(f"[{args.dataset}/{args.method}] {len(records)}/{len(rows)} complete", flush=True)
    metrics = detection_metrics(records) if args.task == "detection" else response_metrics(records)
    report = {"task": args.task, "dataset": args.dataset, "method": args.method,
              "split": args.split, "partial": bool(args.limit), "n": len(records), "metrics": metrics}
    freeze(args.out / "metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
