"""Explicit stages for medical FPQ detection and answer baselines.

No all-in-one stage calls a paid judge. prepare/splits/gate/report/status are
CPU-only; generate/detect/extract load a local causal LM. Well scoring has its
own explicit capped CLI (evaluate_well.py).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_suite import prepare, load_suite, split_plan, export_splits
from src.config import load_config
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import digest, frozen_json, file_digest, output_lock

METHODS = ("plain", "zero_shot_cot", "fp_identification", "extract_verify", "premise_review")


def gpu_guard():
    visible = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if visible not in {"0", "1", "0,1", "1,0"}:
        raise ValueError("Only physical GPUs 0 and 1 are authorized; set CUDA_VISIBLE_DEVICES=0 or 1")


def choose_rows(suite, partition):
    return [q for q in suite["questions"] if partition == "all" or q["partition"] == partition]


def final_budget(out, explicit=None):
    path = Path(out) / "generation_defaults.json"
    if path.exists():
        defaults = json.loads(path.read_text())
        value = defaults["final_tokens"]
        if defaults.get("requires_cache_migration") and not (Path(out) / "cache_migration_complete.json").exists():
            raise ValueError("Cache migration incomplete; rerun prepare_final1024.py first")
        if explicit is not None and explicit != value:
            raise ValueError("Final budget differs from frozen run defaults; use a separate run directory")
    else:
        value = 512 if explicit is None else explicit
    if type(value) is not int or value < 1:
        raise ValueError("Positive integer final token budget required")
    return value


def atomic_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(tmp, rows)
    tmp.replace(path)


def load_gate_inputs(out, signals):
    if not any(s in signals for s in ("hidden", "mean", "direct", "review")):
        return None, {}
    from src import baseline_generation as bg
    features = bg.load_feature_records(out / "model") if any(s in signals for s in ("hidden", "mean")) else None
    records = bg.load_detection_records(out / "model") if any(s in signals for s in ("direct", "review")) else []
    detections = {r["id"]: r for r in records}
    if len(detections) != len(records):
        raise ValueError("Duplicate detection record")
    return features, detections


def run(args):
    out = Path(args.out_dir).resolve()
    if args.stage == "prepare":
        print(json.dumps(prepare(args.manifest, out, exclude_ids=args.exclude_ids,
                                 tpq_path=args.tpq, fpq_source=args.fpq_source), indent=2))
        print("Prepared our question inventory, not Well paper split. GPU/GPT calls: 0.")
        return
    suite = load_suite(out)
    if args.stage == "status":
        from collections import Counter
        from src import baseline_generation as bg
        generation = {}
        for method in METHODS:
            records = bg.load_generation_records(out / "model", method)
            generation[method] = {
                "statuses": dict(Counter(r.get("status", "unknown") for r in records)),
                "final_cap_hits": sum(r.get("details", {}).get("answer", {}).get("cap_hit", False)
                                      for r in records),
                "review_cap_hits": sum(any(r.get("details", {}).get(k, {}).get("cap_hit", False)
                                           for k in ("review", "reasoning")) for r in records),
            }
        print(json.dumps({"questions": len(suite["questions"]),
                          "generation": generation,
                          "detection_valid": len(bg.load_detection_records(out / "model")),
                          "feature_questions": len(bg.load_feature_records(out / "model")),
                          "answer_files": [str(p) for p in sorted((out / "answers").glob("*.jsonl"))],
                          "gate_runs": [str(p.parent) for p in sorted((out / "gates").glob("*/result.json"))],
                          "note": "No calls made; inspect stage logs for active jobs."}, indent=2))
        return
    if args.stage == "splits":
        split = split_plan(suite["questions"], scheme=args.scheme, evaluation=args.evaluation,
                           folds=args.folds, seed=args.seed)
        print(export_splits(out, split))
        print("Each directory has train/calibration/evaluation; suitable for downstream GEPA/finetuning too.")
        return
    if args.stage == "import-answers":
        # Reusing old answers never relabels them as an exact new baseline reproduction.
        if not args.name.startswith("legacy_"):
            raise ValueError("Imported runs must use a legacy_ name to preserve protocol distinctions")
        meta = json.loads(Path(args.run_metadata).read_text())
        lookup = {q["id"]: q for q in suite["questions"]}
        imported, seen = [], set()
        for r in read_jsonl(args.answers):
            if r["id"] in seen or r["id"] not in lookup:
                raise ValueError("Duplicate/unknown imported answer ID")
            seen.add(r["id"])
            q = lookup[r["id"]]
            if r.get("question") != q["question"] or not isinstance(r.get("response"), str) or not r["response"].strip():
                raise ValueError("Imported answers need exact question text and nonempty response")
            imported.append({**r, "method": args.name, "set": q["set"], "partition": q["partition"],
                             "source_method": r.get("method"), "source_metadata_hash": digest(meta)})
        with output_lock(out / "import"):
            frozen_json(out / "imports" / f"{args.name}.json",
                        {"source_sha256": file_digest(args.answers), "metadata": meta,
                         "question_hash": digest(suite["questions"]), "method": args.name})
            atomic_jsonl(out / "answers" / f"{args.name}.jsonl", imported)
        print(f"Imported {len(imported)} saved responses under {args.name}. Generation/judge calls: 0.")
        return
    if args.stage in {"generate", "detect", "extract"}:
        gpu_guard()
        from src import baseline_generation as bg
        cfg = load_config(args.config)
        rows = choose_rows(suite, args.partition)
        if not rows:
            raise ValueError("Empty generation/extraction partition")
        if args.stage == "generate":
            budget = final_budget(out, args.final_tokens)
            print(f"Final-answer cap: {budget}; review cap: {args.review_tokens}. Judge calls: 0.", flush=True)
            bg.run_generation(rows, cfg, out / "model", args.methods,
                              final_tokens=budget, review_tokens=args.review_tokens,
                              extraction_tokens=args.extraction_tokens)
            lookup = {q["id"]: q for q in suite["questions"]}
            with output_lock(out / "export_answers"):
                for method in args.methods:
                    records = bg.load_generation_records(out / "model", method)
                    # Preserve failure rows in source cache; missing responses stay missing in judging.
                    records = [{**r, "method": method, "question": lookup[r["id"]]["question"],
                                "set": lookup[r["id"]]["set"], "partition": lookup[r["id"]]["partition"]}
                               for r in records if r["id"] in lookup and isinstance(r.get("response"), str)
                               and r["response"].strip() and r.get("status", "complete") not in {"failed", "error"}]
                    atomic_jsonl(out / "answers" / f"{method}.jsonl", records)
            print(f"Answers exported to {out / 'answers'}. Judge calls: 0.")
        elif args.stage == "detect":
            bg.run_detection(rows, cfg, out / "model", review_tokens=args.review_tokens)
            print("Continuous direct/review detection scores stored. Judge calls: 0.")
        else:
            layers = args.layers or sorted({round(cfg["source_model"]["n_layers"] * x) for x in (.4, .6, .8)})
            bg.extract_features(rows, cfg, out / "model", layers)
            print("Question prefill states stored. Answers generated: 0. Judge calls: 0.")
        return
    if args.stage == "gate":
        import hashlib
        import sklearn
        import numpy as np
        from src.baseline_gates import run_gate_comparison, gate_report
        split = split_plan(suite["questions"], scheme=args.scheme, evaluation=args.evaluation,
                           folds=args.folds, seed=args.seed)
        export_splits(out, split)
        features, detections = load_gate_inputs(out, args.signals)
        layers = args.layers
        if layers is None and features:
            layers = sorted(set.intersection(*(set(x) for x in features.values())))
        destination = out / "gates" / args.name
        spec = {"suite_hash": digest(suite), "split": split, "signals": args.signals,
                "model_identity": json.loads((out / "model" / "identity.json").read_text())
                                  if any(s != "text" for s in args.signals) else None,
                "layers": layers, "c_grid": args.c_grid, "target_fpr": args.target_fpr,
                "feature_hash": digest({k: {str(layer): hashlib.sha256(np.asarray(v, dtype="<f4").tobytes()).hexdigest() for layer, v in x.items()}
                                        for k, x in features.items()}) if features else None,
                "detection_hash": digest(detections),
                "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
                "implementation_hash": file_digest(ROOT / "src/baseline_gates.py")}
        with output_lock(destination):
            frozen_json(destination / "plan.json", spec)
            result = run_gate_comparison(suite["questions"], split, features=features, detections=detections,
                                        signals=args.signals, layers=layers, c_grid=args.c_grid,
                                        target_fpr=args.target_fpr, seed=args.seed)
            frozen_json(destination / "result.json", result)
            report, summary = gate_report(result, repeats=args.bootstrap)
            if spec["model_identity"]:
                report += "\nModel identity and source hashes: `plan.json`. Layer/C choices and thresholds: `result.json`.\n"
            frozen_json(destination / f"summary_bootstrap{args.bootstrap}.json", summary)
            (destination / "report.md").write_text(report)
        print(report)
        return
    raise ValueError("Unknown stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "status", "splits", "generate", "detect", "extract", "gate", "import-answers"):
        p = sub.add_parser(stage)
        p.add_argument("--out-dir", required=True)
        if stage == "prepare":
            p.add_argument("--manifest", required=True, help="Existing run_pilot split/manifest.json")
            p.add_argument("--tpq", help="Optional Well TPQ JSONL annotation join by exact normalized question")
            p.add_argument("--fpq-source", help="HF Cancer-Myth JSONL with original source_myth and presupposition_correction")
            p.add_argument("--exclude-ids", nargs="*", default=[])
        if stage == "import-answers":
            p.add_argument("--answers", required=True)
            p.add_argument("--run-metadata", required=True)
            p.add_argument("--name", required=True)
        if stage in {"generate", "detect", "extract"}:
            p.add_argument("--config", default=str(ROOT / "configs/qwen25_7b.yaml"))
            p.add_argument("--partition", choices=("all", "fit", "dev", "test"), default="all")
            if stage != "extract":
                p.add_argument("--review-tokens", type=int, default=1024)
            if stage == "generate":
                p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
                p.add_argument("--final-tokens", type=int, default=None,
                               help="Run's generation_defaults.json, otherwise 512; frozen defaults cannot be overridden")
                p.add_argument("--extraction-tokens", type=int, default=512)
            if stage == "extract":
                p.add_argument("--layers", type=int, nargs="+")
        if stage in {"splits", "gate"}:
            p.add_argument("--scheme", choices=("holdout", "crossfit"), default="holdout")
            p.add_argument("--evaluation", choices=("dev", "test"), default="dev")
            p.add_argument("--folds", type=int, default=5)
            p.add_argument("--seed", type=int, default=17)
            if stage == "gate":
                p.add_argument("--name", default="holdout_v1")
                p.add_argument("--signals", nargs="+", choices=("text", "hidden", "mean", "direct", "review"),
                               default=["text", "hidden", "mean", "direct", "review"])
                p.add_argument("--layers", nargs="+", type=int)
                p.add_argument("--c-grid", nargs="+", type=float, default=[.001, .01, .1, 1.0])
                p.add_argument("--target-fpr", type=float, default=.05)
                p.add_argument("--bootstrap", type=int, default=500)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
