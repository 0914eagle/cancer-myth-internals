"""Source-backed Well balanced GEPA, no-RAG; preflight never calls a model."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import freeze, load_rows, validate_splits
from src.fpqa_well_upstream import (
    WELL_REVISION, UncachedCaller, load_upstream, make_args, verify_source, write_inputs,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "optimize"))
    p.add_argument("--dataset", choices=("crepe", "cancer_myth"), required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/fpqa_prompting/well_upstream/luna6.json")
    p.add_argument("--well-root", type=Path, default=ROOT / "external/fpqa_20261002/Well")
    args = p.parse_args()
    args.out = args.out.resolve()
    hashes = verify_source(args.well_root)
    if importlib.metadata.version("gepa") != "0.1.1":
        raise ValueError("Expected Well lockfile GEPA 0.1.1")
    config = json.loads(args.config.read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        if rel.startswith(args.dataset + "/"):
            if hashlib.sha256((args.data / rel).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Changed dataset: {rel}")
    base = args.data / args.dataset
    validate_splits(*(load_rows(base / f"{s}.jsonl") for s in ("train", "dev", "test")))
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out / "STOP").exists():
        raise RuntimeError("STOP requested")
    if args.command == "optimize" and (args.out / "run.json").exists():
        raise ValueError("Use a new output directory; do not silently resume/reuse old calls")
    counts = write_inputs(base, args.out / "inputs")
    caller = UncachedCaller(config, args.out)
    def no_calls(*a, **kw):
        raise RuntimeError("Preflight must not call models")
    eval_number = 0
    def record(candidate, example, result):
        nonlocal eval_number
        eval_number += 1
        freeze(args.out / "evaluations" / f"{eval_number:06d}.json",
               {"candidate": candidate, "id": example["id"],
                "score": result[0], "feedback": result[1]})
    balanced, direct, _ = load_upstream(args.well_root, caller if args.command == "optimize" else no_calls, record)
    family = {"crepe": "CREPE", "cancer_myth": "CancerMyth"}[args.dataset]
    upstream_args = make_args(balanced, family, args.out / "inputs", args.out, config)
    train, val = (balanced.load_balanced_split(upstream_args, s) for s in ("train", "val"))
    # Validate actual upstream templates/messages and schema without calls.
    evaluator = balanced.BalancedDirectQAGEPAEvaluator(upstream_args, "gemini")
    seed = direct.load_seed_prompt(upstream_args)
    for row in train + val:
        kind = row[balanced.EXAMPLE_TYPE_KEY]
        evaluator.template_classes[kind](**row, model_final_answer="placeholder").generate()
        from src.fpqa_cli_backend import envelope
        envelope(direct.build_direct_qa_prompt(seed, row, upstream_args), exact=True)
    plan = {"protocol": "well_upstream_balanced_no_rag_cli_v3", "upstream_revision": WELL_REVISION,
            "source_sha256": hashes, "config": config, "upstream_args": vars(upstream_args),
            "counts": counts, "n_train": len(train), "n_val": len(val),
            "gepa_version": importlib.metadata.version("gepa"),
            "pydantic_version": importlib.metadata.version("pydantic"),
            "adapter_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), ROOT / "src/fpqa_well_upstream.py", ROOT / "src/fpqa_cli_backend.py")},
            "model_and_data_changes": "Configured models and selected prepared question IDs",
            "remaining_transport_differences": [
                "CLI replaces Gemini API; provider-added context cannot be guaranteed identical",
                "Empty reflection system uses a per-call empty base template catalog; persistent Codex context remains",
                "Original API temperature=0; CLI sampling temperature is not exposed by this adapter",
                "Configured CLI reasoning effort medium is not Gemini thinking_level=None",
                "CLI timeout=240s; transport errors stop with saved logs rather than Gemini API-specific retries"],
            "scope": "Optimization only; no test generation or bulk scoring"}
    freeze(args.out / "preflight.json", plan)
    if args.command == "preflight":
        print(json.dumps({k: plan[k] for k in ("protocol", "counts", "n_train", "n_val", "remaining_transport_differences")}, indent=2))
        return
    freeze(args.out / "run.json", plan)
    balanced.run_gepa(upstream_args, backend="gemini")
    print(f"Finished original Well optimization: {args.out}")


if __name__ == "__main__":
    main()
