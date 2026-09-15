"""Snapshot reusable baseline caches into a separate 1024-final-token suite.

CPU/file operations only. Does not edit the source or run models/judges.
Does not relabel 512-token answers as 1024-token generations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline_generation as bg
from src.baseline_suite import load_suite
from src.pilot import digest, file_digest, output_lock


def inventory(source):
    """Only immutable cache/spec files; never copy run locks or final ledgers."""
    paths = [source / name for name in ("suite.json", "questions.jsonl", "model/identity.json")]
    for name in ("shared_generation", "shared_binary"):
        paths.extend(sorted((source / "model" / name).glob("*.json")))
    for name in ("features", "detection"):
        folder = source / "model" / name
        if (folder / "spec.json").exists():
            paths.append(folder / "spec.json")
            paths.extend(sorted((folder / "cache").glob("*.npz" if name == "features" else "*.json")))
    # Existing question-level gate runs are independent of final answer length.
    for folder in sorted((source / "gates").glob("*")):
        if folder.is_dir() and (folder / "result.json").exists() and not Path(str(folder) + ".lock").exists():
            paths.extend(sorted(folder.glob("*.json")))
            if (folder / "report.md").exists():
                paths.append(folder / "report.md")
    return paths


def copy_verified(source, destination, expected):
    if destination.exists():
        if destination.is_symlink() or file_digest(destination) != expected:
            raise ValueError(f"Destination differs: {destination}")
        return
    if source.is_symlink() or file_digest(source) != expected:
        raise ValueError(f"Source snapshot changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
    try:
        shutil.copyfile(source, tmp)
        if file_digest(tmp) != expected:
            raise ValueError(f"Source changed during copy: {source}")
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if source == out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("Source and destination must be separate, non-nested directories")
    with output_lock(Path(str(out) + ".migration")):
        plan_path = out / "cache_migration_plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            if plan["source"] != str(source) or plan["destination"] != str(out):
                raise ValueError("Migration paths changed")
        else:
            if out.exists() and any(out.iterdir()):
                raise ValueError("Destination must be new/empty; do not mix experiments")
            suite = load_suite(source)
            identity = json.loads((source / "model/identity.json").read_text())
            if identity["implementation_sha256"] != file_digest(ROOT / "src/baseline_generation.py"):
                raise ValueError("Source generation implementation changed; use original code")
            if identity["prompts"] != bg.PROMPTS:
                raise ValueError("Source prompt protocol differs")
            original_spec = json.loads((source / "model/generation/plain/spec.json").read_text())
            if original_spec["final_tokens"] != 512 or original_spec["review_tokens"] != 1024:
                raise ValueError("Expected 512-final/1024-review source run")
            files = {}
            for path in inventory(source):
                if path.is_symlink():
                    raise ValueError(f"Do not snapshot symlinks: {path}")
                files[str(path.relative_to(source))] = file_digest(path)
            plan = {"version": "baseline-final1024-migration-v1", "source": str(source),
                    "destination": str(out), "files": files, "questions": len(suite["questions"]),
                    "source_plain_spec_sha256": file_digest(source / "model/generation/plain/spec.json"),
                    "final_tokens": 1024, "review_tokens": 1024, "extraction_tokens": 512,
                    "policy": "Exact cache keys only; no EOS answer rebadging; original run read-only",
                    "calls": {"gpu": 0, "judge": 0}}
            bg._atomic_json(plan_path, plan, frozen=True)
        done = out / "cache_migration_complete.json"
        if done.exists():
            if json.loads(done.read_text())["plan_sha256"] != digest(plan):
                raise ValueError("Migration plan changed after completion")
            print(f"Migration already complete: {out}. No source recopy, no model calls.")
            return
        for relative, sha in plan["files"].items():
            copy_verified(source / relative, out / relative, sha)
        load_suite(out)
        # Validate feature/detection signatures without loading a language model.
        feature_n = len(bg.load_feature_records(out / "model"))
        detection_n = len(bg.load_detection_records(out / "model"))
        defaults = {"final_tokens": 1024, "requires_cache_migration": True}
        bg._atomic_json(out / "generation_defaults.json", defaults, frozen=True)
        summary = {"plan_sha256": digest(plan), "copied_files": len(plan["files"]),
                   "feature_questions": feature_n, "detection_questions": detection_n,
                   "final_tokens": 1024, "judge_calls": 0}
        bg._atomic_json(done, summary, frozen=True)
        print(json.dumps(summary, indent=2))
        print(f"Prepared {out}. Next: SUITE_DIR={out} bash scripts/run_baselines.sh generate")
        print("All methods use final cap 1024. Shared reasoning/extraction/binary caches reuse exact keys.")
        print("512 final answers remain in source; final outputs are generated anew, not continued from saved text.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_dir, args.out_dir)
