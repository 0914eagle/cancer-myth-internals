"""Freeze existing grouped Cancer-Myth splits for question-only prompting."""
import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_fpqa_prompt_data import balanced, norm, sha, write_json, write_rows
from src.fpqa_prompting import load_rows, validate_splits


def select_groups(rows, target, seed):
    """Exact label counts without splitting linked source groups; no model outputs."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    reached = {(0, 0): None}
    for group in keys:
        positive = sum(r["set"] == "fpq" for r in groups[group])
        negative = len(groups[group]) - positive
        for before in list(reached):
            after = (before[0] + positive, before[1] + negative)
            if after[0] <= target[0] and after[1] <= target[1] and after not in reached:
                reached[after] = (before, group)
        if target in reached:
            break
    if target not in reached:
        raise ValueError("Cannot satisfy counts while preserving groups")
    selected = set()
    cursor = target
    while reached[cursor] is not None:
        cursor, group = reached[cursor]
        selected.add(group)
    return selected


def prepare(source, out, well_root, protocol="existing"):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite frozen data")
    original = load_rows(source)
    if protocol == "test100":
        test_groups = select_groups(original, (100, 100), 42)
        remaining = [r for r in original if r["group_id"] not in test_groups]
        dev_groups = select_groups(remaining, (100, 18), 43)
        original = [dict(r, partition="test" if r["group_id"] in test_groups else
                         "dev" if r["group_id"] in dev_groups else "fit") for r in original]
    reserved = set()
    for family in ("CancerMyth", "CancerMythNFP"):
        for row in json.loads((well_root / "data_gen" / family / "eval_few_shot.json").read_text()):
            reserved.add(norm(row["question"]))
    splits = {s: [] for s in ("train", "dev", "test")}
    groups = {}
    for row in original:
        split = {"fit": "train", "dev": "dev", "test": "test"}[row["partition"]]
        group = row["group_id"]
        if groups.setdefault(group, split) != split:
            raise ValueError("Group overlaps splits")
        if norm(row["question"]) in reserved:
            raise ValueError("Question appears in judge demonstrations")
        splits[split].append(dict(id=row["id"], dataset="cancer_myth", split=split,
                                 group_id=group, question=row["question"], label=int(row["set"] == "fpq"),
                                 presuppositions=[row["premise_text"]], reference_answer=row["correction"]))
    validate_splits(*(splits[s] for s in ("train", "dev", "test")))
    for split, rows in splits.items():
        write_rows(out / "cancer_myth" / f"{split}.jsonl", rows)
        write_rows(out / "cancer_myth" / f"{split}.questions.jsonl",
                   [{"id": r["id"], "question": r["question"]} for r in rows])
    sizes = (("train", 30), ("dev", 18)) if protocol == "test100" else (("train", None), ("dev", 30))
    opt_counts = {}
    for split, size in sizes:
        rows = balanced(splits[split], size, 42)
        write_rows(out / "cancer_myth" / f"opt_{split}.jsonl", rows)
        opt_counts[split] = len(rows)
    manifest = dict(seed=42, source=str(source), source_sha256=sha(source),
                    protocol=protocol,
                    note="Group-disjoint split, not Well's unreleased ID split. Previously analyzed benchmark; not a fresh research holdout. No relabeling. test100 uses deterministic group subset counts, not uniform question sampling; old GEPA must not be reused.",
                    counts={s: dict(Counter("fpq" if r["label"] else "nfp" for r in rows)) for s, rows in splits.items()},
                    optimization_train=opt_counts["train"], optimization_dev=opt_counts["dev"],
                    files={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*.jsonl"))})
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=ROOT / "configs/premise_sft/cancer_questions.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--protocol", choices=("existing", "test100"), default="existing")
    p.add_argument("--well-root", type=Path, default=ROOT / "external/fpqa_20261002/Well")
    args = p.parse_args()
    print(json.dumps(prepare(args.source, args.out, args.well_root, args.protocol), indent=2))
