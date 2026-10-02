"""Pin public sources and prepare official splits; no model calls, no test resplitting."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import unicodedata
import urllib.request

WELL_REV = "6c9770f65da6e9c50250cf94e35b07f258288e38"
FALSEQA_REV = "9a9729974d8c5cfbfc6929e5b830111c0ed7c0c7"
CREPE_REV = "cd11ed2eb5b93c3cd9bd4f2a6255e4adee7c658f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def checkout(root, name, repo, revision):
    dest = root / name
    if not dest.exists():
        dest.mkdir(parents=True)
        subprocess.run(["git", "init", str(dest)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", repo], check=True)
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", "FETCH_HEAD"], check=True)
    current = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if current != revision:
        raise ValueError(f"{dest}: expected pinned {revision}, found {current}; use a fresh source directory")


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    checkout(root, "Well", "https://github.com/ShenranTomWang/Well.git", WELL_REV)
    checkout(root, "FalseQA", "https://github.com/thunlp/FalseQA.git", FALSEQA_REV)
    dest = root / "CREPE"
    dest.mkdir(exist_ok=True)
    for split in ("train", "dev", "test"):
        path = dest / f"{split}.jsonl"
        if path.exists():
            continue
        url = f"https://huggingface.co/datasets/tasksource/CREPE/resolve/{CREPE_REV}/{split}.jsonl"
        temp = path.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=240) as response, temp.open("wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
        temp.replace(path)
    write_json(dest / "source.json", {"repo": "tasksource/CREPE", "revision": CREPE_REV})


def crepe_rows(path, split, excluded_ids):
    rows, dropped = [], []
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(raw)
        labels = {x.replace("_", " ").strip().lower() for x in row["labels"]}
        if len(labels) != 1 or not labels <= {"normal", "false presupposition"}:
            dropped.append({"source_line": line, "reason": "ambiguous_or_unknown_label", "id": row.get("id")})
            continue
        if str(row["id"]) in excluded_ids:
            dropped.append({"source_line": line, "reason": "well_few_shot_reserved", "id": row["id"]})
            continue
        rows.append({"id": f"crepe:{row['id']}", "dataset": "crepe", "split": split,
                     "question": row["question"], "label": int("false presupposition" in labels),
                     "presuppositions": row.get("presuppositions", []),
                     "corrections": row.get("corrections", []),
                     "reference_answer": row.get("comment", ""), "source_line": line})
    return rows, dropped


def falseqa_rows(path, split):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f), 2):
            if row["label"] not in {"0", "1"}:
                raise ValueError(f"Unknown FalseQA label at {path}:{line}")
            rows.append({"id": f"falseqa:{split}:{line}", "dataset": "falseqa", "split": split,
                         "question": row["question"], "label": int(row["label"]),
                         "presuppositions": [], "corrections": [],
                         "reference_answer": row["answer"], "source_line": line})
    return rows


def remove_overlap(splits):
    """Preserve official test first, then dev; drop exact normalized overlap from earlier splits.

    No semantic grouping is guessed from CSV order. Conflicting labels are excluded everywhere.
    """
    labels = {}
    for rows in splits.values():
        for r in rows:
            labels.setdefault(norm(r["question"]), set()).add(r["label"])
    conflicts = {q for q, ys in labels.items() if len(ys) > 1}
    seen, cleaned, dropped = set(), {}, []
    for split in ("test", "dev", "train"):
        cleaned[split] = []
        for row in splits[split]:
            key = norm(row["question"])
            reason = "conflicting_question_labels" if key in conflicts else "duplicate_question" if key in seen else None
            if reason:
                dropped.append({"id": row["id"], "split": split, "reason": reason})
                continue
            if not key:
                raise ValueError("Empty question")
            seen.add(key)
            cleaned[split].append(row)
    return cleaned, dropped


def balanced(rows, per_class, seed):
    rng = random.Random(seed)
    groups = [[r for r in rows if r["label"] == y] for y in (1, 0)]
    for group in groups:
        rng.shuffle(group)
    n = min(map(len, groups)) if per_class is None else per_class
    if any(len(g) < n for g in groups):
        raise ValueError(f"Not enough examples for {n} per class")
    return [r for pair in zip(groups[0][:n], groups[1][:n]) for r in pair]


def prepare(sources, out, seed=42, dev_per_class=50):
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite prepared data: {out}")
    for name, revision in (("Well", WELL_REV), ("FalseQA", FALSEQA_REV)):
        actual = subprocess.check_output(["git", "-C", str(sources / name), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"{name}: unexpected revision {actual}")
    shot_path = sources / "Well/data_gen/CREPE/few_shot.json"
    excluded = {str(r["id"]) for r in json.loads(shot_path.read_text())}
    manifest = {"seed": seed, "dev_per_class": dev_per_class, "model_calls": 0,
                "sources": {"Well": WELL_REV, "FalseQA": FALSEQA_REV, "CREPE_HF_mirror": CREPE_REV},
                "note": "Official splits; no 300-character filter. Test untouched except exact duplicates/conflicting labels. No semantic pair IDs inferred.",
                "raw_sha256": {}, "files": {}, "datasets": {}}
    for dataset in ("crepe", "falseqa"):
        splits, exclusions = {}, []
        for split in ("train", "dev", "test"):
            if dataset == "crepe":
                path = sources / "CREPE" / f"{split}.jsonl"
                rows, dropped = crepe_rows(path, split, excluded)
                exclusions.extend(dict(r, split=split) for r in dropped)
            else:
                path = sources / "FalseQA/dataset" / f"{'valid' if split == 'dev' else split}.csv"
                rows = falseqa_rows(path, split)
            manifest["raw_sha256"][str(path.relative_to(sources))] = sha(path)
            splits[split] = rows
        raw_counts = {s: dict(Counter(r["label"] for r in rs)) for s, rs in splits.items()}
        splits, dropped = remove_overlap(splits)
        exclusions.extend(dropped)
        for split, rows in splits.items():
            write_rows(out / dataset / f"{split}.jsonl", rows)
            # The generation input never includes a label, gold premise, or answer.
            write_rows(out / dataset / f"{split}.questions.jsonl", [{"id": r["id"], "question": r["question"]} for r in rows])
        train = balanced(splits["train"], None, seed)
        dev = balanced(splits["dev"], dev_per_class, seed)
        for split, rows in (("train", train), ("dev", dev)):
            write_rows(out / dataset / f"opt_{split}.jsonl", rows)
            write_rows(out / dataset / f"opt_{split}_fpq.jsonl", [r for r in rows if r["label"]])
        write_rows(out / dataset / "exclusions.jsonl", exclusions)
        manifest["datasets"][dataset] = {
            "eligible_before_dedup": raw_counts,
            "counts": {s: {"fpq": sum(r["label"] for r in rs), "nfp": sum(not r["label"] for r in rs)} for s, rs in splits.items()},
            "optimization_train": len(train), "optimization_dev": len(dev),
            "exclusions": dict(Counter(r["reason"] for r in exclusions))}
    for path in sorted(out.rglob("*.jsonl")):
        manifest["files"][str(path.relative_to(out))] = sha(path)
    write_json(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download:
        download(args.sources)
    result = prepare(args.sources, args.out, args.seed)
    print(json.dumps(result["datasets"], indent=2))


if __name__ == "__main__":
    main()
