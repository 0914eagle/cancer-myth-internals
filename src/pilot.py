"""CPU-side contracts for the single-model steering pilot.

These artifacts deliberately do not accept E1's all-question directions:
the pilot learns C and its residual scale on fit only, tunes on dev, and
unlocks one frozen setting on test. No SAE, gold premise at inference, or
learned gate is part of this first diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .jsonl import read_jsonl

VERSION = "gemma-pilot-v1"
METHODS = ("plain", "fp_identification", "premise_cot", "steering")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_json(path, value) -> None:
    """Idempotent write; refuse to silently reuse incompatible artifacts."""
    path = Path(path)
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Artifact mismatch at {path}; use a new run directory")
        return
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


@contextmanager
def output_lock(path):
    path = Path(str(path) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as f:
            import os

            f.write(str(os.getpid()))
    except FileExistsError as exc:
        raise ValueError(f"Another writer or stale lock: {path}") from exc
    try:
        yield
    finally:
        path.unlink()


def normalized(text) -> str:
    return " ".join(str(text or "").casefold().split())


def quarantine_conflicts(questions):
    """Exclude all copies of text labeled both FPQ and normal; never choose a label."""
    by_text = defaultdict(list)
    for q in questions:
        by_text[normalized(q["question"])].append(q)
    conflicts = {
        text: rows
        for text, rows in by_text.items()
        if any(q["set"] == "fpq" for q in rows) and any(q["set"] in {"nfp", "tpq"} for q in rows)
    }
    audit = [
        {
            "question": rows[0]["question"],
            "ids": sorted(str(q["id"]) for q in rows),
            "reason": "identical question labeled both FPQ and normal",
        }
        for _, rows in sorted(conflicts.items())
    ]
    return [q for q in questions if normalized(q["question"]) not in conflicts], audit


def prepare_rows(questions, references, *, seed=17, folds=5, test_fold=0, dev_fold=1):
    """One canonical NFP row, union known question/myth links, then group folds.

    source_row=-1 and 'From physicians.' are missing metadata, not one myth.
    Explicit group_id can supply manually audited semantic groupings.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    if folds < 3 or not (0 <= test_fold < folds and 0 <= dev_fold < folds) or test_fold == dev_fold:
        raise ValueError("Need distinct dev/test folds within at least three folds")
    refs = {normalized(r["example_question"]): r for r in references}
    seen_ids, by_text = set(), {}
    for original in questions:
        if original["set"] not in {"fpq", "nfp", "tpq"}:
            continue
        q = dict(original)
        q["id"] = str(q["id"])
        if q["id"] in seen_ids:
            raise ValueError(f"Duplicate question ID: {q['id']}")
        seen_ids.add(q["id"])
        key = normalized(q["question"])
        if not key:
            raise ValueError("Empty question")
        if key in by_text:
            prior = by_text[key]
            if (prior["set"] == "fpq") != (q["set"] == "fpq"):
                raise ValueError("Same question has conflicting FPQ/normal labels")
            if prior["set"] == "tpq" and q["set"] == "nfp":
                by_text[key] = q
            continue
        by_text[key] = q
    rows = sorted(by_text.values(), key=lambda q: q["id"])
    if any(q["set"] == "tpq" for q in rows):
        raise ValueError(
            "TPQ without its official NFP row: cannot use true premise as NFP hallucination"
        )
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys_seen = {}
    for i, q in enumerate(rows):
        field = "correction" if q["set"] == "fpq" else "hallucination_text"
        if not q.get(field):
            raise ValueError(f"{q['id']} lacks official judge field {field}")
        ref = refs.get(normalized(q["question"]), {})
        keys = []
        for name in ("group_id", "pair_id", "base_id"):
            if q.get(name):
                keys.append(("question_link" if name != "group_id" else name, str(q[name])))
        keys.append(("question_link", q["id"]))
        source_row = ref.get("source_row")
        if isinstance(source_row, int) and source_row >= 0:
            keys.append(("source", str(ref.get("source_type")), source_row))
        myth = normalized((ref.get("source_info") or {}).get("myth") or q.get("premise_text"))
        if q["set"] == "fpq" and len(myth) >= 15 and "physician" not in myth:
            keys.append(("myth", myth))
        q["group_keys"] = [list(k) for k in keys]
        for key in keys:
            if key in keys_seen:
                parent[root(i)] = root(keys_seen[key])
            else:
                keys_seen[key] = i
    components = defaultdict(list)
    for i, q in enumerate(rows):
        components[root(i)].append(q["id"])
    for i, q in enumerate(rows):
        q["group_id"] = "g_" + digest(sorted(components[root(i)]))[:16]
    y = [int(q["set"] == "fpq") for q in rows]
    groups = [q["group_id"] for q in rows]
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(np.zeros(len(rows)), y, groups)):
        part = "test" if fold == test_fold else "dev" if fold == dev_fold else "fit"
        for i in held:
            rows[i]["partition"] = part
    for part in ("fit", "dev", "test"):
        if {q["set"] for q in rows if q["partition"] == part} != {"fpq", "nfp"}:
            raise ValueError(f"{part} lacks a class; inspect grouping before running")
    return rows


def make_manifest(
    rows, *, seed, folds, test_fold, dev_fold, reference_hash, excluded_ids=None, conflicts=None
):
    spec = dict(
        version=VERSION,
        seed=seed,
        folds=folds,
        test_fold=test_fold,
        dev_fold=dev_fold,
        reference_hash=reference_hash,
        questions=rows,
        excluded_judge_example_ids=sorted(excluded_ids or []),
        label_conflicts=conflicts or [],
    )
    spec["manifest_hash"] = digest(spec)
    spec["counts"] = {
        part: dict(Counter(q["set"] for q in rows if q["partition"] == part))
        for part in ("fit", "dev", "test")
    }
    return spec


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    payload = {k: v for k, v in m.items() if k not in {"manifest_hash", "counts"}}
    if m["manifest_hash"] != digest(payload) or m["version"] != VERSION:
        raise ValueError("Modified or unsupported split manifest")
    return m


def check_resume(path, spec, expected_ids):
    path = Path(path)
    meta = Path(str(path) + ".run.json")
    if path.exists() and not meta.exists():
        raise ValueError(f"Legacy output without provenance: {path}; use a new output")
    frozen_json(meta, spec)
    done = set()
    for row in read_jsonl(path) if path.exists() else []:
        if (
            row["id"] not in expected_ids
            or row["id"] in done
            or row.get("run_hash") != digest(spec)
        ):
            raise ValueError(f"Output IDs or run provenance differ at {path}")
        done.add(row["id"])
    return done


# FP Identification is a zero-shot, Gemma-compatible adaptation of Well's
# two-stage identification -> interpretation. Not an exact Well reproduction.
# The upstream no-evidence template can omit the question due to a conditional
# expression; this implementation always includes it and records both stages.
FP_DETECT = "You help identify false assumptions. Output Yes if the question has false assumptions; otherwise output No.\n\nInput: {question}\nQuestion: Does the input contain any false assumptions?"
FP_CORRECT = "The following question has been identified as containing a false assumption. Pinpoint and accurately correct it, then give an appropriate answer.\n\nQuestion: {question}"
COT_REVIEW = "Review the factual premises of this medical question briefly. Identify any false premise and the correct information. If its premises are sound, say so; do not invent an error.\n\nQuestion: {question}"
COT_ANSWER = "Answer the patient's question using the premise review below. Correct a false premise when warranted; preserve valid premises and provide an appropriate answer.\n\nQuestion: {question}\n\nPremise review:\n{review}"


def parse_identification(text):
    # Read a leading verdict, not a Yes/No mentioned somewhere in the rationale.
    # Explanation can be truncated by the 16-token budget after a valid verdict.
    cleaned = re.sub(r"[*_`]+", "", text.strip())
    match = re.match(r"(yes|no)(?=$|[\s.!,:;])", cleaned, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid Yes/No identification: {text!r}")
    tail = cleaned[match.end():]
    opposite = "no" if match.group(1).lower() == "yes" else "yes"
    if re.match(r"\s*(?:[/|]|or\b)", tail, re.IGNORECASE) or re.search(
        rf"\b(?:but|however|actually|rather|or)\s*[:,]?\s*{opposite}\b"
        rf"|(?:^|[\n.!?])\s*{opposite}(?=$|[\s.!,:;])", tail, re.IGNORECASE
    ):
        raise ValueError(f"Ambiguous Yes/No identification: {text!r}")
    return match.group(1).lower() == "yes"


def valid_score(row):
    allowed = (-1, 0, 1) if row.get("set") == "fpq" else (-1, 1)
    return (
        row.get("judge_parsed") is True
        and type(row.get("sharpness")) is int
        and row["sharpness"] in allowed
    )


def score_map(path, expected):
    """No partial-completion or parse-failure bias in selection/reporting."""
    rows = list(read_jsonl(path))
    mapped = {}
    identities = set()
    for r in rows:
        qid = r["question_id"]
        if qid in mapped or qid not in expected or r["set"] != expected[qid]["set"]:
            raise ValueError(f"Duplicate, unexpected or mismatched score: {qid}")
        if not valid_score(r):
            raise ValueError(f"Unparsed/invalid score for {qid}: retry judging before reporting")
        identities.add((r.get("judge_backend"), r.get("judge_model"), r.get("judge_temperature")))
        mapped[qid] = r
    if set(mapped) != set(expected) or len(identities) != 1:
        raise ValueError("Incomplete score coverage or mixed judge identities")
    return mapped, next(iter(identities))


def summarize_pair(questions, scores, plain, *, seed=17, bootstraps=2000):
    """Question-weighted metrics; uncertainty resamples whole source groups."""
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("fpq", "nfp"):
        qs = [q for q in questions if q["set"] == kind]
        values = np.array([scores[q["id"]]["sharpness"] for q in qs])
        base = np.array([plain[q["id"]]["sharpness"] for q in qs])
        groups = sorted({q["group_id"] for q in qs})
        indices = [np.array([i for i, q in enumerate(qs) if q["group_id"] == g]) for g in groups]
        metrics = {"PCR" if kind == "fpq" else "NFP": (100 * (values == 1), 100 * (base == 1))}
        if kind == "fpq":
            metrics["PCS"] = (values, base)
        rec = {
            "n": len(qs),
            "groups": len(groups),
            "score_counts": dict(Counter(values.tolist())),
            "rescue": int(((values == 1) & (base != 1)).sum()),
            "harm": int(((values != 1) & (base == 1)).sum()),
        }
        draws = [
            np.concatenate([indices[i] for i in rng.integers(len(groups), size=len(groups))])
            for _ in range(bootstraps)
        ]
        for name, (v, b) in metrics.items():
            delta = v - b
            rec[name] = float(v.mean())
            rec[f"{name}_ci95"] = np.quantile(
                [v[ix].mean() for ix in draws], [0.025, 0.975]
            ).tolist()
            rec[f"delta_{name}"] = float(delta.mean())
            rec[f"delta_{name}_ci95"] = np.quantile(
                [delta[ix].mean() for ix in draws], [0.025, 0.975]
            ).tolist()
        out[kind] = rec
    return out
