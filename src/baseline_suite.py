"""Frozen question inventory and leak-aware evaluation splits for baseline comparisons.

These are our splits, NOT the unpublished Well-Actually paper split. No model
downloads, GPU work, or judge calls occur in this module.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from src.jsonl import read_jsonl, write_jsonl
from src.pilot import digest, file_digest, frozen_json, load_manifest, normalized

VERSION = "medical-baselines-v1"


def check_questions(rows):
    rows = [dict(q) for q in rows]
    ids, texts, group_parts = set(), set(), {}
    for q in rows:
        if not isinstance(q.get("id"), str) or not q["id"] or q["id"] in ids:
            raise ValueError("Empty/duplicate/non-string question ID")
        ids.add(q["id"])
        if q.get("set") not in {"fpq", "nfp"}:
            raise ValueError("Only canonical FPQ/NFP rows; TPQ annotations are not extra questions")
        if not isinstance(q.get("question"), str) or not normalized(q["question"]):
            raise ValueError("Empty question")
        text = normalized(q["question"])
        if text in texts:
            raise ValueError("Duplicate question text; canonicalize before preparing")
        texts.add(text)
        if not q.get("group_id") or q.get("partition") not in {"fit", "dev", "test"}:
            raise ValueError("Every row needs a source group and original fit/dev/test partition")
        if group_parts.setdefault(q["group_id"], q["partition"]) != q["partition"]:
            raise ValueError("Source group leaks across original partitions")
        if q["set"] == "fpq" and not q.get("correction"):
            raise ValueError("FPQ needs original reference correction for answer evaluation")
    if not rows or {q["set"] for q in rows} != {"fpq", "nfp"}:
        raise ValueError("Need both FPQ and NFP questions")
    return sorted(rows, key=lambda q: q["id"])


def prepare(manifest_path, out, *, exclude_ids=(), tpq_path=None, fpq_source=None):
    """Keep existing canonical labels/groups. Optionally JOIN TPQ, never append it."""
    m = load_manifest(manifest_path)
    rows = check_questions(m["questions"])
    excluded = sorted(set(exclude_ids))
    if set(excluded) - {q["id"] for q in rows}:
        raise ValueError("Excluded ID is absent from source manifest")
    rows = [q for q in rows if q["id"] not in excluded]
    demo_exclusions = []
    # These few-shot questions are part of the frozen evaluator, not test items.
    from src.well_eval import example_questions
    demo_texts = {normalized(q) for q in example_questions()}
    demo_exclusions = [q["id"] for q in rows if normalized(q["question"]) in demo_texts]
    rows = [q for q in rows if q["id"] not in demo_exclusions]
    if fpq_source:
        refs = {}
        for ref in read_jsonl(fpq_source):
            key = normalized(ref.get("question", ""))
            if not key or key in refs:
                raise ValueError("Empty/duplicate question in FPQ reference file")
            refs[key] = ref
        for q in rows:
            if q["set"] != "fpq":
                continue
            ref = refs.get(normalized(q["question"]))
            if not ref or not ref.get("source_myth") or not ref.get("presupposition_correction"):
                raise ValueError(f"Missing original FPQ premise/correction for {q['id']}")
            q["legacy_correction"] = q["correction"]
            q["premise_text"] = ref["source_myth"]
            q["correction"] = ref["presupposition_correction"]
    join_audit = []
    if tpq_path:
        tpq = list(read_jsonl(tpq_path))
        by_text = {}
        for q in tpq:
            key = normalized(q.get("question", q.get("example_question", "")))
            if not key or key in by_text:
                raise ValueError("Empty/duplicate TPQ question in annotation file")
            by_text[key] = q
        for q in rows:
            other = by_text.get(normalized(q["question"]))
            if other and q["set"] == "nfp":
                q["tpq_annotation"] = {
                    k: other[k] for k in ("id", "presuppositions", "doctor_suggestion") if k in other
                }
                join_audit.append(q["id"])
    rows = check_questions(rows)
    out = Path(out)
    spec = {"version": VERSION, "source_manifest_sha256": file_digest(manifest_path),
            "source_manifest_hash": m["manifest_hash"], "questions": rows,
            "excluded_ids": excluded,
            "excluded_well_demo_ids": demo_exclusions,
            "fpq_reference_sha256": file_digest(fpq_source) if fpq_source else None,
            "tpq_sha256": file_digest(tpq_path) if tpq_path else None,
            "tpq_joined_ids": join_audit,
            "source_exclusions": {k: m.get(k) for k in ("excluded_judge_example_ids", "label_conflicts")},
            "warning": "Our existing grouped split; not Well paper IDs. Dev previously inspected."}
    frozen_json(out / "suite.json", spec)
    write_frozen_jsonl(out / "questions.jsonl", rows)
    for part in ("fit", "dev", "test"):
        write_frozen_jsonl(out / "questions" / f"{part}.jsonl", [q for q in rows if q["partition"] == part])
    return {part: dict(Counter(q["set"] for q in rows if q["partition"] == part))
            for part in ("fit", "dev", "test")}


def write_frozen_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        if list(read_jsonl(path)) != rows:
            raise ValueError(f"Existing JSONL differs: {path}")
    else:
        write_jsonl(path, rows)


def load_suite(out):
    out = Path(out)
    spec = json.loads((out / "suite.json").read_text())
    if spec["version"] != VERSION or list(read_jsonl(out / "questions.jsonl")) != spec["questions"]:
        raise ValueError("Modified/unsupported question inventory")
    check_questions(spec["questions"])
    return spec


def _group_folds(rows, n, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    if n < 2:
        raise ValueError("Need at least two folds")
    y = np.array([q["set"] == "fpq" for q in rows], dtype=int)
    groups = [q["group_id"] for q in rows]
    splits = list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
        np.zeros(len(rows)), y, groups))
    for train, held in splits:
        if set(y[train]) != {0, 1} or set(y[held]) != {0, 1}:
            raise ValueError("A fold lacks a class; fewer folds or more source groups needed")
    return splits


def split_plan(rows, *, scheme="holdout", evaluation="dev", folds=5, seed=17):
    """Nested role export shared by probes, GEPA and finetuning.

    For probes, setting selection uses further group CV inside train. Calibration
    sets thresholds only. GEPA/finetuning may use calibration as their dev set.
    Evaluation IDs are never used in either role.
    """
    rows = check_questions(rows)
    if scheme not in {"holdout", "crossfit"} or evaluation not in {"dev", "test"}:
        raise ValueError("Unsupported scheme/evaluation")
    outer = []
    if scheme == "holdout":
        outer = [([q for q in rows if q["partition"] == "fit"],
                  [q for q in rows if q["partition"] == evaluation])]
    else:
        outer = [([rows[i] for i in tr], [rows[i] for i in te])
                 for tr, te in _group_folds(rows, folds, seed)]
    specs = []
    for k, (pool, held) in enumerate(outer):
        if not pool or not held:
            raise ValueError("Empty training/evaluation partition")
        tr, cal = _group_folds(pool, 3, seed + k)[0]
        roles = {"train_ids": [pool[i]["id"] for i in tr],
                 "calibration_ids": [pool[i]["id"] for i in cal],
                 "evaluation_ids": [q["id"] for q in held]}
        validate_roles(rows, roles)
        specs.append({"fold": k, **roles})
    if scheme == "crossfit":
        evaluated = [qid for s in specs for qid in s["evaluation_ids"]]
        if len(evaluated) != len(rows) or set(evaluated) != {q["id"] for q in rows}:
            raise ValueError("Each question must have exactly one out-of-fold prediction")
    return {"scheme": scheme, "evaluation": evaluation if scheme == "holdout" else "out_of_fold",
            "seed": seed, "folds": specs, "question_hash": digest(rows),
            "note": "Cross-fitting does not erase earlier dev inspection; no paper-split equivalence."}


def validate_roles(rows, roles):
    by_id = {q["id"]: q for q in rows}
    used_ids, used_groups = set(), set()
    for name in ("train_ids", "calibration_ids", "evaluation_ids"):
        ids = roles[name]
        if len(ids) != len(set(ids)) or not set(ids) <= by_id.keys():
            raise ValueError("Invalid role IDs")
        gs = {by_id[i]["group_id"] for i in ids}
        if used_ids & set(ids) or used_groups & gs:
            raise ValueError("Group/ID leakage between train, calibration, evaluation")
        if {by_id[i]["set"] for i in ids} != {"fpq", "nfp"}:
            raise ValueError("Every role needs both classes")
        used_ids.update(ids)
        used_groups.update(gs)


def export_splits(out, spec):
    suite = load_suite(out)
    if spec["question_hash"] != digest(suite["questions"]):
        raise ValueError("Split inventory differs")
    base = Path(out) / "splits" / (spec["scheme"] + "_" + spec["evaluation"])
    frozen_json(base / "plan.json", spec)
    lookup = {q["id"]: q for q in suite["questions"]}
    for fold in spec["folds"]:
        for role in ("train", "calibration", "evaluation"):
            write_frozen_jsonl(base / f"fold_{fold['fold']}" / f"{role}.jsonl",
                              [lookup[i] for i in fold[f"{role}_ids"]])
    return base
