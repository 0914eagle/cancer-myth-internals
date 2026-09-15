"""Contracts for common inventories and grouped train/calibration/evaluation roles."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from src import baseline_suite as suite
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import make_manifest


@pytest.fixture
def questions():
    # Two related, differently labelled questions per source group ensure that
    # merely disjoint IDs cannot conceal source leakage.
    rows = []
    for g in range(90):
        part = "fit" if g < 60 else "dev" if g < 80 else "test"
        for kind in ("fpq", "nfp"):
            rows.append({
                "id": f"{kind}_{g:03}", "set": kind,
                "question": f"Distinct {kind} patient question number {g}",
                "correction": "Reference correction" if kind == "fpq" else "",
                "hallucination_text": "Unsupported universal claim",
                "partition": part, "group_id": f"source_{g:03}",
            })
    return suite.check_questions(rows)


def test_nested_crossfit_has_no_group_leakage_and_one_oof_result(questions):
    plan = suite.split_plan(questions, scheme="crossfit", folds=5)
    lookup = {q["id"]: q for q in questions}
    observed = Counter()
    for fold in plan["folds"]:
        groups = []
        ids = []
        for role in ("train", "calibration", "evaluation"):
            members = fold[f"{role}_ids"]
            ids.append(set(members))
            groups.append({lookup[i]["group_id"] for i in members})
            assert {lookup[i]["set"] for i in members} == {"fpq", "nfp"}
        assert set.union(*ids) == set(lookup)
        for a in range(3):
            for b in range(a):
                assert groups[a].isdisjoint(groups[b])
                assert ids[a].isdisjoint(ids[b])
        observed.update(fold["evaluation_ids"])
    assert observed == Counter({qid: 1 for qid in lookup})
    assert suite.split_plan(questions, scheme="crossfit", folds=5) == plan


def test_holdout_keeps_test_untouched_and_calibration_inside_fit(questions):
    plan = suite.split_plan(questions, scheme="holdout", evaluation="dev")
    lookup = {q["id"]: q for q in questions}
    fold = plan["folds"][0]
    for role in ("train_ids", "calibration_ids"):
        assert {lookup[i]["partition"] for i in fold[role]} == {"fit"}
    assert {lookup[i]["partition"] for i in fold["evaluation_ids"]} == {"dev"}
    assert not any(lookup[i]["partition"] == "test" for role in
                   ("train_ids", "calibration_ids", "evaluation_ids") for i in fold[role])


def test_roles_reject_source_leakage_even_when_ids_are_different(questions):
    fold = suite.split_plan(questions)["folds"][0]
    changed = deepcopy(questions)
    lookup = {q["id"]: q for q in changed}
    lookup[fold["evaluation_ids"][0]]["group_id"] = lookup[fold["train_ids"][0]]["group_id"]
    with pytest.raises(ValueError, match="Group/ID leakage"):
        suite.validate_roles(changed, fold)


def test_roles_reject_duplicate_id_and_missing_class(questions):
    fold = suite.split_plan(questions)["folds"][0]
    duplicate = deepcopy(fold)
    duplicate["train_ids"].append(duplicate["train_ids"][0])
    with pytest.raises(ValueError, match="Invalid role IDs"):
        suite.validate_roles(questions, duplicate)
    one_class = deepcopy(fold)
    one_class["calibration_ids"] = [i for i in one_class["calibration_ids"] if i.startswith("fpq")]
    with pytest.raises(ValueError, match="both classes"):
        suite.validate_roles(questions, one_class)


def test_inventory_rejects_duplicate_text_cross_partition_group_and_missing_label(questions):
    for mutate, match in (
        (lambda q: q[1].update(question="  " + q[0]["question"].upper() + "  "), "Duplicate question"),
        (lambda q: q[-1].update(group_id=q[0]["group_id"]), "leaks across original partitions"),
        (lambda q: q[0].update(set="tpq"), "canonical FPQ/NFP"),
    ):
        changed = deepcopy(questions)
        mutate(changed)
        with pytest.raises(ValueError, match=match):
            suite.check_questions(changed)


def test_prepare_excludes_ids_and_joins_tpq_without_adding_questions(questions, tmp_path):
    manifest = make_manifest(questions, seed=17, folds=5, test_fold=0, dev_fold=1,
                             reference_hash="source", excluded_ids=["original-example"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    tpq_path = tmp_path / "tpq.jsonl"
    lookup = {q["id"]: q for q in questions}
    write_jsonl(tpq_path, [
        {"id": "well-normal-000", "question": lookup["nfp_000"]["question"].upper(),
         "presuppositions": ["A patient situation"], "doctor_suggestion": False},
        {"id": "unmatched", "question": "A question outside the canonical inventory",
         "presuppositions": ["Must not append this"]},
        {"id": "wrong-label", "question": lookup["fpq_001"]["question"],
         "presuppositions": ["Must not overwrite an FPQ label"]},
    ])
    out = tmp_path / "suite"
    counts = suite.prepare(manifest_path, out, exclude_ids=["fpq_000"], tpq_path=tpq_path)
    spec = suite.load_suite(out)
    assert len(spec["questions"]) == len(questions) - 1
    assert sum(sum(part.values()) for part in counts.values()) == len(questions) - 1
    assert spec["excluded_ids"] == ["fpq_000"]
    assert spec["source_exclusions"]["excluded_judge_example_ids"] == ["original-example"]
    assert spec["tpq_joined_ids"] == ["nfp_000"]
    saved = {q["id"]: q for q in spec["questions"]}
    assert "fpq_000" not in saved
    assert saved["nfp_000"]["set"] == "nfp"
    assert saved["nfp_000"]["tpq_annotation"]["presuppositions"] == ["A patient situation"]
    assert "tpq_annotation" not in saved["fpq_001"]
    assert suite.prepare(manifest_path, out, exclude_ids=["fpq_000"], tpq_path=tpq_path) == counts
    with pytest.raises(ValueError, match="absent"):
        suite.prepare(manifest_path, tmp_path / "bad", exclude_ids=["unknown"])
    with pytest.raises(ValueError, match="Artifact mismatch"):
        suite.prepare(manifest_path, out, tpq_path=tpq_path)
    plan = suite.split_plan(spec["questions"], scheme="crossfit")
    base = suite.export_splits(out, plan)
    assert "fpq_000" not in {q["id"] for p in base.glob("fold_*/*.jsonl") for q in read_jsonl(p)}


def test_export_rejects_stale_inventory_hash(questions, tmp_path):
    manifest = make_manifest(questions, seed=17, folds=5, test_fold=0, dev_fold=1,
                             reference_hash="source")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    out = tmp_path / "suite"
    suite.prepare(path, out)
    plan = suite.split_plan(questions)
    plan["question_hash"] = "different questions"
    with pytest.raises(ValueError, match="inventory differs"):
        suite.export_splits(out, plan)


def test_original_reference_join_and_demo_exclusion(questions, tmp_path):
    from src.well_eval import example_questions
    rows = deepcopy(questions)
    rows[0]["question"] = example_questions()[0]
    demo_id = rows[0]["id"]
    manifest = make_manifest(rows, seed=17, folds=5, test_fold=0, dev_fold=1,
                             reference_hash="original")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    refs = [{"question": q["question"].upper(), "source_myth": "Original myth " + q["id"],
             "presupposition_correction": "Original correction " + q["id"]}
            for q in rows if q["set"] == "fpq" and q["id"] != demo_id]
    refpath = tmp_path / "references.jsonl"
    write_jsonl(refpath, refs)
    suite.prepare(path, tmp_path / "out", fpq_source=refpath)
    saved = suite.load_suite(tmp_path / "out")
    assert saved["excluded_well_demo_ids"] == [demo_id]
    assert saved["fpq_reference_sha256"] == suite.file_digest(refpath)
    assert demo_id not in {q["id"] for q in saved["questions"]}
    for q in saved["questions"]:
        if q["set"] == "fpq":
            assert q["premise_text"] == "Original myth " + q["id"]
            assert q["correction"] == "Original correction " + q["id"]
            assert q["legacy_correction"] == "Reference correction"
    write_jsonl(refpath, refs[:-1])
    with pytest.raises(ValueError, match="Missing original"):
        suite.prepare(path, tmp_path / "missing", fpq_source=refpath)
    write_jsonl(refpath, refs + refs[:1])
    with pytest.raises(ValueError, match="duplicate"):
        suite.prepare(path, tmp_path / "duplicate", fpq_source=refpath)
