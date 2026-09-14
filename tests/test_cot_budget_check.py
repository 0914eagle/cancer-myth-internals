import json
from pathlib import Path

import pytest

from scripts.run_pilot import generation_rows
from scripts import cot_budget_check as mod
from src.jsonl import write_jsonl, load_json
from src.pilot import digest, file_digest


def test_generation_passes_1024_to_only_selected_dev_and_resumes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_pilot as runner
    from src import pilot_model

    qs = [
        {"id": i, "partition": p, "set": "fpq", "question": i}
        for i, p in [("a", "dev"), ("b", "dev"), ("c", "test")]
    ]
    monkeypatch.setattr(runner, "load_manifest", lambda _: {"questions": qs, "manifest_hash": "m"})
    monkeypatch.setattr(runner, "load_config", lambda _: {"source_model": {"model_id": "fake"}})
    monkeypatch.setattr(pilot_model, "load_model", lambda _: (None, None))
    monkeypatch.setattr(pilot_model, "model_identity", lambda *a: {"checkpoint": "same"})
    calls = []

    def fake_generate(model, tokenizer, questions, **kwargs):
        calls.append([q["id"] for q in questions])
        assert kwargs["review_tokens"] == 1024 and kwargs["max_new_tokens"] == 512
        return ["answer" for _ in questions], [{"review": "review"} for _ in questions]

    monkeypatch.setattr(pilot_model, "generate_batch", fake_generate)
    ids = tmp_path / "ids.json"
    ids.write_text('["b"]')
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"identity": {"checkpoint": "same"}}))
    args = SimpleNamespace(
        config="unused",
        manifest="unused",
        partition="dev",
        method="premise_cot",
        max_new_tokens=512,
        review_tokens=1024,
        batch_size=1,
        question_ids=ids,
        identity_reference=identity,
        output=tmp_path / "new.jsonl",
    )
    runner.generate(args)
    runner.generate(args)
    assert calls == [["b"]]
    assert load_json(str(args.output) + ".run.json")["review_tokens"] == 1024
    identity.write_text(json.dumps({"identity": {"checkpoint": "changed"}}))
    args.output = tmp_path / "mismatch.jsonl"
    with pytest.raises(ValueError, match="identity differs"):
        runner.generate(args)
    assert calls == [["b"]]


def test_subset_is_dev_cot_only_preserves_order_and_rejects_invalid_ids(tmp_path):
    manifest = {
        "questions": [
            {"id": i, "partition": p} for i, p in [("a", "dev"), ("b", "dev"), ("c", "test")]
        ]
    }
    path = tmp_path / "ids.json"
    path.write_text(json.dumps(["b", "a"]))
    assert [q["id"] for q in generation_rows(manifest, "dev", "premise_cot", path)] == ["a", "b"]
    for partition, method in [("test", "premise_cot"), ("dev", "plain")]:
        with pytest.raises(ValueError, match="only available"):
            generation_rows(manifest, partition, method, path)
    for ids in ([], ["a", "a"], ["c"], [1], {"a": 1}):
        path.write_text(json.dumps(ids))
        with pytest.raises(ValueError, match="unique nonempty"):
            generation_rows(manifest, "dev", "premise_cot", path)
    assert len(generation_rows(manifest, "dev", "plain")) == 2


@pytest.mark.parametrize("code_changed", [False, True])
def test_report_pairs_old_new_without_judging_and_preserves_human_edits(
    tmp_path, monkeypatch, code_changed
):
    monkeypatch.setattr(
        mod.subprocess, "run", lambda *a, **k: pytest.fail("report must not call models")
    )
    source = tmp_path / "source"
    source.write_text("baseline")
    old = {
        "identity": {"model": "frozen"},
        "manifest_hash": "m",
        "partition": "dev",
        "method": "premise_cot",
        "max_new_tokens": 512,
        "batch_size": 1,
        "seed": 17,
        "decoding": "greedy_cache",
        "review_tokens": 128,
    }
    plan = {
        "old_run": old,
        "question_ids": ["a"],
        "sources": {str(source): file_digest(source)},
        "cases": [
            {
                "id": "a",
                "question": "q",
                "reference": "ref",
                "old_review": "short",
                "old_answer": "old",
                "old_review_output_tokens": 128,
            }
        ],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "question_ids.json").write_text(json.dumps(["a"]))
    run = {
        **old,
        "review_tokens": 1024,
        "question_ids": ["a"],
        "diagnostic_subset_hash": file_digest(tmp_path / "question_ids.json"),
    }
    if code_changed:
        run["identity"] = {**old["identity"], "implementation_hash": "updated-code"}
        control_run = {**run, "review_tokens": 128}
        control_path = tmp_path / "premise_cot_r128_control.jsonl"
        Path(str(control_path) + ".run.json").write_text(json.dumps(control_run))
        write_jsonl(
            control_path,
            [
                {
                    "id": "a",
                    "question": "q",
                    "run_hash": digest(control_run),
                    "review": "short",
                    "response": "old",
                }
            ],
        )
    path = tmp_path / "premise_cot_r1024.jsonl"
    Path(str(path) + ".run.json").write_text(json.dumps(run))
    write_jsonl(
        path,
        [
            {
                "id": "a",
                "question": "q",
                "run_hash": digest(run),
                "review": "longer",
                "response": "new",
                "review_output_tokens": 170,
            }
        ],
    )
    mod.report(tmp_path)
    summary = load_json(tmp_path / "summary.json")
    assert summary["answers_changed"] == 1 and summary["reviews_at_1024_tokens"] == 0
    assert summary["judge_calls"] == 0 and summary["scores_assigned"] == 0
    if code_changed:
        assert summary["control_audit"]["changed_ids"] == []
    review = tmp_path / "review.md"
    review.write_text(review.read_text() + "human note")
    mod.report(tmp_path)
    assert review.read_text().endswith("human note")
    run["max_new_tokens"] = 1024
    Path(str(path) + ".run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="max_new_tokens"):
        mod.report(tmp_path)
