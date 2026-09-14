import json

import numpy as np
import pytest

from src import overnight_c_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def direction(name="C", **kwargs):
    return {"name": name, "layer": 21, "path": "fit.npz", "key": "C", **kwargs}


def dose(task_id, alpha, question="q1", policy="all", **kwargs):
    return {"id": task_id, "kind": "dose", "question_id": question,
            "positive": "correct", "negative": "wrong", "alpha": alpha,
            "direction": direction(), "policy": policy, **kwargs}


def likelihood(p, n):
    return {side: {"mean_logp": value, "sum_logp": value * 40, "token_count": 40,
                   "prefix32": {"mean_logp": value, "token_count": 32},
                   "rest": {"mean_logp": value, "token_count": 8}}
            for side, value in (("positive", p), ("negative", n))}


def artifact(out, task, result=None, status="complete", **kwargs):
    write(out / "tasks" / f"{task['id']}.json",
          {"id": task["id"], "task": task, "result": result or {}, "status": status, **kwargs})


@pytest.fixture(autouse=True)
def no_plot(monkeypatch):
    monkeypatch.setattr(report, "_plot", lambda *args: (None, None))


def read_summary(out, tasks, **kwargs):
    write(out / "plan.json", {"tasks": tasks, "judge_calls": 0, "warnings": ["Selected dev only"], **kwargs})
    paths = report.write_report(out)
    assert paths["report"].exists()
    return json.loads(paths["summary"].read_text())


def test_dose_requires_matched_control_and_keeps_both_changes(tmp_path):
    tasks = [dose("zero", 0), dose("dose", 0.3), dose("prefill", 0.3, policy="prefill"),
             dose("missing_zero", 0.3, question="q2"), dose("failed", 0.3, question="q3"),
             dose("pending", 0.3, question="q4")]
    artifact(tmp_path, tasks[0], likelihood(-2, -3))
    artifact(tmp_path, tasks[1], likelihood(-3, -5))
    artifact(tmp_path, tasks[2], likelihood(-1, -3))
    artifact(tmp_path, tasks[3], likelihood(-1, -2))
    artifact(tmp_path, tasks[4], status="failed", error="out of memory")
    summary = read_summary(tmp_path, tasks)
    row = next(r for r in summary["dose_groups"] if r["alpha"] == 0.3 and r["policy"] == "all")
    assert row["planned"] == 4
    assert row["complete"] == 2
    assert row["paired"] == 1
    assert row["full"]["delta_margin"] == 1
    # Increasing margin alone must not hide worsening correction likelihood.
    assert row["full"]["positive_delta"] == -1
    assert row["full"]["negative_delta"] == -2
    prefill = next(r for r in summary["dose_pairs"] if r["policy"] == "prefill")
    assert prefill["control_task_id"] == "zero"
    assert summary["statuses"] == {"complete": 4, "failed": 1, "timed_out": 0, "skipped": 0, "pending": 1}
    text = (tmp_path / "report.md").read_text()
    assert "out of memory" in text and "No completed artifact" in text
    assert "not PCR" in text


def test_control_never_reused_for_different_pair_or_random_direction(tmp_path):
    tasks = [dose("zero", 0), dose("other_pair", 0.1, positive="another answer"),
             dose("random", 0.1, direction=direction("random", random_seed=0))]
    for t in tasks:
        artifact(tmp_path, t, likelihood(-1, -2))
    summary = read_summary(tmp_path, tasks)
    assert all(r["paired"] == 0 for r in summary["dose_groups"] if r["alpha"] == 0.1)


def test_projection_excludes_zero_nfp_missing_and_handles_ties(tmp_path):
    tasks = [{"id": str(i), "kind": "project", "label": label, "set": kind,
              "direction": direction()} for i, (label, kind) in enumerate(
                  [(1, "fpq"), (-1, "fpq"), (0, "fpq"), (1, "nfp"), (None, "fpq")])]
    for task in tasks:
        artifact(tmp_path, task, {"prefix32": {"dot": 1}, "full": {"dot": 2}})
    summary = read_summary(tmp_path, tasks)
    for row in summary["projection_groups"]:
        assert row["positive"] == row["negative"] == 1
        assert row["auroc"] == 0.5
    assert report.descriptive_auc([1], [1]) is None
    assert report.descriptive_auc([0, 2, 1], [-1, 1, 1]) == 1


def test_generations_unjudged_corrupt_failed_pending_all_visible(tmp_path):
    tasks = [{"id": str(i), "kind": "generate", "direction": direction(), "alpha": 0.1,
              "baseline_response": "same"} for i in range(5)]
    artifact(tmp_path, tasks[0], {"response": "same", "output_tokens": 10, "hit_token_cap": False})
    artifact(tmp_path, tasks[1], {"response": "different", "output_tokens": 20, "hit_token_cap": True})
    artifact(tmp_path, tasks[2], status="timed_out", error="nine hour budget")
    (tmp_path / "tasks" / "3.json").write_text("not json")
    write(tmp_path / "tasks" / "extra.json", {})
    summary = read_summary(tmp_path, tasks)
    row = summary["generation_groups"][0]
    assert row["planned"] == 5 and row["complete"] == 2
    assert row["compared_to_plain"] == 2 and row["changed_from_plain"] == 1
    assert row["mean_output_tokens"] == 15 and row["cap_hits"] == 1
    assert summary["statuses"]["failed"] == 1
    assert summary["statuses"]["timed_out"] == 1
    assert summary["statuses"]["pending"] == 1
    assert "Unplanned artifact excluded" in summary["warnings"][1]
    assert not any(k in row for k in ("pcr", "success", "score"))
    before = (tmp_path / "tasks" / "0.json").read_bytes()
    report.write_report(tmp_path)
    assert (tmp_path / "tasks" / "0.json").read_bytes() == before


def test_cosines_unavailable_recorded_without_hiding_report(tmp_path):
    np.savez(tmp_path / "d.npz", a=np.array([1., 0.]), b=np.array([0., 1.]), zero=np.zeros(2))
    comparisons = [{"name": "orthogonal", "left": {"path": "d.npz", "key": "a"},
                    "right": {"path": "d.npz", "key": "b"}},
                   {"name": "zero", "left": {"path": "d.npz", "key": "a"},
                    "right": {"path": "d.npz", "key": "zero"}}]
    summary = read_summary(tmp_path, [], direction_comparisons=comparisons)
    assert summary["direction_comparisons"][0]["cosine"] == 0
    assert summary["direction_comparisons"][1]["status"] == "unavailable"
    assert "zero direction" in summary["direction_comparisons"][1]["error"]


def test_task_identity_mismatch_and_duplicate_plan_ids_rejected(tmp_path):
    task = dose("one", 0)
    artifact(tmp_path, {**task, "alpha": 0.3}, likelihood(-1, -2))
    summary = read_summary(tmp_path, [task])
    assert summary["statuses"]["failed"] == 1
    assert "differs from frozen plan" in summary["tasks"][0]["error"]
    write(tmp_path / "plan.json", {"tasks": [task, task]})
    with pytest.raises(ValueError, match="unique"):
        report.write_report(tmp_path)


def test_ablation_at_alpha_zero_is_compared_to_nonablated_control(tmp_path):
    tasks = [dose("zero", 0), dose("ablate", 0, policy="ablate")]
    artifact(tmp_path, tasks[0], likelihood(-1, -2))
    artifact(tmp_path, tasks[1], likelihood(-3, -3))
    summary = read_summary(tmp_path, tasks)
    ablated = next(r for r in summary["dose_pairs"] if r["policy"] == "ablate")
    assert ablated["control_task_id"] == "zero"
    assert ablated["full"]["delta_margin"] == -1


def test_plain_direction_none_and_fit_dev_examples_separated(tmp_path):
    tasks = [{"id": "fit", "kind": "generate", "direction": None, "partition": "fit", "method": "fit_plain"},
             {"id": "dev", "kind": "generate", "direction": None, "partition": "dev", "question_id": "q1",
              "question": "Question?", "baseline_response": "Saved plain"}]
    artifact(tmp_path, tasks[0], {"response": "FIT TEXT", "output_tokens": 10})
    artifact(tmp_path, tasks[1], {"response": "Dev ``` fenced answer", "output_tokens": 10})
    summary = read_summary(tmp_path, tasks)
    assert len(summary["generation_groups"]) == 2
    text = (tmp_path / "generated_examples.md").read_text()
    assert "FIT TEXT" not in text
    assert "Saved plain" in text and "Dev ``` fenced answer" in text
    assert "````text" in text


def measured_audit(n, delta, ratio, maximum, policy="all", alpha_abs=33):
    return {"modified_predictor_positions": n, "mean_actual_delta_norm": delta,
            "mean_original_hidden_norm": 100 if n else None,
            "mean_actual_relative_delta": ratio, "max_actual_relative_delta": maximum,
            "policy": policy, "alpha_abs": alpha_abs}


def test_actual_intervention_weighted_by_positions_and_sequences_separate(tmp_path):
    tasks = [dose("short", .1), dose("long", .1, question="q2"), dose("missing", .1, question="q3"),
             {"id": "gen", "kind": "generate", "direction": direction(), "alpha": .1, "policy": "all"}]
    for task, n, delta, ratio in [(tasks[0], 1, 10, .1), (tasks[1], 3, 30, .3)]:
        result = likelihood(-1, -2)
        result["steering_audit"] = {"positive": measured_audit(n, delta, ratio, ratio + .05),
                                    "negative": measured_audit(n, 2 * delta, 2 * ratio, 2 * ratio + .05)}
        artifact(tmp_path, task, result)
    artifact(tmp_path, tasks[2], likelihood(-1, -2))
    artifact(tmp_path, tasks[3], {"response": "answer", "steering_audit": measured_audit(2, 5, .05, .07)})
    summary = read_summary(tmp_path, tasks)
    rows = summary["steering_audit_groups"]
    positive = next(row for row in rows if row["side"] == "positive")
    assert positive["valid_audits"] == 2 and positive["complete"] == 3
    assert positive["modified_predictor_positions"] == 4
    assert positive["mean_actual_delta_norm"] == 25  # Not unweighted mean 20.
    assert positive["mean_actual_relative_delta"] == pytest.approx(.25)
    assert positive["max_actual_relative_delta"] == pytest.approx(.35)
    assert positive["alpha_abs_values"] == [33]
    assert positive["invalid_audits"] == [{"task_id": "missing", "reason": "missing steering audit"}]
    negative = next(row for row in rows if row["side"] == "negative")
    assert negative["mean_actual_delta_norm"] == 50
    generated = next(row for row in rows if row["kind"] == "generate")
    assert generated["mean_actual_delta_norm"] == 5
    text = (tmp_path / "report.md").read_text()
    assert "Actual residual intervention" in text and "missing steering audit" in text


def test_actual_audit_zero_and_ablation_and_no_hook_not_conflated(tmp_path):
    tasks = [dose("zero", 0), dose("ablate", 0, policy="ablate"),
             {"id": "plain", "kind": "generate", "direction": None, "alpha": 0}]
    for task, audit in [(tasks[0], measured_audit(0, 0, 0, 0, alpha_abs=0)),
                        (tasks[1], measured_audit(5, 2, .02, .03, policy="ablate", alpha_abs=None))]:
        result = likelihood(-1, -2)
        result["steering_audit"] = {"positive": audit, "negative": audit}
        artifact(tmp_path, task, result)
    artifact(tmp_path, tasks[2], {"response": "plain", "steering_audit": None})
    rows = read_summary(tmp_path, tasks)["steering_audit_groups"]
    zero = next(row for row in rows if row["kind"] == "dose" and row["policy"] == "all")
    assert zero["valid_audits"] == 1 and zero["modified_predictor_positions"] == 0
    assert zero["mean_actual_delta_norm"] == 0 and zero["mean_original_hidden_norm"] is None
    ablate = next(row for row in rows if row["policy"] == "ablate")
    assert ablate["mean_actual_delta_norm"] == 2 and ablate["alpha_abs_values"] == []
    plain = next(row for row in rows if row["kind"] == "generate")
    assert plain["not_applicable"] == 1 and plain["mean_actual_delta_norm"] is None
