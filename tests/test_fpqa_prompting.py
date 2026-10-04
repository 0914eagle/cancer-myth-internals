import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prepare_fpqa_prompt_data import balanced, crepe_rows, remove_overlap
from scripts.run_fpqa_prompt_experiment import optimize
from src.fpqa_prompting import (
    DIRECT, Evaluator, detection_metrics, freeze, parse_detection, parse_rating,
    response_metrics, seed_prompt, task_messages, validate_splits, well_crepe_judge,
)
from src.fpqa_cli_backend import CLIBackend, envelope, parse_claude, parse_codex


def row(i, label=1, split="train"):
    return {"id": str(i), "dataset": "crepe", "question": f"Question {i}?", "label": label,
            "presuppositions": ["GOLD PREMISE MUST NOT LEAK"], "corrections": ["OTHER CORRECTION"],
            "reference_answer": "GOLD ANSWER MUST NOT LEAK", "split": split}


def test_cli_backends_receive_identical_experiment_bytes(monkeypatch):
    observed = []
    def run(cmd, prompt, cwd, timeout):
        if cmd[0] == "claude":
            system = cmd[cmd.index("--system-prompt") + 1]
            result = {"subtype": "success", "stop_reason": "end_turn", "num_turns": 1,
                      "modelUsage": {"claude-sonnet-5-5": {}}, "result": "Yes"}
        else:
            path = next(x.split("=", 1)[1] for x in cmd if x.startswith("model_instructions_file="))
            system = Path(json.loads(path)).read_text()
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text("Yes")
            result = None
        observed.append((system, prompt, timeout))
        return (json.dumps(result) if result else '\n'.join(map(json.dumps, [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Yes"}},
            {"type": "turn.completed"}])), "")
    monkeypatch.setattr("src.fpqa_cli_backend.run_process", run)
    monkeypatch.setattr("src.fpqa_cli_backend.subprocess.run", lambda *a, **kw: SimpleNamespace(stdout="test-version"))
    for backend, model in [("codex", "gpt-6-luna"), ("claude", "claude-sonnet-5-5")]:
        call = CLIBackend({"backend": backend, "model": model, "effort": "medium", "timeout_seconds": 240})
        answer, _ = call(task_messages(DIRECT, row(1)))
        assert answer == "Yes"
    assert observed[0] == observed[1]
    assert "GOLD" not in str(observed)


def test_cli_rejects_model_fallback_and_tool_use():
    good = {"subtype": "success", "stop_reason": "end_turn", "num_turns": 1,
            "modelUsage": {"claude-sonnet-5-5": {}}, "result": "Yes"}
    assert parse_claude(json.dumps(good), "claude-sonnet-5-5")[0] == "Yes"
    with pytest.raises(ValueError, match="served model"):
        parse_claude(json.dumps(good), "claude-sonnet-5")
    with pytest.raises(ValueError, match="tool attempt"):
        parse_claude(json.dumps({**good, "num_turns": 2}), "claude-sonnet-5-5")
    events = [{"type": "item.completed", "item": {"type": "command_execution"}}, {"type": "turn.completed"}]
    with pytest.raises(ValueError, match="tool item"):
        parse_codex('\n'.join(map(json.dumps, events)), "Yes")


def test_fewshot_envelope_preserves_roles_order_and_content():
    messages = [{"role": "system", "content": "Grade"}, {"role": "user", "content": "example"},
                {"role": "assistant", "content": "Rating: 5"}, {"role": "user", "content": "actual"}]
    system, user = envelope(messages)
    assert "Grade" in system
    assert json.loads(user.split('\n', 1)[1]) == messages[1:]


def test_all_presets_share_judge_reflector_and_budget():
    root = Path(__file__).resolve().parents[1] / "configs/fpqa_prompting"
    configs = [json.loads(p.read_text()) for p in root.glob("*.json")]
    assert len(configs) == 6
    for c in configs:
        assert c["judge_model"] == c["reflection_model"] == configs[0]["judge_model"]
        assert c["judge_model"]["model"] == "claude-sonnet-5-5"
        assert (c["max_metric_calls"], c["seed"], c["workers"]) == (500, 42, 5)


def test_task_only_question_even_with_gold_present():
    messages = task_messages(DIRECT, row(1))
    assert messages[1]["content"] == "Question 1?"
    assert "GOLD" not in json.dumps(messages)


@pytest.mark.parametrize("text,pred", [("Yes", 1), ("no.", 0), ("Yes and No", None), ("Unknown", None), ('{"yes": true}', None)])
def test_detection_is_not_permissive(text, pred):
    assert parse_detection(text) == pred


def test_invalid_does_not_become_normal_or_disappear():
    result = detection_metrics([{"label": 1, "prediction": 1}, {"label": 0, "prediction": None}])
    assert result["balanced_accuracy"] == .5
    assert result["fpr"] == 0
    assert result["nfp_error_including_invalid"] == 1
    assert result["invalid_nfp"] == 1
    assert result["auroc"] is None


def test_rating_rejects_ambiguous_or_fractional_score():
    assert parse_rating("Explanation. Rating: 4") == 4
    for text in ("Rating: 4.5", "Rating: 4\nRating: 5", "Rating: 9", "four"):
        with pytest.raises(ValueError):
            parse_rating(text)


def test_review_has_two_calls_and_no_gold():
    calls = []
    def call(role, messages):
        calls.append(messages)
        return "The premise is false." if len(calls) == 1 else "Yes"
    score, record = Evaluator("detection", "cot_2step", call, None)(DIRECT, row(1))
    assert score == 1 and record["review"]
    assert len(calls) == 2
    assert "GOLD" not in json.dumps(calls)
    assert "The premise is false." in calls[1][1]["content"]


def test_conflicting_label_and_cross_split_duplicates():
    a, b = row("same", 1), row("same", 0)
    train = [a, b, row("duplicate", 1), row("unique", 0)]
    cleaned, dropped = remove_overlap({"train": train, "dev": [], "test": [row("duplicate", 1)]})
    assert [r["id"] for r in cleaned["train"]] == ["unique"]
    assert len(cleaned["test"]) == 1
    assert len(dropped) == 3


def test_crepe_ambiguous_labels_reserved_and_underscore(tmp_path):
    raw = [{"id": "1", "question": "a", "labels": ["normal", "false_presupposition"]},
           {"id": "2", "question": "b", "labels": ["false_presupposition"]},
           {"id": "3", "question": "c", "labels": ["normal"]}]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(map(json.dumps, raw)))
    rows, dropped = crepe_rows(path, "train", {"3"})
    assert len(rows) == 1 and rows[0]["label"] == 1
    assert len(dropped) == 2


def test_balanced_selection_and_split_validation():
    rows = [row(i, i % 2) for i in range(20)]
    sample = balanced(rows, 4, 42)
    assert len(sample) == 8 and sum(r["label"] for r in sample) == 4
    assert sample == balanced(rows, 4, 42)
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(rows[:4], rows[:4], rows[4:])


def test_freeze_rejects_changed_protocol(tmp_path):
    path = tmp_path / "run.json"
    freeze(path, {"model": "a"})
    with pytest.raises(ValueError):
        freeze(path, {"model": "b"})


def test_response_summary_keeps_zero():
    result = response_metrics([{"label": 1, "score": 0}, {"label": 1, "score": 5}])
    assert result["fpq"]["ge4"] == .5
    assert result["fpq"]["mean"] == 2.5


def test_well_renderer_matches_original_generate_method():
    root = Path("external/fpqa_20261002/Well")
    if not root.exists():
        pytest.skip("Run source setup to check upstream renderer equivalence")
    for y, kind in ((0, "TPQ"), (1, "FPQ")):
        tree = ast.parse((root / f"data_gen/template/CREPE{kind}_template.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        original.returns = None
        scope = {}
        # Compile only the pinned generate method for equivalence; no upstream imports.
        exec(compile(ast.Module(body=[original], type_ignores=[]), "original_generate", "exec"), scope)
        r = row(y, y)
        attrs = SimpleNamespace(question=r["question"], model_final_answer="ANSWER",
                                presupposition=r["presuppositions"][0], correction=r["reference_answer"],
                                answer=r["reference_answer"], system_role="system", user_role="user", model_role="assistant")
        assert well_crepe_judge(root, r, "ANSWER") == scope["generate"](attrs)


def test_falseqa_does_not_get_invented_gold_premise():
    r = row(1)
    r["dataset"] = "falseqa"
    with pytest.raises(ValueError, match="FalseQA"):
        well_crepe_judge(None, r, "answer")


def test_real_gepa_loop_with_mock_models(tmp_path):
    pytest.importorskip("gepa")
    calls = []
    def call(role, messages):
        calls.append((role, messages))
        return "Yes" if "IMPROVED" in messages[0]["content"] else "No"
    evaluator = Evaluator("detection", "gepa", call, None)
    reflections = []
    def reflect(messages):
        reflections.append(messages)
        return "```\nIMPROVED: Reply Yes.\n```"
    # Synthetic all-positive fixture exercises engine plumbing, not a research result.
    optimize(evaluator, DIRECT, [row(1), row(2)], [row(3), row(4)],
             {"max_metric_calls": 14, "seed": 42}, tmp_path, reflect)
    saved = json.loads((tmp_path / "optimized_prompt.json").read_text())
    assert calls and reflections
    assert "IMPROVED" in saved["system_prompt"]
    assert saved["validation_score"] == 1


def test_prewome_three_calls_working_memory_and_no_gold():
    calls = []
    def call(role, messages):
        calls.append((role, messages))
        if len(calls) == 1:
            return "- X is true"
        if len(calls) == 2:
            return "Feedback:\n- X is true: false - reason\nAction: correct X"
        if role == "task":
            return "Final answer."
        return "Rating: 4"
    ev = Evaluator("response", "prewome", call, None)
    import src.fpqa_prompting as fp
    orig = fp.well_crepe_judge
    fp.well_crepe_judge = lambda root, ex, ans: [{"role": "user", "content": ans}]
    try:
        score, record = ev(seed_prompt("response", "prewome"), row(1))
    finally:
        fp.well_crepe_judge = orig
    assert [r for r, _ in calls] == ["task", "task", "task", "judge"]
    assert "Action: correct X" in calls[2][1][1]["content"] and "- X is true" in calls[2][1][1]["content"]
    assert record["review"]["feedback"].startswith("Feedback")
    assert "GOLD" not in json.dumps([m for r, m in calls if r == "task"])
    assert score == 0.8
