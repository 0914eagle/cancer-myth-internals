"""Server-run protocol tests with a fake runtime; no torch/model downloads."""
import json

import pytest

from scripts import check_final_budget as check


def questions():
    return [{"id": f"{kind}_{i}", "set": kind, "partition": part,
             "group_id": f"{part}_{kind}_{i}", "question": f"Question {part} {kind} {i}?",
             "correction": "GOLD_DO_NOT_PROMPT"}
            for part in ("fit", "dev", "test") for kind in ("fpq", "nfp") for i in range(12)]


class FakeRuntime:
    identity = {"model": "fake"}

    def __init__(self):
        self.calls = []

    def generate(self, prompt, budget):
        self.calls.append((prompt, budget))
        is_review = prompt.endswith("Let's think step by step.")
        text = "REASONING_TRACE" if is_review else "Short answer." + (" Extra detail." if budget == 1024 else "")
        return {"text": text, "input_tokens": 20, "output_tokens": budget if budget == 512 else 40,
                "max_new_tokens": budget, "ended_with_eos": budget != 512,
                "cap_hit": budget == 512, "stop_reason": "length" if budget == 512 else "eos"}


def test_selection_is_fit_only_unique_and_order_independent():
    rows = questions()
    picked = check.select_questions(rows)
    assert picked == check.select_questions(list(reversed(rows)))
    assert len(picked) == len({q["group_id"] for q in picked}) == 12
    assert all(q["partition"] == "fit" for q in picked)
    assert sum(q["set"] == "fpq" for q in picked) == 8


def plan():
    cases = check.select_questions(questions())
    return {"cases": cases, "config": {}, "source_identity": {"identity": FakeRuntime.identity},
            "source_plain": {q["id"]: {"response": "Short answer."} for q in cases},
            "sources": {}, "implementation": {}, "review_tokens": 1024}


def test_paired_prompts_shared_reviews_resume_and_report(tmp_path, monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(check.bg, "make_runtime", lambda cfg: fake)
    p = plan()
    check.generate(tmp_path, p)
    assert len(fake.calls) == 60
    assert sum(prompt.endswith("Let's think step by step.") for prompt, _ in fake.calls) == 12
    assert all("GOLD_DO_NOT_PROMPT" not in prompt for prompt, _ in fake.calls)
    assert all(" dev " not in prompt and " test " not in prompt for prompt, _ in fake.calls)
    for q in p["cases"]:
        a, b = [json.loads(check.record_path(tmp_path, "zero_shot_cot", budget, q["id"]).read_text())
                for budget in check.BUDGETS]
        assert a["details"]["reasoning"] == b["details"]["reasoning"]
        assert a["details"]["final_prompt_sha256"] == b["details"]["final_prompt_sha256"]
    monkeypatch.setattr(check.bg, "make_runtime", lambda cfg: pytest.fail("Must not reload on completed resume"))
    check.generate(tmp_path, p)
    check.report(tmp_path, p)
    assert "paired=12" in (tmp_path / "report.md").read_text()
    assert "Reasoning mismatch IDs: []" in (tmp_path / "report.md").read_text()
    assert "different IDs: []" in (tmp_path / "report.md").read_text()
    assert "GOLD_DO_NOT_PROMPT" in (tmp_path / "examples.md").read_text()
    changed = {**p, "review_tokens": 128}
    with pytest.raises(ValueError, match="signature mismatch"):
        check.generate(tmp_path, changed)


def test_partial_report_and_source_change(tmp_path):
    p = plan()
    check.report(tmp_path, p)
    assert "0/12" in (tmp_path / "report.md").read_text()
    assert "MISSING" in (tmp_path / "examples.md").read_text()
    source = tmp_path / "source.json"
    source.write_text("old")
    p["sources"][str(source)] = check.file_digest(source)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen source/code changed"):
        check.verify(p)
