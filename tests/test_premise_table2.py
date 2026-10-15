"""Matched prompt and score contracts; no model loading or external calls."""
import json
from pathlib import Path

import pytest

from scripts import premise_table2 as t
from scripts import premise_sft as sft
from src import baseline_generation as bg
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import file_digest


@pytest.mark.parametrize("text,expected", [
    ('{"has_false_premise":true,"false_premises":["X"]}', "diagnostic_only"),
    ('```json\n{"has_false_premise":false}\n```', "diagnostic_only"),
    ('```\n{"has_false_premise":false}\n```', "diagnostic_only"),
    ('{"has_false_premise":true}\n\nAnswer:', "diagnostic_only"),
    ('{"has_false_premise":true}\n\nAnswer:\nAn answer.', "diagnostic_with_answer"),
    ('```json\n{"has_false_premise":false}\n```\nA real answer.', "diagnostic_with_answer"),
    ('{\n  "has_false_premise":', "diagnostic_incomplete"),
    ("An ordinary answer.", "text"), ("", "empty"),
])
def test_output_form_does_not_promote_diagnostic_only(text, expected):
    assert t.output_form(text) == expected


class FakeRuntime:
    def __init__(self, cfg):
        self.identity = {"condition": cfg["condition"], "seed": cfg["seed"]}
        self.condition = cfg["condition"]
        self.calls = []

    def generate(self, prompt, budget):
        self.calls.append((prompt, budget))
        assert "GOLD_SECRET" not in prompt
        diagnostic = self.condition != "base" and prompt in {"False question?", "Normal question?"}
        text = '{"has_false_premise":false,"false_premises":[]}' if diagnostic else "An answer."
        return {"text": text, "input_tokens": 10, "output_tokens": 4,
                "max_new_tokens": budget, "ended_with_eos": True, "budget_reached": False,
                "cap_hit": False, "stop_reason": "eos", "eos_token_id": 9}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    lock = tmp_path / "downloads.json"
    lock.write_text(json.dumps({"model_id": sft.MODEL, "model_revision": sft.MODEL_REVISION,
                                "model_path": "/unused"}))
    qs = [{"id": "f", "set": "fpq", "partition": "test", "question": "False question?",
           "premise_text": "GOLD_SECRET", "correction": "A correction."},
          {"id": "n", "set": "nfp", "partition": "test", "question": "Normal question?"}]
    out = tmp_path / "experiment"
    out.mkdir()
    conditions = {"base": None}
    for name in ("cancer", "crepe"):
        run = tmp_path / name
        run.mkdir()
        (run / "complete.json").write_text(json.dumps({"smoke_only": False, "last_epoch": 3}))
        (run / "train_manifest.json").write_text(json.dumps({"model_revision": sft.MODEL_REVISION,
                                                             "ids": ["fit_only"]}))
        conditions[name] = str(run)
    plan = {"version": t.VERSION, "implementation_sha256": file_digest(t.__file__),
            "baseline_generation_sha256": file_digest(bg.__file__), "model_lock": str(lock),
            "model_lock_sha256": file_digest(lock), "questions": qs, "seed": 17,
            "conditions": conditions, "final_tokens": 1024, "review_tokens": 1024}
    (out / "experiment.json").write_text(json.dumps(plan))
    write_jsonl(out / "questions.jsonl", qs)
    runtimes = []
    def factory(cfg):
        runtime = FakeRuntime(cfg)
        runtimes.append(runtime)
        return runtime
    monkeypatch.setattr(t, "PinnedRuntime", factory)
    return out, runtimes


def test_matched_prompts_two_stage_cot_and_resume(experiment):
    out, runtimes = experiment
    for condition in t.CONDITIONS:
        t.generate(out, condition, list(t.METHODS))
    expected_prompts = []
    for method in t.METHODS:
        for question in ("False question?", "Normal question?"):
            if method == "plain":
                expected_prompts.append((question, 1024))
            elif method == "zero_shot_cot":
                expected_prompts.extend([(bg.COT_REASON.format(question=question), 1024),
                    (bg.COT_FINAL.format(question=question, review="An answer."), 1024)])
            else:
                expected_prompts.append((bg.FP_ANSWER.format(question=question), 1024))
    assert all(runtime.calls == expected_prompts for runtime in runtimes)
    t.generate(out, "cancer", list(t.METHODS))
    assert runtimes[-1].calls == []
    rows = list(read_jsonl(out / "answers/cancer_plain.jsonl"))
    assert all(r["output_form"] == "diagnostic_only" and not r["final_answer_present"] for r in rows)
    assert list(read_jsonl(out / "answers/base_plain.jsonl"))[0]["method"] == "plain"


def test_diagnostic_only_raw_well_five_is_not_answer_success(experiment):
    out, _ = experiment
    for condition in t.CONDITIONS:
        t.generate(out, condition, list(t.METHODS))
    result = t.judge_plan(out, "claude", "claude-sonnet-5")
    assert result["answer_rows"] == 18 and result["unique_calls"] <= 18
    t.well.score(out / "well", max_calls=18, caller=lambda prompt: ("Rating: 5", "claude-sonnet-5"))
    result = t.report(out)
    assert result["complete"]
    for row in result["rows"]:
        if row["condition"] != "base" and row["prompt_method"] == "plain":
            assert row["raw_judge_counts_0_to_5"][5] == 1
            assert row["answer_s4_count"] == row["answer_s5_count"] == 0
            assert row["paired_base"]["s4_harm"] == 1
        else:
            assert row["answer_s5_over_valid"] == 1


def test_incomplete_training_and_unauthorized_gpu_rejected(experiment):
    out, _ = experiment
    plan = t.load_plan(out)
    p = Path(plan["conditions"]["cancer"]) / "complete.json"
    p.write_text(json.dumps({"smoke_only": True, "last_epoch": 1}))
    with pytest.raises(ValueError, match="completed 3-epoch"):
        t.generate(out, "cancer", ["plain"])
    with pytest.raises(ValueError, match="authorized"):
        bg.gpu_guard({"3"})
    with pytest.raises(ValueError, match="authorized"):
        bg.gpu_guard()  # Original baseline still permits only 0/1 by default.


@pytest.fixture
def answer_experiment(experiment, tmp_path, monkeypatch):
    from scripts import premise_table2_answer as answer
    out, runtimes = experiment
    csv_path = tmp_path / "base.csv"
    csv_path.write_text("model,method,id,set,score\n" + "".join(
        f"qwen,{m},{qid},{kind},5\n" for m in t.METHODS for qid, kind in (("f", "fpq"), ("n", "nfp"))))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"judge": "claude-sonnet-5"}))
    monkeypatch.setattr(answer, "BASE_CSV", csv_path)
    monkeypatch.setattr(answer, "BASE_PROVENANCE", provenance)
    protocol = {"implementation_sha256": file_digest(answer.__file__),
                "answer_instruction": answer.INSTRUCTION, "base_csv_sha256": file_digest(csv_path),
                "comparison_note": "Historical Base does not have the added instruction."}
    (out / "answer_instruction_protocol.json").write_text(json.dumps(protocol))
    return out, runtimes, answer


def test_answer_instruction_only_on_final_prompts_and_sft_only(answer_experiment):
    out, runtimes, answer = answer_experiment
    for condition in answer.CONDITIONS:
        answer.generate(out, condition)
    expected = []
    for method in t.METHODS:
        for question in ("False question?", "Normal question?"):
            if method == "plain":
                expected.append((question + "\n\nAnswer the question.", 1024))
            elif method == "zero_shot_cot":
                expected.extend([(bg.COT_REASON.format(question=question), 1024),
                    (bg.COT_FINAL.format(question=question, review="An answer.") + "\n\nAnswer the question.", 1024)])
            else:
                expected.append((bg.FP_ANSWER.format(question=question) + "\n\nAnswer the question.", 1024))
    assert len(runtimes) == 2 and all(r.calls == expected for r in runtimes)
    assert not (out / "models/base").exists()
    with pytest.raises(ValueError, match="Only SFT"):
        answer.generate(out, "base")
    answer.generate(out, "cancer")
    assert runtimes[-1].calls == []


def test_answer_instruction_well_scores_only_sft_and_reuses_base(answer_experiment):
    out, _, answer = answer_experiment
    for condition in answer.CONDITIONS:
        answer.generate(out, condition)
    plan = answer.judge_plan(out)
    assert plan["answer_rows"] == 12
    judge, _ = t.well.load_plan(out / "well")
    assert len(judge["mapping"]) == 6 and all(not k.startswith("base") for k in judge["mapping"])
    t.well.score(out / "well", max_calls=12, caller=lambda _: ("Rating: 4", "claude-sonnet-5"))
    result = answer.report(out)
    assert result["complete"]
    for row in result["rows"]:
        if row["condition"] == "historical_base":
            assert row["s5_over_valid"] == 1 and not row["answer_instruction_added"]
        else:
            assert row["s4_over_valid"] == 1 and row["s5_over_valid"] == 0
            assert row["answer_instruction_added"]
    cases = list(read_jsonl(out / "table2_cases.jsonl"))
    assert all(r["historical_base_rating"] == 5 and not r["matched_prompt_comparison"] for r in cases)
