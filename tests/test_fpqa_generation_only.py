import json
from pathlib import Path

from scripts.generate_fpqa_baselines import generate_one, record_path, reuse_answers
from scripts.prepare_cancer_prompt_data import prepare
from scripts.judge_fpqa_saved import score_one
from src.fpqa_prompting import digest, seed_prompt, task_messages, well_crepe_judge
from scripts.run_fpqa_luna_suite import commands


ROOT = Path(__file__).resolve().parents[1]


def test_generation_calls_only_task_without_gold(tmp_path):
    calls = []
    def caller(role, messages):
        calls.append((role, messages))
        return "Final answer"
    row = dict(id="x", question="Question?", label=1,
               presuppositions=["SECRET PREMISE"], reference_answer="SECRET REFERENCE")
    job = ("cancer_myth", "balanced", row)
    generate_one(caller, tmp_path, job)
    assert len(calls) == 1 and calls[0][0] == "task"
    assert "SECRET" not in json.dumps(calls)
    record = json.loads(record_path(tmp_path, job).read_text())
    assert record["answer"] == "Final answer"
    assert "score" not in record and "label" not in record


def test_cancer_frozen_groups_and_optimization_subsets(tmp_path):
    result = prepare(ROOT / "configs/premise_sft/cancer_questions.jsonl", tmp_path / "data",
                     ROOT / "external/fpqa_20261002/Well")
    assert result["counts"]["test"] == {"fpq": 117, "nfp": 30}
    root = tmp_path / "data/cancer_myth"
    def load(name):
        return [json.loads(x) for x in (root / (name + ".jsonl")).read_text().splitlines()]
    group_sets = [{r["group_id"] for r in load(s)} for s in ("train", "dev", "test")]
    assert not any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i))
    for split, n in (("train", 178), ("dev", 60)):
        selected = load("opt_" + split)
        assert len(selected) == n and sum(r["label"] for r in selected) == n // 2
        assert {r["id"] for r in selected} <= {r["id"] for r in load(split)}


def test_saved_answer_scoring_does_not_generate():
    row = dict(id="x", dataset="cancer_myth", question="QUESTION",
               label=1, presuppositions=["PREMISE"], reference_answer="CORRECTION")
    record = dict(id="x", method="plain", question="QUESTION", answer="EXISTING ANSWER",
                  messages_sha256=digest(task_messages(seed_prompt("response", "plain"), row)))
    calls = []
    def call(role, messages):
        calls.append((role, messages))
        return "Explanation. Rating: 4"
    result = score_one(call, ROOT / "external/fpqa_20261002/Well", row, record)
    assert result["score"] == 4 and result["answer"] == "EXISTING ANSWER"
    assert len(calls) == 1 and calls[0][0] == "judge"
    assert "CORRECTION" in str(calls[0][1])


def test_cancer_judge_matches_pinned_original():
    import ast
    from types import SimpleNamespace
    well = ROOT / "external/fpqa_20261002/Well"
    for label, name in ((1, "CancerMyth"), (0, "CancerMythNFP")):
        tree = ast.parse((well / f"data_gen/template/{name}_template.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "generate")
        fn.returns = None
        scope = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "original", "exec"), scope)
        row = dict(dataset="cancer_myth", label=label, question="QUESTION", presuppositions=["PREMISE"], reference_answer="CORRECTION")
        attrs = SimpleNamespace(question="QUESTION", model_final_answer="ANSWER", presupposition="PREMISE",
                                correction="CORRECTION", answer="CORRECTION", system_role="system", user_role="user", model_role="assistant")
        assert well_crepe_judge(well, row, "ANSWER") == scope["generate"](attrs)


def test_cancer_test100_split(tmp_path):
    out = tmp_path / "data"
    result = prepare(ROOT / "configs/premise_sft/cancer_questions.jsonl", out,
                     ROOT / "external/fpqa_20261002/Well", "test100")
    assert result["counts"] == {"train": {"fpq": 383, "nfp": 31},
                                "dev": {"fpq": 100, "nfp": 18}, "test": {"fpq": 100, "nfp": 100}}
    assert (result["optimization_train"], result["optimization_dev"]) == (60, 36)
    groups = []
    for split in ("train", "dev", "test"):
        rows = [json.loads(x) for x in (out / "cancer_myth" / f"{split}.jsonl").read_text().splitlines()]
        groups.append({r["group_id"] for r in rows})
    assert not(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_reuse_requires_same_model_and_prompt(tmp_path):
    import pytest
    src, out = tmp_path / "old", tmp_path / "new"
    src.mkdir()
    config = {"task_model": {"model": "gpt-6-luna"}}
    (src / "run.json").write_text(json.dumps({"config": config}))
    row = dict(id="x", question="Q")
    job = ("cancer_myth", "plain", row)
    generate_one(lambda role, messages: "SAVED", src, job)
    assert reuse_answers(src, out, [job], config) == 1
    record = json.loads(record_path(out, job).read_text())
    assert record["answer"] == "SAVED" and record["reused_source_sha256"]
    with pytest.raises(ValueError, match="different task model"):
        reuse_answers(src, out, [job], {"task_model": {"model": "different"}})
    with pytest.raises(ValueError, match="identity"):
        reuse_answers(src, tmp_path / "changed", [("cancer_myth", "plain", {**row, "question": "Changed"})], config)


def test_luna_suite_prioritizes_optimization_and_uses_new_split():
    phases = commands(Path('/tmp/test_luna_suite'))
    assert set(phases[0]) == {"cancer_myth_optimize", "crepe_optimize", "baseline_generation"}
    assert "baseline_judgments" in phases[1]
    for phase in phases:
        for name, cmd in phase.items():
            assert cmd[cmd.index("--config") + 1].endswith("luna_only/luna6.json")
            if name.startswith("cancer"):
                assert cmd[cmd.index("--data") + 1].endswith("cancer_data_test100_v2")
    c = json.loads((ROOT / 'configs/fpqa_prompting/luna_only/luna6.json').read_text())
    assert all(c[r]["model"] == "gpt-6-luna" and c[r]["backend"] == "codex"
               for r in ("task_model", "judge_model", "reflection_model"))
