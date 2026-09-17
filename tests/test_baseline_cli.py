"""One offline end-to-end run through preparation, caches, gates and judge plan."""
import json
import sys

import numpy as np

from scripts import run_baseline_suite as cli
from src import baseline_generation as bg
from src import well_eval
from src.pilot import make_manifest


def test_suite_cli_connects_generation_detection_features_and_evaluation(tmp_path, monkeypatch):
    rows = []
    for i in range(45):
        part = "fit" if i < 30 else "dev" if i < 40 else "test"
        for kind in ("fpq", "nfp"):
            rows.append({"id": f"{kind}_{i}", "set": kind, "group_id": f"g{i}",
                         "partition": part, "question": f"Patient {kind} question {i}",
                         "premise_text": "GOLD premise", "correction": "GOLD correction"})
    manifest = make_manifest(rows, seed=17, folds=5, dev_fold=1, test_fold=0, reference_hash="synthetic")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    out = tmp_path / "suite"

    class Runtime:
        identity = {"model": "synthetic-offline"}
        def generate(self, prompt, budget):
            assert "GOLD" not in prompt
            text = '{"premises": []}' if prompt.startswith("Extract") else "Synthetic response."
            return {"text": text, "output_tokens": 3, "cap_hit": False}
        def binary(self, prompt, positive="Yes", negative="No"):
            assert "GOLD" not in prompt
            return {"score": .8 if "fpq" in prompt else .2}
        def features(self, prompt, layers):
            assert "GOLD" not in prompt
            return {k: np.array([1 if "fpq" in prompt else -1, int(prompt.split()[-1])]) for k in layers}

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(bg, "make_runtime", lambda cfg: Runtime())
    def command(stage, *extra):
        monkeypatch.setattr(sys, "argv", ["suite", stage, "--out-dir", str(out), *extra])
        cli.main()
    command("prepare", "--manifest", str(path))
    command("splits")
    command("generate")
    command("detect")
    command("extract", "--layers", "1", "2")
    command("gate", "--layers", "1", "2", "--c-grid", "0.01", "--bootstrap", "5")
    command("status")
    gate = json.loads((out / "gates/holdout_v1/result.json").read_text())
    assert len(gate["predictions"]) == 5 * 20
    assert {p["signal"] for p in gate["predictions"]} == {"text", "hidden", "mean", "direct", "review"}
    judge_out = out / "judge"
    plan = well_eval.prepare(out / "questions.jsonl", sorted((out / "answers").glob("*.jsonl")), judge_out)
    assert plan["calls_now"] == 0 and plan["shared_calls"] > 0
    report = well_eval.report(judge_out)
    assert report["attempted"] == 0 and report["unstarted"] == plan["unique_calls"]
    assert len(report["methods"]) == len(bg.METHODS)
    for sets in report["methods"].values():
        assert sets["fpq"]["expected"] == sets["nfp"]["expected"] == 45
        assert sets["fpq"]["valid"] == sets["nfp"]["valid"] == 0
