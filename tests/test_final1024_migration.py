"""Server tests: real cache paths/signatures, fake generation, no torch/model."""
import json

import numpy as np
import pytest

from scripts.prepare_final1024 import prepare
from scripts.run_baseline_suite import final_budget
from src import baseline_generation as bg
from src.baseline_suite import VERSION
from src.pilot import file_digest


class Runtime:
    identity = {"model": "fake"}

    def __init__(self):
        self.calls = []

    def generate(self, prompt, budget):
        self.calls.append((prompt, budget))
        return {"text": f"Answer under budget {budget}.", "output_tokens": 6,
                "max_new_tokens": budget, "ended_with_eos": True,
                "cap_hit": False, "stop_reason": "eos"}

    def features(self, question, layers):
        return {layer: np.array([layer, len(question)], dtype=np.float32) for layer in layers}


def source_suite(tmp_path, monkeypatch):
    source = tmp_path / "old"
    source.mkdir()
    rows = [{"id": k, "set": k, "partition": "fit", "group_id": k,
             "question": f"Question {k}?", "correction": "GOLD"} for k in ("fpq", "nfp")]
    (source / "suite.json").write_text(json.dumps({"version": VERSION, "questions": rows}))
    (source / "questions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    runtime = Runtime()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(bg, "make_runtime", lambda cfg: runtime)
    bg.run_generation(rows, {}, source / "model", ["plain", "zero_shot_cot"])
    bg.extract_features(rows, {}, source / "model", [11, 17])
    return source, rows, runtime


def test_reuses_reasoning_not_old_final_answers(tmp_path, monkeypatch):
    source, rows, runtime = source_suite(tmp_path, monkeypatch)
    before = {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()}
    assert len(runtime.calls) == 6
    out = tmp_path / "new"
    prepare(source, out)
    assert not (out / "model/generation").exists()
    assert not (out / "answers").exists()
    assert len(bg.load_feature_records(out / "model")) == 2
    assert final_budget(out) == 1024
    with pytest.raises(ValueError, match="frozen"):
        final_budget(out, 512)
    bg.run_generation(rows, {}, out / "model", ["plain", "zero_shot_cot"], final_tokens=final_budget(out))
    assert len(runtime.calls) == 10  # Four new finals; both reasoning traces reused.
    assert all(budget == 1024 for _, budget in runtime.calls[6:])
    for method in ("plain", "zero_shot_cot"):
        assert all(r["details"]["answer"]["max_new_tokens"] == 1024
                   for r in bg.load_generation_records(out / "model", method))
    prepare(source, out)  # No new snapshot after run starts.
    bg.run_generation(rows, {}, out / "model", ["plain", "zero_shot_cot"], final_tokens=1024)
    assert len(runtime.calls) == 10
    after = {str(p.relative_to(source)): file_digest(p) for p in source.rglob("*") if p.is_file()}
    assert before == after


def test_partial_copy_resumes_and_nested_paths_rejected(tmp_path, monkeypatch):
    source, _, _ = source_suite(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="non-nested"):
        prepare(source, source / "nested")
    out = tmp_path / "new"
    from scripts import prepare_final1024 as migration
    original = migration.copy_verified
    count = 0
    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("simulated interruption")
        return original(*args)
    monkeypatch.setattr(migration, "copy_verified", interrupted)
    with pytest.raises(RuntimeError):
        prepare(source, out)
    assert not (out / "cache_migration_complete.json").exists()
    monkeypatch.setattr(migration, "copy_verified", original)
    prepare(source, out)
    assert final_budget(out) == 1024


def test_defaults_and_incomplete_migration(tmp_path):
    assert final_budget(tmp_path) == 512
    assert final_budget(tmp_path, 1024) == 1024
    (tmp_path / "generation_defaults.json").write_text(json.dumps(
        {"final_tokens": 1024, "requires_cache_migration": True}))
    with pytest.raises(ValueError, match="incomplete"):
        final_budget(tmp_path)
