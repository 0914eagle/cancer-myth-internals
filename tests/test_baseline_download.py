"""Dataset provenance and resume checks without network or model dependencies."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import download_baseline_references as download


def test_reference_download_is_revision_pinned_and_resumes_offline(tmp_path, monkeypatch):
    out = tmp_path / "references.jsonl"
    calls = []
    def info(name, *, revision):
        calls.append((name, revision))
        return SimpleNamespace(sha="fixed-commit")
    def dataset(name, *, revision, split):
        assert revision == "fixed-commit" and split == "validation"
        calls.append((name, revision, split))
        return [{"question": "Question?", "source_myth": "Original false premise",
                 "presupposition_correction": "Original correction"}]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=info)))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=dataset))
    monkeypatch.setattr(sys, "argv", ["download", "--output", str(out)])
    download.main()
    assert len(calls) == 2
    meta = json.loads(out.with_suffix(".jsonl.source.json").read_text())
    assert meta["resolved_revision"] == "fixed-commit" and meta["rows"] == 1
    download.main()
    assert len(calls) == 2
    out.write_text(out.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        download.main()


def test_partial_reference_file_is_not_silently_replaced(tmp_path, monkeypatch):
    out = tmp_path / "references.jsonl"
    out.write_text('{"partial": true}\n')
    monkeypatch.setattr(sys, "argv", ["download", "--output", str(out)])
    with pytest.raises(ValueError, match="Incomplete"):
        download.main()
    assert out.read_text() == '{"partial": true}\n'
