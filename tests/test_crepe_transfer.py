import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ct", ROOT / "scripts" / "crepe_transfer.py")
ct = importlib.util.module_from_spec(spec); spec.loader.exec_module(ct)


def test_label_detection_variants(tmp_path):
    rows = [
        {"question": "Why do we only use 10% of our brain?", "labels": ["false presupposition"]},
        {"question": "Why is the sky blue?", "labels": ["normal"]},
        {"question": "Why do cats purr?", "presuppositions": []},
        {"question": "Why did Einstein fail math?", "presuppositions": ["Einstein failed math"]},
        {"question": "Why is the sky blue?", "labels": ["normal"]},          # duplicate
        {"input": "no label here"},
    ]
    (tmp_path / "dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out, skipped, files = ct.load_crepe(tmp_path)
    assert [r["label"] for r in out] == [1, 0, 0, 1]
    assert skipped == {"duplicate": 1, "unclear_label": 1}
    assert files == {"dev.jsonl": 4}
    # explicit key override
    out2, _, _ = ct.load_crepe(tmp_path, label_key="labels", pos_re="false", neg_re="normal")
    assert [r["label"] for r in out2] == [1, 0]
