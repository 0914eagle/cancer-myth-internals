import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pp", ROOT / "scripts" / "probe_positions.py")
pp = importlib.util.module_from_spec(spec); spec.loader.exec_module(pp)


def test_span_shift_and_token_positions():
    content = "My father has cancer. Chemo spreads it. What now?"
    rendered = "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
    ss, se, cs, ce = pp.shift_span(rendered, content, (22, 39))
    assert rendered[ss:se] == "Chemo spreads it." and rendered[cs:ce] == content
    offsets = [(0, 5), (5, 10), (10, 24), (24, 30), (30, 39), (39, 45), (45, 45)]
    assert pp.token_positions(offsets, 24, 39) == [3, 4]
    assert pp.token_positions(offsets, 45, 45) == []
    with pytest.raises(ValueError):
        pp.shift_span("something else", content, (0, 1))


def test_pool_uses_block_outputs_only():
    hidden = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)  # (L+1=3, T=4, d=2)
    last = pp.pool(hidden, [3], "last")
    assert last.shape == (2, 2) and np.allclose(last[0], hidden[1, 3])
    mean = pp.pool(hidden, [1, 2], "mean")
    assert np.allclose(mean[1], hidden[2, 1:3].mean(axis=0))
    with pytest.raises(ValueError):
        pp.pool(hidden, [], "mean")


def test_build_rows_maps_spans_and_pairs():
    suite = [{"id": "fpq_1", "set": "fpq", "question": "Q one text here", "premise_text": "Myth statement number one.", "correction": "Correction statement number one."},
             {"id": "fpq_2", "set": "fpq", "question": "Q two text here", "premise_text": "short", "correction": "Correction statement number two."},
             {"id": "nfp_1", "set": "nfp", "question": "N one"}]
    e1 = [{"id": "e1_1", "question": "Q one text here", "premise_span": [2, 5]}, {"id": "e1_2", "question": "Q two text here", "premise_span": [0, 3]}]
    twins = [{"id": "e1_1_true", "set": "tpair", "pair_id": "e1_1", "question": "Q ONE text here", "premise_span": [2, 5], "replaced_with": "ONE"},
             {"id": "e1_1_fpara", "label_false_premise": 1, "pair_id": "e1_1", "question": "Q uno text here", "premise_span": [2, 5], "replaced_with": "uno"}]
    rows = pp.build_rows(suite, e1, twins)
    kinds = sorted((r["kind"], r["label"], r["origin"]) for r in rows)
    assert kinds == [("fpara", 1, "fpq_1"), ("isolated", 0, "fpq_1"), ("isolated", 1, "fpq_1"),
                     ("isolated_clean", 0, "fpq_1"), ("isolated_clean", 1, "fpq_1"), ("natural", 1, "fpq_1"), ("twin", 0, "fpq_1")]
    clean = {r["id"]: r["text"] for r in rows if r["kind"] == "isolated_clean"}
    assert clean == {"fpq_1_fspan": "uno", "fpq_1_tspan": "ONE"}
