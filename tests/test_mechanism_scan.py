import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ms", ROOT / "scripts" / "mechanism_scan.py")
ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)


def test_span_mass_and_rows():
    attn = np.zeros((2, 3, 5)); attn[0, 1, [2, 3]] = 0.25; attn[1, 0, 4] = 1.0
    m = ms.span_mass(attn, [2, 3])
    assert m.shape == (2, 3) and m[0, 1] == 0.5 and m[1, 0] == 0.0
    suite = [{"id": "fpq_1", "set": "fpq", "question": "Q one"}, {"id": "nfp_1", "set": "nfp", "question": "N"}]
    e1 = [{"id": "e1_1", "question": "Q one", "premise_span": [0, 1]}]
    twins = [{"id": "e1_1_true", "set": "tpair", "pair_id": "e1_1", "question": "Q ONE", "premise_span": [2, 5]},
             {"id": "e1_1_fpara", "label_false_premise": 1, "pair_id": "e1_1", "question": "Q uno", "premise_span": [2, 5]}]
    rows = ms.build_rows(suite, e1, twins)
    assert sorted((r["kind"], r["label"]) for r in rows) == [("fpara", 1), ("natural", 0), ("natural", 1), ("twin", 0)]
