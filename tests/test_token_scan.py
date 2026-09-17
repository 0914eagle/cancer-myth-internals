import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ts", ROOT / "scripts" / "token_scan.py")
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)


def test_build_rows_covers_every_source_with_origins():
    suite = [{"id": "fpq_1", "set": "fpq", "question": "Q one"}, {"id": "nfp_1", "set": "nfp", "question": "N one"}]
    e1 = [{"id": "e1_1", "question": "Q one", "premise_span": [0, 1]}]
    twins = [{"id": "e1_1_true", "set": "tpair", "pair_id": "e1_1", "question": "Q ONE", "premise_span": [2, 5]},
             {"id": "e1_1_fpara", "label_false_premise": 1, "pair_id": "e1_1", "question": "Q uno", "premise_span": [2, 5]}]
    para = [{"id": "fpq_1_para", "set": "fpq", "paraphrase_of": "fpq_1", "question": "P one"},
            {"id": "nfp_1_para", "set": "nfp", "paraphrase_of": "nfp_1", "question": "P n"}]
    crepe = [{"id": "crepe_0", "set": "nfp", "label": 0, "question": "Why?"}]
    rows = ts.build_rows(suite, e1, twins, para, crepe)
    got = {(r["kind"], r["id"]): (r["label"], r["origin"], bool(r.get("span"))) for r in rows}
    assert got[("natural", "fpq_1")] == (1, "fpq_1", True)
    assert got[("natural", "nfp_1")] == (0, "nfp_1", False)
    assert got[("twin", "e1_1_true")] == (0, "fpq_1", True)
    assert got[("fpara", "e1_1_fpara")] == (1, "fpq_1", True)
    assert got[("para", "fpq_1_para")] == (1, "fpq_1", False)
    assert got[("crepe", "crepe_0")] == (0, "crepe_0", False)
