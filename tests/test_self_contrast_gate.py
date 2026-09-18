import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("scg", ROOT / "scripts" / "self_contrast_gate.py")
scg = importlib.util.module_from_spec(spec); spec.loader.exec_module(scg)


def test_gate_score_and_helpers():
    # belief first: P(A)=0.1 means the alternative (B) wins -> 0.9; alt first: P(A)=0.9 -> 0.9
    assert abs(scg.gate_score(0.1, 0.9) - 0.9) < 1e-9
    assert abs(scg.gate_score(0.5, 0.5) - 0.5) < 1e-9
    assert scg.clean('  "Chemo spreads cancer."\nExplanation...') == "Chemo spreads cancer."
    assert scg.jaccard("Chemotherapy makes cancer spread faster.", "Chemotherapy causes cancer to spread faster") > 0.5


def test_build_rows_all_sources():
    suite = [{"id": "fpq_1", "set": "fpq", "question": "Q one", "premise_text": "M"}, {"id": "nfp_1", "set": "nfp", "question": "N one"}]
    e1 = [{"id": "e1_1", "question": "Q one"}]
    twins = [{"id": "e1_1_true", "set": "tpair", "pair_id": "e1_1", "question": "Q ONE"},
             {"id": "e1_1_fpara", "label_false_premise": 1, "pair_id": "e1_1", "question": "Q uno"}]
    para = [{"id": "fpq_1_para", "set": "fpq", "paraphrase_of": "fpq_1", "question": "P"}]
    rows = scg.build_rows(suite, e1, twins, para)
    assert sorted((r["kind"], r["label"], r["origin"]) for r in rows) == [
        ("fpara", 1, "fpq_1"), ("natural", 0, "nfp_1"), ("natural", 1, "fpq_1"), ("para", 1, "fpq_1"), ("twin", 0, "fpq_1")]
