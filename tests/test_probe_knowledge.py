import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pk", ROOT / "scripts" / "probe_knowledge.py")
pk = importlib.util.module_from_spec(spec); spec.loader.exec_module(pk)


def test_parse_and_statement_filter():
    assert pk.parse_verdict("False. Chemotherapy does not spread cancer.") == "False"
    assert pk.parse_verdict("  **True** — this is accurate") == "True"
    assert pk.parse_verdict("Truthfully, it depends") is None
    assert pk.parse_verdict("") is None
    assert not pk.usable_statement("From physicians.")
    assert pk.usable_statement("Lung cancer only affects older people.")


def test_labels_require_both_directions():
    def rec(qid, s, o, v, pf):
        return {"id": qid, "statement": s, "order": o, "verdict": v, "p_false": pf}
    records = [
        # knows: myth False both orders, correction True both orders
        rec("a", "myth", "tf", "False", .9), rec("a", "myth", "ft", "False", .8),
        rec("a", "correction", "tf", "True", .1), rec("a", "correction", "ft", "True", .2),
        # says False to everything -> unsure, not knows
        rec("b", "myth", "tf", "False", .9), rec("b", "myth", "ft", "False", .9),
        rec("b", "correction", "tf", "False", .9), rec("b", "correction", "ft", "False", .9),
        # no: myth True, correction False
        rec("c", "myth", "tf", "True", .1), rec("c", "myth", "ft", "True", .2),
        rec("c", "correction", "tf", "False", .8), rec("c", "correction", "ft", "False", .7),
        # order flip on myth -> unsure
        rec("d", "myth", "tf", "False", .6), rec("d", "myth", "ft", "True", .4),
        rec("d", "correction", "tf", "True", .1), rec("d", "correction", "ft", "True", .1),
    ]
    labels = {l["id"]: l for l in pk.label_rows(records)}
    assert labels["a"]["knows"] == "yes" and labels["a"]["k2_margin"] > 0
    assert labels["b"]["knows"] == "unsure" and abs(labels["b"]["k2_margin"]) < 1e-9
    assert labels["c"]["knows"] == "no" and labels["c"]["k2_margin"] < 0
    assert labels["d"]["knows"] == "unsure"


def test_forced_choice_labels_and_margin():
    def rec(qid, s, o, v, **kw):
        return {"id": qid, "statement": s, "order": o, "verdict": v, **kw}
    records = [
        rec("a", "myth", "tf", "False", p_false=.9), rec("a", "myth", "ft", "False", p_false=.9),
        rec("a", "correction", "tf", "False", p_false=.9), rec("a", "correction", "ft", "False", p_false=.9),
        # mc: myth is A, correction is B -> correct answer B; cm -> A
        rec("a", "pair", "mc", "B", p_a=.2), rec("a", "pair", "cm", "A", p_a=.8),
        rec("b", "pair", "mc", "A", p_a=.7), rec("b", "pair", "cm", "A", p_a=.7),   # always A: position bias
    ]
    labels = {l["id"]: l for l in pk.label_rows(records)}
    assert labels["a"]["knows"] == "unsure" and labels["a"]["knows_fc"] == "yes"
    assert abs(labels["a"]["fc_margin"] - 0.3) < 1e-9
    assert labels["b"]["knows_fc"] == "unsure" and labels["b"]["knows"] is None
    assert pk.parse_choice("B. The second statement is accurate.") == "B"
    assert pk.parse_choice("Statement A is accurate") == "A"
    assert pk.parse_choice("Both are wrong") is None
