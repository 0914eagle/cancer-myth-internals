"""Offline tests for the truth x expression 2x2 helpers (no LLM, no torch)."""
import numpy as np
import pytest

from src import truth_form as tf


def test_parse_variants_and_verdict():
    reply = 'Sure:\n```json\n{"FA": "sugar feeds my tumour", "FH": "I read that sugar feeds tumours", "TA": "sugar does not feed tumours", "TH": "I was told sugar does not feed tumours"}\n```'
    v = tf.parse_variants(reply)
    assert v["FA"] == "sugar feeds my tumour" and v["TH"].startswith("I was told")
    assert tf.parse_variants('{"FA": "x", "FH": "y"}') is None and tf.parse_variants("no json") is None
    assert tf.parse_verdict("CLAIM=FALSE; STANCE=HEDGED") == ("FALSE", "HEDGED")
    assert tf.parse_verdict("claim = true ; stance = assertive") == ("TRUE", "ASSERTIVE")
    assert tf.parse_verdict("I think it is false") is None
    assert tf.cell_ok("FH", ("FALSE", "HEDGED")) and not tf.cell_ok("FH", ("NEITHER", "HEDGED")) and not tf.cell_ok("TA", ("TRUE", "HEDGED"))


def test_splice_keeps_outside_text():
    q, span = tf.splice("Since [X] happens, what now?", (6, 9), "[YY]")
    assert q == "Since [YY] happens, what now?" and span == [6, 10]


def test_two_by_two_separates_truth_from_expression():
    rng = np.random.default_rng(0)
    truth_gate, style_gate, topic_gate = {}, {}, {}
    for o in range(30):
        base = rng.normal(0, 1)
        truth_gate[f"o{o}"] = {"FA": base + 1 + rng.normal(0, .1), "FH": base + 1 + rng.normal(0, .1), "TA": base - 1 + rng.normal(0, .1), "TH": base - 1 + rng.normal(0, .1)}
        style_gate[f"o{o}"] = {"FA": base + 1 + rng.normal(0, .1), "FH": base - 1 + rng.normal(0, .1), "TA": base + 1 + rng.normal(0, .1), "TH": base - 1 + rng.normal(0, .1)}
        topic_gate[f"o{o}"] = {c: base + rng.normal(0, .1) for c in tf.CELLS}
    r = tf.two_by_two(truth_gate)
    # pooled AUROC mixes in the origin base variance; the paired order is the within-origin check
    assert r["truth_effect"] > 1.8 and abs(r["expression_effect"]) < 0.2 and r["auroc"]["truth|assertive"] > 0.8
    assert r["paired_order"]["truth|assertive"] == 1.0 and r["variance_share"]["truth"] > 0.9
    r = tf.two_by_two(style_gate)
    assert r["expression_effect"] > 1.8 and abs(r["truth_effect"]) < 0.2 and r["paired_order"]["expression|false"] == 1.0 and r["variance_share"]["expression"] > 0.9
    r = tf.two_by_two(topic_gate)
    assert abs(r["truth_effect"]) < 0.15 and abs(r["expression_effect"]) < 0.15 and r["truth_ci"][0] < 0 < r["truth_ci"][1]
    assert tf.report_lines("g", r)[0].startswith("| g | 30 |")
    with pytest.raises(ValueError):
        tf.two_by_two({"a": {"FA": 1, "FH": 1, "TA": 1, "TH": 1}})


def test_assembled_design_helpers():
    props = tf.parse_props('{"F": "Bronchial tumors affect only older people.", "T": "bronchial tumors do not affect only older people", "TAIL": "How can I support my grandmother?"}')
    assert props["F"] == "Bronchial tumors affect only older people" and props["TAIL"].startswith("How")
    assert tf.parse_props('{"F": "x", "T": "x", "TAIL": "y"}') is None  # identical clauses rejected
    built = tf.assemble(props)
    assert built["FD"] == "Given that bronchial tumors affect only older people, how can I support my grandmother?"
    assert built["FS"] == "Given that I've read that bronchial tumors affect only older people, how can I support my grandmother?"
    assert built["TS"].replace(tf.SOURCE_PHRASE, "") == built["TD"]  # source phrase is the only D/S difference
    v = tf.parse_pair_verdict("SAME_CLAIM=YES; F_FALSE=YES; T_TRUE=yes; TAIL_NEUTRAL=NO")
    assert v == {"SAME_CLAIM": True, "F_FALSE": True, "T_TRUE": True, "TAIL_NEUTRAL": False}
    assert tf.parse_pair_verdict("SAME_CLAIM=YES") is None
    scores = {f"o{i}": {"FD": 1.0 + i, "FS": 1.1 + i, "TD": -1.0 + i, "TS": -0.9 + i} for i in range(5)}
    r = tf.two_by_two(scores, tf.DESIGNS["assembled"]["cells"])
    assert r["cells"] == ["FD", "FS", "TD", "TS"] and r["truth_effect"] == pytest.approx(2.0) and "residual" not in r["variance_share"]
