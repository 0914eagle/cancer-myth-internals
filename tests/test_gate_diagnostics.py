import numpy as np
import pytest

from scripts.gate_diagnostics import benefit_readouts, make_curves, selection_counts


def test_rescue_is_not_success_or_net_gain():
    plain = np.array([-1, -1, 1, 1, 1])
    action = np.array([1, 0, 1, -1, -1])
    kinds = np.array(["fpq"] * 4 + ["nfp"])
    scores = np.array([5, 3, 4, 2, 1])
    r = benefit_readouts(scores, plain, action, kinds)
    assert r["FP ID succeeds (FPQ only)"]["positive"] == 2
    assert r["FP ID rescues (FPQ only)"]["positive"] == 1
    assert r["FP ID rescues (Plain-failed FPQ only)"] == {"n": 2, "positive": 1, "auroc": 1.0}
    assert r["FP ID harms (all dev; high score predicts harm)"]["positive"] == 2
    counts = selection_counts(np.ones(5, dtype=bool), plain, action, kinds)
    assert counts["fpq"] == {"selected": 4, "rescue": 1, "harm": 1, "net_gain": 0}
    assert counts["all"]["net_gain"] == -1
    assert (
        benefit_readouts(scores[:1], plain[:1], action[:1], kinds[:1])["FP ID rescues (FPQ only)"][
            "auroc"
        ]
        is None
    )


def test_budget_endpoints_ties_and_random_exact_expectation():
    ids = ["b", "a", "c"]
    kinds = np.array(["fpq", "fpq", "nfp"])
    values = {"plain": np.array([1, -1, 1]), "fp_identification": np.array([-1, 1, -1])}
    curves = make_curves(ids, kinds, values, {"hidden": np.array([1, 1, 0])})
    hidden = [r for r in curves if r["signal"] == "hidden"]
    random = [r for r in curves if r["signal"] == "random expectation"]
    assert (hidden[0]["pcr"], hidden[0]["nfp"]) == (50, 100)
    assert hidden[1]["pcr"] == 100  # ID a beats b on a score tie, irrespective of labels
    assert (hidden[-1]["pcr"], hidden[-1]["nfp"]) == (50, 0)
    assert random[1]["pcr"] == 50
    assert random[1]["nfp"] == pytest.approx(200 / 3)
    for key in ("pcr", "nfp", "pcs", "fpq_rescue", "fpq_harm", "nfp_harm"):
        assert hidden[-1][key] == random[-1][key]
