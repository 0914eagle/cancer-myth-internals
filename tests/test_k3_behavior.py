from src.k3_behavior import chi2_independence, crosstab, k3_group, permutation_p, transitions


def _row(qid, side, score, alignment, model="qwen25", dataset="fpq"):
    return {"model": model, "id": qid, "dataset": dataset, "side": side, "original_score": str(score), "alignment": alignment}


def test_groups_are_task_results_not_knowledge():
    assert k3_group(None) == "excluded"
    assert k3_group({"knows_fc": None}) == "not_run"
    assert k3_group({"knows_fc": "yes"}) == "stable_choice"
    assert k3_group({"knows_fc": "unsure"}) == "unsure"
    assert k3_group({"knows_fc": "no"}) == "wrong"


def test_crosstab_and_transitions():
    labels = {"fpq_1": {"knows_fc": "yes"}, "fpq_2": {"knows_fc": "no"}}
    rows = [_row("fpq_1", "plain", 2, "misses_target"), _row("fpq_1", "alternative", 5, "corrects_target"),
            _row("fpq_2", "plain", 1, "endorses_target_error"), _row("fpq_2", "alternative", 1, "endorses_target_error"),
            _row("fpq_3", "plain", 4, "corrects_target"), _row("nfp_1", "plain", 5, "x", dataset="nfp"),
            _row("fpq_1", "plain", 1, "misses_target", model="gemma")]
    ct = crosstab(rows, labels, "qwen25", "plain")
    assert ct["stable_choice"]["low_buckets"]["misses_target"] == 1
    assert ct["wrong"]["low_buckets"]["endorses_target_error"] == 1
    assert ct["excluded"]["high"] == 1 and ct["excluded"]["n"] == 1
    tr = transitions(rows, labels, "qwen25")
    assert tr["stable_choice"]["low->high"] == 1 and tr["wrong"]["low->low"] == 1


def test_chi2_and_permutation():
    stat, dof, min_e = chi2_independence([[10, 0], [0, 10]])
    assert dof == 1 and abs(stat - 20.0) < 1e-9 and min_e == 5.0
    assert chi2_independence([[3, 0], [0, 0]]) is None
    _, p_same = permutation_p(["a"] * 10 + ["b"] * 10, ["a"] * 10 + ["b"] * 10, n_perm=200)
    _, p_diff = permutation_p(["a"] * 20, ["b"] * 20, n_perm=200)
    assert p_same > 0.5 and p_diff < 0.05
