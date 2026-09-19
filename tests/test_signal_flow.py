"""Offline tests for the signal-flow helpers: no torch or GPU."""
import numpy as np

from src import signal_flow as sf


def test_token_roles_and_bins_cover_prompt_regions():
    # rendered = "<s>" + "PRE " + "abc DEF ghi jkl" + " POST<e>" with question "abc DEF ghi jkl" and span "DEF"
    rendered = "<s>PRE abc DEF ghi jkl POST<e>"
    q0, q1 = sf.content_positions(rendered, "abc DEF ghi jkl")
    # tokens: <s>(0,0 special) PRE(3,6) abc(7,10) DEF(11,14) ghi(15,18) jkl(19,22) POST(23,27) <e>(27,27)
    offsets = [(0, 0), (3, 6), (7, 10), (11, 14), (15, 18), (19, 22), (23, 27), (27, 27)]
    span_abs = (q0 + 4, q0 + 7)
    roles = sf.token_roles(offsets, q0, q1, span_abs)
    assert roles == [sf.ROLE_PRE, sf.ROLE_PRE, sf.ROLE_Q_BEFORE, sf.ROLE_SPAN, sf.ROLE_Q_AFTER, sf.ROLE_Q_AFTER, sf.ROLE_POST, sf.ROLE_LAST]
    bins = sf.bin_indices(roles)
    assert bins["span"] == [3] and bins["q_before"] == [2] and bins["last"] == [7] and bins["pre"] == [0, 1]
    assert bins["q_after_1"] == [4] and bins["q_after_2"] == [5] and "q_after_3" not in bins  # two tokens -> two thirds
    # no span: whole question is q_before
    roles2 = sf.token_roles(offsets, q0, q1, None)
    assert [r for r in roles2[2:6]] == [sf.ROLE_Q_BEFORE] * 4 and roles2[-1] == sf.ROLE_LAST
    means = sf.bin_means(np.arange(16, dtype=float).reshape(2, 8), bins)
    assert means["pre"].tolist() == [0.5, 8.5] and means["last"].tolist() == [7.0, 15.0]


def test_direction_fit_separates_and_projection_is_held_out_by_fold():
    rng = np.random.default_rng(0)
    pos = rng.normal(0, 1, (40, 6)) + np.array([2, 0, 0, 0, 0, 0])
    neg = rng.normal(0, 1, (40, 6)) - np.array([2, 0, 0, 0, 0, 0])
    d, center = sf.fit_direction(pos, neg, C=0.1)
    assert abs(np.linalg.norm(d) - 1) < 1e-5 and abs(d[0]) > 0.9
    scores = np.concatenate([(pos - center) @ d, (neg - center) @ d])
    assert sf.auroc([1] * 40 + [0] * 40, scores) > 0.95
    assert sf.auroc([1, 1], [0.1, 0.2]) is None  # one class
    folds = {sf.fold_of(f"fpq_{i}") for i in range(50)}
    assert folds == {0, 1} and sf.fold_of("fpq_1") == sf.fold_of("fpq_1")


def test_transfer_fraction_and_heatmap_lines():
    tf = sf.transfer_fraction(np.array([1.0, 0.5, 3.0]), np.array([0.0, 0.0, 3.0]), np.array([2.0, 1.0, 3.0]))
    assert tf[0] == 0.5 and tf[1] == 0.5 and np.isnan(tf[2])
    lines = sf.heatmap_lines(["L1"], ["span", "last"], {(0, "span"): 0.97, (0, "last"): None}, "T")
    assert lines[0] == "## T" and lines[-1] == "| L1 | 0.970 | NA |"
