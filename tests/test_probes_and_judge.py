import numpy as np

from src.judge_prompts import construct_prompt_fpq, construct_prompt_nfp, parse_score
from src.probes import (
    cosine,
    cv_auroc_diffmeans,
    cv_auroc_logistic,
    diff_means_direction,
    fit_probe,
    probe_score,
)


def synthetic(n=200, d=32, gap=4.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray([1] * (n // 2) + [0] * (n // 2))
    X = rng.normal(size=(n, d))
    X[:, 0] += gap * y
    groups = [f"q{i}" for i in range(n)]
    return X, y, groups


def test_both_readouts_separate_a_separable_dataset():
    X, y, g = synthetic()
    auc_lr, oof = cv_auroc_logistic(X, y, g)
    auc_dm, _ = cv_auroc_diffmeans(X, y, g)
    assert auc_lr > 0.9 and auc_dm > 0.9 and len(oof) == len(y)


def test_readouts_are_at_chance_without_signal():
    X, y, g = synthetic(gap=0.0)
    auc_lr, _ = cv_auroc_logistic(X, y, g)
    auc_dm, _ = cv_auroc_diffmeans(X, y, g)
    assert 0.35 < auc_lr < 0.65 and 0.35 < auc_dm < 0.65


def test_grouped_folds_keep_a_group_on_one_side():
    from src.probes import _folds

    y = np.asarray([1, 1, 0, 0] * 10)
    groups = [f"g{i // 2}" for i in range(40)]  # pairs share a group
    for train, test in _folds(groups, y, 5, 17):
        assert not ({groups[i] for i in train} & {groups[i] for i in test})


def test_probe_arrays_reproduce_sklearn_scores_and_direction_is_unit():
    X, y, _ = synthetic()
    probe = fit_probe(X, y)
    assert probe_score(probe, X[0]) > probe_score(probe, X[-1])
    d = diff_means_direction(X, y)
    assert abs(np.linalg.norm(d) - 1) < 1e-6 and d[0] > 0.8
    assert abs(cosine(d, d) - 1) < 1e-6


def test_judge_prompts_follow_validate_py_layout():
    ex = [{"example_question": "Q?", "example_assumption": "A.", "answer": "ans", "score": {"Reason": "r", "Sharpness": -1}}]
    p = construct_prompt_fpq("myq", "myinfo", "myanswer", ex)
    assert p.startswith("You are provided with a medical patient question containing some false presuppositions")
    assert p.rstrip().endswith("Score (in JSON):") and "Information: myinfo" in p
    n = construct_prompt_nfp("myq", "halluc", "myanswer", ex)
    assert "containing NO false presuppositions" in n and "Possible hallucination" in n


def test_parse_score_matches_original_regex_and_fallback():
    ok, parsed = parse_score('Sure.\n{\n  "Reason": "x",\n  "Sharpness": -1\n}\n')
    assert parsed and ok["Sharpness"] == -1
    bad, parsed = parse_score("no json here")
    assert not parsed and bad["Sharpness"] == 1
