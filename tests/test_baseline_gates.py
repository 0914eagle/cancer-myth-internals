"""Offline readout checks, especially avoiding evaluation-driven fitting."""

from copy import deepcopy

import numpy as np
import pytest

from src import baseline_gates as gates
from src import baseline_suite as suite


@pytest.fixture
def gate_data():
    rows = []
    features = {}
    detections = {}
    rng = np.random.default_rng(102)
    for g in range(90):
        for kind in ("fpq", "nfp"):
            qid = f"{kind}_{g:03}"
            y = 1 if kind == "fpq" else -1
            rows.append({"id": qid, "set": kind, "group_id": f"source_{g:03}",
                         "question": f"{'False assumption' if y > 0 else 'Normal context'} patient number {g}",
                         "partition": "fit" if g < 60 else "dev" if g < 80 else "test",
                         "correction": "correct statement" if y > 0 else ""})
            features[qid] = {11: np.array([3 * y, *rng.normal(size=3)]),
                             17: rng.normal(size=4)}
            detections[qid] = {"direct_score": float(y + g / 1000), "review_score": float(y + g / 500)}
    rows = suite.check_questions(rows)
    return rows, suite.split_plan(rows), features, detections


def test_fpr_threshold_uses_strict_inequality_and_never_breaks_ties():
    tied = np.array([0.1] * 18 + [0.9] * 2)
    threshold = gates.threshold_at_fpr(tied, 0.05)
    assert threshold == 0.9
    assert (tied > threshold).sum() == 0
    thirty = np.arange(30, dtype=float)
    assert (thirty > gates.threshold_at_fpr(thirty, 0.05)).sum() == 1
    assert (thirty > gates.threshold_at_fpr(thirty, 0)).sum() == 0
    for bad in ([], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            gates.threshold_at_fpr(bad)
    for target in (-0.1, 1.0, np.nan):
        with pytest.raises(ValueError):
            gates.threshold_at_fpr(thirty, target)


def test_layer_regularization_and_threshold_do_not_use_evaluation(gate_data):
    rows, split, features, _ = gate_data
    kwargs = dict(signals=("hidden", "mean"), layers=(17, 11), c_grid=(1.0, 0.001))
    baseline = gates.run_gate_comparison(rows, split, features=features, **kwargs)
    changed = deepcopy(features)
    for qid in split["folds"][0]["evaluation_ids"]:
        for layer in changed[qid]:
            changed[qid][layer] = -100 * changed[qid][layer]
    rerun = gates.run_gate_comparison(rows, split, features=changed, **kwargs)
    # Includes train-CV scores, chosen C/layer and calibration threshold.
    assert baseline["selections"] == rerun["selections"]
    assert any(a["score"] != b["score"] for a, b in zip(baseline["predictions"], rerun["predictions"]))
    for selection in baseline["selections"]:
        assert selection["best"]["layer"] == 11


def test_inner_model_selection_never_fits_on_calibration_or_evaluation(gate_data, monkeypatch):
    rows, split, features, _ = gate_data
    real = gates._fit_predict
    outer_train = set(split["folds"][0]["train_ids"])
    calls = []

    def record(kind, train_rows, predict_rows, **kwargs):
        fitted = {q["id"] for q in train_rows}
        assert fitted <= outer_train
        assert fitted.isdisjoint(q["id"] for q in predict_rows)
        calls.append(fitted)
        return real(kind, train_rows, predict_rows, **kwargs)

    monkeypatch.setattr(gates, "_fit_predict", record)
    gates.run_gate_comparison(rows, split, features=features, signals=("text", "hidden"),
                              layers=(11,), c_grid=(0.01, 1.0))
    assert len(calls) == 14  # two C settings x three train folds + final fit, per signal
    assert sum(fitted == outer_train for fitted in calls) == 2


def test_calibration_positive_scores_cannot_set_detection_threshold(gate_data):
    rows, split, _, detections = gate_data
    first = gates.run_gate_comparison(rows, split, detections=detections, signals=("direct", "review"))
    changed = deepcopy(detections)
    for qid in split["folds"][0]["calibration_ids"]:
        if qid.startswith("fpq"):
            changed[qid] = {"direct_score": -1e6, "review_score": 1e6}
    second = gates.run_gate_comparison(rows, split, detections=changed, signals=("direct", "review"))
    assert first["selections"] == second["selections"]
    assert first["predictions"] == second["predictions"]


@pytest.mark.parametrize("bad", [None, True, "0.9", float("nan"), float("inf")])
def test_missing_or_malformed_calibration_is_not_silently_dropped(gate_data, bad):
    rows, split, _, detections = gate_data
    qid = split["folds"][0]["calibration_ids"][0]
    detections[qid]["direct_score"] = bad
    with pytest.raises(ValueError, match="incomplete calibration"):
        gates.run_gate_comparison(rows, split, detections=detections, signals=("direct",))


def test_incomplete_evaluation_is_not_a_gate_pass_or_complete_metric(gate_data):
    rows, split, _, detections = gate_data
    qid = next(i for i in split["folds"][0]["evaluation_ids"] if i.startswith("nfp"))
    detections[qid]["direct_score"] = None
    result = gates.run_gate_comparison(rows, split, detections=detections, signals=("direct",))
    invalid = next(p for p in result["predictions"] if p["id"] == qid)
    assert invalid["score"] is None and invalid["gate_on"] is None
    report, summary = gates.gate_report(result, repeats=20)
    assert "| direct | 39/40 | incomplete | incomplete | incomplete |" in report
    assert summary["direct"]["valid"] == 39
    assert summary["direct"]["nfp_n"] == 19


def test_crossfit_auroc_does_not_pool_incomparable_fold_score_scales():
    predictions = [
        {"id": "a0", "group_id": "a", "fold": 0, "set": "nfp", "score": 100, "gate_on": False},
        {"id": "a1", "group_id": "b", "fold": 0, "set": "fpq", "score": 101, "gate_on": True},
        {"id": "b0", "group_id": "c", "fold": 1, "set": "nfp", "score": -101, "gate_on": False},
        {"id": "b1", "group_id": "d", "fold": 1, "set": "fpq", "score": -100, "gate_on": True},
    ]
    summary = gates.summarize_predictions(predictions)
    assert summary["auroc"] == 1.0  # a pooled ranking would incorrectly give 0.75
    assert summary["tpr"] == 1.0 and summary["fpr"] == 0.0
    assert gates.bootstrap_ci(predictions, repeats=30, seed=9) == gates.bootstrap_ci(predictions, repeats=30, seed=9)


def test_crossfit_produces_one_prediction_per_question_and_signal(gate_data):
    rows, _, _, detections = gate_data
    split = suite.split_plan(rows, scheme="crossfit", folds=5)
    result = gates.run_gate_comparison(rows, split, detections=detections, signals=("direct", "review"))
    for signal in ("direct", "review"):
        predictions = [p for p in result["predictions"] if p["signal"] == signal]
        assert len(predictions) == len(rows)
        assert len({p["id"] for p in predictions}) == len(rows)
    assert len(result["selections"]) == 10
