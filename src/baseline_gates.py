"""Question-level FPQ detection, separate from final-answer scoring.

Hyperparameters/layers are chosen by group CV inside train; thresholds use only
calibration negatives. No evaluation labels influence fitting or calibration.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.baseline_suite import _group_folds, validate_roles


def threshold_at_fpr(negative_scores, target=0.05):
    x = np.asarray(negative_scores, dtype=float)
    if not len(x) or not np.isfinite(x).all() or not 0 <= target < 1:
        raise ValueError("Need finite calibration negatives and target in [0,1)")
    return float(np.sort(x)[::-1][math.floor(target * len(x))])


def _fit_predict(kind, train_rows, predict_rows, *, features, layer, c, seed):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    y = np.asarray([q["set"] == "fpq" for q in train_rows], dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Training fold needs both classes")
    settings = dict(C=c, class_weight="balanced", max_iter=5000, random_state=seed)
    if kind == "text":
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1),
                              LogisticRegression(**settings))
        train = [q["question"] for q in train_rows]
        test = [q["question"] for q in predict_rows]
    else:
        train = np.stack([features[q["id"]][layer] for q in train_rows])
        test = np.stack([features[q["id"]][layer] for q in predict_rows])
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError("Nonfinite hidden state")
        if kind == "mean":
            # Positive pole FPQ minus NFP. This is a readout, NOT steering C.
            direction = train[y == 1].mean(0) - train[y == 0].mean(0)
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError("Degenerate mean-difference direction")
            return test @ (direction / norm)
        model = make_pipeline(StandardScaler(), LogisticRegression(**settings))
    model.fit(train, y)
    return model.decision_function(test)


def run_gate_comparison(rows, split, *, features=None, detections=None,
                        signals=("text", "hidden", "mean", "direct", "review"),
                        layers=(11, 17, 22), c_grid=(0.001, 0.01, 0.1, 1.0),
                        target_fpr=0.05, seed=17):
    from sklearn.metrics import roc_auc_score
    from src.pilot import digest

    if split["question_hash"] != digest(rows):
        raise ValueError("Split questions changed")
    if (not signals or len(signals) != len(set(signals))
            or set(signals) - {"text", "hidden", "mean", "direct", "review"}):
        raise ValueError("Unknown/duplicate signal")
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError("Positive finite regularization grid required")
    if any(s in signals for s in ("hidden", "mean")) and (not features or not layers):
        raise ValueError("Hidden signals need extracted features/layers")
    if any(s in signals for s in ("direct", "review")) and detections is None:
        raise ValueError("Output elicitation signals need detection records")
    if not 0 <= target_fpr < 1:
        raise ValueError("Invalid FPR target")
    by_id = {q["id"]: q for q in rows}
    results, selections = [], []
    for fold in split["folds"]:
        validate_roles(rows, fold)
        train = [by_id[i] for i in fold["train_ids"]]
        cal = [by_id[i] for i in fold["calibration_ids"]]
        held = [by_id[i] for i in fold["evaluation_ids"]]
        for kind in signals:
            selection = {"fold": fold["fold"], "signal": kind}
            if kind in {"direct", "review"}:
                key = kind + "_score"
                def score(q):
                    value = detections.get(q["id"], {}).get(key)
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                        return None
                    return float(value)
                cal_scores, held_scores = [score(q) for q in cal], [score(q) for q in held]
                # A calibrated FPR computed only on convenient valid negatives is misleading.
                if any(s is None for s in cal_scores):
                    raise ValueError(f"{kind}: incomplete calibration detection; cannot set threshold")
                selection["selection"] = "fixed elicitation; no evaluation tuning"
            else:
                inner = _group_folds(train, 3, seed + fold["fold"])
                candidates = []
                for layer in ((None,) if kind == "text" else sorted(set(layers))):
                    for c in ((1.0,) if kind == "mean" else sorted(set(c_grid))):
                        aucs = []
                        for tr, va in inner:
                            scores = _fit_predict(kind, [train[i] for i in tr], [train[i] for i in va],
                                                  features=features, layer=layer, c=c, seed=seed)
                            aucs.append(roc_auc_score([train[i]["set"] == "fpq" for i in va], scores))
                        candidates.append({"layer": layer, "C": c, "train_cv_auroc": float(np.mean(aucs))})
                # Stable tie-break: lower layer then smaller C from ordered candidates.
                best = max(candidates, key=lambda x: x["train_cv_auroc"])
                scores = _fit_predict(kind, train, cal + held, features=features,
                                      layer=best["layer"], c=best["C"], seed=seed)
                cal_scores, held_scores = scores[:len(cal)].tolist(), scores[len(cal):].tolist()
                selection.update(best=best, candidates=candidates)
            negatives = [s for q, s in zip(cal, cal_scores) if q["set"] == "nfp"]
            threshold = threshold_at_fpr(negatives, target_fpr)
            selection.update(threshold=threshold, calibration_negative_n=len(negatives),
                             calibration_fp=sum(s > threshold for s in negatives),
                             train_ids=fold["train_ids"], calibration_ids=fold["calibration_ids"],
                             evaluation_ids=fold["evaluation_ids"])
            selections.append(selection)
            for q, score in zip(held, held_scores):
                results.append({"id": q["id"], "group_id": q["group_id"], "set": q["set"],
                                "fold": fold["fold"], "signal": kind, "score": score,
                                "threshold": threshold, "gate_on": score > threshold if score is not None else None})
    return {"split": split, "target_fpr": target_fpr, "selections": selections, "predictions": results}


def summarize_predictions(predictions):
    """Fold-weighted AUROC avoids ranking incomparable raw scores across folds."""
    from sklearn.metrics import roc_auc_score
    valid = [p for p in predictions if p["score"] is not None]
    aucs, weights = [], []
    for fold in sorted({p["fold"] for p in predictions}):
        subset = [p for p in valid if p["fold"] == fold]
        if {p["set"] for p in subset} != {"fpq", "nfp"}:
            continue
        aucs.append(float(roc_auc_score([p["set"] == "fpq" for p in subset], [p["score"] for p in subset])))
        weights.append(len(subset))
    counts = Counter(p["set"] for p in valid)
    tp = sum(p["set"] == "fpq" and p["gate_on"] for p in valid)
    fp = sum(p["set"] == "nfp" and p["gate_on"] for p in valid)
    return {"expected": len(predictions), "valid": len(valid), "fpq_n": counts["fpq"],
            "nfp_n": counts["nfp"], "tp": tp, "fp": fp,
            "auroc": float(np.average(aucs, weights=weights)) if aucs else None,
            "tpr": tp / counts["fpq"] if counts["fpq"] else None,
            "fpr": fp / counts["nfp"] if counts["nfp"] else None}


def bootstrap_ci(predictions, *, repeats=500, seed=17):
    """Group bootstrap within each held fold; conditional on fitted readouts."""
    if repeats < 1:
        return {}
    rng = np.random.default_rng(seed)
    folds = sorted({p["fold"] for p in predictions})
    grouped = {}
    for fold in folds:
        grouped[fold] = {}
        for p in predictions:
            if p["fold"] == fold:
                grouped[fold].setdefault(p["group_id"], []).append(p)
    values = {name: [] for name in ("auroc", "tpr", "fpr")}
    for _ in range(repeats):
        sample = []
        for groups in grouped.values():
            keys = list(groups)
            for i in rng.integers(0, len(keys), len(keys)):
                sample.extend(groups[keys[i]])
        summary = summarize_predictions(sample)
        for name in values:
            if summary[name] is not None:
                values[name].append(summary[name])
    return {name: np.quantile(v, [0.025, 0.975]).tolist() for name, v in values.items() if v}


def gate_report(result, *, repeats=500):
    lines = ["# Question-level gate comparison", "",
             "Question labels only. No answer-quality scoring, routing or steering in this table.",
             "Our grouped split, not the Well paper split. Earlier dev inspection remains exploratory.",
             "Crossfit AUROC is evaluation-size-weighted within-fold AUROC (not pooled raw scores).",
             "95% intervals bootstrap source groups within folds, conditional on fitted models; not training uncertainty.",
             "Thresholds use calibration negatives only. An empirical FPR target is not a population guarantee.", "",
             "| Signal | Valid/expected | AUROC [95% CI] | TPR [95% CI] | FPR [95% CI] | TP/FPQ | FP/NFP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    summaries = {}
    def cell(summary, ci, key):
        if summary[key] is None:
            return "unavailable"
        interval = ci.get(key)
        return f"{summary[key]:.3f}" + (f" [{interval[0]:.3f}, {interval[1]:.3f}]" if interval else "")
    for signal in dict.fromkeys(p["signal"] for p in result["predictions"]):
        ps = [p for p in result["predictions"] if p["signal"] == signal]
        summary = summarize_predictions(ps)
        ci = bootstrap_ci(ps, repeats=repeats)
        summaries[signal] = {**summary, "ci": ci}
        incomplete = summary["valid"] != summary["expected"]
        vals = ["incomplete" if incomplete else cell(summary, ci, key) for key in ("auroc", "tpr", "fpr")]
        lines.append(f"| {signal} | {summary['valid']}/{summary['expected']} | " + " | ".join(vals)
                     + f" | {summary['tp']}/{summary['fpq_n']} | {summary['fp']}/{summary['nfp_n']} |")
    return "\n".join(lines) + "\n", summaries
