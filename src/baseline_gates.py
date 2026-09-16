"""Question-level FPQ detection, separate from final-answer scoring.

Hyperparameters/layers are chosen by group CV inside train; thresholds use only
calibration negatives. No evaluation labels influence fitting or calibration.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from src.baseline_suite import _group_folds, validate_roles

# Signals that read the question text only (no hidden states, no elicitation).
# text: word 1-2gram TF-IDF; style/masked: content-blind (src/style_features.py).
TEXT_SIGNALS = ("text", "style", "masked")
ALL_SIGNALS = TEXT_SIGNALS + ("hidden", "mean", "direct", "review")


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
    elif kind == "style":
        from src.style_features import style_vector
        model = make_pipeline(StandardScaler(), LogisticRegression(**settings))
        train = np.stack([style_vector(q["question"]) for q in train_rows])
        test = np.stack([style_vector(q["question"]) for q in predict_rows])
    elif kind == "masked":
        from src.style_features import mask_content
        model = make_pipeline(TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 3), max_features=5000,
                                              min_df=1, lowercase=False),
                              LogisticRegression(**settings))
        train = [mask_content(q["question"]) for q in train_rows]
        test = [mask_content(q["question"]) for q in predict_rows]
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
            or set(signals) - set(ALL_SIGNALS)):
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
                for layer in ((None,) if kind in TEXT_SIGNALS else sorted(set(layers))):
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


def matched_operating_point(predictions, fpr):
    """TPR when each held fold flags the same share of ITS evaluation negatives.

    The calibration threshold targets an FPR but lands elsewhere out of fold
    (5% became 8-10% in crossfit_v1), and it lands differently per signal, so
    the TPR column is not comparable across rows. Here the threshold is set
    post hoc on the fold's evaluation negatives so that floor(fpr * n_neg)
    of them are flagged (strict >), the same count for every signal. This is
    an oracle operating point for comparison, not a deployable threshold.
    Fold-wise because raw scores are not comparable across folds."""
    if not 0 <= fpr < 1:
        raise ValueError("fpr must be in [0, 1)")
    valid = [p for p in predictions if p["score"] is not None]
    tp = fp = n_pos = n_neg = 0
    for fold in sorted({p["fold"] for p in valid}):
        subset = [p for p in valid if p["fold"] == fold]
        negatives = [p["score"] for p in subset if p["set"] == "nfp"]
        positives = [p["score"] for p in subset if p["set"] == "fpq"]
        if not negatives or not positives:
            continue
        threshold = threshold_at_fpr(negatives, fpr)
        tp += sum(s > threshold for s in positives)
        fp += sum(s > threshold for s in negatives)
        n_pos += len(positives)
        n_neg += len(negatives)
    if not n_pos or not n_neg:
        return {"fpr_target": fpr, "tpr": None, "fpr": None, "tp": tp, "fp": fp, "fpq_n": n_pos, "nfp_n": n_neg}
    return {"fpr_target": fpr, "tpr": tp / n_pos, "fpr": fp / n_neg, "tp": tp, "fp": fp,
            "fpq_n": n_pos, "nfp_n": n_neg}


def partial_auroc(predictions, max_fpr=0.1):
    """Fold-weighted standardized partial AUROC over FPR in [0, max_fpr]
    (sklearn's McClish correction; 0.5 = chance, 1 = perfect in that range)."""
    from sklearn.metrics import roc_auc_score
    if not 0 < max_fpr <= 1:
        raise ValueError("max_fpr must be in (0, 1]")
    valid = [p for p in predictions if p["score"] is not None]
    aucs, weights = [], []
    for fold in sorted({p["fold"] for p in valid}):
        subset = [p for p in valid if p["fold"] == fold]
        if {p["set"] for p in subset} != {"fpq", "nfp"}:
            continue
        aucs.append(float(roc_auc_score([p["set"] == "fpq" for p in subset], [p["score"] for p in subset],
                                        max_fpr=max_fpr)))
        weights.append(len(subset))
    return float(np.average(aucs, weights=weights)) if aucs else None


def _group_resamples(predictions, repeats, seed):
    rng = np.random.default_rng(seed)
    grouped = {}
    for p in predictions:
        grouped.setdefault(p["fold"], {}).setdefault(p["group_id"], []).append(p)
    for _ in range(repeats):
        sample = []
        for groups in grouped.values():
            keys = list(groups)
            for i in rng.integers(0, len(keys), len(keys)):
                sample.extend(groups[keys[i]])
        yield sample


def bootstrap_matched(predictions, *, fprs=(0.05, 0.08, 0.10), max_fpr=0.1, repeats=500, seed=17):
    """Group bootstrap (within folds) of the matched-FPR TPRs and partial AUROC."""
    out = {"matched": {str(f): matched_operating_point(predictions, f) for f in fprs},
           "partial_auroc": {"max_fpr": max_fpr, "value": partial_auroc(predictions, max_fpr)}}
    if repeats < 1:
        return out
    draws = {str(f): [] for f in fprs}
    pauc = []
    for sample in _group_resamples(predictions, repeats, seed):
        for f in fprs:
            value = matched_operating_point(sample, f)["tpr"]
            if value is not None:
                draws[str(f)].append(value)
        value = partial_auroc(sample, max_fpr)
        if value is not None:
            pauc.append(value)
    for f in fprs:
        out["matched"][str(f)]["ci"] = np.quantile(draws[str(f)], [0.025, 0.975]).tolist() if draws[str(f)] else None
    out["partial_auroc"]["ci"] = np.quantile(pauc, [0.025, 0.975]).tolist() if pauc else None
    return out


def matched_report(result, *, fprs=(0.05, 0.08, 0.10), max_fpr=0.1, repeats=500):
    """Appendix table: every signal at the same evaluation-negative FPR."""
    lines = ["", "## Matched operating points (appendix)", "",
             "Thresholds here are set post hoc on each held fold's evaluation negatives so every signal flags the",
             "same share of normal questions; this compares rankings at equal FPR and is NOT a deployable threshold.",
             f"pAUROC: standardized partial AUROC over FPR in [0, {max_fpr:g}] (0.5 = chance).",
             "Intervals: group bootstrap within folds.", "",
             "| Signal | " + " | ".join(f"TPR @ FPR {f:.0%} [95% CI] (TP/FPQ, FP/NFP)" for f in fprs)
             + f" | pAUROC@{max_fpr:g} [95% CI] |",
             "|---|" + "---:|" * (len(fprs) + 1)]
    summaries = {}
    for signal in dict.fromkeys(p["signal"] for p in result["predictions"]):
        ps = [p for p in result["predictions"] if p["signal"] == signal]
        summary = bootstrap_matched(ps, fprs=fprs, max_fpr=max_fpr, repeats=repeats)
        summaries[signal] = summary
        cells = []
        for f in fprs:
            m = summary["matched"][str(f)]
            if m["tpr"] is None:
                cells.append("unavailable")
                continue
            ci = m.get("ci")
            cells.append(f"{m['tpr']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
                         + f" ({m['tp']}/{m['fpq_n']}, {m['fp']}/{m['nfp_n']})")
        pa = summary["partial_auroc"]
        ci = pa.get("ci")
        cells.append("unavailable" if pa["value"] is None else
                     f"{pa['value']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
        lines.append(f"| {signal} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", summaries


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
    lines.append("")
    lines.append("style/masked rows (if present) see no content words: the register floor every gate must clear.")
    matched_text, matched = matched_report(result, repeats=repeats)
    for signal, extra in matched.items():
        summaries.setdefault(signal, {})["matched"] = extra
    return "\n".join(lines) + "\n" + matched_text, summaries
