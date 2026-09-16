"""Three tests of "text beats hidden because of provenance style".

Crossfit on the natural suite gave text 0.755 > hidden 0.693 > DiM 0.681.
FPQ and NFP come from different writers, so the text gate may read
register, not premise. Each condition below re-measures every gate on a
comparison where register is controlled, using the SAME fold assignment
as the natural run (every rewrite inherits its source question's fold, so
no origin leaks across train and evaluation).

  natural        FPQ vs NFP as in crossfit (reference row)
  twins          FPQ vs its true-premise twin (same writer, same text
                 outside the premise span): register held fixed
  para           paraphrased FPQ vs paraphrased NFP (one writer for both
                 classes): provenance register removed
  natural->twins gate fitted on natural, scored on FPQ-vs-twin
  natural->para  gate fitted on natural, scored on the paraphrases
  para->natural  gate fitted on paraphrases, scored on natural
  twins->natural gate fitted on FPQ-vs-twin, scored on natural

Signals: text (TF-IDF), style and masked (content-blind register floor),
hidden and mean (Qwen prefill states; need extracted features for every
row of the condition). The number to read is each signal's AUROC minus
text's, with a paired group bootstrap. Interpretation table: docs/29.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.baseline_gates import TEXT_SIGNALS, _fit_predict
from src.baseline_suite import _group_folds

CONDITIONS = ("natural", "twins", "para", "natural->twins", "natural->para", "para->natural", "twins->natural")


def _tag(rows, source):
    out = []
    for q in rows:
        origin = q.get("paraphrase_of") or q.get("pair_id") or q["id"]
        out.append({"id": q["id"], "question": q["question"], "set": q["set"], "origin": origin,
                    "source": source, "group_id": q.get("group_id") or origin})
    return out


def assemble(natural, twins=(), para=()):
    """Tag rows with their origin and source; validate that every rewrite
    traces to a natural question and keeps the right class."""
    by_id = {q["id"]: q for q in natural}
    nat = _tag(natural, "natural")
    tw = _tag(twins, "twins")
    pa = _tag(para, "para")
    for r in tw:
        if r["set"] != "tpair" or by_id.get(r["origin"], {}).get("set") != "fpq":
            raise ValueError(f"Twin {r['id']} must be set=tpair derived from an FPQ")
    for r in pa:
        if r["origin"] not in by_id or r["set"] != by_id[r["origin"]]["set"]:
            raise ValueError(f"Paraphrase {r['id']} must keep its source's set")
    ids = [r["id"] for r in nat + tw + pa]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate row ID across sources")
    return nat, tw, pa


def fold_assignment(natural, *, folds=5, seed=17):
    """Fold per natural question ID (StratifiedGroupKFold on group_id).
    Rewrites inherit their origin's fold."""
    assignment = {}
    for k, (_, held) in enumerate(_group_folds(natural, folds, seed)):
        for i in held:
            assignment[natural[i]["id"]] = k
    if len(assignment) != len(natural):
        raise ValueError("Every natural question needs exactly one fold")
    return assignment


def condition_rows(name, nat, tw, pa):
    """(train_pool, eval_pool) row lists for one condition, before fold split."""
    fpq_with_twin = [r for r in nat if r["set"] == "fpq" and r["id"] in {t["origin"] for t in tw}]
    pairs = fpq_with_twin + tw
    table = {
        "natural": (nat, nat),
        "twins": (pairs, pairs),
        "para": (pa, pa),
        "natural->twins": (nat, pairs),
        "natural->para": (nat, pa),
        "para->natural": (pa, nat),
        "twins->natural": (pairs, nat),
    }
    if name not in table:
        raise ValueError(f"Unknown condition {name}")
    train, evaluation = table[name]
    if not train or not evaluation:
        return None
    if {r["set"] == "fpq" for r in train} != {True, False} or {r["set"] == "fpq" for r in evaluation} != {True, False}:
        return None
    return train, evaluation


def _select(kind, train, *, features, layers, c_grid, seed):
    from sklearn.metrics import roc_auc_score
    inner = _group_folds(train, 3, seed)
    candidates = []
    for layer in ((None,) if kind in TEXT_SIGNALS else sorted(set(layers))):
        for c in ((1.0,) if kind == "mean" else sorted(set(c_grid))):
            aucs = []
            for tr, va in inner:
                scores = _fit_predict(kind, [train[i] for i in tr], [train[i] for i in va],
                                      features=features, layer=layer, c=c, seed=seed)
                aucs.append(roc_auc_score([train[i]["set"] == "fpq" for i in va], scores))
            candidates.append({"layer": layer, "C": c, "train_cv_auroc": float(np.mean(aucs))})
    return max(candidates, key=lambda x: x["train_cv_auroc"]), candidates


def run_conditions(nat, tw, pa, *, assignment, signals=("text", "style", "masked", "hidden", "mean"),
                   features=None, layers=(), c_grid=(0.001, 0.01, 0.1, 1.0), conditions=CONDITIONS, seed=17):
    """Out-of-fold scores for every condition x signal. Hidden signals run
    only where every row of the condition has features; the result lists
    what was skipped and why."""
    features = features or {}
    predictions, selections, skipped = [], [], []
    for name in conditions:
        pools = condition_rows(name, nat, tw, pa)
        if pools is None:
            skipped.append({"condition": name, "reason": "no rows / missing class"})
            continue
        train_pool, eval_pool = pools
        for kind in signals:
            if kind not in TEXT_SIGNALS:
                missing = [r["id"] for r in train_pool + eval_pool if r["id"] not in features]
                if missing:
                    skipped.append({"condition": name, "signal": kind,
                                    "reason": f"{len(missing)} rows without features (e.g. {missing[0]})"})
                    continue
            folds = sorted(set(assignment.values()))
            for k in folds:
                train = [r for r in train_pool if assignment[r["origin"]] != k]
                held = [r for r in eval_pool if assignment[r["origin"]] == k]
                if {r["set"] == "fpq" for r in train} != {True, False} or {r["set"] == "fpq" for r in held} != {True, False}:
                    raise ValueError(f"{name}/{kind}: fold {k} lacks a class")
                best, candidates = _select(kind, train, features=features, layers=layers, c_grid=c_grid, seed=seed + k)
                scores = _fit_predict(kind, train, held, features=features, layer=best["layer"], c=best["C"], seed=seed)
                selections.append({"condition": name, "signal": kind, "fold": k, "best": best,
                                   "candidates": candidates, "train_n": len(train), "eval_n": len(held)})
                for r, s in zip(held, scores):
                    predictions.append({"condition": name, "signal": kind, "fold": k, "id": r["id"],
                                        "origin": r["origin"], "group_id": r["group_id"], "set": r["set"],
                                        "source": r["source"], "score": float(s)})
    return {"predictions": predictions, "selections": selections, "skipped": skipped,
            "signals": list(signals), "conditions": list(conditions), "layers": list(layers), "c_grid": list(c_grid)}


def fold_weighted_auroc(preds):
    from sklearn.metrics import roc_auc_score
    aucs, weights = [], []
    for k in sorted({p["fold"] for p in preds}):
        subset = [p for p in preds if p["fold"] == k]
        y = [p["set"] == "fpq" for p in subset]
        if len(set(y)) < 2:
            continue
        aucs.append(float(roc_auc_score(y, [p["score"] for p in subset])))
        weights.append(len(subset))
    return float(np.average(aucs, weights=weights)) if aucs else None


def summarize(result, *, repeats=500, seed=17, reference="text"):
    """Per condition: AUROC per signal, and (signal - reference) with a paired
    group bootstrap: the same resample of origin groups (within folds) scores
    both signals, so the interval is for the difference, not two marginals."""
    rng = np.random.default_rng(seed)
    out = {}
    preds = result["predictions"]
    for name in dict.fromkeys(p["condition"] for p in preds):
        cond = [p for p in preds if p["condition"] == name]
        by_signal = {}
        for p in cond:
            by_signal.setdefault(p["signal"], []).append(p)
        summary = {"signals": {}, "delta_vs_" + reference: {}}
        for sig, ps in by_signal.items():
            summary["signals"][sig] = {"auroc": fold_weighted_auroc(ps), "n": len(ps),
                                       "counts": dict(Counter(p["set"] for p in ps))}
        # Paired resampling: group (origin) keys within folds, shared across signals.
        keys = sorted({(p["fold"], p["group_id"]) for p in cond})
        index = {}
        for p in cond:
            index.setdefault((p["fold"], p["group_id"]), {}).setdefault(p["signal"], []).append(p)
        draws = {sig: [] for sig in by_signal}
        for _ in range(max(repeats, 0)):
            picked = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
            for sig in by_signal:
                sample = [p for key in picked for p in index[key].get(sig, [])]
                value = fold_weighted_auroc(sample)
                draws[sig].append(np.nan if value is None else value)
        for sig in by_signal:
            values = np.asarray(draws[sig], dtype=float)
            values = values[np.isfinite(values)]
            summary["signals"][sig]["ci"] = np.quantile(values, [0.025, 0.975]).tolist() if len(values) else None
            if reference in by_signal and sig != reference:
                a, b = np.asarray(draws[sig], dtype=float), np.asarray(draws[reference], dtype=float)
                ok = np.isfinite(a) & np.isfinite(b)
                point = (summary["signals"][sig]["auroc"] or np.nan) - (summary["signals"][reference]["auroc"] or np.nan)
                summary["delta_vs_" + reference][sig] = {
                    "delta": None if not np.isfinite(point) else float(point),
                    "ci": np.quantile(a[ok] - b[ok], [0.025, 0.975]).tolist() if ok.any() else None}
        out[name] = summary
    return out


def report(result, summaries, *, reference="text"):
    lines = ["# Style-confound controls", "",
             "Same fold assignment as the natural crossfit; every rewrite inherits its source question's fold.",
             "AUROC is evaluation-size-weighted within-fold AUROC. Deltas use a paired group bootstrap (same",
             f"resample scores both signals). style/masked see no content words. Reference signal: {reference}.", "",
             "| Condition | Signal | n (fpq/neg) | AUROC [95% CI] | AUROC - " + reference + " [95% CI] | chosen |",
             "|---|---|---:|---:|---:|---|"]
    chosen = {}
    for s in result["selections"]:
        chosen.setdefault((s["condition"], s["signal"]), []).append(s["best"])
    for name, summary in summaries.items():
        for sig, info in summary["signals"].items():
            counts = info["counts"]
            neg = sum(v for k, v in counts.items() if k != "fpq")
            ci = info.get("ci")
            auroc = "unavailable" if info["auroc"] is None else f"{info['auroc']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
            d = summary["delta_vs_" + reference].get(sig)
            delta = "-" if not d or d["delta"] is None else f"{d['delta']:+.3f}" + (f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d.get("ci") else "")
            picks = chosen.get((name, sig), [])
            pick = ", ".join(sorted({f"L{b['layer']}/C{b['C']:g}" if b["layer"] is not None else f"C{b['C']:g}" for b in picks}))
            lines.append(f"| {name} | {sig} | {counts.get('fpq', 0)}/{neg} | {auroc} | {delta} | {pick} |")
    if result["skipped"]:
        lines += ["", "Skipped:"] + [f"- {s['condition']}" + (f"/{s['signal']}" if s.get("signal") else "") + f": {s['reason']}"
                                     for s in result["skipped"]]
    lines += ["", "Read: the register floor is max(style, masked). A signal whose margin over that floor survives",
              "twins and para is reading the premise; one whose margin vanishes there was reading provenance."]
    return "\n".join(lines) + "\n"
