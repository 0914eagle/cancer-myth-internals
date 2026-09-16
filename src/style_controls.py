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
  edited         false paraphrase (premise span rewritten, still false)
                 vs true twin (span rewritten, now true): BOTH spans are
                 LLM-edited, so editing traces cancel and only truth is
                 left. The cleanest minimal pair when both exist.
  para           paraphrased FPQ vs paraphrased NFP (one writer for both
                 classes): provenance register removed
  natural->twins gate fitted on natural, scored on FPQ-vs-twin
  natural->edited  gate fitted on natural, scored on false-para-vs-twin
  natural->para  gate fitted on natural, scored on the paraphrases
  para->natural  gate fitted on paraphrases, scored on natural
  twins->natural gate fitted on FPQ-vs-twin, scored on natural
  same_writer    FPQ vs only the NFP written by the FPQ writer (nfp.json
                 from_model == gpt-4o, 31 of 149): no rewriting, same
                 generator on both sides, so generator fingerprints cancel.
                 Few negatives, wide intervals.
  natural->same_writer  gate fitted on all natural, scored on FPQ vs
                 same-writer NFP only

Class label: `label` = 1 for FPQ and for any rewrite marked
label_false_premise=1 (false paraphrases), 0 otherwise (NFP, true twins).

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

CONDITIONS = ("natural", "twins", "edited", "para", "same_writer", "natural->twins", "natural->edited",
              "natural->para", "natural->same_writer", "para->natural", "twins->natural",
              "twins->para", "edited->para")


def row_label(q):
    if q.get("set") == "fpq":
        return 1
    if q.get("set") == "tpair":
        return 0
    if q.get("label_false_premise") in (0, 1):
        return int(q["label_false_premise"])
    return 1 if q.get("set") == "fpq" else 0


def _tag(rows, source):
    out = []
    for q in rows:
        origin = q.get("paraphrase_of") or q.get("pair_id") or q["id"]
        out.append({"id": q["id"], "question": q["question"], "set": q["set"], "label": row_label(q),
                    "origin": origin, "source": source, "group_id": q.get("group_id") or origin,
                    "writer": q.get("from_model")})
    return out


def split_twin_file(rows):
    """A questions_twins.jsonl may hold true twins (set=tpair, label 0) and
    false paraphrases (span rewritten, still false, label_false_premise=1).
    Returns (true_twins, false_paraphrases); anything else is an error."""
    twins, fparas, other = [], [], []
    for r in rows:
        if r.get("set") == "tpair":
            twins.append(r)
        elif r.get("label_false_premise") == 1 and r.get("pair_id"):
            fparas.append(r)
        else:
            other.append(r.get("id"))
    if other:
        raise ValueError(f"Unrecognized twin-file rows (neither tpair nor false paraphrase): {other[:3]}")
    return twins, fparas


def assemble(natural, twins=(), para=(), fparas=()):
    """Tag rows with their origin and source; validate that every rewrite
    traces to a natural question and keeps the right class."""
    by_id = {q["id"]: q for q in natural}
    nat = _tag(natural, "natural")
    tw = _tag(twins, "twins")
    pa = _tag(para, "para")
    fp = _tag(fparas, "fparas")
    for r in tw:
        if r["set"] != "tpair" or by_id.get(r["origin"], {}).get("set") != "fpq":
            raise ValueError(f"Twin {r['id']} must be set=tpair derived from an FPQ")
    for r in fp:
        if r["label"] != 1 or by_id.get(r["origin"], {}).get("set") != "fpq":
            raise ValueError(f"False paraphrase {r['id']} must be label 1 derived from an FPQ")
    for r in pa:
        if r["origin"] not in by_id or r["set"] != by_id[r["origin"]]["set"]:
            raise ValueError(f"Paraphrase {r['id']} must keep its source's set")
    ids = [r["id"] for r in nat + tw + pa + fp]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate row ID across sources")
    return nat, tw, pa, fp


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


def condition_rows(name, nat, tw, pa, fp=(), *, fpq_writer="gpt-4o"):
    """(train_pool, eval_pool) row lists for one condition, before fold split."""
    twin_origins = {t["origin"] for t in tw}
    # Same-writer negatives: NFP rows whose recorded generator equals the FPQ generator.
    # FPQ rows without a from_model field are assumed to be the FPQ writer (Cancer-Myth: GPT-4o).
    same = [r for r in nat if r["set"] == "fpq" and r["writer"] in (None, fpq_writer)] + \
           [r for r in nat if r["set"] == "nfp" and r["writer"] == fpq_writer]
    fpq_with_twin = [r for r in nat if r["set"] == "fpq" and r["id"] in twin_origins]
    pairs = fpq_with_twin + tw
    # edited: only origins that have BOTH a false paraphrase and a true twin.
    both = twin_origins & {f["origin"] for f in fp}
    edited = [f for f in fp if f["origin"] in both] + [t for t in tw if t["origin"] in both]
    table = {
        "natural": (nat, nat),
        "twins": (pairs, pairs),
        "edited": (edited, edited),
        "para": (pa, pa),
        "natural->twins": (nat, pairs),
        "natural->edited": (nat, edited),
        "natural->para": (nat, pa),
        "same_writer": (same, same),
        "natural->same_writer": (nat, same),
        "para->natural": (pa, nat),
        "twins->natural": (pairs, nat),
        # Truth-trained gates scored on the one-writer distribution with the
        # natural class mix: does premise signal learned on pairs carry over?
        "twins->para": (pairs, pa),
        "edited->para": (edited, pa),
    }
    if name not in table:
        raise ValueError(f"Unknown condition {name}")
    train, evaluation = table[name]
    if not train or not evaluation:
        return None
    if {r["label"] for r in train} != {0, 1} or {r["label"] for r in evaluation} != {0, 1}:
        return None
    return train, evaluation


def _as_fit_rows(rows):
    """baseline_gates._fit_predict reads the class from set == 'fpq'; give it
    rows whose set encodes our label without touching the originals."""
    return [{**r, "set": "fpq" if r["label"] == 1 else "nfp"} for r in rows]


def _select(kind, train, *, features, layers, c_grid, seed):
    from sklearn.metrics import roc_auc_score
    train = _as_fit_rows(train)
    inner = _group_folds(train, 3, seed)
    candidates = []
    for layer in ((None,) if kind in TEXT_SIGNALS else sorted(set(layers))):
        for c in ((1.0,) if kind == "mean" else sorted(set(c_grid))):
            aucs = []
            for tr, va in inner:
                scores = _fit_predict(kind, [train[i] for i in tr], [train[i] for i in va],
                                      features=features, layer=layer, c=c, seed=seed)
                aucs.append(roc_auc_score([train[i]["label"] for i in va], scores))
            candidates.append({"layer": layer, "C": c, "train_cv_auroc": float(np.mean(aucs))})
    return max(candidates, key=lambda x: x["train_cv_auroc"]), candidates


def run_conditions(nat, tw, pa, fp=(), *, assignment, signals=("text", "style", "masked", "hidden", "mean"),
                   features=None, layers=(), c_grid=(0.001, 0.01, 0.1, 1.0), conditions=CONDITIONS, seed=17,
                   fpq_writer="gpt-4o"):
    """Out-of-fold scores for every condition x signal. Hidden signals run
    only where every row of the condition has features; the result lists
    what was skipped and why."""
    import time
    features = features or {}
    predictions, selections, skipped = [], [], []
    started = time.time()
    n_layers = max(1, len(set(layers or ())))
    for name in conditions:
        pools = condition_rows(name, nat, tw, pa, fp, fpq_writer=fpq_writer)
        if pools is None:
            skipped.append({"condition": name, "reason": "no rows / missing class"})
            continue
        train_pool, eval_pool = pools
        print(f"[controls] {name}: train pool {len(train_pool)}, eval pool {len(eval_pool)}", flush=True)
        for kind in signals:
            # Cost note: each fold fits (layers x C-grid x 3 inner) + 1 models;
            # hidden on 3584-dim states is the slow one, minutes per fold.
            fits = (n_layers if kind not in TEXT_SIGNALS else 1) * (1 if kind == "mean" else len(set(c_grid))) * 3 + 1
            t0 = time.time()
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
                if {r["label"] for r in train} != {0, 1} or {r["label"] for r in held} != {0, 1}:
                    skipped.append({"condition": name, "signal": kind, "reason": f"fold {k} lacks a class"})
                    continue
                try:
                    best, candidates = _select(kind, train, features=features, layers=layers, c_grid=c_grid, seed=seed + k)
                except ValueError as exc:  # inner CV cannot split (too few negatives)
                    skipped.append({"condition": name, "signal": kind, "reason": f"fold {k}: {exc}"})
                    continue
                scores = _fit_predict(kind, _as_fit_rows(train), _as_fit_rows(held), features=features,
                                      layer=best["layer"], c=best["C"], seed=seed)
                selections.append({"condition": name, "signal": kind, "fold": k, "best": best,
                                   "candidates": candidates, "train_n": len(train), "eval_n": len(held)})
                for r, s in zip(held, scores):
                    predictions.append({"condition": name, "signal": kind, "fold": k, "id": r["id"],
                                        "origin": r["origin"], "group_id": r["group_id"], "set": r["set"],
                                        "label": r["label"], "source": r["source"], "score": float(s)})
            print(f"[controls]   {kind}: {len(folds)} folds x {fits} fits in {time.time() - t0:.0f}s "
                  f"(elapsed {time.time() - started:.0f}s)", flush=True)
    return {"predictions": predictions, "selections": selections, "skipped": skipped,
            "signals": list(signals), "conditions": list(conditions), "layers": list(layers), "c_grid": list(c_grid)}


def fold_weighted_auroc(preds):
    from sklearn.metrics import roc_auc_score
    aucs, weights = [], []
    for k in sorted({p["fold"] for p in preds}):
        subset = [p for p in preds if p["fold"] == k]
        y = [p["label"] for p in subset]
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
                                       "counts": dict(Counter(p["set"] for p in ps)),
                                       "positives": sum(p["label"] for p in ps),
                                       "negatives": sum(1 - p["label"] for p in ps)}
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
             "| Condition | Signal | n (pos/neg) | AUROC [95% CI] | AUROC - " + reference + " [95% CI] | chosen |",
             "|---|---|---:|---:|---:|---|"]
    chosen = {}
    for s in result["selections"]:
        chosen.setdefault((s["condition"], s["signal"]), []).append(s["best"])
    for name, summary in summaries.items():
        for sig, info in summary["signals"].items():
            ci = info.get("ci")
            auroc = "unavailable" if info["auroc"] is None else f"{info['auroc']:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")
            d = summary["delta_vs_" + reference].get(sig)
            delta = "-" if not d or d["delta"] is None else f"{d['delta']:+.3f}" + (f" [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if d.get("ci") else "")
            picks = chosen.get((name, sig), [])
            pick = ", ".join(sorted({f"L{b['layer']}/C{b['C']:g}" if b["layer"] is not None else f"C{b['C']:g}" for b in picks}))
            lines.append(f"| {name} | {sig} | {info['positives']}/{info['negatives']} | {auroc} | {delta} | {pick} |")
    if result["skipped"]:
        lines += ["", "Skipped:"] + [f"- {s['condition']}" + (f"/{s['signal']}" if s.get("signal") else "") + f": {s['reason']}"
                                     for s in result["skipped"]]
    lines += ["", "Read: the register floor is max(style, masked). A signal whose margin over that floor survives",
              "twins/edited and para is reading the premise; one whose margin vanishes there was reading provenance.",
              "edited (false paraphrase vs true twin) is the cleanest pair: both spans LLM-written, only truth differs."]
    return "\n".join(lines) + "\n"
