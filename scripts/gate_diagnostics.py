"""Offline dev diagnosis of the frozen L21 routing pilot. No model or judge calls.

Replays signed saved answers; never reads test features or chooses a dev threshold.
Optional C selection uses grouped CV inside gate train, preserving calibration.
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pilot_gate import (
    backbone_fingerprint,
    metrics,
    split_fit,
    threshold_at_fpr,
    SEED,
)
from src.jsonl import load_json, read_jsonl
from src.pilot import digest, file_digest, frozen_json, load_manifest, output_lock

VERSION = "gate-dev-offline-diagnostics-v1"
CANDIDATE_C = (0.001, 0.01, 0.1, 1.0, 10.0)
FPR_GRID = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75)


def load_inputs(pilot, gate_dir, nfp_dir):
    from scripts.audit_pilot_judge import audit
    from scripts.reevaluate_pilot_nfp import get_plan, load_generation, METHODS
    from scripts.check_terra_judge import MODEL, events_by_id, parse_nfp_json

    manifest = load_manifest(pilot / "split/manifest.json")
    rows, split = split_fit(manifest)
    features = load_json(gate_dir / "features.json")
    gate = load_json(gate_dir / "gate.json")
    prior = load_json(gate_dir / "report.json")
    if (
        features["manifest_hash"] != manifest["manifest_hash"]
        or features["question_hash"] != digest(rows)
        or gate["features_hash"] != digest(features)
        or gate["split"] != split
        or prior["gate_hash"] != digest(gate)
    ):
        raise ValueError("Gate/report/manifest mismatch")
    sources = dict(prior["sources"])
    for path, expected in sources.items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Source changed since original gate report: {path}")
    extra = [
        pilot / "split/manifest.json",
        gate_dir / "features.json",
        gate_dir / "gate.json",
        gate_dir / "report.json",
        nfp_dir / "plan.json",
        nfp_dir / "run.json",
        nfp_dir / "attempts.jsonl",
    ]
    # Require the very same NFP ledger, not a substitute with a similar aggregate.
    for path in (nfp_dir / "plan.json", nfp_dir / "attempts.jsonl"):
        if sources.get(str(path.resolve())) != file_digest(path):
            raise ValueError("Use the original gate report's NFP directory")
    plan, cases = get_plan(nfp_dir)
    nrun = load_json(nfp_dir / "run.json")
    if nrun != {"plan_hash": digest(plan)} or plan["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("NFP provenance mismatch")
    nfp = {}
    for key, history in events_by_id(nfp_dir, plan, digest(nrun)).items():
        event = history[-1]
        parsed, ok = parse_nfp_json(event.get("raw"))
        if event["status"] == "finished" and event.get("model") == MODEL and ok:
            nfp[key] = parsed["Sharpness"]
    questions = {q["id"]: q for q in rows if q["partition"] == "dev"}
    ids = sorted(questions)
    kinds = np.array([questions[i]["set"] for i in ids])
    answers, values = {}, {}
    for method in METHODS:
        audit(pilot, "gpt-5.6-sol", method)
        responses, hashes = load_generation(pilot, method, manifest, questions)
        sources.update(hashes)
        run = load_json(pilot / "dev" / f"{method}.jsonl.run.json")
        if backbone_fingerprint(run["identity"]) != backbone_fingerprint(features["identity"]):
            raise ValueError("Feature/answer backbone mismatch")
        sp = pilot / "dev" / f"{method}_judge_codex_gpt-5.6-sol.jsonl"
        extra += [sp, Path(str(sp) + ".run.json")]
        sol = {r["question_id"]: r for r in read_jsonl(sp)}
        answers[method] = responses
        v = []
        for qid in ids:
            q = questions[qid]
            if q["set"] == "fpq":
                score = sol[qid]["sharpness"]
            else:
                key = plan["mapping"][method][qid]
                case = cases[key]
                if (
                    case["question"] != q["question"]
                    or case["reference"] != q["hallucination_text"]
                    or case["answer"] != responses[qid]["response"]
                    or key not in nfp
                ):
                    raise ValueError("NFP question/reference/answer mismatch or missing score")
                score = nfp[key]
            v.append(score)
        values[method] = np.array(v)
    for j, qid in enumerate(ids):
        donors = {}
        for method in METHODS:
            answer = answers[method][qid]["response"]
            if answer in donors:
                values[method][j] = values[donors[answer]][j]
            else:
                donors[answer] = method
    for method, label in [
        ("plain", "plain"),
        ("fp_identification", "always FP Identification"),
        ("premise_cot", "premise CoT"),
    ]:
        if metrics(values[method], values["plain"], kinds) != prior["summary"][label]:
            raise ValueError("Reconstructed baseline differs from original gate report")
    for name, g in gate["gates"].items():
        if not np.isfinite(list(g["scores"].values())).all():
            raise ValueError("Non-finite gate scores")
        if any(g["selected"][i] != (g["scores"][i] > g["threshold"]) for i in ids):
            raise ValueError("Saved gate mask differs from scores")
    sources.update({str(p.resolve()): file_digest(p) for p in extra})
    return rows, split, ids, kinds, features, gate, values, sources


def safe_auc(y, scores):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None


def benefit_readouts(scores, plain, action, kinds):
    fp = kinds == "fpq"
    failed = fp & (plain != 1)
    rescue = (plain != 1) & (action == 1)
    harm = (plain == 1) & (action != 1)
    targets = {
        "FPQ label (all dev)": (fp, np.ones(len(fp), dtype=bool)),
        "FP ID succeeds (FPQ only)": (action == 1, fp),
        "FP ID rescues (FPQ only)": (rescue, fp),
        "FP ID rescues (Plain-failed FPQ only)": (rescue, failed),
        "FP ID harms (all dev; high score predicts harm)": (harm, np.ones(len(fp), dtype=bool)),
    }
    return {
        name: {
            "n": int(mask.sum()),
            "positive": int(y[mask].sum()),
            "auroc": safe_auc(y[mask], scores[mask]),
        }
        for name, (y, mask) in targets.items()
    }


def selection_counts(mask, plain, action, kinds):
    rescue = mask & (plain != 1) & (action == 1)
    harm = mask & (plain == 1) & (action != 1)
    return {
        kind: {
            "selected": int((mask & group).sum()),
            "rescue": int((rescue & group).sum()),
            "harm": int((harm & group).sum()),
            "net_gain": int((rescue & group).sum() - (harm & group).sum()),
        }
        for kind, group in [
            ("all", np.ones(len(mask), dtype=bool)),
            ("fpq", kinds == "fpq"),
            ("nfp", kinds == "nfp"),
        ]
    }


def make_curves(ids, kinds, values, score_sets):
    plain, action = values["plain"], values["fp_identification"]
    base, end = metrics(plain, plain, kinds), metrics(action, plain, kinds)
    curves = []
    for name, scores in score_sets.items():
        # Retrospective exact-budget ranking. ID tie-break is independent of labels/outcomes.
        order = sorted(range(len(ids)), key=lambda j: (-scores[j], ids[j]))
        mask = np.zeros(len(ids), dtype=bool)
        for k in range(len(ids) + 1):
            if k:
                mask[order[k - 1]] = True
            m = metrics(np.where(mask, action, plain), plain, kinds)
            curves.append(
                {
                    "signal": name,
                    "k": k,
                    "pcr": m["fpq"]["pass"],
                    "nfp": m["nfp"]["pass"],
                    "pcs": m["pcs"],
                    "fpq_rescue": m["fpq"]["rescue"],
                    "fpq_harm": m["fpq"]["harm"],
                    "nfp_rescue": m["nfp"]["rescue"],
                    "nfp_harm": m["nfp"]["harm"],
                }
            )
    for k in range(len(ids) + 1):
        fraction = k / len(ids)
        curves.append(
            {
                "signal": "random expectation",
                "k": k,
                "pcr": base["fpq"]["pass"] + fraction * (end["fpq"]["pass"] - base["fpq"]["pass"]),
                "nfp": base["nfp"]["pass"] + fraction * (end["nfp"]["pass"] - base["nfp"]["pass"]),
                "pcs": base["pcs"] + fraction * (end["pcs"] - base["pcs"]),
                **{
                    f"{kind}_{effect}": fraction * end[kind][effect]
                    for kind in ("fpq", "nfp")
                    for effect in ("rescue", "harm")
                },
            }
        )
    return curves


def regularization_check(rows, split, features, gate, gate_dir):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def estimator(c):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=SEED),
        )

    ids = [q["id"] for q in rows]
    index = {qid: j for j, qid in enumerate(ids)}
    x = []
    for qid in ids:
        path = gate_dir / "cache" / f"{digest(qid)}.npz"
        if file_digest(path) != gate["cache_hashes"][qid]:
            raise ValueError(f"Feature cache changed: {qid}")
        with np.load(path, allow_pickle=False) as z:
            if str(z["qid"]) != qid or str(z["plan_hash"]) != digest(features):
                raise ValueError("Feature cache provenance mismatch")
            x.append(z["x"].copy())
    x = np.stack(x)
    if not np.isfinite(x).all():
        raise ValueError("Invalid feature vector")
    y = np.array([q["set"] == "fpq" for q in rows])
    train = np.array([index[i] for i in split["train_ids"]])
    groups = np.array([q["group_id"] for q in rows])[train]
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            x[train], y[train], groups
        )
    )
    candidates = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for c in CANDIDATE_C:
            aucs = []
            for fit, valid in folds:
                if len(np.unique(y[train][fit])) != 2 or len(np.unique(y[train][valid])) != 2:
                    raise ValueError("Both classes required in each train-only CV fold")
                model = estimator(c).fit(x[train][fit], y[train][fit])
                aucs.append(safe_auc(y[train][valid], model.decision_function(x[train][valid])))
            candidates.append({"C": c, "fold_auc": aucs, "mean_auc": float(np.mean(aucs))})
        best = max(candidates, key=lambda c: (c["mean_auc"], -c["C"]))
        model = estimator(best["C"]).fit(x[train], y[train])
    scores = dict(zip(ids, map(float, model.decision_function(x))))
    cal_neg = [scores[i] for i in split["calibration_ids"] if not y[index[i]]]
    threshold = threshold_at_fpr(cal_neg)
    return {
        "C_candidates": candidates,
        "chosen_C": best["C"],
        "selection": "mean grouped 3-fold AUROC inside gate train only; ties smaller C",
        "scores": scores,
        "threshold": threshold,
        "calibration_nfp": len(cal_neg),
        "calibration_false_positives": sum(s > threshold for s in cal_neg),
        "parameters": {
            "mean": model[0].mean_.tolist(),
            "scale": model[0].scale_.tolist(),
            "coef": model[1].coef_.tolist(),
            "intercept": model[1].intercept_.tolist(),
        },
    }


def diagnose(pilot, gate_dir, nfp_dir, out, check_c=False, plot=False):
    import sklearn

    pilot, gate_dir, nfp_dir, out = map(Path, (pilot, gate_dir, nfp_dir, out))
    rows, split, ids, kinds, features, gate, values, sources = load_inputs(pilot, gate_dir, nfp_dir)
    gates = dict(gate["gates"])
    c_result = None
    if check_c:
        c_result = regularization_check(rows, split, features, gate, gate_dir)
        gates["hidden train-CV C"] = c_result
    plain, action = values["plain"], values["fp_identification"]
    oracles = {
        "FPQ-label oracle": np.where(kinds == "fpq", action, plain),
        "Observed-benefit oracle": np.where((action == 1) & (plain != 1), action, plain),
    }
    result = {
        "version": VERSION,
        "implementation_hash": file_digest(Path(__file__)),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "sources": sources,
        "check_c": check_c,
        "split": split,
        "oracle_metrics": {name: metrics(v, plain, kinds) for name, v in oracles.items()},
        "regularization": c_result,
        "signals": {},
        "calibration_sweep": [],
    }
    score_sets = {}
    by_id = {q["id"]: q for q in rows}
    for name, g in gates.items():
        scores = np.array([g["scores"][i] for i in ids])
        score_sets[name] = scores
        mask = scores > g["threshold"]
        result["signals"][name] = {
            "benefit_auroc": benefit_readouts(scores, plain, action, kinds),
            "selection": selection_counts(mask, plain, action, kinds),
            "metrics": metrics(np.where(mask, action, plain), plain, kinds),
        }
        cal_neg = [g["scores"][i] for i in split["calibration_ids"] if by_id[i]["set"] == "nfp"]
        for target in FPR_GRID:
            threshold = threshold_at_fpr(cal_neg, target)
            mask = scores > threshold
            result["calibration_sweep"].append(
                {
                    "signal": name,
                    "target_calibration_fpr": target,
                    "threshold": threshold,
                    "calibration_fp": sum(s > threshold for s in cal_neg),
                    "calibration_nfp": len(cal_neg),
                    "dev_selected": int(mask.sum()),
                    "dev_tpr": float(mask[kinds == "fpq"].mean()),
                    "dev_fpr": float(mask[kinds == "nfp"].mean()),
                    "metrics": metrics(np.where(mask, action, plain), plain, kinds),
                }
            )
    curves = make_curves(ids, kinds, values, score_sets)
    result["budget_curves"] = curves
    cot = metrics(values["premise_cot"], plain, kinds)
    result["cot_metrics"] = cot
    # A description of this viewed dev set, never a selected deployment threshold.
    result["dev_curve_points_strictly_above_cot_pcr_without_lower_nfp"] = {
        name: [
            r["k"]
            for r in curves
            if r["signal"] == name
            and r["pcr"] > cot["fpq"]["pass"] + 1e-9
            and r["nfp"] >= cot["nfp"]["pass"] - 1e-9
        ]
        for name in score_sets
    }
    lines = [
        "# Offline routing diagnosis — inspected dev only",
        "",
        "GPT calls: 0. Gemma forwards/generation: 0. Test not opened. Original artifacts unchanged.",
        "FPQ uses Sol; NFP uses Terra v2 reference-targeted rubric. Annotation/evidence limitations remain.",
        "Oracles use ground truth or observed outcomes; neither is a deployable gate.",
        "",
        "| Oracle | PCR % | NFP pass % | FPQ rescue/harm |",
        "|---|---:|---:|---:|",
    ]
    for name, m in result["oracle_metrics"].items():
        lines.append(
            f"| {name} | {m['fpq']['pass']:.1f} | {m['nfp']['pass']:.1f} | {m['fpq']['rescue']}/{m['fpq']['harm']} |"
        )
    lines += [
        "",
        "Label-oracle success does not establish that the intervention is error-free.",
        "",
        "| Signal | Population | Selected | Rescued | Harmed | Net gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, info in result["signals"].items():
        for group, s in info["selection"].items():
            lines.append(
                f"| {name} | {group} | {s['selected']} | {s['rescue']} | {s['harm']} | {s['net_gain']} |"
            )
    lines += [
        "",
        "Rescue precision = rescued / selected. Net gain / selected subtracts harms; it is a different quantity.",
        "",
        "| Signal | Prediction target | n | Positive | AUROC |",
        "|---|---|---:|---:|---:|",
    ]
    for name, info in result["signals"].items():
        for target, a in info["benefit_auroc"].items():
            auc = "NA (one class/empty)" if a["auroc"] is None else f"{a['auroc']:.3f}"
            lines.append(f"| {name} | {target} | {a['n']} | {a['positive']} | {auc} |")
    lines += [
        "",
        "## Budget curves",
        "",
        "budget_curves.csv: exact top-k ranking on viewed dev; ties use ascending question ID, never labels/outcomes.",
        "This retrospective batch policy is distinct from a calibrated score threshold. No best dev k is selected.",
        "Random is the exact expectation for uniformly selecting k of all dev questions; it has no confidence band.",
        f"CoT comparison point: PCR {cot['fpq']['pass']:.1f}, NFP {cot['nfp']['pass']:.1f}.",
        "Viewed dev k values with strictly higher PCR and no lower NFP:",
        str(result["dev_curve_points_strictly_above_cot_pcr_without_lower_nfp"]),
        "",
        "## Calibration-only threshold sweep",
        "",
        "Fixed FPR grid; thresholds use calibration NFP only. All dev results are descriptive, not a test guarantee.",
        "| Signal | Cal target | Cal FP/n | Threshold | Dev selected | Dev PCR | Dev NFP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in result["calibration_sweep"]:
        lines.append(
            f"| {r['signal']} | {r['target_calibration_fpr']:.2f} | {r['calibration_fp']}/{r['calibration_nfp']} | {r['threshold']:.6f} | {r['dev_selected']} | {r['metrics']['fpq']['pass']:.1f} | {r['metrics']['nfp']['pass']:.1f} |"
        )
    if c_result:
        lines += [
            "",
            "## CPU regularization check",
            "",
            f"Chosen C: {c_result['chosen_C']}; {c_result['selection']}.",
            "Scaler/LR fit within each train fold. Calibration is reserved for the 5% threshold; dev is not used to select C.",
            "This is an exploratory follow-up after viewing the original dev result; it is not independent validation.",
            str(c_result["C_candidates"]),
        ]
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / "diagnostics"):
        frozen_json(out / "diagnostics.json", result)
        with (out / "budget_curves.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        (out / "report.md").write_text("\n".join(lines) + "\n")
        if plot:
            plot_curves(curves, cot, out)
    print(f"Offline diagnosis complete. GPT/Gemma calls: 0. {out / 'report.md'}")
    return result


def plot_curves(curves, cot, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in dict.fromkeys(r["signal"] for r in curves):
        group = [r for r in curves if r["signal"] == name]
        k, pcr, nfp = ([r[key] for r in group] for key in ("k", "pcr", "nfp"))
        style = "--" if name == "random expectation" else "-"
        axes[0].plot(k, pcr, style, label=name)
        axes[1].plot(k, nfp, style, label=name)
        axes[2].plot(nfp, pcr, style, label=name)
    axes[0].axhline(cot["fpq"]["pass"], color="gray", linestyle=":", label="CoT")
    axes[1].axhline(cot["nfp"]["pass"], color="gray", linestyle=":")
    axes[2].scatter(
        [cot["nfp"]["pass"]], [cot["fpq"]["pass"]], marker="*", s=100, color="black", label="CoT"
    )
    for ax, xlabel, ylabel in zip(
        axes,
        ["Selected questions (k)"] * 2 + ["NFP pass (%)"],
        ["PCR (%)", "NFP pass (%)", "PCR (%)"],
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Inspected dev: saved-answer routing, mixed judge protocols (exploratory)")
    fig.tight_layout()
    fig.savefig(out / "budget_curves.png", dpi=180)
    fig.savefig(out / "budget_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--gate-dir", required=True)
    parser.add_argument("--nfp-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--regularization-check", action="store_true")
    parser.add_argument("--plot", action="store_true")
    a = parser.parse_args()
    diagnose(a.pilot_dir, a.gate_dir, a.nfp_dir, a.out_dir, a.regularization_check, a.plot)


if __name__ == "__main__":
    main()
