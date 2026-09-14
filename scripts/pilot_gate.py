"""Single-layer gate pilot: question prefill -> fit/calibrate -> saved-answer routing.

No generation/judge calls. Fit is subdivided by group; dev is never used to set
thresholds. Baseline action is the existing two-stage FP Identification pipeline,
not a forced FP correction prompt. Original test stays locked.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.jsonl import load_json, read_jsonl
from src.pilot import digest, file_digest, frozen_json, load_manifest, output_lock

VERSION = "single-layer-question-gate-v1"
LAYER = 21
SEED = 17


def backbone_fingerprint(identity):
    # Implementation hashes can differ between saved action pipelines; compare the
    # actual model/tokenizer checkpoint and chat template, retaining full identities.
    return {
        "model_id": identity.get("source_model", {}).get("model_id"),
        **{
            key: identity.get(key)
            for key in ("resolved_revision", "tokenizer_revision", "chat_template_hash")
        },
    }


def split_fit(manifest):
    from sklearn.model_selection import StratifiedGroupKFold

    rows = sorted(
        (q for q in manifest["questions"] if q["partition"] in ("fit", "dev")),
        key=lambda q: q["id"],
    )
    if any(q["set"] not in ("fpq", "nfp") for q in rows):
        raise ValueError("Only FPQ/NFP labels are supported")
    fit = [q for q in rows if q["partition"] == "fit"]
    dev = [q for q in rows if q["partition"] == "dev"]
    if not fit or not dev or {q["group_id"] for q in fit} & {q["group_id"] for q in dev}:
        raise ValueError("Need nonempty group-disjoint fit/dev")
    y = np.array([q["set"] == "fpq" for q in fit], dtype=int)
    train, cal = next(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
            np.zeros(len(fit)), y, [q["group_id"] for q in fit]
        )
    )
    for indices in (train, cal):
        if set(y[indices]) != {0, 1}:
            raise ValueError("Both classes required in gate train and calibration")
    train_ids, cal_ids = [fit[i]["id"] for i in train], [fit[i]["id"] for i in cal]
    return rows, {
        "train_ids": train_ids,
        "calibration_ids": cal_ids,
        "dev_ids": [q["id"] for q in dev],
    }


def threshold_at_fpr(scores, target=0.05):
    """Strict 'score > threshold'; ties are excluded together, never randomized."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 <= target < 1:
        raise ValueError("Invalid calibration scores/target")
    allowed = math.floor(target * len(scores))
    # Excludes the (allowed+1)-th largest negative and all ties.
    return float(np.sort(scores)[::-1][allowed])


def extract(pilot, out, config):
    import torch
    from src.pilot_model import load_model, model_identity, render
    from src.modeling import decoder_layers

    pilot, out = Path(pilot), Path(out)
    manifest = load_manifest(pilot / "split/manifest.json")
    rows, split = split_fit(manifest)
    cfg = load_config(config)
    if cfg["source_model"]["model_id"] != "google/gemma-2-9b-it":
        raise ValueError("This fixed pilot is Gemma-2-9B-it only")
    with output_lock(out / "extract"):
        model, tok = load_model(cfg)
        identity = model_identity(model, tok, cfg)
        baseline = load_json(pilot / "dev/plain.jsonl.run.json")["identity"]
        for key in ("resolved_revision", "tokenizer_revision", "chat_template_hash"):
            if identity[key] != baseline[key]:
                raise ValueError(f"Baseline model/tokenizer differs: {key}")
        if identity["source_model"]["model_id"] != baseline["source_model"]["model_id"]:
            raise ValueError("Baseline model differs")
        if not 1 <= LAYER < len(decoder_layers(model)):
            raise ValueError("Layer must precede final normalization")
        plan = {
            "version": VERSION,
            "manifest_hash": manifest["manifest_hash"],
            "identity": identity,
            "layer": LAYER,
            "position": "last token of original-question chat prefill; block output before final norm",
            "seed": SEED,
            "split": split,
            "question_ids": [q["id"] for q in rows],
            "question_hash": digest(rows),
            "implementation_hash": file_digest(Path(__file__)),
        }
        frozen_json(out / "features.json", plan)
        cache = out / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        vector = []

        def capture(_module, _inputs, outputs):
            h = outputs[0] if isinstance(outputs, tuple) else outputs
            vector.append(h[0, -1].detach().float().cpu().numpy().copy())

        hook = decoder_layers(model)[LAYER - 1].register_forward_hook(capture)
        try:
            for i, q in enumerate(rows, 1):
                path = cache / f"{digest(q['id'])}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as z:
                        if (
                            str(z["plan_hash"].item()) != digest(plan)
                            or str(z["qid"].item()) != q["id"]
                        ):
                            raise ValueError("Feature cache signature mismatch")
                        if (
                            z["x"].shape != (model.config.hidden_size,)
                            or not np.isfinite(z["x"]).all()
                        ):
                            raise ValueError("Invalid cached vector")
                else:
                    ids = tok(render(tok, q["question"]), add_special_tokens=False)["input_ids"]
                    if len(ids) > model.config.max_position_embeddings:
                        raise ValueError("Question too long; no silent truncation")
                    tensor = torch.tensor([ids], device=model.device)
                    vector.clear()
                    with torch.inference_mode():
                        # Base model avoids LM logits and autoregressive answer generation.
                        result = model.base_model(
                            input_ids=tensor,
                            attention_mask=torch.ones_like(tensor),
                            use_cache=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    del result
                    if len(vector) != 1 or not np.isfinite(vector[0]).all():
                        raise ValueError("Invalid captured feature")
                    with path.with_suffix(".tmp").open("wb") as f:
                        np.savez(f, x=vector[0], qid=q["id"], plan_hash=digest(plan))
                    path.with_suffix(".tmp").replace(path)
                if i % 10 == 0 or i == len(rows):
                    print(f"[gate prefill L{LAYER}] {i}/{len(rows)}", flush=True)
        finally:
            hook.remove()
    print("Extraction done. No answers generated; GPT calls: 0. Next: fit, report.")


def fit(out, manifest_path):
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    out = Path(out)
    manifest = load_manifest(manifest_path)
    rows, split = split_fit(manifest)
    with output_lock(out / "fit"):
        plan = load_json(out / "features.json")
        if (
            plan["version"] != VERSION
            or plan["manifest_hash"] != manifest["manifest_hash"]
            or plan["question_hash"] != digest(rows)
            or plan["split"] != split
        ):
            raise ValueError("Features differ from manifest/split")
        X = []
        cache_hashes = {}
        for q in rows:
            p = out / "cache" / f"{digest(q['id'])}.npz"
            with np.load(p, allow_pickle=False) as z:
                if str(z["plan_hash"].item()) != digest(plan) or str(z["qid"].item()) != q["id"]:
                    raise ValueError("Feature signature mismatch")
                X.append(z["x"].copy())
            cache_hashes[q["id"]] = file_digest(p)
        X = np.stack(X)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError("Invalid feature matrix")
        lookup = {q["id"]: i for i, q in enumerate(rows)}
        train = np.array([lookup[i] for i in split["train_ids"]])
        cal = np.array([lookup[i] for i in split["calibration_ids"]])
        y = np.array([q["set"] == "fpq" for q in rows], dtype=int)
        settings = dict(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=3000, random_state=SEED
        )
        hidden = make_pipeline(StandardScaler(), LogisticRegression(**settings))
        bow = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000),
            LogisticRegression(**settings),
        )
        texts = np.array([q["question"] for q in rows])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            hidden.fit(X[train], y[train])
            bow.fit(texts[train], y[train])
        outputs = {}
        for name, clf, values in [("hidden", hidden, X), ("text", bow, texts)]:
            decision = clf.decision_function(values)
            threshold = threshold_at_fpr(decision[cal][y[cal] == 0])
            selected = decision > threshold
            outputs[name] = {
                "threshold": threshold,
                "target_calibration_fpr": 0.05,
                "calibration_nfp": int(sum(y[cal] == 0)),
                "calibration_false_positives": int(sum(selected[cal] & (y[cal] == 0))),
                "scores": {q["id"]: float(decision[i]) for i, q in enumerate(rows)},
                "selected": {q["id"]: bool(selected[i]) for i, q in enumerate(rows)},
            }
        result = {
            "version": VERSION,
            "features_hash": digest(plan),
            "cache_hashes": cache_hashes,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "settings": settings,
            "split": split,
            "gates": outputs,
        }
        frozen_json(out / "gate.json", result)
        # Numeric/JSON parameters only: no executable pickle. Kept for reproducibility.
        params = {
            "hidden_mean": hidden[0].mean_.tolist(),
            "hidden_scale": hidden[0].scale_.tolist(),
            "hidden_coef": hidden[1].coef_.tolist(),
            "hidden_intercept": hidden[1].intercept_.tolist(),
            "text_vocabulary": {k: int(v) for k, v in bow[0].vocabulary_.items()},
            "text_idf": bow[0].idf_.tolist(),
            "text_coef": bow[1].coef_.tolist(),
            "text_intercept": bow[1].intercept_.tolist(),
        }
        frozen_json(out / "parameters.json", params)
    print(
        f"Gate fit: train {len(train)}, calibration {len(cal)}; dev unused for fitting/thresholds. GPT calls: 0."
    )
    for n, g in outputs.items():
        print(
            f"{n}: calibration FPR {g['calibration_false_positives']}/{g['calibration_nfp']}; threshold={g['threshold']:.5f}"
        )


def metrics(values, base, kinds):
    result = {}
    for kind in ("fpq", "nfp"):
        mask = kinds == kind
        a, b = values[mask], base[mask]
        result[kind] = {
            "n": int(mask.sum()),
            "pass": float(100 * np.mean(a == 1)),
            "rescue": int(sum((a == 1) & (b != 1))),
            "harm": int(sum((a != 1) & (b == 1))),
        }
    result["pcs"] = float(np.mean(values[kinds == "fpq"]))
    return result


def report(pilot, out, nfp_dir):
    from sklearn.metrics import roc_auc_score
    from scripts.audit_pilot_judge import audit
    from scripts.reevaluate_pilot_nfp import get_plan, load_generation, METHODS
    from scripts.check_terra_judge import MODEL, events_by_id, parse_nfp_json, evidence_issue

    pilot, out, nfp_dir = map(Path, (pilot, out, nfp_dir))
    manifest = load_manifest(pilot / "split/manifest.json")
    allrows, split = split_fit(manifest)
    features, gate = load_json(out / "features.json"), load_json(out / "gate.json")
    if (
        features["manifest_hash"] != manifest["manifest_hash"]
        or features["question_hash"] != digest(allrows)
        or gate["features_hash"] != digest(features)
        or gate["split"] != split
    ):
        raise ValueError("Gate/manifest mismatch")
    questions = {q["id"]: q for q in allrows if q["partition"] == "dev"}
    ids = sorted(questions)
    kinds = np.array([questions[i]["set"] for i in ids])
    plan, cases = get_plan(nfp_dir)
    nrun = load_json(nfp_dir / "run.json")
    if nrun != {"plan_hash": digest(plan)} or plan["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("NFP score provenance mismatch")
    events = events_by_id(nfp_dir, plan, digest(nrun))
    nfp, flags = {}, []
    for key, eventrows in events.items():
        event = eventrows[-1]
        parsed, ok = parse_nfp_json(event.get("raw"))
        if event["status"] == "finished" and event.get("model") == MODEL and ok:
            nfp[key] = parsed["Sharpness"]
            if evidence_issue(parsed, cases[key]["answer"]):
                flags.append(key)
    answers, values, originals = {}, {}, {}
    source_hashes = {
        str(p.resolve()): file_digest(p)
        for p in (nfp_dir / "plan.json", nfp_dir / "attempts.jsonl", out / "gate.json")
    }
    for method in METHODS:
        audit(pilot, "gpt-5.6-sol", method)
        responses, hashes = load_generation(pilot, method, manifest, questions)
        saved_identity = load_json(pilot / "dev" / f"{method}.jsonl.run.json")["identity"]
        if backbone_fingerprint(saved_identity) != backbone_fingerprint(features["identity"]):
            raise ValueError("Baseline and feature backbone identities differ")
        source_hashes.update(hashes)
        sp = pilot / "dev" / f"{method}_judge_codex_gpt-5.6-sol.jsonl"
        source_hashes[str(sp.resolve())] = file_digest(sp)
        sol = {r["question_id"]: r for r in read_jsonl(sp)}
        answers[method] = responses
        values[method] = []
        for qid in ids:
            if questions[qid]["set"] == "fpq":
                value = sol[qid]["sharpness"]
            else:
                key = plan["mapping"][method][qid]
                case = cases[key]
                if (
                    case["question"] != questions[qid]["question"]
                    or case["reference"] != questions[qid]["hallucination_text"]
                    or case["answer"] != responses[qid]["response"]
                    or key not in nfp
                ):
                    raise ValueError("NFP answer/reference changed or score missing")
                value = nfp[key]
            values[method].append(value)
        values[method] = np.array(values[method])
        originals[method] = values[method].copy()
    # Identical answer must carry the same judgment within this routing comparison.
    # Plain then FP Identification then CoT is a fixed donor order, never best-score selection.
    canonicalized = []
    for i, qid in enumerate(ids):
        seen = {}
        for method in METHODS:
            text = answers[method][qid]["response"]
            if text in seen:
                donor = seen[text]
                if values[method][i] != values[donor][i]:
                    canonicalized.append(
                        {
                            "question_id": qid,
                            "method": method,
                            "donor": donor,
                            "old": int(values[method][i]),
                            "shared": int(values[donor][i]),
                        }
                    )
                values[method][i] = values[donor][i]
            else:
                seen[text] = method
    y = kinds == "fpq"
    lines = [
        "# Gate pilot — dev (exploratory)",
        "",
        "Actions replay existing final answers. No generation, no judge calls, no residual steering.",
        "Gate ON uses the saved FP Identification pipeline (including its own Yes/No stage); OFF uses Plain.",
        "FPQ: original Sol rubric. NFP: Terra v2 reference-targeted objection rubric. Mixed protocols; not a final paper table.",
        "Labels include previously flagged ambiguous items. Gate calibration uses fit only; dev remains a repeatedly inspected pilot.",
        f"L{LAYER} fixed; train={len(split['train_ids'])}; calibration={len(split['calibration_ids'])}; dev={len(ids)}.",
        "Small calibration/test negative counts: 5% empirical calibration target is not a population FPR guarantee.",
        f"Identical-answer score discrepancies shared by fixed donor order: {len(canonicalized)}. Original files unchanged.",
        f"NFP evidence-format flags retained: {len(flags)}; semantic and annotation issues still require review.",
        "",
        "| Signal | Dev AUROC | Dev TPR | Dev FPR | Selected | Calibration FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected = {}
    for name, g in gate["gates"].items():
        scores = np.array([g["scores"][i] for i in ids])
        z = scores > g["threshold"]
        if any(bool(z[j]) != g["selected"][i] for j, i in enumerate(ids)):
            raise ValueError("Saved gate decisions differ from threshold")
        selected[name] = z
        lines.append(
            f"| {name} | {roc_auc_score(y, scores):.3f} | {sum(z & y)}/{sum(y)} | {sum(z & ~y)}/{sum(~y)} | {sum(z)}/{len(z)} | {g['calibration_false_positives']}/{g['calibration_nfp']} |"
        )
    routed = {
        "plain": values["plain"],
        "always FP Identification": values["fp_identification"],
        "premise CoT": values["premise_cot"],
    }
    for name, z in selected.items():
        routed[f"{name} gate x FP Identification"] = np.where(
            z, values["fp_identification"], values["plain"]
        )
    lines += [
        "",
        "| Method | PCR % | PCS | NFP pass % | FPQ rescue/harm | NFP rescue/harm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    summaries = {}
    for name, v in routed.items():
        m = metrics(v, values["plain"], kinds)
        summaries[name] = m
        lines.append(
            f"| {name} | {m['fpq']['pass']:.1f} | {m['pcs']:.3f} | {m['nfp']['pass']:.1f} | {m['fpq']['rescue']}/{m['fpq']['harm']} | {m['nfp']['rescue']}/{m['nfp']['harm']} |"
        )
    rng = np.random.default_rng(SEED)
    draws = []
    k = int(sum(selected["hidden"]))
    for _ in range(500):
        z = np.zeros(len(ids), dtype=bool)
        z[rng.choice(len(ids), size=k, replace=False)] = True
        m = metrics(
            np.where(z, values["fp_identification"], values["plain"]), values["plain"], kinds
        )
        draws.append([m["fpq"]["pass"], m["nfp"]["pass"]])
    arr = np.array(draws)
    lines += [
        "",
        f"Matched random gate: exactly {k} of {len(ids)} questions per draw, 500 draws; no model calls.",
        f"PCR mean {arr[:, 0].mean():.1f} (5–95% draw range {np.quantile(arr[:, 0], 0.05):.1f}–{np.quantile(arr[:, 0], 0.95):.1f}); "
        f"NFP mean {arr[:, 1].mean():.1f} (range {np.quantile(arr[:, 1], 0.05):.1f}–{np.quantile(arr[:, 1], 0.95):.1f}).",
        "Random draw ranges are not confidence intervals; they isolate question selection from intervention count.",
        "",
        "## Identical-answer sharing audit",
        "",
        str(canonicalized),
        "",
        "## Question-level routing",
        "",
        "| ID | Kind | Hidden on | Text on | Plain | FP ID | Hidden result |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for j, qid in enumerate(ids):
        lines.append(
            f"| {qid} | {kinds[j]} | {int(selected['hidden'][j])} | {int(selected['text'][j])} | {values['plain'][j]} | {values['fp_identification'][j]} | {routed['hidden gate x FP Identification'][j]} |"
        )
    result = {
        "gate_hash": digest(gate),
        "sources": source_hashes,
        "summary": summaries,
        "identical_answer_sharing": canonicalized,
        "nfp_evidence_flags": flags,
        "original_baselines": {
            m: metrics(v, originals["plain"], kinds) for m, v in originals.items()
        },
        "random_gate_selected_count": k,
        "random_draw_metrics": draws,
    }
    with output_lock(out / "report"):
        frozen_json(out / "report.json", result)
        (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("## Question-level routing")]))
    print(f"Full report: {out / 'report.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    a = sub.add_parser("extract")
    a.add_argument("--pilot-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--config", default="configs/gemma2_9b.yaml")
    a = sub.add_parser("fit")
    a.add_argument("--out-dir", required=True)
    a.add_argument("--manifest", required=True)
    a = sub.add_parser("report")
    a.add_argument("--pilot-dir", required=True)
    a.add_argument("--out-dir", required=True)
    a.add_argument("--nfp-dir", required=True)
    a = p.parse_args()
    if a.stage == "extract":
        extract(a.pilot_dir, a.out_dir, a.config)
    elif a.stage == "fit":
        fit(a.out_dir, a.manifest)
    else:
        report(a.pilot_dir, a.out_dir, a.nfp_dir)


if __name__ == "__main__":
    main()
