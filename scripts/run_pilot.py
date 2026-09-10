"""Single-Gemma pilot: prepare -> fit -> generate dev -> select -> generate test -> report.

Run each subcommand with --help. Judging uses scripts/run_judge.py between
generation and selection/report. All expensive operations are explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl, write_jsonl
from src.pilot import (
    METHODS,
    VERSION,
    check_resume,
    digest,
    file_digest,
    frozen_json,
    load_manifest,
    make_manifest,
    normalized,
    output_lock,
    prepare_rows,
    quarantine_conflicts,
    score_map,
    summarize_pair,
)


def prepare(args):
    cfg = load_config(args.config)
    raw_path = Path(args.references or Path(cfg["data"]["cancer_myth_repo"]) / "data/all_data.json")
    references = load_json(raw_path)
    if args.questions:
        questions = list(read_jsonl(args.questions))
    else:
        from src.rows import load_nfp

        questions = [
            {
                "id": f"fpq_{r['QID']}",
                "set": "fpq",
                "question": r["example_question"].strip(),
                "correction": r["example_assumption"],
                "category": r.get("category"),
            }
            for r in references
        ] + load_nfp(cfg)
    questions, conflicts = quarantine_conflicts(questions)
    example_texts = {
        normalized(e["example_question"])
        for name in ("examples_fpq", "examples_nfp")
        for e in load_json(cfg["judge"][name])
    }
    excluded_ids = [q["id"] for q in questions if normalized(q["question"]) in example_texts]
    questions = [q for q in questions if normalized(q["question"]) not in example_texts]
    rows = prepare_rows(
        questions,
        references,
        seed=args.seed,
        folds=args.folds,
        test_fold=args.test_fold,
        dev_fold=args.dev_fold,
    )
    manifest = make_manifest(
        rows,
        seed=args.seed,
        folds=args.folds,
        test_fold=args.test_fold,
        dev_fold=args.dev_fold,
        reference_hash=file_digest(raw_path),
        excluded_ids=excluded_ids,
        conflicts=conflicts,
    )
    out = Path(args.out_dir)
    frozen_json(out / "manifest.json", manifest)
    for part in ("fit", "dev", "test"):
        subset = [q for q in rows if q["partition"] == part]
        path = out / f"{part}.jsonl"
        if path.exists():
            if list(read_jsonl(path)) != subset:
                raise ValueError(f"Modified partition file: {path}")
        else:
            write_jsonl(path, subset)
    print(json.dumps(manifest["counts"], indent=2))
    if conflicts:
        print(f"[quarantine] {json.dumps(conflicts, ensure_ascii=False)}")
    print(f"[prepared] {out / 'manifest.json'} ({manifest['manifest_hash'][:12]})")


def fit(args):
    from src.pilot_model import learn_direction, load_model

    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)
    raw_path = Path(args.references or Path(cfg["data"]["cancer_myth_repo"]) / "data/all_data.json")
    if file_digest(raw_path) != manifest["reference_hash"]:
        raise ValueError("Reference file changed since preparing the split")
    with output_lock(Path(args.out_dir) / "fit"):
        model, tokenizer = load_model(cfg)
        learn_direction(
            model,
            tokenizer,
            cfg,
            manifest,
            load_json(raw_path),
            layers=args.layers,
            prefix_tokens=args.prefix_tokens,
            out_dir=args.out_dir,
        )


def generate(args):
    import numpy as np
    import torch
    from src.pilot_model import generate_batch, load_model, model_identity
    from src.steering import SteerSpec

    cfg = load_config(args.config)
    manifest = load_manifest(args.manifest)
    if args.method == "steering" and not args.fit_dir:
        raise ValueError("Steering needs --fit-dir (pilot artifacts, not E1 directions)")
    if args.partition == "test":
        if not args.selection:
            raise ValueError("Test is locked: select a dev setting first, then pass --selection")
        selection = load_json(args.selection)
        if selection["manifest_hash"] != manifest["manifest_hash"]:
            raise ValueError("Selection belongs to a different split")
        if selection["selected"] is None:
            raise ValueError("No dev candidate met the diagnostic criterion; test remains locked")
        if digest(cfg["source_model"]) != digest(selection["identity"]["source_model"]):
            raise ValueError("Test model differs from dev")
        if args.method == "steering":
            picked = selection["selected"]
            args.layer, args.alpha = picked["layer"], picked["alpha"]
            if file_digest(Path(args.fit_dir) / "directions.npz") != picked["direction_hash"]:
                raise ValueError("Test direction differs from dev selection")
        if (
            args.max_new_tokens != selection["max_new_tokens"]
            or args.review_tokens != selection["review_tokens"]
        ):
            raise ValueError("Test generation budgets differ from dev")
    if args.max_new_tokens < 1 or args.review_tokens < 1 or args.batch_size < 1:
        raise ValueError("Token budgets and batch size must be positive")
    rows = [q for q in manifest["questions"] if q["partition"] == args.partition]
    run = {
        "version": VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "partition": args.partition,
        "method": args.method,
        "max_new_tokens": args.max_new_tokens,
        "review_tokens": args.review_tokens,
        "batch_size": args.batch_size,
        "seed": int(cfg.get("seed", 17)),
        "decoding": "greedy_cache",
        "question_ids": [q["id"] for q in rows],
    }
    if args.partition == "test":
        run["selection_hash"] = file_digest(args.selection)
    spec = None
    with output_lock(args.output):
        model, tokenizer = load_model(cfg)
        run["identity"] = model_identity(model, tokenizer, cfg)
        if args.partition == "test" and run["identity"] != selection["identity"]:
            raise ValueError("Resolved model/tokenizer revision changed since dev")
        if args.method == "steering":
            if (
                args.layer is None
                or args.alpha is None
                or not np.isfinite(args.alpha)
                or args.alpha < 0
            ):
                raise ValueError("Dev steering requires --layer and nonnegative finite --alpha")
            fit_dir = Path(args.fit_dir)
            fit_spec = load_json(fit_dir / "fit.json")
            if (
                fit_spec["manifest_hash"] != manifest["manifest_hash"]
                or fit_spec["identity"] != run["identity"]
            ):
                raise ValueError("Direction model/split differs from current run")
            key = f"L{args.layer}_c_pair{fit_spec['prefix_tokens']}"
            with np.load(fit_dir / "directions.npz") as arrays:
                direction = torch.tensor(arrays[key], dtype=torch.float32)
                scale = float(arrays[f"L{args.layer}_norm_scale"])
            if not torch.isfinite(direction).all() or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid fitted direction or scale")
            spec = SteerSpec(args.layer, direction, args.alpha * scale)
            run.update(
                layer=args.layer,
                alpha=args.alpha,
                alpha_abs=spec.alpha,
                norm_scale=scale,
                direction_key=key,
                direction_hash=file_digest(fit_dir / "directions.npz"),
            )
        done = check_resume(args.output, run, set(run["question_ids"]))
        # Keep original batches on resume: skip complete batches and recompute
        # incomplete batches without appending already completed items.
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            if all(q["id"] in done for q in batch):
                continue
            texts, records = generate_batch(
                model,
                tokenizer,
                batch,
                method=args.method,
                max_new_tokens=args.max_new_tokens,
                review_tokens=args.review_tokens,
                spec=spec,
            )
            for q, text, record in zip(batch, texts, records):
                if q["id"] in done:
                    continue
                if not text:
                    raise ValueError(f"Empty final answer for {q['id']}")
                append_jsonl(
                    args.output,
                    {
                        "id": q["id"],
                        "set": q["set"],
                        "question": q["question"],
                        "response": text,
                        "model_id": cfg["source_model"]["model_id"],
                        "method": args.method,
                        "run_hash": digest(run),
                        **record,
                    },
                )
            print(
                f"[{args.method}/{args.partition}] {min(start + args.batch_size, len(rows))}/{len(rows)}",
                flush=True,
            )


def load_scored(path, manifest, partition):
    meta = load_json(str(path) + ".run.json")
    run = meta.get("response_run")
    if (
        not run
        or run["manifest_hash"] != manifest["manifest_hash"]
        or run["partition"] != partition
    ):
        raise ValueError(f"Scores are not a {partition} pilot run for this manifest: {path}")
    expected = {q["id"]: q for q in manifest["questions"] if q["partition"] == partition}
    scores, judge = score_map(path, expected)
    if any(r.get("response_run_hash") != digest(run) for r in scores.values()):
        raise ValueError(f"Score provenance differs from generation run: {path}")
    return scores, judge, run


def select(args):
    manifest = load_manifest(args.manifest)
    questions = [q for q in manifest["questions"] if q["partition"] == "dev"]
    plain, judge, base_run = load_scored(args.plain_scores, manifest, "dev")
    if base_run["method"] != "plain" or args.nfp_margin_pp < 0:
        raise ValueError("Need Plain dev scores and a nonnegative margin in percentage points")
    candidates = []
    for path in args.candidate_scores:
        scores, other_judge, run = load_scored(path, manifest, "dev")
        if (
            other_judge != judge
            or run["identity"] != base_run["identity"]
            or run["method"] != "steering"
            or run["max_new_tokens"] != base_run["max_new_tokens"]
        ):
            raise ValueError(
                "Candidate and Plain differ in judge/model/budget or candidate is not steering"
            )
        if run["alpha"] == 0 and any(
            scores[qid].get("response_text_hash") != plain[qid].get("response_text_hash")
            for qid in plain
        ):
            raise ValueError(
                "alpha=0 answers differ from Plain; inspect batch/decoding reproducibility"
            )
        summary = summarize_pair(questions, scores, plain)
        eligible = (
            run["alpha"] > 0
            and summary["fpq"]["delta_PCR"] > 0
            and summary["nfp"]["delta_NFP"] >= -args.nfp_margin_pp
        )
        candidates.append(
            {"scores_hash": file_digest(path), "run": run, "summary": summary, "eligible": eligible}
        )
    eligible = [c for c in candidates if c["eligible"]]
    # Diagnostic point-estimate rule, not a noninferiority or significance claim.
    winner = max(
        eligible,
        key=lambda c: (
            c["summary"]["fpq"]["PCR"],
            c["summary"]["fpq"]["PCS"],
            c["summary"]["nfp"]["NFP"],
            -c["run"]["alpha"],
            -c["run"]["layer"],
        ),
        default=None,
    )
    result = {
        "version": VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "identity": base_run["identity"],
        "judge": list(judge),
        "max_new_tokens": base_run["max_new_tokens"],
        "review_tokens": base_run["review_tokens"],
        "nfp_margin_pp": args.nfp_margin_pp,
        "plain_scores_hash": file_digest(args.plain_scores),
        "selection_rule": "dev point estimates: delta_PCR>0 and delta_NFP>=-margin; maximize PCR, PCS, NFP, smaller alpha/layer",
        "selected": winner["run"] if winner else None,
        "candidates": candidates,
    }
    frozen_json(args.output, result)
    print(json.dumps({"selected": result["selected"], "n_candidates": len(candidates)}, indent=2))


def report(args):
    manifest = load_manifest(args.manifest)
    questions = [q for q in manifest["questions"] if q["partition"] == args.partition]
    loaded = [(path, *load_scored(path, manifest, args.partition)) for path in args.scores]
    bases = [entry for entry in loaded if entry[3]["method"] == "plain"]
    if len(bases) != 1:
        raise ValueError("Report needs exactly one Plain file")
    _, plain, judge, base_run = bases[0]
    results = []
    for path, scores, other_judge, run in loaded:
        if (
            other_judge != judge
            or run["identity"] != base_run["identity"]
            or run["max_new_tokens"] != base_run["max_new_tokens"]
        ):
            raise ValueError("Cannot mix judges, models or final-answer budgets")
        label = run["method"]
        if label == "steering":
            label += f"_L{run['layer']}_a{run['alpha']:g}"
        results.append(
            {
                "method": label,
                "run": run,
                "scores_hash": file_digest(path),
                **summarize_pair(questions, scores, plain),
            }
        )
    payload = {
        "partition": args.partition,
        "manifest_hash": manifest["manifest_hash"],
        "judge": list(judge),
        "results": results,
    }
    frozen_json(args.output, payload)
    lines = [
        f"# Gemma pilot — {args.partition}",
        "",
        "Percentages are 0–100; PCS is −1 to +1. CIs resample source groups.",
        "This is a pilot, not an exact Well split/protocol reproduction or a preservation guarantee.",
        "",
        "| Method | FPQ n | PCR [95% CI] | PCS | NFP n | NFP [95% CI] | FPQ rescue/harm | NFP rescue/harm |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]

    def ci(d, k):
        return f"{d[k]:.1f} [{d[k + '_ci95'][0]:.1f}, {d[k + '_ci95'][1]:.1f}]"

    for r in results:
        f, n = r["fpq"], r["nfp"]
        lines.append(
            f"| {r['method']} | {f['n']} | {ci(f, 'PCR')} | {f['PCS']:.3f} | {n['n']} | {ci(n, 'NFP')} | {f['rescue']}/{f['harm']} | {n['rescue']}/{n['harm']} |"
        )
    md = Path(args.output).with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--config", default="configs/gemma2_9b.yaml")
    p.add_argument(
        "--questions", help="Existing E0 questions; otherwise use public Cancer-Myth JSONs"
    )
    p.add_argument("--references")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-fold", type=int, default=0)
    p.add_argument("--dev-fold", type=int, default=1)
    p.set_defaults(func=prepare)
    p = subs.add_parser("fit")
    p.add_argument("--config", default="configs/gemma2_9b.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--references")
    p.add_argument("--layers", type=int, nargs="+", default=[14, 21, 28])
    p.add_argument("--prefix-tokens", type=int, default=32)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=fit)
    p = subs.add_parser("generate")
    p.add_argument("--config", default="configs/gemma2_9b.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--partition", choices=["dev", "test"], default="dev")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--fit-dir")
    p.add_argument("--layer", type=int)
    p.add_argument("--alpha", type=float, help="Fraction of the fit-only residual norm")
    p.add_argument("--selection", help="Required for test; overrides steering layer/alpha")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--review-tokens", type=int, default=128)
    p.add_argument("--output", required=True)
    p.set_defaults(func=generate)
    p = subs.add_parser("select")
    p.add_argument("--manifest", required=True)
    p.add_argument("--plain-scores", required=True)
    p.add_argument("--candidate-scores", nargs="+", required=True)
    p.add_argument(
        "--nfp-margin-pp",
        type=float,
        required=True,
        help="Pilot dev point-estimate tolerance; not a guarantee",
    )
    p.add_argument("--output", required=True)
    p.set_defaults(func=select)
    p = subs.add_parser("report")
    p.add_argument("--manifest", required=True)
    p.add_argument("--partition", choices=["dev", "test"], required=True)
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
