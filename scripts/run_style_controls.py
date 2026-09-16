"""Run the three style-confound tests on a prepared suite (CPU; judge calls 0).

    python scripts/run_style_controls.py --suite-dir $SUITE_DIR \\
        --twins $ROWS/questions_twins.jsonl \\
        --paraphrases $SUITE_DIR/variants/para/questions_para.jsonl \\
        --name v1

Natural features come from <suite>/model (the `extract` stage); twin and
paraphrase features from <suite>/model_variants/twins and .../para (the
`extract` stage with --variant-file/--variant-name). Without them, the
hidden/mean rows are skipped for those conditions and the report says so;
text/style/masked always run. Output: <suite>/style_controls/<name>/
{result.json, summary.json, report.md}. See src/style_controls.py, docs/29.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_suite import load_suite
from src.jsonl import read_jsonl
from src.pilot import digest, file_digest, frozen_json, output_lock
from src.style_controls import CONDITIONS, assemble, fold_assignment, report, run_conditions, split_twin_file, summarize


def remap_twins(rows, by_text, source, suffix):
    """Point pair_id at the suite question with the same text. Only pair_id is
    rewritten: the row's own id must stay as in the twins file, because the
    `extract --variant-file` features are keyed by it (v1 lost every twin
    hidden row by renaming ids here)."""
    out = []
    for t in rows:
        suite_id = by_text.get(source.get(t.get("pair_id"), ""))
        if suite_id:
            out.append({**t, "pair_id": suite_id})
    print(f"[twins] {len(out)}/{len(rows)} {suffix} rows mapped to suite questions by text", flush=True)
    return out


def load_features(path):
    from src import baseline_generation as bg
    path = Path(path)
    return bg.load_feature_records(path) if (path / "features" / "spec.json").exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--twins", help="questions_twins.jsonl (set=tpair, pair_id=<fpq id>)")
    parser.add_argument("--twins-questions", help="The questions.jsonl the twins were built from; maps pair_id to suite IDs by exact normalized text when ID schemes differ")
    parser.add_argument("--paraphrases", help="questions_para.jsonl (paraphrase_of=<id>)")
    parser.add_argument("--features-natural", help="default <suite>/model")
    parser.add_argument("--features-twins", help="default <suite>/model_variants/twins")
    parser.add_argument("--features-para", help="default <suite>/model_variants/para")
    parser.add_argument("--signals", nargs="+", default=["text", "style", "masked", "hidden", "mean"],
                        choices=("text", "style", "masked", "hidden", "mean"))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[.001, .01, .1, 1.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--name", default="v1")
    parser.add_argument("--fpq-writer", default="gpt-4o", help="from_model value of the FPQ generator; NFP rows with the same value form the same_writer control")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite = load_suite(suite_dir)
    natural = suite["questions"]
    twin_rows = list(read_jsonl(args.twins)) if args.twins else []
    twins, fparas = split_twin_file(twin_rows)
    print(f"[twins] file: {len(twins)} true twins, {len(fparas)} false paraphrases", flush=True)
    # Twins may come from the E1 row set, which is larger than the suite; keep those whose FPQ is in the suite.
    ids = {q["id"] for q in natural}
    if args.twins_questions:
        from src.pilot import normalized
        by_text = {normalized(q["question"]): q["id"] for q in natural}
        source = {q["id"]: normalized(q["question"]) for q in read_jsonl(args.twins_questions)}
        twins, fparas = remap_twins(twins, by_text, source, "true"), remap_twins(fparas, by_text, source, "fpara")
    dropped = [t for t in twins + fparas if t.get("pair_id") not in ids]
    twins = [t for t in twins if t.get("pair_id") in ids]
    fparas = [t for t in fparas if t.get("pair_id") in ids]
    if dropped:
        print(f"[twins] {len(dropped)} twin-file rows whose FPQ is not in the suite were dropped", flush=True)
    para = list(read_jsonl(args.paraphrases)) if args.paraphrases else []
    nat, tw, pa, fp = assemble(natural, twins, para, fparas)

    features = {}
    hidden_wanted = any(s in args.signals for s in ("hidden", "mean"))
    if hidden_wanted:
        for label, default, override in (("natural", suite_dir / "model", args.features_natural),
                                         ("twins", suite_dir / "model_variants" / "twins", args.features_twins),
                                         ("para", suite_dir / "model_variants" / "para", args.features_para)):
            loaded = load_features(override or default)
            overlap = set(loaded) & set(features)
            if overlap:
                raise ValueError(f"Feature ID overlap between sources: {sorted(overlap)[:3]}")
            features.update(loaded)
            print(f"[features] {label}: {len(loaded)} rows", flush=True)
    layers = args.layers
    if hidden_wanted and features and not layers:
        layers = sorted(set.intersection(*(set(x) for x in features.values())))
    if hidden_wanted and not layers:
        layers = ()

    assignment = fold_assignment(natural, folds=args.folds, seed=args.seed)
    same_writer_nfp = sum(1 for q in natural if q["set"] == "nfp" and q.get("from_model") == args.fpq_writer)
    print(f"[same_writer] NFP written by {args.fpq_writer}: {same_writer_nfp}", flush=True)
    result = run_conditions(nat, tw, pa, fp, assignment=assignment, signals=args.signals, features=features,
                            layers=layers, c_grid=args.c_grid, conditions=args.conditions, seed=args.seed,
                            fpq_writer=args.fpq_writer)
    summaries = summarize(result, repeats=args.bootstrap, seed=args.seed)
    text = report(result, summaries)

    destination = suite_dir / "style_controls" / args.name
    spec = {"suite_hash": digest(suite), "twins_sha256": file_digest(args.twins) if args.twins else None,
            "paraphrases_sha256": file_digest(args.paraphrases) if args.paraphrases else None,
            "twins_n": len(tw), "false_paraphrases_n": len(fp), "para_n": len(pa),
            "signals": args.signals, "conditions": args.conditions, "fpq_writer": args.fpq_writer,
            "layers": list(layers or ()), "c_grid": args.c_grid, "folds": args.folds, "seed": args.seed,
            "fold_assignment": assignment,
            "implementation_hash": file_digest(ROOT / "src/style_controls.py")}
    with output_lock(destination):
        frozen_json(destination / "plan.json", spec)
        frozen_json(destination / "result.json", result)
        frozen_json(destination / f"summary_bootstrap{args.bootstrap}.json", summaries)
        (destination / "report.md").write_text(text)
    print(text)
    print(f"Written to {destination}. Judge calls: 0.")


if __name__ == "__main__":
    main()
