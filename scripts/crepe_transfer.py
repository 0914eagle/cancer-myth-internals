"""CREPE <-> Cancer-Myth probe transfer (docs/32 §7, 33 §4).

Does a false-presupposition probe learned where no template/provenance
confound exists (CREPE: both classes are Reddit questions) read premise truth
in Cancer-Myth? The clean readout is the twin pair (only truth differs).

    python scripts/crepe_transfer.py inspect --crepe-dir $DATA/external/CREPE
    CUDA_VISIBLE_DEVICES=1 python scripts/crepe_transfer.py extract --crepe-dir ... \\
        --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml
    python scripts/crepe_transfer.py eval --suite-dir $SUITE_DIR \\
        --twins $DATA/e1_rows_v1/questions_twins.jsonl --twins-questions $DATA/e1_rows_v1/questions.jsonl \\
        --paraphrases $SUITE_DIR/variants/para/questions_para.jsonl

Label detection: --label-key/--positive-regex if given; else a `labels`
field (positive when any label matches /false/i, negative when it matches
/normal|true|no[_ ]?false|none/i); else a non-empty `presuppositions`-like
field. Rows without a clear label are skipped and counted. Judge calls: 0.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl
from src.pilot import digest, file_digest, frozen_json, normalized, output_lock

QUESTION_KEYS = ("question", "input", "query", "text")
PRESUP_KEYS = ("false_presuppositions", "presuppositions", "presupposition", "false_presupposition")
POS_DEFAULT = r"false"
NEG_DEFAULT = r"normal|true|no[_ ]?false|none|valid"


def _iter_rows(crepe_dir):
    for path in sorted(Path(crepe_dir).rglob("*")):
        if path.suffix == ".jsonl":
            for i, row in enumerate(read_jsonl(path)):
                yield path, i, row
        elif path.suffix == ".json" and path.stat().st_size < 200_000_000:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(data, list):
                for i, row in enumerate(data):
                    if isinstance(row, dict):
                        yield path, i, row


def _question(row):
    for k in QUESTION_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _label(row, label_key=None, pos_re=POS_DEFAULT, neg_re=NEG_DEFAULT):
    """1 = has a false presupposition, 0 = none, None = unclear."""
    if label_key:
        v = row.get(label_key)
        vals = v if isinstance(v, list) else [v]
        s = " ".join(str(x) for x in vals if x is not None)
        if re.search(pos_re, s, re.IGNORECASE):
            return 1
        if re.search(neg_re, s, re.IGNORECASE) or (isinstance(v, list) and not v):
            return 0
        return None
    if "labels" in row or "label" in row:
        return _label(row, "labels" if "labels" in row else "label", pos_re, neg_re)
    for k in PRESUP_KEYS:
        if k in row:
            v = row[k]
            if isinstance(v, list):
                return 1 if any(str(x).strip() for x in v) else 0
            if isinstance(v, str):
                return 1 if v.strip() else 0
    return None


def load_crepe(crepe_dir, *, label_key=None, pos_re=POS_DEFAULT, neg_re=NEG_DEFAULT, split_regex=None):
    rows, seen, skipped, files = [], set(), Counter(), Counter()
    for path, i, row in _iter_rows(crepe_dir):
        if split_regex and not re.search(split_regex, str(path)):
            continue
        q = _question(row)
        if not q:
            skipped["no_question"] += 1
            continue
        y = _label(row, label_key, pos_re, neg_re)
        if y is None:
            skipped["unclear_label"] += 1
            continue
        key = normalized(q)
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        files[path.name] += 1
        rows.append({"id": f"crepe_{len(rows):05d}", "question": q, "set": "fpq" if y else "nfp",
                     "label": y, "source_file": path.name, "source_line": i})
    return rows, skipped, files


def _load_features(path):
    from src import baseline_generation as bg
    path = Path(path)
    return bg.load_feature_records(path) if (path / "features" / "spec.json").exists() else {}


def _remap_twins(rows, by_text, source):
    out = []
    for t in rows:
        suite_id = by_text.get(source.get(t.get("pair_id"), ""))
        if suite_id:
            out.append({**t, "pair_id": suite_id})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("inspect", "extract"):
        p = sub.add_parser(name)
        p.add_argument("--crepe-dir", required=True)
        p.add_argument("--label-key")
        p.add_argument("--positive-regex", default=POS_DEFAULT)
        p.add_argument("--negative-regex", default=NEG_DEFAULT)
        p.add_argument("--split-regex", help="only files whose path matches (e.g. 'dev|test')")
        if name == "extract":
            p.add_argument("--suite-dir", required=True)
            p.add_argument("--config", required=True)
            p.add_argument("--layers", nargs="+", type=int)
            p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("fetch-hf", help="Download a Hugging Face mirror into <crepe-dir>/data/<split>.jsonl")
    p.add_argument("--crepe-dir", required=True)
    p.add_argument("--dataset", default="tasksource/CREPE")
    p.add_argument("--config-name", default=None)
    p = sub.add_parser("eval")
    p.add_argument("--suite-dir", required=True)
    p.add_argument("--twins")
    p.add_argument("--twins-questions")
    p.add_argument("--paraphrases")
    p.add_argument("--signals", nargs="+", default=["text", "style", "masked", "hidden", "mean"])
    p.add_argument("--c-grid", nargs="+", type=float, default=[.001, .01, .1, 1.0])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--name", default="v1")
    p.add_argument("--crepe-max", type=int, default=0, help="stratified subsample of CREPE rows for the probes (0 = all; 8,466 x inner CV on 3584 dims takes hours)")
    args = parser.parse_args()

    if args.stage == "fetch-hf":
        from datasets import load_dataset
        data_dir = Path(args.crepe_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(args.dataset, args.config_name) if args.config_name else load_dataset(args.dataset)
        for split, table in ds.items():
            path = data_dir / f"hf_{args.dataset.replace('/', '__')}_{split}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in table:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[fetch-hf] {split}: {len(table)} rows -> {path}; columns {table.column_names}", flush=True)
        return
    if args.stage in ("inspect", "extract"):
        rows, skipped, files = load_crepe(args.crepe_dir, label_key=args.label_key, pos_re=args.positive_regex,
                                          neg_re=args.negative_regex, split_regex=args.split_regex)
        counts = Counter(r["set"] for r in rows)
        print(f"[crepe] rows {len(rows)}: fpq {counts.get('fpq', 0)}, nfp {counts.get('nfp', 0)}; skipped {dict(skipped)}; files {dict(files)}", flush=True)
        if args.stage == "inspect":
            shown = 0
            for path, i, row in _iter_rows(args.crepe_dir):
                print(f"--- {path.name}:{i} keys={sorted(row)}")
                print(json.dumps(row, ensure_ascii=False)[:600])
                shown += 1
                if shown >= 3:
                    break
            for r in rows[:3]:
                print(r)
            return
        if not rows or len(counts) < 2:
            raise SystemExit("CREPE labels not detected on both classes; use inspect and pass --label-key/--positive-regex")
        if args.limit:
            rows = rows[:args.limit]
        from src import baseline_generation as bg
        from src.config import load_config
        cfg = load_config(args.config)
        layers = args.layers or sorted({round(cfg["source_model"]["n_layers"] * x) for x in (.4, .6, .8)})
        out = Path(args.suite_dir).resolve() / "crepe"
        out.mkdir(parents=True, exist_ok=True)
        with (out / "questions.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        frozen_json(out / "source.json", {"crepe_dir": str(Path(args.crepe_dir).resolve()), "rows": len(rows),
                                          "counts": dict(counts), "skipped": dict(skipped), "files": dict(files),
                                          "label_key": args.label_key, "positive_regex": args.positive_regex,
                                          "negative_regex": args.negative_regex, "layers": layers})
        bg.extract_features(rows, cfg, out / "model", layers)
        print(f"CREPE prefill states stored under {out / 'model'}. Judge calls: 0.")
        return

    # ---- eval (CPU) ----
    from src.baseline_suite import load_suite
    from src.style_controls import (TEXT_SIGNALS, _fit_predict, _select, _as_fit_rows, assemble, fold_assignment,
                                    report, split_twin_file, summarize)
    suite_dir = Path(args.suite_dir).resolve()
    suite = load_suite(suite_dir)
    natural = suite["questions"]
    crepe_rows = list(read_jsonl(suite_dir / "crepe" / "questions.jsonl"))
    twins, fparas = split_twin_file(list(read_jsonl(args.twins))) if args.twins else ([], [])
    if args.twins_questions:
        by_text = {normalized(q["question"]): q["id"] for q in natural}
        source = {q["id"]: normalized(q["question"]) for q in read_jsonl(args.twins_questions)}
        twins, fparas = _remap_twins(twins, by_text, source), _remap_twins(fparas, by_text, source)
    ids = {q["id"] for q in natural}
    twins = [t for t in twins if t.get("pair_id") in ids]
    fparas = [t for t in fparas if t.get("pair_id") in ids]
    para = list(read_jsonl(args.paraphrases)) if args.paraphrases else []
    nat, tw, pa, fp = assemble(natural, twins, para, fparas)
    cr = [{"id": r["id"], "question": r["question"], "set": r["set"], "label": r["label"], "origin": r["id"],
           "source": "crepe", "group_id": r["id"], "writer": None} for r in crepe_rows]
    if args.crepe_max and len(cr) > args.crepe_max:
        rng = np.random.default_rng(args.seed)
        pos = [r for r in cr if r["label"] == 1]; neg = [r for r in cr if r["label"] == 0]
        share = len(pos) / len(cr)
        k_pos = int(round(args.crepe_max * share)); k_neg = args.crepe_max - k_pos
        cr = [pos[i] for i in sorted(rng.choice(len(pos), min(k_pos, len(pos)), replace=False))] + \
             [neg[i] for i in sorted(rng.choice(len(neg), min(k_neg, len(neg)), replace=False))]
        print(f"[crepe] subsampled to {len(cr)} (pos {k_pos}, neg {k_neg}) with seed {args.seed}", flush=True)
    features = {}
    for label, path in (("natural", suite_dir / "model"), ("twins", suite_dir / "model_variants" / "twins"),
                        ("para", suite_dir / "model_variants" / "para"), ("crepe", suite_dir / "crepe" / "model")):
        loaded = _load_features(path)
        if set(loaded) & set(features):
            raise ValueError("Feature ID overlap")
        features.update(loaded)
        print(f"[features] {label}: {len(loaded)}", flush=True)
    layers = sorted(set.intersection(*(set(x) for x in features.values()))) if features else []

    twin_origins = {t["origin"] for t in tw}
    pairs = [r for r in nat if r["set"] == "fpq" and r["id"] in twin_origins] + tw
    both = twin_origins & {f["origin"] for f in fp}
    edited = [f for f in fp if f["origin"] in both] + [t for t in tw if t["origin"] in both]
    transfers = {"crepe->natural": (cr, nat), "crepe->twins": (cr, pairs), "crepe->edited": (cr, edited),
                 "crepe->para": (cr, pa), "natural->crepe": (nat, cr), "twins->crepe": (pairs, cr),
                 "edited->crepe": (edited, cr), "para->crepe": (pa, cr)}
    predictions, selections, skipped = [], [], []

    def run_pair(name, train, held, fold, kind):
        if not train or not held or {r["label"] for r in train} != {0, 1} or {r["label"] for r in held} != {0, 1}:
            skipped.append({"condition": name, "signal": kind, "reason": "no rows / missing class"}); return
        if kind not in TEXT_SIGNALS:
            missing = [r["id"] for r in train + held if r["id"] not in features]
            if missing:
                skipped.append({"condition": name, "signal": kind, "reason": f"{len(missing)} rows without features"}); return
        try:
            best, candidates = _select(kind, train, features=features, layers=layers, c_grid=args.c_grid, seed=args.seed + fold)
        except ValueError as exc:
            skipped.append({"condition": name, "signal": kind, "reason": str(exc)}); return
        scores = _fit_predict(kind, _as_fit_rows(train), _as_fit_rows(held), features=features,
                              layer=best["layer"], c=best["C"], seed=args.seed)
        selections.append({"condition": name, "signal": kind, "fold": fold, "best": best, "candidates": candidates,
                           "train_n": len(train), "eval_n": len(held)})
        for r, s in zip(held, scores):
            predictions.append({"condition": name, "signal": kind, "fold": fold, "id": r["id"], "origin": r["origin"],
                                "group_id": r["group_id"], "set": r["set"], "label": r["label"], "source": r["source"],
                                "score": float(s)})

    # Within CREPE: 5-fold (register floor and probe on data without the template confound).
    assignment = fold_assignment(cr, folds=5, seed=args.seed)
    for kind in args.signals:
        t0 = __import__("time").time()
        for k in sorted(set(assignment.values())):
            run_pair("crepe", [r for r in cr if assignment[r["id"]] != k], [r for r in cr if assignment[r["id"]] == k], k, kind)
        print(f"[crepe] within-CREPE {kind}: {__import__('time').time() - t0:.0f}s", flush=True)
    for name, (train, held) in transfers.items():
        for kind in args.signals:
            t0 = __import__("time").time()
            run_pair(name, train, held, 0, kind)
            print(f"[crepe] {name} {kind}: {__import__('time').time() - t0:.0f}s", flush=True)
    result = {"predictions": predictions, "selections": selections, "skipped": skipped, "signals": args.signals,
              "conditions": ["crepe"] + list(transfers), "layers": layers, "c_grid": args.c_grid}
    summaries = summarize(result, repeats=args.bootstrap, seed=args.seed)
    text = report(result, summaries).replace("# Style-confound controls", "# CREPE <-> Cancer-Myth transfer")
    text += ("\nRead: crepe (within) gives the register floor and probe where both classes share a source. "
             "crepe->edited/twins is the clean transfer readout (only truth differs); ~0.5 means no transferable "
             "truth direction at the last prompt token for this model on medical narratives.\n")
    destination = suite_dir / "crepe" / "transfer" / args.name
    with output_lock(destination):
        frozen_json(destination / "plan.json", {"suite_hash": digest(suite), "crepe_n": len(cr), "twins_n": len(tw),
                                                "fparas_n": len(fp), "para_n": len(pa), "layers": layers,
                                                "signals": args.signals, "c_grid": args.c_grid, "seed": args.seed,
                                                "implementation_hash": file_digest(Path(__file__))})
        frozen_json(destination / "result.json", result)
        frozen_json(destination / f"summary_bootstrap{args.bootstrap}.json", summaries)
        (destination / "report.md").write_text(text)
    print(text)
    print(f"Written to {destination}. Judge calls: 0.")


if __name__ == "__main__":
    main()
