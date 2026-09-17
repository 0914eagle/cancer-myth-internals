"""Token-scan gate: score every PROMPT token with a probe learned on premise
spans; the gate is the max (or top-k mean) over tokens. No span is needed at
test time; the argmax says where the premise is. Prefill only, no extra
forward pass, judge calls 0. (docs/34 §5)

    CUDA_VISIBLE_DEVICES=1 python scripts/token_scan.py extract --suite-dir $SUITE_DIR \\
        --config configs/qwen25_7b.yaml --e1-questions $DATA/e1_rows_v1/questions.jsonl \\
        --twins $DATA/e1_rows_v1/questions_twins.jsonl \\
        --paraphrases $SUITE_DIR/variants/para/questions_para.jsonl --layers 17 20 --name scan_v1
    python scripts/token_scan.py eval --suite-dir $SUITE_DIR --name scan_v1

Rows: natural 732 (FPQ span from the E1 rows where aligned), true twins,
false paraphrases, paraphrases, optionally CREPE (--crepe-max N).
Per row and layer: content-token states (T, d) float16, last-token state,
token char offsets, span mask.

Eval, per fold of the natural suite (rewrites inherit their origin's fold):
  S  span probe   trained on span_mean: natural FPQ + false paraphrase (1) vs true twin (0)
  Q  q_mean probe trained on natural FPQ vs NFP question means
Scorers on held rows: S_max, S_top5, S_qmean, S_last, Q_qmean, Q_last.
Eval sets: natural (FPQ vs NFP), twins, edited, para, [crepe].
Localization: for FPQ rows with a span, share of S argmax tokens inside the span.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl
from src.pilot import frozen_json, normalized, output_lock

SCORERS = ("S_max", "S_top5", "S_qmean", "S_last", "Q_qmean", "Q_last")


def shift_span(rendered, content, span):
    start = rendered.find(content)
    if start < 0:
        raise ValueError("Rendered prompt does not contain the user content verbatim")
    return start + int(span[0]), start + int(span[1]), start, start + len(content)


def token_positions(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if a != b and a < end and b > start]


def build_rows(suite_questions, e1_rows, twin_rows, para_rows, crepe_rows=()):
    by_text = {normalized(q["question"]): q["id"] for q in suite_questions}
    e1_text = {r["id"]: normalized(r.get("question", "")) for r in e1_rows}
    e1_span = {}
    for r in e1_rows:
        sid = by_text.get(normalized(r.get("question", "")))
        if sid and r.get("premise_span"):
            e1_span[sid] = r["premise_span"]
    rows = []
    for q in suite_questions:
        rows.append({"id": q["id"], "kind": "natural", "set": q["set"], "label": 1 if q["set"] == "fpq" else 0,
                     "origin": q["id"], "text": q["question"], "span": e1_span.get(q["id"])})
    for t in twin_rows:
        sid = by_text.get(e1_text.get(t.get("pair_id"), "")) or t.get("pair_id")
        if sid not in {q["id"] for q in suite_questions}:
            continue
        kind = "twin" if t.get("set") == "tpair" else ("fpara" if t.get("label_false_premise") == 1 else None)
        if kind:
            rows.append({"id": t["id"], "kind": kind, "set": t.get("set"), "label": 0 if kind == "twin" else 1,
                         "origin": sid, "text": t["question"], "span": t.get("premise_span")})
    for p in para_rows:
        rows.append({"id": p["id"], "kind": "para", "set": p["set"], "label": 1 if p["set"] == "fpq" else 0,
                     "origin": p["paraphrase_of"], "text": p["question"], "span": None})
    for c in crepe_rows:
        rows.append({"id": c["id"], "kind": "crepe", "set": c["set"], "label": c["label"], "origin": c["id"],
                     "text": c["question"], "span": None})
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate row IDs")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True)
    p.add_argument("--e1-questions", required=True); p.add_argument("--twins", required=True)
    p.add_argument("--paraphrases"); p.add_argument("--crepe-max", type=int, default=0)
    p.add_argument("--layers", nargs="+", type=int, default=[17, 20]); p.add_argument("--name", default="scan_v1")
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--suite-dir", required=True); p.add_argument("--name", default="scan_v1")
    p.add_argument("--C", type=float, default=0.01); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17); p.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    out = S / "tokens" / args.name

    if args.stage == "extract":
        from src import baseline_generation as bg
        from src.baseline_suite import load_suite
        from src.config import load_config
        from src.pilot_model import render
        suite = load_suite(S)
        e1 = list(read_jsonl(args.e1_questions)); tw = list(read_jsonl(args.twins))
        pa = list(read_jsonl(args.paraphrases)) if args.paraphrases else []
        cr = []
        if args.crepe_max and (S / "crepe" / "questions.jsonl").exists():
            allc = list(read_jsonl(S / "crepe" / "questions.jsonl"))
            rng = np.random.default_rng(args.crepe_max)
            pos = [c for c in allc if c["label"] == 1]; neg = [c for c in allc if c["label"] == 0]
            kp = int(round(args.crepe_max * len(pos) / len(allc))); kn = args.crepe_max - kp
            cr = [pos[i] for i in sorted(rng.choice(len(pos), min(kp, len(pos)), replace=False))] + \
                 [neg[i] for i in sorted(rng.choice(len(neg), min(kn, len(neg)), replace=False))]
        rows = build_rows(suite["questions"], e1, tw, pa, cr)
        if args.limit:
            rows = rows[:args.limit]
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"[scan] rows {len(rows)}: {dict(kinds)}; layers {args.layers}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cfg = load_config(args.config)
        runtime = bg.make_runtime(cfg)
        frozen_json(out / "identity.json", {**runtime.identity, "layers": args.layers, "dtype": "float16", "tokens": "content (user question) tokens + last prompt token"})
        tok, model, torch = runtime.tokenizer, runtime.model, runtime.torch
        cache = out / "cache"; cache.mkdir(exist_ok=True)
        with output_lock(out / "extract"):
            for i, r in enumerate(rows, 1):
                path = cache / f"{r['id']}.npz"
                if path.exists():
                    continue
                rendered = render(tok, r["text"])
                enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
                offsets = enc["offset_mapping"]
                _, _, cs, ce = shift_span(rendered, r["text"], (0, len(r["text"])))
                content_idx = token_positions(offsets, cs, ce)
                span_mask = np.zeros(len(content_idx), dtype=bool)
                if r.get("span"):
                    ss, se, _, _ = shift_span(rendered, r["text"], r["span"])
                    inside = set(token_positions(offsets, ss, se))
                    span_mask = np.asarray([j in inside for j in content_idx], dtype=bool)
                ids = torch.tensor([enc["input_ids"]], device=model.get_input_embeddings().weight.device)
                with torch.inference_mode():
                    o = model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                         output_hidden_states=True, return_dict=True)
                    arrays = {}
                    for L in args.layers:
                        h = o.hidden_states[L][0].float().cpu().numpy()
                        arrays[f"L{L}"] = h[content_idx].astype(np.float16)
                        arrays[f"last_L{L}"] = h[-1].astype(np.float16)
                del o
                rel = np.asarray([(offsets[j][0] - cs, offsets[j][1] - cs) for j in content_idx], dtype=np.int32)
                np.savez_compressed(path, **arrays, offsets=rel, span=span_mask, id=np.asarray(r["id"]))
                if i % 50 == 0 or i == len(rows):
                    print(f"[scan] {i}/{len(rows)}", flush=True)
        print(f"Written to {out}. Judge calls: 0.")
        return

    # ---- eval ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.style_controls import fold_assignment
    rows = [r for r in read_jsonl(out / "rows.jsonl") if (out / "cache" / f"{r['id']}.npz").exists()]
    layers = json.load(open(out / "identity.json"))["layers"]
    natural = [r for r in rows if r["kind"] == "natural"]
    assignment = fold_assignment([{"id": r["id"], "set": r["set"], "group_id": r["id"]} for r in natural], folds=args.folds, seed=args.seed)
    print(f"[scan] rows {len(rows)}; natural {len(natural)}; layers {layers}", flush=True)

    def load(r, L):
        z = np.load(out / "cache" / f"{r['id']}.npz")
        return z[f"L{L}"].astype(np.float32), z[f"last_L{L}"].astype(np.float32), z["span"], z["offsets"]

    def fold_of(r):
        return assignment.get(r["origin"], -1)

    results = {}
    localization = {}
    for L in layers:
        feats = {r["id"]: load(r, L) for r in rows}
        scores = defaultdict(dict)   # scorer -> id -> score
        argmax_in_span = []
        first_clfs = None
        for k in sorted(set(assignment.values())):
            train = [r for r in rows if fold_of(r) != k and fold_of(r) >= 0]
            held = [r for r in rows if fold_of(r) == k]
            # S: span probe
            Xs, ys = [], []
            for r in train:
                if r["kind"] in ("natural", "fpara", "twin") and r.get("span") and r["label"] in (0, 1):
                    if r["kind"] == "natural" and r["set"] != "fpq":
                        continue
                    T, _, mask, _ = feats[r["id"]]
                    if mask.any():
                        Xs.append(T[mask].mean(0)); ys.append(r["label"])
            Sclf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000, random_state=args.seed)).fit(np.stack(Xs), ys)
            # Q: question-mean probe on natural
            Xq, yq = [], []
            for r in train:
                if r["kind"] == "natural":
                    T, _, _, _ = feats[r["id"]]
                    Xq.append(T.mean(0)); yq.append(r["label"])
            Qclf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000, random_state=args.seed)).fit(np.stack(Xq), yq)
            if first_clfs is None:
                first_clfs = (Sclf, Qclf)
            for r in held:
                T, last, mask, _ = feats[r["id"]]
                tok_scores = Sclf.decision_function(T)
                scores["S_max"][r["id"]] = float(tok_scores.max())
                scores["S_top5"][r["id"]] = float(np.sort(tok_scores)[-args.topk:].mean())
                scores["S_qmean"][r["id"]] = float(Sclf.decision_function(T.mean(0)[None])[0])
                scores["S_last"][r["id"]] = float(Sclf.decision_function(last[None])[0])
                scores["Q_qmean"][r["id"]] = float(Qclf.decision_function(T.mean(0)[None])[0])
                scores["Q_last"][r["id"]] = float(Qclf.decision_function(last[None])[0])
                if r["label"] == 1 and mask.any():
                    argmax_in_span.append(bool(mask[int(tok_scores.argmax())]))
        # CREPE rows are outside the natural folds: score them with the first fold's probes (never trained on CREPE).
        if first_clfs is not None:
            Sclf, Qclf = first_clfs
            for r in rows:
                if r["kind"] != "crepe":
                    continue
                T, last, _, _ = feats[r["id"]]
                tok_scores = Sclf.decision_function(T)
                scores["S_max"][r["id"]] = float(tok_scores.max())
                scores["S_top5"][r["id"]] = float(np.sort(tok_scores)[-args.topk:].mean())
                scores["S_qmean"][r["id"]] = float(Sclf.decision_function(T.mean(0)[None])[0])
                scores["S_last"][r["id"]] = float(Sclf.decision_function(last[None])[0])
                scores["Q_qmean"][r["id"]] = float(Qclf.decision_function(T.mean(0)[None])[0])
                scores["Q_last"][r["id"]] = float(Qclf.decision_function(last[None])[0])
        localization[L] = (sum(argmax_in_span), len(argmax_in_span))
        by_origin = defaultdict(dict)
        for r in rows:
            by_origin[r["origin"]][r["kind"]] = r
        eval_sets = {
            "natural (FPQ vs NFP)": [r for r in natural],
            "twins (FPQ vs true twin)": [x for o, d in by_origin.items() if "twin" in d and "natural" in d and d["natural"]["set"] == "fpq" for x in (d["natural"], d["twin"])],
            "edited (false para vs true twin)": [x for o, d in by_origin.items() if "twin" in d and "fpara" in d for x in (d["fpara"], d["twin"])],
            "para (FPQ vs NFP, one writer)": [r for r in rows if r["kind"] == "para"],
            "crepe": [r for r in rows if r["kind"] == "crepe"],
        }
        for name, es in eval_sets.items():
            es = [r for r in es if fold_of(r) >= 0 or r["kind"] == "crepe"]
            if not es or len({r["label"] for r in es}) < 2:
                continue
            for sc in SCORERS:
                y = [r["label"] for r in es]; s = [scores[sc].get(r["id"]) for r in es]
                keep = [i for i, v in enumerate(s) if v is not None]
                y = [y[i] for i in keep]; s = [s[i] for i in keep]
                results[(name, sc, L)] = float(roc_auc_score(y, s)) if len(set(y)) == 2 else None
        print(f"[scan] layer {L} done", flush=True)
    lines = [f"# Token-scan gate ({args.name}; C={args.C}, {args.folds} folds by natural origin; rewrites inherit folds)", "",
             "S = probe on premise-span means (FPQ + false paraphrase vs true twin); Q = probe on natural question means (FPQ vs NFP). "
             "S_max/S_top5 scan every prompt token (no span at test time). Reference last-token gates: natural hidden 0.72, text 0.75; twins 0.75; edited 0.76 (29).", ""]
    names = list(dict.fromkeys(n for (n, _, _) in results))
    for L in layers:
        lines += [f"## Layer {L}", "", "| eval set | " + " | ".join(SCORERS) + " |", "|---|" + "|".join("---:" for _ in SCORERS) + "|"]
        for n in names:
            vals = [results.get((n, sc, L)) for sc in SCORERS]
            lines.append(f"| {n} | " + " | ".join(f"{v:.3f}" if v is not None else "NA" for v in vals) + " |")
        hit, tot = localization[L]
        lines += ["", f"Localization: S argmax token inside the premise span for {hit}/{tot} FPQ rows with a span ({hit / max(tot, 1):.1%}).", ""]
    lines += ["Read: S_max on natural and para says whether a span-trained scanner separates FPQ from NFP without any span; "
              "S_max on twins/edited says whether it separates truth alone; natural->crepe rows in 34 §2 are the last-token reference."]
    frozen_json(out / "result.json", {"|".join(map(str, k)): v for k, v in results.items()})
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
