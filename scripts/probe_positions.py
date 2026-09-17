"""Where does the falsity signal live? Position x layer probes (docs/32 §7, chat 9/17).

    CUDA_VISIBLE_DEVICES=1 python scripts/probe_positions.py extract --suite-dir $SUITE_DIR \\
        --config configs/qwen25_7b.yaml --e1-questions $DATA/e1_rows_v1/questions.jsonl \\
        --twins $DATA/e1_rows_v1/questions_twins.jsonl --name pos_v1
    python scripts/probe_positions.py eval --suite-dir $SUITE_DIR --name pos_v1

Rows and positions (every transformer block output, stored float16):
  isolated   myth statement (label 1) vs its correction (label 0), bare user
             message: `last` (assistant-prefill token), `mean` (statement tokens)
  narrative  original FPQ (1) vs true twin (0) [twins]; false paraphrase (1) vs
             true twin (0) [edited]: `span_last`, `span_mean` (premise span),
             `q_mean` (question tokens), `last`
Conditions in eval: isolated, twins, edited (5-fold by origin), and
isolated->twins / isolated->edited (train on isolated `last`/`mean`, score
narrative `span_last`/`span_mean`). Logistic regression, StandardScaler,
fixed C (no inner CV: 28 layers x 4 positions x 5 folds already). Judge calls: 0.
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

POS_ISOLATED = ("last", "mean")
POS_NARRATIVE = ("span_last", "span_mean", "q_mean", "last")


def shift_span(rendered, content, span):
    """Character span inside the rendered chat prompt for a span inside `content`."""
    start = rendered.find(content)
    if start < 0:
        raise ValueError("Rendered prompt does not contain the user content verbatim")
    return start + int(span[0]), start + int(span[1]), start, start + len(content)


def token_positions(offsets, start, end):
    """Token indices whose character span overlaps [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if a != b and a < end and b > start]


def pool(hidden, idx, how):
    """hidden: (L+1, T, d) numpy; returns (L, d) over block outputs 1..L."""
    if not idx:
        raise ValueError("Empty token index set")
    h = hidden[1:, idx, :]
    return h[:, -1, :] if how == "last" else h.mean(axis=1)


def build_rows(suite_questions, e1_rows, twin_rows):
    """Isolated statement rows + narrative rows with spans mapped to suite IDs."""
    by_text = {normalized(q["question"]): q["id"] for q in suite_questions}
    suite = {q["id"]: q for q in suite_questions}
    rows = []
    for q in suite_questions:
        if q["set"] != "fpq":
            continue
        m, c = (q.get("premise_text") or "").strip(), (q.get("correction") or "").strip()
        if len(m) >= 20 and len(m.split()) >= 4 and len(c) >= 20 and len(c.split()) >= 4:
            rows.append({"id": f"{q['id']}_myth", "kind": "isolated", "label": 1, "origin": q["id"], "text": m})
            rows.append({"id": f"{q['id']}_corr", "kind": "isolated", "label": 0, "origin": q["id"], "text": c})
    e1_by_suite = {}
    for r in e1_rows:
        sid = by_text.get(normalized(r.get("question", "")))
        if sid and r.get("premise_span"):
            e1_by_suite[sid] = r
    twins, fparas = {}, {}
    for t in twin_rows:
        sid = by_text.get(normalized((next((r["question"] for r in e1_rows if r["id"] == t.get("pair_id")), "")))) or t.get("pair_id")
        if sid not in suite or not t.get("premise_span"):
            continue
        if t.get("set") == "tpair":
            twins[sid] = t
        elif t.get("label_false_premise") == 1:
            fparas[sid] = t
    # isolated_clean: the two Sonnet-written spans as bare statements (false paraphrase span vs true twin span).
    # The myth/correction pair is confounded by the correction field's "The presupposition is that ..." form.
    for sid in twins:
        if sid in fparas and twins[sid].get("replaced_with") and fparas[sid].get("replaced_with"):
            rows.append({"id": f"{sid}_fspan", "kind": "isolated_clean", "label": 1, "origin": sid, "text": fparas[sid]["replaced_with"].strip()})
            rows.append({"id": f"{sid}_tspan", "kind": "isolated_clean", "label": 0, "origin": sid, "text": twins[sid]["replaced_with"].strip()})
    for sid, e1 in e1_by_suite.items():
        if sid in twins:
            rows.append({"id": sid, "kind": "natural", "label": 1, "origin": sid, "text": suite[sid]["question"], "span": e1["premise_span"]})
            t = twins[sid]
            rows.append({"id": t["id"], "kind": "twin", "label": 0, "origin": sid, "text": t["question"], "span": t["premise_span"]})
            if sid in fparas:
                f = fparas[sid]
                rows.append({"id": f["id"], "kind": "fpara", "label": 1, "origin": sid, "text": f["question"], "span": f["premise_span"]})
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
    p.add_argument("--name", default="pos_v1"); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--suite-dir", required=True); p.add_argument("--name", default="pos_v1")
    p.add_argument("--layers", nargs="+", type=int, help="default: every block")
    p.add_argument("--C", type=float, default=0.01); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    out = S / "positions" / args.name

    if args.stage == "extract":
        from src import baseline_generation as bg
        from src.baseline_suite import load_suite
        from src.config import load_config
        from src.pilot_model import render
        suite = load_suite(S)
        e1 = list(read_jsonl(args.e1_questions)); tw = list(read_jsonl(args.twins))
        rows = build_rows(suite["questions"], e1, tw)
        if args.limit:
            rows = rows[:args.limit]
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"[positions] rows {len(rows)}: {dict(kinds)}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cfg = load_config(args.config)
        runtime = bg.make_runtime(cfg)
        frozen_json(out / "identity.json", {**runtime.identity, "positions": {"isolated": POS_ISOLATED, "narrative": POS_NARRATIVE}, "dtype": "float16"})
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
                ids = torch.tensor([enc["input_ids"]], device=model.get_input_embeddings().weight.device)
                with torch.inference_mode():
                    o = model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                         output_hidden_states=True, return_dict=True)
                    hidden = torch.stack(o.hidden_states, 0)[:, 0].float().cpu().numpy()  # (L+1, T, d)
                del o
                arrays = {}
                if r["kind"] == "isolated":
                    _, _, cs, ce = shift_span(rendered, r["text"], (0, len(r["text"])))
                    arrays["last"] = pool(hidden, [hidden.shape[1] - 1], "last")
                    arrays["mean"] = pool(hidden, token_positions(offsets, cs, ce), "mean")
                else:
                    ss, se, cs, ce = shift_span(rendered, r["text"], r["span"])
                    span_idx = token_positions(offsets, ss, se)
                    arrays["span_last"] = pool(hidden, span_idx, "last")
                    arrays["span_mean"] = pool(hidden, span_idx, "mean")
                    arrays["q_mean"] = pool(hidden, token_positions(offsets, cs, ce), "mean")
                    arrays["last"] = pool(hidden, [hidden.shape[1] - 1], "last")
                np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in arrays.items()}, id=np.asarray(r["id"]))
                if i % 25 == 0 or i == len(rows):
                    print(f"[positions] {i}/{len(rows)}", flush=True)
        print(f"Written to {out}. Judge calls: 0.")
        return

    # ---- eval ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rows = list(read_jsonl(out / "rows.jsonl"))
    feats = {}
    for r in rows:
        path = out / "cache" / f"{r['id']}.npz"
        if path.exists():
            z = np.load(path)
            feats[r["id"]] = {k: z[k].astype(np.float32) for k in z.files if k != "id"}
    rows = [r for r in rows if r["id"] in feats]
    n_layers = next(iter(feats.values()))["last"].shape[0]
    layers = args.layers or list(range(1, n_layers + 1))
    print(f"[positions] {len(rows)} rows with features; {n_layers} blocks; layers {layers[0]}..{layers[-1]}", flush=True)

    def fit_score(train, test, pos_train, pos_test, layer):
        Xtr = np.stack([feats[r["id"]][pos_train][layer - 1] for r in train]); ytr = [r["label"] for r in train]
        Xte = np.stack([feats[r["id"]][pos_test][layer - 1] for r in test]); yte = [r["label"] for r in test]
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            return None
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000, random_state=args.seed))
        clf.fit(Xtr, ytr)
        return float(roc_auc_score(yte, clf.decision_function(Xte)))

    def cv(cond_rows, pos):
        groups = [r["origin"] for r in cond_rows]
        res = {}
        for layer in layers:
            aucs = []
            for tr, te in GroupKFold(n_splits=args.folds).split(cond_rows, groups=groups):
                a = fit_score([cond_rows[i] for i in tr], [cond_rows[i] for i in te], pos, pos, layer)
                if a is not None:
                    aucs.append(a)
            res[layer] = float(np.mean(aucs)) if aucs else None
        return res

    iso = [r for r in rows if r["kind"] == "isolated"]
    iso_clean = [r for r in rows if r["kind"] == "isolated_clean"]
    nat = {r["origin"]: r for r in rows if r["kind"] == "natural"}
    twn = {r["origin"]: r for r in rows if r["kind"] == "twin"}
    fpa = {r["origin"]: r for r in rows if r["kind"] == "fpara"}
    twins_rows = [nat[o] for o in nat if o in twn] + [twn[o] for o in nat if o in twn]
    edited_rows = [fpa[o] for o in fpa if o in twn] + [twn[o] for o in fpa if o in twn]
    results = {}
    for pos in POS_ISOLATED:
        results[("isolated", pos)] = cv(iso, pos); print(f"[positions] isolated/{pos} done", flush=True)
        if iso_clean:
            results[("isolated_clean", pos)] = cv(iso_clean, pos); print(f"[positions] isolated_clean/{pos} done", flush=True)
    for name, cond_rows in (("twins", twins_rows), ("edited", edited_rows)):
        for pos in POS_NARRATIVE:
            results[(name, pos)] = cv(cond_rows, pos); print(f"[positions] {name}/{pos} done", flush=True)
    # transfer: isolated -> narrative span (statement-final token -> span-final token; mean -> mean)
    for name, cond_rows in (("twins", twins_rows), ("edited", edited_rows)):
        for pi, pn in (("last", "span_last"), ("mean", "span_mean")):
            res = {}
            for layer in layers:
                res[layer] = fit_score(iso, cond_rows, pi, pn, layer)
            results[(f"isolated->{name}", f"{pi}->{pn}")] = res
    # Lexical baselines on the SAME pairs: TF-IDF / style / masked on the span substring and on the whole question.
    from src.baseline_gates import _fit_predict
    def text_cv(cond_rows, field, kind):
        groups = [r["origin"] for r in cond_rows]
        aucs = []
        for tr, te in GroupKFold(n_splits=args.folds).split(cond_rows, groups=groups):
            def as_fit(idx):
                return [{"id": cond_rows[i]["id"], "set": "fpq" if cond_rows[i]["label"] == 1 else "nfp",
                         "question": field(cond_rows[i])} for i in idx]
            train, test = as_fit(tr), as_fit(te)
            y = [1 if r["set"] == "fpq" else 0 for r in test]
            if len(set(y)) < 2 or len({1 if r["set"] == "fpq" else 0 for r in train}) < 2:
                continue
            aucs.append(float(roc_auc_score(y, _fit_predict(kind, train, test, features={}, layer=None, c=args.C if kind == "text" else 0.001, seed=args.seed))))
        return float(np.mean(aucs)) if aucs else None
    span_text = lambda r: r["text"][int(r["span"][0]):int(r["span"][1])]
    whole = lambda r: r["text"]
    lex = {}
    for name, cond_rows in (("twins", twins_rows), ("edited", edited_rows)):
        for kind in ("text", "style", "masked"):
            lex[(name, f"{kind} on span text")] = text_cv(cond_rows, span_text, kind)
            lex[(name, f"{kind} on whole question")] = text_cv(cond_rows, whole, kind)
    if iso_clean:
        for kind in ("text", "style", "masked"):
            lex[("isolated_clean", f"{kind} on statement")] = text_cv(iso_clean, whole, kind)
    lines = [f"# Position x layer probes ({args.name}; C={args.C}, {args.folds}-fold by origin; AUROC mean over folds)", "",
             f"isolated {len(iso)} rows ({len(iso)//2} myth/correction pairs; CONFOUNDED: corrections start 'The presupposition is that ...'); "
             f"isolated_clean {len(iso_clean)//2} pairs (false span vs true span, both Sonnet-written); twins {len(twins_rows)//2} pairs; edited {len(edited_rows)//2} pairs", "",
             "| condition | position | best layer | best AUROC | " + " | ".join(f"L{l}" for l in layers) + " |",
             "|---|---|---|---|" + "|".join("---:" for _ in layers) + "|"]
    for (cond, pos), res in results.items():
        vals = [res[l] for l in layers]
        ok = [(l, v) for l, v in zip(layers, vals) if v is not None]
        best = max(ok, key=lambda x: x[1]) if ok else (None, None)
        lines.append(f"| {cond} | {pos} | {best[0]} | {best[1]:.3f} |" + "|".join(f" {v:.3f} " if v is not None else " NA " for v in vals) + "|")
    lines += ["", "## Lexical floor on the same pairs (no hidden states)", "", "| condition | readout | AUROC |", "|---|---|---:|"]
    for (cond, what), v in lex.items():
        lines.append(f"| {cond} | {what} | {v:.3f} |" if v is not None else f"| {cond} | {what} | NA |")
    lines += ["", "Read: isolated says whether the model's state marks a myth as false at all (Marks&Tegmark-style). "
              "twins/edited at span_* vs last says where the signal dies inside the narrative. isolated->twins says whether the "
              "isolated falsity direction transfers to the embedded span. For reference, the last-token probe at L11/17/22 "
              "with inner-CV C gave twins 0.754 / edited 0.764 (29 §7.2); text 0.767 / 0.790."]
    frozen_json(out / "result.json", {"|".join(k): {str(l): v for l, v in res.items()} for k, res in results.items()})
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
