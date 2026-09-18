"""Mechanism scan in one run (chat 9/18): attention, head-level probes, and a
span->last-token transplant pilot. Judge calls 0.

    CUDA_VISIBLE_DEVICES=0 python scripts/mechanism_scan.py extract --suite-dir $SUITE_DIR \\
        --config configs/qwen25_7b.yaml --e1-questions $DATA/e1_rows_v1/questions.jsonl \\
        --twins $DATA/e1_rows_v1/questions_twins.jsonl --name mech_v1
    CUDA_VISIBLE_DEVICES=0 python scripts/mechanism_scan.py patch --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml --name mech_v1
    python scripts/mechanism_scan.py eval --suite-dir $SUITE_DIR --name mech_v1

Two GPUs: extract with --shard 0/2 on GPU 0 and --shard 1/2 on GPU 1 (each
shard takes its own lock and skips cached rows); patch with --layers 17 on one
card and --layers 20 on the other (results land in patch_results_L{L}.json and
eval merges every patch_results*.json).

Rows: FPQ with a twin (span from E1), true twins, false paraphrases, NFP.
extract (eager attention): per row and layer/head, attention mass from the
last prompt token to the premise span (①); per-head attention output at the
last token (②); residuals at last token, span mean, question mean.
patch (③): learn the span-mean truth direction d_l per layer on pairs; during
the DIRECT Yes/No readout add alpha * z * 0.05 * ||h|| * d_l to the last token,
where z is the row's own standardized projection of its span mean (natural NFP:
question mean) on d_l. If the readout AUROC rises with alpha, the signal is
computed at the span but not delivered to the answer position.
eval: AUROC tables for ①, ②, ③ (twins / edited / natural).
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

DIRECT = "Does the following question rest on a false premise? Answer only Yes or No.\n\nQuestion: {question}"


def shift_span(rendered, content, span):
    start = rendered.find(content)
    if start < 0:
        raise ValueError("Rendered prompt does not contain the user content verbatim")
    return start + int(span[0]), start + int(span[1]), start, start + len(content)


def token_positions(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if a != b and a < end and b > start]


def span_mass(attn_last, idx):
    """attn_last: (L, H, T) attention from the last token. Sum over token idx -> (L, H)."""
    if not idx:
        raise ValueError("Empty index set")
    return attn_last[:, :, idx].sum(axis=-1)


def parse_shard(text):
    k, n = (int(x) for x in str(text).split("/"))
    if n < 1 or not 0 <= k < n:
        raise ValueError(f"--shard must be k/n with 0 <= k < n, got {text!r}")
    return k, n


def build_rows(suite_questions, e1_rows, twin_rows):
    by_text = {normalized(q["question"]): q["id"] for q in suite_questions}
    e1_text = {r["id"]: normalized(r.get("question", "")) for r in e1_rows}
    e1_span = {by_text[normalized(r["question"])]: r["premise_span"] for r in e1_rows
               if normalized(r.get("question", "")) in by_text and r.get("premise_span")}
    suite = {q["id"]: q for q in suite_questions}
    twins, fparas = {}, {}
    for t in twin_rows:
        sid = by_text.get(e1_text.get(t.get("pair_id"), "")) or t.get("pair_id")
        if sid not in suite or not t.get("premise_span"):
            continue
        if t.get("set") == "tpair":
            twins[sid] = t
        elif t.get("label_false_premise") == 1:
            fparas[sid] = t
    rows = []
    for sid, t in twins.items():
        if sid in e1_span:
            rows.append({"id": sid, "kind": "natural", "set": "fpq", "label": 1, "origin": sid, "text": suite[sid]["question"], "span": e1_span[sid]})
            rows.append({"id": t["id"], "kind": "twin", "set": "tpair", "label": 0, "origin": sid, "text": t["question"], "span": t["premise_span"]})
            if sid in fparas:
                f = fparas[sid]
                rows.append({"id": f["id"], "kind": "fpara", "set": "fpq", "label": 1, "origin": sid, "text": f["question"], "span": f["premise_span"]})
    for q in suite_questions:
        if q["set"] == "nfp":
            rows.append({"id": q["id"], "kind": "natural", "set": "nfp", "label": 0, "origin": q["id"], "text": q["question"], "span": None})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs")
    return rows


def merge_patch_results(out):
    """patch_results_L{L}.json per layer (plus a legacy patch_results.json) -> one dict, layers ascending."""
    merged = {}
    paths = sorted(Path(out).glob("patch_results_L*.json"), key=lambda p: int(p.stem.split("_L")[1]))
    legacy = Path(out) / "patch_results.json"
    for path in ([legacy] if legacy.exists() else []) + paths:
        merged.update(json.loads(path.read_text()))
    return merged


def _load_runtime(config_path):
    from src import baseline_generation as bg
    from src.config import load_config
    cfg = load_config(config_path)
    cfg["source_model"]["attn_implementation"] = "eager"
    return bg.make_runtime(cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True)
    p.add_argument("--e1-questions", required=True); p.add_argument("--twins", required=True)
    p.add_argument("--name", default="mech_v1"); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--shard", default="0/1", help="k/n: process rows k::n (run one process per GPU)")
    p = sub.add_parser("patch")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True)
    p.add_argument("--name", default="mech_v1"); p.add_argument("--layers", nargs="+", type=int, default=[17, 20])
    p.add_argument("--alphas", nargs="+", type=float, default=[0, 1, 2, 4]); p.add_argument("--scale", type=float, default=0.05)
    p = sub.add_parser("eval")
    p.add_argument("--suite-dir", required=True); p.add_argument("--name", default="mech_v1")
    p.add_argument("--C", type=float, default=0.01)
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    out = S / "mechanism" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"; cache.mkdir(exist_ok=True)

    if args.stage == "extract":
        from src.baseline_suite import load_suite
        from src.pilot_model import render
        suite = load_suite(S)
        rows = build_rows(suite["questions"], list(read_jsonl(args.e1_questions)), list(read_jsonl(args.twins)))
        if args.limit:
            rows = rows[:args.limit]
        rows_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        rows_path = out / "rows.jsonl"
        if rows_path.exists() and rows_path.read_text(encoding="utf-8") != rows_text:
            raise ValueError(f"{rows_path} exists with different rows; use a new --name")
        rows_path.write_text(rows_text, encoding="utf-8")
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"] + "/" + r["set"]] += 1
        k, n = parse_shard(args.shard)
        rows = rows[k::n]
        print(f"[mech] rows {sum(kinds.values())} {dict(kinds)}; shard {k}/{n} -> {len(rows)} rows", flush=True)
        runtime = _load_runtime(args.config)
        tok, model, torch = runtime.tokenizer, runtime.model, runtime.torch
        from src.modeling import decoder_layers
        blocks = decoder_layers(model)
        n_heads = model.config.num_attention_heads
        frozen_json(out / "identity.json", {**runtime.identity, "attn": "eager", "n_layers": len(blocks), "n_heads": n_heads})
        head_z = {}
        def make_hook(layer):
            def hook(_m, inputs, _out):
                head_z[layer] = inputs[0][0, -1].detach().float().cpu().numpy().reshape(n_heads, -1)
            return hook
        handles = [blk.self_attn.o_proj.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
        try:
            with output_lock(out / f"extract.shard{k}of{n}"):
                for i, r in enumerate(rows, 1):
                    path = cache / f"{r['id']}.npz"
                    if path.exists():
                        continue
                    rendered = render(tok, r["text"])
                    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
                    offsets = enc["offset_mapping"]
                    _, _, cs, ce = shift_span(rendered, r["text"], (0, len(r["text"])))
                    content_idx = token_positions(offsets, cs, ce)
                    span_idx = []
                    if r.get("span"):
                        ss, se, _, _ = shift_span(rendered, r["text"], r["span"])
                        span_idx = token_positions(offsets, ss, se)
                    ids = torch.tensor([enc["input_ids"]], device=model.get_input_embeddings().weight.device)
                    head_z.clear()
                    with torch.inference_mode():
                        o = model.base_model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                             output_hidden_states=True, output_attentions=True, return_dict=True)
                        attn_last = torch.stack([a[0, :, -1, :] for a in o.attentions], 0).float().cpu().numpy()  # (L, H, T)
                        hid = torch.stack(o.hidden_states[1:], 0)[:, 0].float().cpu().numpy()  # (L, T, d)
                    del o
                    arrays = {
                        "attn_span": span_mass(attn_last, span_idx) if span_idx else np.full(attn_last.shape[:2], np.nan, dtype=np.float32),
                        "attn_content": span_mass(attn_last, content_idx),
                        "head_z_last": np.stack([head_z[l] for l in range(len(blocks))], 0).astype(np.float16),
                        "resid_last": hid[:, -1, :].astype(np.float16),
                        "resid_q_mean": hid[:, content_idx, :].mean(1).astype(np.float16),
                        "resid_span_mean": (hid[:, span_idx, :].mean(1) if span_idx else np.full(hid.shape[::2], np.nan)).astype(np.float16),
                        "n_span_tokens": np.asarray(len(span_idx)), "n_content_tokens": np.asarray(len(content_idx)),
                    }
                    np.savez_compressed(path, **arrays, id=np.asarray(r["id"]))
                    if i % 25 == 0 or i == len(rows):
                        print(f"[mech] {i}/{len(rows)}", flush=True)
        finally:
            for h in handles:
                h.remove()
        print(f"Written to {out}. Judge calls: 0.")
        return

    rows = [r for r in read_jsonl(out / "rows.jsonl") if (cache / f"{r['id']}.npz").exists()]
    by_origin = defaultdict(dict)
    for r in rows:
        by_origin[r["origin"]][r["kind"]] = r
    pairs_twins = [(d["natural"], d["twin"]) for d in by_origin.values() if "natural" in d and "twin" in d and d["natural"]["set"] == "fpq"]
    pairs_edited = [(d["fpara"], d["twin"]) for d in by_origin.values() if "fpara" in d and "twin" in d]
    nfp = [r for r in rows if r["kind"] == "natural" and r["set"] == "nfp"]
    load = lambda r: np.load(cache / f"{r['id']}.npz")

    if args.stage == "patch":
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        runtime = _load_runtime(args.config)
        model, torch = runtime.model, runtime.torch
        from src.modeling import decoder_layers
        blocks = decoder_layers(model)
        results = {}
        for L in args.layers:
            if (out / f"patch_results_L{L}.json").exists():
                print(f"[patch] L{L} already done; skipping", flush=True)
                continue
            with output_lock(out / f"patch_L{L}"):
                # direction from span means: false (natural FPQ + fpara) vs true (twin)
                X, y = [], []
                for a, b in pairs_twins + pairs_edited:
                    X.append(load(a)["resid_span_mean"][L - 1].astype(np.float32)); y.append(1)
                    X.append(load(b)["resid_span_mean"][L - 1].astype(np.float32)); y.append(0)
                X = np.stack(X); sc = StandardScaler().fit(X)
                clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
                d = clf.coef_[0] / sc.scale_  # back to raw space
                d = d / np.linalg.norm(d)
                proj_train = X @ d
                mu, sd = proj_train.mean(), proj_train.std() + 1e-8
                state = {"delta": None}
                def hook(_m, _inp, output):
                    if state["delta"] is None:
                        return output
                    h = output[0] if isinstance(output, tuple) else output
                    h[:, -1, :] = h[:, -1, :] + state["delta"].to(h.dtype)
                    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
                handle = blocks[L - 1].register_forward_hook(hook)
                try:
                    d_t = torch.tensor(d, device=model.get_input_embeddings().weight.device, dtype=torch.float32)
                    eval_rows = [x for pr in pairs_twins for x in pr] + [x for pr in pairs_edited for x in pr] + nfp
                    seen = set(); uniq = []
                    for r in eval_rows:
                        if r["id"] not in seen:
                            seen.add(r["id"]); uniq.append(r)
                    scores = {a: {} for a in args.alphas}
                    for i, r in enumerate(uniq, 1):
                        z = load(r)
                        src = z["resid_span_mean"][L - 1] if not np.isnan(z["resid_span_mean"][L - 1]).any() else z["resid_q_mean"][L - 1]
                        zscore = float((src.astype(np.float32) @ d - mu) / sd)
                        hnorm = float(np.linalg.norm(z["resid_last"][L - 1].astype(np.float32)))
                        for a in args.alphas:
                            state["delta"] = None if a == 0 else (a * zscore * args.scale * hnorm) * d_t
                            scores[a][r["id"]] = runtime.binary(DIRECT.format(question=r["text"]))["score"]
                        if i % 50 == 0 or i == len(uniq):
                            print(f"[patch] L{L} {i}/{len(uniq)}", flush=True)
                    state["delta"] = None
                finally:
                    handle.remove()
                layer_results = {}
                for a in args.alphas:
                    sc_ = scores[a]
                    def auc(items):
                        y_ = [lab for lab, _ in items]; s_ = [v for _, v in items]
                        return float(roc_auc_score(y_, s_)) if len(set(y_)) == 2 else None
                    layer_results[f"L{L}|alpha{a}"] = {
                        "twins": auc([(1, sc_[x["id"]]) for x, _ in pairs_twins] + [(0, sc_[t["id"]]) for _, t in pairs_twins]),
                        "edited": auc([(1, sc_[x["id"]]) for x, _ in pairs_edited] + [(0, sc_[t["id"]]) for _, t in pairs_edited]),
                        "natural": auc([(1, sc_[x["id"]]) for x, _ in pairs_twins] + [(0, sc_[n["id"]]) for n in nfp]),
                        "mean_p_yes_fpq": float(np.mean([sc_[x["id"]] for x, _ in pairs_twins])),
                        "mean_p_yes_twin": float(np.mean([sc_[t["id"]] for _, t in pairs_twins])),
                        "mean_p_yes_nfp": float(np.mean([sc_[n["id"]] for n in nfp])),
                    }
                (out / f"patch_results_L{L}.json").write_text(json.dumps(layer_results, indent=2))
            results.update(layer_results)
        print(json.dumps(results, indent=2))
        print(f"Written to {out}. Judge calls: 0.")
        return

    # ---- eval (CPU) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lines = [f"# Mechanism scan ({args.name})", "", f"twins pairs {len(pairs_twins)}; edited pairs {len(pairs_edited)}; NFP {len(nfp)}", ""]
    def pair_auc(pairs, key, fn):
        y, s = [], []
        for a, b in pairs:
            y += [1, 0]; s += [fn(load(a)[key]), fn(load(b)[key])]
        s = np.asarray(s, dtype=float)
        ok = ~np.isnan(s)
        return float(roc_auc_score(np.asarray(y)[ok], s[ok])) if ok.sum() and len(set(np.asarray(y)[ok])) == 2 else None
    # ① attention mass last -> span, per layer/head
    any_z = load(pairs_twins[0][0]); L, H = any_z["attn_span"].shape
    lines += ["## ① Attention from the last prompt token to the premise span (FPQ vs true twin)", "",
              "| layer | mean mass FPQ | mean mass twin | AUROC (sum over heads) | best head (AUROC) |", "|---|---:|---:|---:|---|"]
    per_head = {}
    for l in range(L):
        m_f = np.mean([load(a)["attn_span"][l].sum() for a, _ in pairs_twins]); m_t = np.mean([load(b)["attn_span"][l].sum() for _, b in pairs_twins])
        a_sum = pair_auc(pairs_twins, "attn_span", lambda v, l=l: v[l].sum())
        best = None
        for h in range(H):
            a_h = pair_auc(pairs_twins, "attn_span", lambda v, l=l, h=h: v[l, h])
            per_head[(l + 1, h)] = a_h
            if a_h is not None and (best is None or abs(a_h - .5) > abs(best[1] - .5)):
                best = (h, a_h)
        lines.append(f"| L{l + 1} | {m_f:.3f} | {m_t:.3f} | {a_sum:.3f} | H{best[0]} ({best[1]:.3f}) |")
    top = sorted(per_head.items(), key=lambda kv: -abs((kv[1] or .5) - .5))[:10]
    lines += ["", "Top heads by |AUROC-0.5| (attention-to-span, twins): " + ", ".join(f"L{l}H{h} {a:.3f}" for (l, h), a in top), ""]
    # ② per-head output probes at the last token (twins, 5-fold by origin)
    lines += ["## ② Per-head attention-output probes at the last token (twins, 5-fold by origin, top 12) vs residual probe", "",
              "| layer/head | AUROC |", "|---|---:|"]
    groups = [a["origin"] for a, _ in pairs_twins for _ in range(2)]
    rows_lin = [x for pr in pairs_twins for x in pr]
    y_lin = np.array([1, 0] * len(pairs_twins))
    def cv_auc(X):
        aucs = []
        for tr, te in GroupKFold(n_splits=5).split(X, groups=groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000)).fit(X[tr], y_lin[tr])
            aucs.append(roc_auc_score(y_lin[te], clf.decision_function(X[te])))
        return float(np.mean(aucs))
    Z = np.stack([load(r)["head_z_last"].astype(np.float32) for r in rows_lin])  # (N, L, H, dh)
    head_auc = {}
    for l in range(L):
        for h in range(H):
            head_auc[(l + 1, h)] = cv_auc(Z[:, l, h, :])
    for (l, h), a in sorted(head_auc.items(), key=lambda kv: -kv[1])[:12]:
        lines.append(f"| L{l}H{h} | {a:.3f} |")
    R = np.stack([load(r)["resid_last"].astype(np.float32) for r in rows_lin])
    for l in (11, 17, 20):
        lines.append(f"| residual L{l} (reference) | {cv_auc(R[:, l - 1, :]):.3f} |")
    # ③ patch results if present
    pr = merge_patch_results(out)
    if pr:
        lines += ["", "## ③ Span->last transplant: DIRECT Yes/No readout AUROC", "",
                  "| layer | alpha | twins | edited | natural (FPQ vs NFP) | P(Yes) FPQ / twin / NFP |", "|---|---:|---:|---:|---:|---|"]
        for k, v in pr.items():
            Lk, ak = k.split("|")
            f = lambda x: f"{x:.3f}" if x is not None else "NA"
            lines.append(f"| {Lk} | {ak[5:]} | {f(v['twins'])} | {f(v['edited'])} | {f(v['natural'])} | {v['mean_p_yes_fpq']:.2f} / {v['mean_p_yes_twin']:.2f} / {v['mean_p_yes_nfp']:.2f} |")
    else:
        lines += ["", "(③ no patch_results_L*.json found; run the patch stage)"]
    lines += ["", "Read: ① says whether the answer position even looks at the premise, and whether it looks differently at false vs true spans. "
              "② says whether any single head's output carries truth at the last token better than the whole residual (0.75-0.78). "
              "③: if the readout AUROC rises with alpha, the truth signal is computed at the span but not delivered; alpha=0 is the unpatched direct readout (~0.5)."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
