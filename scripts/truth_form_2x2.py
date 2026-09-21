"""Truth x expression 2x2 control for FIXED gates (9/21 review). Judge calls 0; Sonnet
writes and checks the span variants (make stage, ~5 calls per origin).

    export OUT=$SUITE_DIR/controls/truth_form_v1
    # 1) make: pick N FPQ origins that have a twin; Sonnet writes FA/FH/TA/TH span replacements,
    #    a second call checks claim + stance; only origins where all four cells pass are kept.
    python scripts/truth_form_2x2.py make --suite-dir $SUITE_DIR --e1-questions $ROWS/questions.jsonl \\
        --twins $ROWS/questions_twins.jsonl --out $OUT --n 80 --backend claude
    # 2) score (GPU): natural gates refit on suite rows EXCLUDING the selected origins (text TF-IDF,
    #    hidden L17/L22, DiM L17), then applied unchanged to the four cells; plus DIRECT P(Yes) and the
    #    signal_flow span direction (held-out fold).
    CUDA_VISIBLE_DEVICES=0 python scripts/truth_form_2x2.py score --suite-dir $SUITE_DIR --config configs/qwen25_7b.yaml \\
        --out $OUT --flow $SUITE_DIR/signal_flow/flow_v1
    # 3) eval: truth effect, expression effect, within-condition AUROCs, variance shares per gate.
    python scripts/truth_form_2x2.py eval --out $OUT

Decision rule (docs/34 §11): a gate that moves with truth inside a fixed expression uses the wanted
signal; one that moves with expression inside fixed truth reads style; one that moves with neither
reads topic. No new probe is trained on the variants.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src import lora_twin as lt
from src import truth_form as tf
from src.jsonl import read_jsonl
from src.pilot import output_lock


def _append(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stage_make(args, out):
    from scripts.make_rows import make_llm
    from src.baseline_suite import load_suite
    from src.pilot import normalized
    suite = load_suite(args.suite_dir)
    questions = suite["questions"]
    e1 = list(read_jsonl(args.e1_questions))
    twins, _ = lt.map_twin_rows(list(read_jsonl(args.twins)), e1, questions)
    e1_by_text = {normalized(r.get("question", "")): r for r in e1}
    by_id = {q["id"]: q for q in questions}
    candidates = []
    for origin in sorted(twins):
        r = e1_by_text.get(normalized(by_id[origin]["question"]))
        if r and r.get("premise_span") and r.get("premise_text") and r.get("correction"):
            candidates.append((origin, r))
    random.Random(args.seed).shuffle(candidates)
    chosen = candidates[:args.n]
    done = {}
    vpath = out / "variants.jsonl"
    if vpath.exists():
        for row in read_jsonl(vpath):
            done.setdefault(row["origin"], []).append(row)
    llm = make_llm(args.backend, args.model, args.codex_cmd, timeout=args.timeout, claude_cmd=args.claude_cmd)
    if llm is None:
        raise SystemExit(f"backend {args.backend} not available")
    audit = {"selected": len(chosen), "kept": 0, "failed": {}}
    with output_lock(out / "make"):
        for i, (origin, r) in enumerate(chosen, 1):
            if origin in done:
                audit["kept"] += any(x.get("status") == "ok" for x in done[origin])
                continue
            q = by_id[origin]["question"]
            s, e = int(r["premise_span"][0]), int(r["premise_span"][1])
            marked = q[:s] + "[[" + q[s:e] + "]]" + q[e:]
            reply = llm(tf.GEN_PROMPT.format(marked=marked, premise=r["premise_text"], correction=r["correction"]))
            variants = tf.parse_variants(reply)
            if variants is None:
                _append(vpath, {"origin": origin, "status": "unparseable", "raw": (reply or "")[:500]})
                audit["failed"]["unparseable"] = audit["failed"].get("unparseable", 0) + 1
                print(f"[make] {i}/{len(chosen)} {origin}: unparseable", flush=True)
                continue
            rows, bad = [], None
            for cell in tf.CELLS:
                text, span = tf.splice(q, (s, e), variants[cell])
                verdict = tf.parse_verdict(llm(tf.CHECK_PROMPT.format(premise=r["premise_text"], correction=r["correction"], question=text)))
                rows.append({"origin": origin, "cell": cell, "id": f"{origin}_{cell}", "question": text, "span": span,
                             "span_text": variants[cell], "verdict": verdict, "premise_text": r["premise_text"],
                             "correction": r["correction"], "partition": by_id[origin].get("partition")})
                if not tf.cell_ok(cell, verdict):
                    bad = bad or f"{cell}:{verdict}"
            status = "ok" if bad is None else f"check_failed({bad})"
            for row in rows:
                _append(vpath, {**row, "status": status})
            if bad is None:
                audit["kept"] += 1
            else:
                audit["failed"]["check_failed"] = audit["failed"].get("check_failed", 0) + 1
            print(f"[make] {i}/{len(chosen)} {origin}: {status}", flush=True)
    audit["llm_failures"] = getattr(llm, "failures", {}).get("n")
    (out / "make_audit.json").write_text(json.dumps(audit, indent=2))
    kept = [r for r in read_jsonl(vpath) if r.get("status") == "ok"]
    sample = {}
    for r in kept[:8]:
        sample.setdefault(r["origin"], {})[r["cell"]] = r["span_text"]
    lines = [f"# truth x form variants ({out.name})", "", f"selected {audit['selected']}, kept (all four cells pass) {audit['kept']}, failed {audit['failed']}", ""]
    for o, cells in sample.items():
        lines += [f"**{o}**"] + [f"- {c}: {cells.get(c, '')}" for c in tf.CELLS] + [""]
    (out / "make_audit.md").write_text("\n".join(lines))
    print(json.dumps(audit, indent=2)); print(f"Written to {out}. Judge calls: 0.")


def stage_score(args, out):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src import baseline_generation as bg
    from src import signal_flow as sf
    from src.baseline_suite import load_suite
    from src.config import load_config
    suite = load_suite(args.suite_dir)
    questions = suite["questions"]
    variants = [r for r in read_jsonl(out / "variants.jsonl") if r.get("status") == "ok"]
    if not variants:
        raise ValueError("No kept variants; run make first")
    origins = sorted({r["origin"] for r in variants})
    feats = bg.load_feature_records(Path(args.suite_dir) / "model")
    layers = sorted(next(iter(feats.values())).keys()) if feats else []
    hidden_layers = [l for l in args.hidden_layers if l in layers]
    if not hidden_layers:
        raise ValueError(f"Suite features lack layers {args.hidden_layers}; have {layers}")
    train = [q for q in questions if q["id"] not in set(origins) and q["id"] in feats]
    y = np.array([q["set"] == "fpq" for q in train], dtype=int)
    print(f"[score] gates refit on {len(train)} natural rows (excluding {len(origins)} origins); hidden layers {hidden_layers}", flush=True)
    text_gate = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1),
                              LogisticRegression(C=args.text_C, class_weight="balanced", max_iter=5000, random_state=17)).fit([q["question"] for q in train], y)
    hidden_gate, dim = {}, {}
    for l in hidden_layers:
        X = np.stack([feats[q["id"]][l] for q in train])
        hidden_gate[l] = make_pipeline(StandardScaler(), LogisticRegression(C=args.hidden_C, class_weight="balanced", max_iter=5000, random_state=17)).fit(X, y)
        d = X[y == 1].mean(0) - X[y == 0].mean(0); dim[l] = d / np.linalg.norm(d)
    # model
    runtime = bg.make_runtime(load_config(args.config))
    spec = importlib.util.spec_from_file_location("signal_flow_cli", ROOT / "scripts" / "signal_flow.py")
    sfc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sfc)
    flow = Path(args.flow) if args.flow else None
    dirs = np.load(flow / "directions.npz") if flow and (flow / "directions.npz").exists() else None
    # rows to score: the four cells + the original FPQ (cell ORIG_F) + its twin (ORIG_T)
    by_id = {q["id"]: q for q in questions}
    twin_rows = {}
    if args.twins:
        from src import lora_twin as lt2
        tw, _ = lt2.map_twin_rows(list(read_jsonl(args.twins)), list(read_jsonl(args.e1_questions)) if args.e1_questions else [], questions)
        twin_rows = {o: tw[o] for o in origins if o in tw}
    rows = list(variants)
    for o in origins:
        rows.append({"origin": o, "cell": "ORIG_F", "id": o, "question": by_id[o]["question"], "span": None})
        if o in twin_rows:
            rows.append({"origin": o, "cell": "ORIG_T", "id": twin_rows[o]["id"], "question": twin_rows[o]["question"], "span": twin_rows[o].get("premise_span")})
    spath = out / "scores.jsonl"
    done = {r["id"] for r in read_jsonl(spath)} if spath.exists() else set()
    with output_lock(out / "score"):
        for i, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            rec = {"origin": r["origin"], "cell": r["cell"], "id": r["id"]}
            rec["text"] = float(text_gate.decision_function([r["question"]])[0])
            hv = feats.get(r["id"]) if r["cell"] == "ORIG_F" else runtime.features(r["question"], hidden_layers)
            for l in hidden_layers:
                rec[f"hidden_L{l}"] = float(hidden_gate[l].decision_function(hv[l][None])[0])
                rec[f"dim_L{l}"] = float(hv[l] @ dim[l])
            # DIRECT prompt: P(Yes) and, when a span is known, the span-direction projection (held-out fold)
            ids, roles, span_idx = sfc.encode_row(runtime.tokenizer, "direct", {"text": r["question"], "span": r.get("span")})
            hid, p_yes = sfc.forward_states(runtime, ids)
            rec["p_yes"] = p_yes
            if dirs is not None and span_idx:
                fold = sf.fold_of(r["origin"])
                sm = hid[:, span_idx, :].mean(1)
                for l in (17, 20):
                    rec[f"span_d_L{l}"] = float((sm[l - 1] - dirs["center"][fold, l - 1]) @ dirs["d"][fold, l - 1])
            _append(spath, rec)
            if i % 20 == 0 or i == len(rows):
                print(f"[score] {i}/{len(rows)}", flush=True)
    print(f"Written to {spath}. Judge calls: 0.")


def stage_eval(args, out):
    recs = list(read_jsonl(out / "scores.jsonl"))
    gates = sorted({k for r in recs for k in r if k not in ("origin", "cell", "id")})
    lines = [f"# Truth x expression 2x2 on fixed gates ({out.name})", "",
             "Cells: FA false/assertive, FH false/hedged, TA true/assertive, TH true/hedged (same question, only the premise span replaced).",
             "truth effect = mean(F) - mean(T); expression effect = mean(A) - mean(H); AUROC pairs are truth within assertive / hedged and expression within false / true;",
             "variance shares = truth / expression / interaction / residual of the within-origin deviations. Gates were fitted on natural rows excluding these origins and NOT refit here.", "",
             "| gate | n | cell means FA / FH / TA / TH | truth effect [95% CI] | expression effect [95% CI] | AUROC truth (A / H) | AUROC expr (F / T) | var share T / E / I / R |",
             "|---|---:|---|---|---|---|---|---|"]
    results = {}
    for g in gates:
        scores = {}
        for r in recs:
            if r["cell"] in tf.CELLS and g in r and r[g] is not None:
                scores.setdefault(r["origin"], {})[r["cell"]] = r[g]
        try:
            res = tf.two_by_two(scores)
        except ValueError as exc:
            lines.append(f"| {g} | - | {exc} | | | | | |"); continue
        results[g] = res
        lines += tf.report_lines(g, res)
    orig = {}
    for r in recs:
        if r["cell"] in ("ORIG_F", "ORIG_T"):
            orig.setdefault(r["origin"], {})[r["cell"]] = r
    both = [o for o, v in orig.items() if "ORIG_F" in v and "ORIG_T" in v]
    if both:
        lines += ["", f"Reference (original FPQ vs its spliced true twin, n={len(both)}): mean score F - T per gate: "
                  + ", ".join(f"{g} {np.mean([orig[o]['ORIG_F'].get(g, np.nan) - orig[o]['ORIG_T'].get(g, np.nan) for o in both]):+.2f}"
                              for g in gates if all(g in orig[o]['ORIG_F'] and g in orig[o]['ORIG_T'] for o in both))]
    lines += ["", "Read: truth effect with a CI excluding 0 and AUROC truth well above 0.5 inside BOTH expressions = the gate tracks the wanted signal; "
              "expression effect with CI excluding 0 = it reads stance; both near 0 = topic only (the origin main effect absorbs the score)."]
    (out / "report.md").write_text("\n".join(lines) + "\n"); (out / "report.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines)); print(f"Written to {out / 'report.md'}. Judge calls: 0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("make")
    p.add_argument("--suite-dir", required=True); p.add_argument("--e1-questions", required=True); p.add_argument("--twins", required=True)
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=80); p.add_argument("--seed", type=int, default=17)
    p.add_argument("--backend", default="claude"); p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--codex-cmd", default="codex"); p.add_argument("--claude-cmd", default="claude"); p.add_argument("--timeout", type=int, default=180)
    p = sub.add_parser("score")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--flow", default=None, help="signal_flow run dir with directions.npz (span direction, held-out fold)")
    p.add_argument("--twins", default=None); p.add_argument("--e1-questions", default=None)
    p.add_argument("--hidden-layers", nargs="+", type=int, default=[17, 22]); p.add_argument("--text-C", type=float, default=0.1); p.add_argument("--hidden-C", type=float, default=0.01)
    p = sub.add_parser("eval"); p.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    {"make": stage_make, "score": stage_score, "eval": stage_eval}[args.stage](args, out)


if __name__ == "__main__":
    main()
