"""Knowledge labels for FPQ myths: does the backbone know the myth is false
when asked directly, without the patient narrative? (docs/31 §2)

    CUDA_VISIBLE_DEVICES=1 python scripts/probe_knowledge.py --suite-dir $SUITE_DIR \\
        --config configs/qwen25_7b.yaml --name qwen25_7b

Per FPQ with premise_text and correction: two statements (myth, correction) x
two answer orders ("True or False" / "False or True"). K1 = greedy generation,
first True/False parsed; K2 = P(False) - P(True) at the first answer token.
knows = myth answered False in BOTH orders AND correction answered True in
BOTH orders; no = neither; unsure = otherwise. Judge calls: 0. Resumable: one
JSONL record per (id, statement, order); existing records are skipped.
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

from src.baseline_suite import load_suite
from src.config import load_config
from src.jsonl import read_jsonl
from src.pilot import frozen_json, output_lock

PROMPT = {
    "tf": ("Is the following statement medically accurate? Answer \"True\" or \"False\" first, "
           "then explain in one or two sentences.\nStatement: {statement}"),
    "ft": ("Is the following statement medically accurate? Answer \"False\" or \"True\" first, "
           "then explain in one or two sentences.\nStatement: {statement}"),
}
EXPECTED = {"myth": "False", "correction": "True"}
_LEAD = re.compile(r"^\W*(true|false)\b", re.IGNORECASE)
_LEAD_AB = re.compile(r"^\W*(?:statement\s+)?([AB])\b", re.IGNORECASE)

# K3 forced choice: the direct question has a large negative bias (Qwen calls
# ~96% of myths AND ~80% of corrections False; Well Table 1). Presenting both
# statements and asking which is accurate removes that bias; two orders
# remove position bias. knows_fc = picks the correction in both orders.
FC_PROMPT = ("Two statements about a medical topic follow. Exactly one is accurate. Which one? "
             "Answer \"A\" or \"B\" first, then explain in one sentence.\nA: {a}\nB: {b}")


def parse_verdict(text):
    m = _LEAD.match((text or "").strip())
    return m.group(1).capitalize() if m else None


def parse_choice(text):
    m = _LEAD_AB.match((text or "").strip())
    return m.group(1).upper() if m else None


def usable_statement(text):
    """Reject non-statements (fpq_0 'From physicians.', 29 §6). A verb list was
    too strict (it dropped "isn't", "lose", "works", "goes": 58 rows on 9/17);
    length and word count are enough to catch the real defects."""
    t = (text or "").strip()
    return len(t) >= 20 and len(t.split()) >= 4


def label_rows(records):
    """records: list of dicts with id, statement, order, verdict, p_false. Returns per-id labels."""
    by_id = {}
    for r in records:
        by_id.setdefault(r["id"], {}).setdefault(r["statement"], {})[r["order"]] = r
    labels = []
    for qid, stmts in sorted(by_id.items()):
        pair = stmts.pop("pair", {})
        verdicts = {s: {o: r.get("verdict") for o, r in orders.items()} for s, orders in stmts.items()}
        def correct(s):
            vs = verdicts.get(s, {})
            return len(vs) == 2 and all(v == EXPECTED[s] for v in vs.values())
        myth_ok, corr_ok = correct("myth"), correct("correction")
        if not stmts:
            knows = None
        elif myth_ok and corr_ok:
            knows = "yes"
        elif not myth_ok and not corr_ok:
            knows = "no"
        else:
            knows = "unsure"
        margin = None
        if "myth" in stmts and "correction" in stmts:
            pf = lambda s: sum(r.get("p_false", 0.0) for r in stmts[s].values()) / max(len(stmts[s]), 1)
            margin = pf("myth") - pf("correction")  # >0: separates in the right direction
        # Forced choice: order "mc" = myth is A, correction is B (correct answer B); "cm" the reverse (A).
        fc = {o: r.get("verdict") for o, r in pair.items()}
        fc_correct = {o: (fc.get(o) == ("B" if o == "mc" else "A")) for o in fc}
        knows_fc = None if len(fc) < 2 else ("yes" if all(fc_correct.values()) else ("no" if not any(fc_correct.values()) else "unsure"))
        fc_margin = None
        if len(pair) == 2:
            # p_a = P(A). Probability mass on the correction: order mc -> 1 - p_a, order cm -> p_a.
            fc_margin = ((1 - pair["mc"]["p_a"]) + pair["cm"]["p_a"]) / 2 - 0.5
        labels.append({"id": qid, "knows": knows, "myth_correct": myth_ok, "correction_correct": corr_ok,
                       "verdicts": verdicts, "k2_margin": margin,
                       "knows_fc": knows_fc, "fc_verdicts": fc, "fc_margin": fc_margin})
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", default="qwen25_7b")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0, help="first N FPQ only (smoke)")
    parser.add_argument("--report-only", action="store_true", help="no model; rebuild labels/report from the checkpoint")
    args = parser.parse_args()

    suite_dir = Path(args.suite_dir).resolve()
    suite = load_suite(suite_dir)
    fpq = [q for q in suite["questions"] if q["set"] == "fpq"]
    excluded = [q["id"] for q in fpq if not (usable_statement(q.get("premise_text")) and usable_statement(q.get("correction")))]
    rows = [q for q in fpq if q["id"] not in set(excluded)]
    if args.limit:
        rows = rows[:args.limit]
    out = suite_dir / "knowledge" / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "knowledge.jsonl"
    done = {(r["id"], r["statement"], r["order"]) for r in read_jsonl(ckpt)} if ckpt.exists() else set()
    jobs = [(q, s, o) for q in rows for s in ("myth", "correction") for o in ("tf", "ft")
            if (q["id"], s, o) not in done]
    jobs += [(q, "pair", o) for q in rows for o in ("mc", "cm") if (q["id"], "pair", o) not in done]
    print(f"[knowledge] FPQ {len(fpq)}, excluded {len(excluded)}, rows {len(rows)}, jobs {len(jobs)} (done {len(done)})", flush=True)

    if jobs and not args.report_only:
        from src import baseline_generation as bg
        cfg = load_config(args.config)
        runtime = bg.make_runtime(cfg)
        frozen_json(out / "identity_k3.json", {**runtime.identity, "prompts": {**PROMPT, "fc": FC_PROMPT}, "max_new_tokens": args.max_new_tokens})
        with output_lock(out / "run"):
            for i, (q, s, o) in enumerate(jobs, 1):
                if s == "pair":
                    a, b = (q["premise_text"], q["correction"]) if o == "mc" else (q["correction"], q["premise_text"])
                    prompt = FC_PROMPT.format(a=a, b=b)
                    gen = runtime.generate(prompt, args.max_new_tokens)
                    k3 = runtime.binary(prompt, positive="A", negative="B")
                    rec = {"id": q["id"], "statement": s, "order": o, "text": f"A: {a[:80]} | B: {b[:80]}",
                           "generated": gen["text"], "verdict": parse_choice(gen["text"]),
                           "p_a": k3["score"], "a_logit": k3.get("positive_logit"), "b_logit": k3.get("negative_logit")}
                    with ckpt.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if i % 20 == 0 or i == len(jobs):
                        print(f"[knowledge] {i}/{len(jobs)}", flush=True)
                    continue
                statement = q["premise_text"] if s == "myth" else q["correction"]
                prompt = PROMPT[o].format(statement=statement)
                gen = runtime.generate(prompt, args.max_new_tokens)
                # K2: first answer token. positive="False" so score = P(False)/(P(False)+P(True)).
                k2 = runtime.binary(prompt, positive="False", negative="True")
                rec = {"id": q["id"], "statement": s, "order": o, "text": statement,
                       "generated": gen["text"], "verdict": parse_verdict(gen["text"]),
                       "p_false": k2["score"], "false_logit": k2.get("positive_logit"), "true_logit": k2.get("negative_logit")}
                with ckpt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 20 == 0 or i == len(jobs):
                    print(f"[knowledge] {i}/{len(jobs)}", flush=True)

    records = list(read_jsonl(ckpt)) if ckpt.exists() else []
    labels = label_rows(records)
    with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    counts = Counter(l["knows"] for l in labels)
    counts_fc = Counter(l["knows_fc"] for l in labels)
    unparsed = sum(1 for r in records if r.get("verdict") is None)
    order_flips = sum(1 for l in labels for s, vs in l["verdicts"].items() if len(vs) == 2 and len(set(vs.values())) > 1)
    lines = [f"# Knowledge probe: {args.name}", "",
             f"FPQ {len(fpq)}; excluded (non-statement premise/correction) {len(excluded)}: {', '.join(excluded[:12])}{' ...' if len(excluded) > 12 else ''}",
             f"records {len(records)} (expected {4 * len(rows)}); unparsed verdicts {unparsed}; statements whose verdict flips with answer order {order_flips}", "",
             "| knows | n | share |", "|---|---:|---:|"]
    for k in ("yes", "unsure", "no"):
        lines.append(f"| {k} | {counts.get(k, 0)} | {counts.get(k, 0) / max(len(labels), 1):.1%} |")
    myth_false = sum(1 for l in labels if l["myth_correct"]); corr_true = sum(1 for l in labels if l["correction_correct"])
    lines += ["", f"myth answered False in both orders: {myth_false}/{len(labels)}; correction answered True in both orders: {corr_true}/{len(labels)}",
              "(a model that says False to everything scores high on the first and ~0 on the second; knows requires both)", "",
              "K2 margin = mean P(False|myth) - mean P(False|correction); positive = separates in the right direction."]
    margins = [l["k2_margin"] for l in labels if l["k2_margin"] is not None]
    import numpy as np
    if margins:
        m = np.asarray(margins)
        lines.append(f"K2 margin: median {np.median(m):+.3f}, share > 0: {(m > 0).mean():.1%}")
    lines += ["", "## K3 forced choice (both statements shown, which is accurate; two orders)", "",
              "| knows_fc | n | share |", "|---|---:|---:|"]
    for k in ("yes", "unsure", "no", None):
        lines.append(f"| {k if k else 'not run'} | {counts_fc.get(k, 0)} | {counts_fc.get(k, 0) / max(len(labels), 1):.1%} |")
    fcm = [l["fc_margin"] for l in labels if l["fc_margin"] is not None]
    if fcm:
        m = np.asarray(fcm)
        lines.append(f"\nFC margin (mean prob on the correction − 0.5): median {np.median(m):+.3f}, share > 0: {(m > 0).mean():.1%}")
    both = Counter((l["knows"], l["knows_fc"]) for l in labels if l["knows_fc"])
    if both:
        lines += ["", "K1 x K3 (knows, knows_fc): " + ", ".join(f"{a}/{b}={n}" for (a, b), n in sorted(both.items(), key=lambda x: -x[1]))]
    lines += ["", "## Sample (first 10 with generated text)", ""]
    for r in records[:10]:
        lines.append(f"- {r['id']} {r['statement']}/{r['order']}: **{r['verdict']}** (p_false {r['p_false']:.2f}) — {r['text'][:100]} → {r['generated'][:160]!r}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
