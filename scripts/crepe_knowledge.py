"""Knowledge readouts on CREPE's human-written presuppositions and corrections
(docs/34 §7 read 3: is 'direct question is biased, comparison is accurate'
an artifact of our data?). Judge calls 0.

    CUDA_VISIBLE_DEVICES=1 python scripts/crepe_knowledge.py --suite-dir $SUITE_DIR \\
        --crepe-dir $CREPE_DIR --config configs/qwen25_7b.yaml --max 300

Per CREPE false-presupposition question with a presupposition and a
correction: K1 direct ("Is the following statement accurate?") on both, two
answer orders; K3 forced choice presupposition vs correction, two orders.
Same label rules as probe_knowledge (knows = both directions in both orders;
knows_fc = correction chosen in both orders).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl
from src.pilot import frozen_json, output_lock

_spec = importlib.util.spec_from_file_location("pk", ROOT / "scripts" / "probe_knowledge.py")
pk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pk)

PROMPT = {
    "tf": "Is the following statement accurate? Answer \"True\" or \"False\" first, then explain in one or two sentences.\nStatement: {statement}",
    "ft": "Is the following statement accurate? Answer \"False\" or \"True\" first, then explain in one or two sentences.\nStatement: {statement}",
}
FC_PROMPT = ("Two statements follow. Exactly one is accurate. Which one? Answer \"A\" or \"B\" first, then explain in one sentence.\nA: {a}\nB: {b}")


def load_pairs(crepe_dir, max_n, seed=17):
    rows = []
    for path in sorted(Path(crepe_dir).glob("data/hf_*.jsonl")):
        for r in read_jsonl(path):
            labels = r.get("labels") or []
            if not any("false" in str(l).lower() for l in labels):
                continue
            pres = [p for p in (r.get("presuppositions") or []) if str(p).strip()]
            cors = [c for c in (r.get("corrections") or []) if str(c).strip()]
            if pres and cors:
                rows.append({"id": r.get("id") or f"{path.stem}_{len(rows)}", "question": r.get("question"),
                             "premise_text": str(pres[0]).strip(), "correction": str(cors[0]).strip()})
    rng = np.random.default_rng(seed)
    if max_n and len(rows) > max_n:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), max_n, replace=False))]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", required=True); ap.add_argument("--crepe-dir", required=True)
    ap.add_argument("--config", required=True); ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=96); ap.add_argument("--name", default="crepe_qwen25_7b")
    args = ap.parse_args()
    out = Path(args.suite_dir).resolve() / "knowledge" / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in load_pairs(args.crepe_dir, args.max) if pk.usable_statement(r["premise_text"]) and pk.usable_statement(r["correction"])]
    ckpt = out / "knowledge.jsonl"
    done = {(r["id"], r["statement"], r["order"]) for r in read_jsonl(ckpt)} if ckpt.exists() else set()
    jobs = [(q, s, o) for q in rows for s in ("myth", "correction") for o in ("tf", "ft") if (q["id"], s, o) not in done]
    jobs += [(q, "pair", o) for q in rows for o in ("mc", "cm") if (q["id"], "pair", o) not in done]
    print(f"[crepe-knowledge] pairs {len(rows)}; jobs {len(jobs)} (done {len(done)})", flush=True)
    if jobs:
        from src import baseline_generation as bg
        from src.config import load_config
        runtime = bg.make_runtime(load_config(args.config))
        frozen_json(out / "identity.json", {**runtime.identity, "prompts": {**PROMPT, "fc": FC_PROMPT}, "max_new_tokens": args.max_new_tokens})
        with output_lock(out / "run"):
            for i, (q, s, o) in enumerate(jobs, 1):
                if s == "pair":
                    a, b = (q["premise_text"], q["correction"]) if o == "mc" else (q["correction"], q["premise_text"])
                    gen = runtime.generate(FC_PROMPT.format(a=a, b=b), args.max_new_tokens)
                    k3 = runtime.binary(FC_PROMPT.format(a=a, b=b), positive="A", negative="B")
                    rec = {"id": q["id"], "statement": s, "order": o, "text": f"A: {a[:80]} | B: {b[:80]}", "generated": gen["text"],
                           "verdict": pk.parse_choice(gen["text"]), "p_a": k3["score"]}
                else:
                    statement = q["premise_text"] if s == "myth" else q["correction"]
                    prompt = PROMPT[o].format(statement=statement)
                    gen = runtime.generate(prompt, args.max_new_tokens)
                    k2 = runtime.binary(prompt, positive="False", negative="True")
                    rec = {"id": q["id"], "statement": s, "order": o, "text": statement, "generated": gen["text"],
                           "verdict": pk.parse_verdict(gen["text"]), "p_false": k2["score"]}
                with ckpt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 50 == 0 or i == len(jobs):
                    print(f"[crepe-knowledge] {i}/{len(jobs)}", flush=True)
    records = list(read_jsonl(ckpt))
    labels = pk.label_rows(records)
    c1, c3 = Counter(l["knows"] for l in labels), Counter(l["knows_fc"] for l in labels)
    myth_false = sum(1 for l in labels if l["myth_correct"]); corr_true = sum(1 for l in labels if l["correction_correct"])
    fcm = [l["fc_margin"] for l in labels if l["fc_margin"] is not None]
    lines = [f"# CREPE knowledge readouts ({args.name}; {len(labels)} human-written presupposition/correction pairs)", "",
             "| readout | result |", "|---|---|",
             f"| K1 presupposition -> False (both orders) | {myth_false}/{len(labels)} ({myth_false / max(len(labels), 1):.1%}) |",
             f"| K1 correction -> True (both orders) | {corr_true}/{len(labels)} ({corr_true / max(len(labels), 1):.1%}) |",
             f"| K1 knows (both) | {c1.get('yes', 0)} ({c1.get('yes', 0) / max(len(labels), 1):.1%}); unsure {c1.get('unsure', 0)}; no {c1.get('no', 0)} |",
             f"| K3 forced choice knows_fc | {c3.get('yes', 0)} ({c3.get('yes', 0) / max(len(labels), 1):.1%}); unsure {c3.get('unsure', 0)}; no {c3.get('no', 0)} |",
             f"| FC margin median | {np.median(fcm):+.3f} (share > 0: {(np.asarray(fcm) > 0).mean():.1%}) |" if fcm else "| FC margin | NA |",
             "", "Cancer-Myth reference (31 §8): K1 myth False 96%, correction True 20%, K3 knows_fc 91%.", "", "## Sample", ""]
    for r in records[:8]:
        lines.append(f"- {r['id']} {r['statement']}/{r['order']}: **{r['verdict']}** — {r['text'][:100]} → {r['generated'][:140]!r}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
