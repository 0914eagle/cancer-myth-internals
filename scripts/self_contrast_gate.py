"""Self-contrast premise gate (candidate A; docs/32 §10, chat 9/18). Judge calls 0.

    CUDA_VISIBLE_DEVICES=0 python scripts/self_contrast_gate.py extract --suite-dir $SUITE_DIR \\
        --config configs/qwen25_7b.yaml --e1-questions $DATA/e1_rows_v1/questions.jsonl \\
        --twins $DATA/e1_rows_v1/questions_twins.jsonl --paraphrases $SUITE_DIR/variants/para/questions_para.jsonl
    python scripts/self_contrast_gate.py eval --suite-dir $SUITE_DIR

Per question, three short steps on the backbone itself:
  1. belief      "Which general medical belief does the patient take for granted?"  (one sentence)
  2. alternative "State the opposite of that belief."                                (one sentence)
  3. choice      A/B forced choice between belief and alternative, both orders, first-token logits
Gate score = mean probability mass on the alternative (= the patient's belief is inaccurate).
Eval: AUROC on natural (FPQ vs NFP), twins, edited, para; belief quality vs the annotated myth
(word Jaccard, examples); routing with the ALREADY-JUDGED answers (gate on -> fp_unconditional
score, off -> plain score) at several operating points. No new generation of answers, no judge.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl
from src.pilot import frozen_json, normalized, output_lock

BELIEF_PROMPT = ("Read the patient's message. Which general medical belief does the patient take for granted "
                 "as if it were established fact? State that belief in ONE sentence as a general claim about "
                 "medicine (not about this particular patient). Do not evaluate it. Return only the sentence.\n\n"
                 "MESSAGE: {question}")
ALT_PROMPT = ("Write ONE sentence that states the opposite of the following claim, as a general medical claim. "
              "Return only the sentence.\n\nCLAIM: {belief}")
FC_PROMPT = ("Two statements about a medical topic follow. Exactly one is accurate. Which one? "
             "Answer \"A\" or \"B\" first, then explain in one sentence.\nA: {a}\nB: {b}")
_LEAD_AB = re.compile(r"^\W*(?:statement\s+)?([AB])\b", re.IGNORECASE)
_W = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in on at to for from with by is are was were be been being it its this that these those and or but if as than then so do does did has have had can could may might will would should must not no i my me we our you your he she they them their there here what how why which who when where".split())


def content(t):
    return {w for w in _W.findall((t or "").lower()) if w not in STOP}


def jaccard(a, b):
    a, b = content(a), content(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def clean(t):
    t = (t or "").strip().strip("\"'“”‘’ ")
    return t.split("\n")[0].strip()


def build_rows(suite_questions, e1_rows, twin_rows, para_rows):
    by_text = {normalized(q["question"]): q["id"] for q in suite_questions}
    e1_text = {r["id"]: normalized(r.get("question", "")) for r in e1_rows}
    ids = {q["id"] for q in suite_questions}
    rows = [{"id": q["id"], "kind": "natural", "set": q["set"], "label": 1 if q["set"] == "fpq" else 0,
             "origin": q["id"], "question": q["question"], "premise_text": q.get("premise_text")} for q in suite_questions]
    for t in twin_rows:
        sid = by_text.get(e1_text.get(t.get("pair_id"), "")) or t.get("pair_id")
        if sid not in ids:
            continue
        kind = "twin" if t.get("set") == "tpair" else ("fpara" if t.get("label_false_premise") == 1 else None)
        if kind:
            rows.append({"id": t["id"], "kind": kind, "set": t.get("set"), "label": 0 if kind == "twin" else 1,
                         "origin": sid, "question": t["question"], "premise_text": t.get("premise_text")})
    for p in para_rows:
        rows.append({"id": p["id"], "kind": "para", "set": p["set"], "label": 1 if p["set"] == "fpq" else 0,
                     "origin": p["paraphrase_of"], "question": p["question"], "premise_text": p.get("premise_text")})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs")
    return rows


def gate_score(p_a_belief_first, p_a_alt_first):
    """Mass on the alternative: order 'bf' (belief=A) -> 1 - P(A); order 'af' (alt=A) -> P(A)."""
    return ((1.0 - p_a_belief_first) + p_a_alt_first) / 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--suite-dir", required=True); p.add_argument("--config", required=True)
    p.add_argument("--e1-questions", required=True); p.add_argument("--twins", required=True)
    p.add_argument("--paraphrases"); p.add_argument("--name", default="scg_v1")
    p.add_argument("--max-new-tokens", type=int, default=60); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--suite-dir", required=True); p.add_argument("--name", default="scg_v1")
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    out = S / "gates" / args.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "records.jsonl"

    if args.stage == "extract":
        from src import baseline_generation as bg
        from src.baseline_suite import load_suite
        from src.config import load_config
        suite = load_suite(S)
        rows = build_rows(suite["questions"], list(read_jsonl(args.e1_questions)), list(read_jsonl(args.twins)),
                          list(read_jsonl(args.paraphrases)) if args.paraphrases else [])
        if args.limit:
            rows = rows[:args.limit]
        with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done = {r["id"] for r in read_jsonl(ckpt)} if ckpt.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        kinds = defaultdict(int)
        for r in rows:
            kinds[r["kind"]] += 1
        print(f"[scg] rows {len(rows)} {dict(kinds)}; todo {len(todo)}", flush=True)
        cfg = load_config(args.config)
        runtime = bg.make_runtime(cfg)
        frozen_json(out / "identity.json", {**runtime.identity, "prompts": {"belief": BELIEF_PROMPT, "alt": ALT_PROMPT, "fc": FC_PROMPT},
                                            "max_new_tokens": args.max_new_tokens})
        with output_lock(out / "extract"):
            for i, r in enumerate(todo, 1):
                belief = clean(runtime.generate(BELIEF_PROMPT.format(question=r["question"]), args.max_new_tokens)["text"])
                alt = clean(runtime.generate(ALT_PROMPT.format(belief=belief), args.max_new_tokens)["text"]) if belief else ""
                rec = {"id": r["id"], "kind": r["kind"], "label": r["label"], "origin": r["origin"], "belief": belief, "alternative": alt}
                if belief and alt:
                    bf = runtime.binary(FC_PROMPT.format(a=belief, b=alt), positive="A", negative="B")
                    af = runtime.binary(FC_PROMPT.format(a=alt, b=belief), positive="A", negative="B")
                    rec.update(p_a_bf=bf["score"], p_a_af=af["score"], score=gate_score(bf["score"], af["score"]))
                else:
                    rec.update(p_a_bf=None, p_a_af=None, score=None)
                with ckpt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 25 == 0 or i == len(todo):
                    print(f"[scg] {i}/{len(todo)}", flush=True)
        print(f"Written to {out}. Judge calls: 0.")
        return

    # ---- eval ----
    from sklearn.metrics import roc_auc_score
    rows = {r["id"]: r for r in read_jsonl(out / "rows.jsonl")}
    recs = {r["id"]: r for r in read_jsonl(ckpt)}
    lines = [f"# Self-contrast premise gate ({args.name})", "", f"rows {len(rows)}; scored {sum(1 for r in recs.values() if r.get('score') is not None)}; "
             f"empty belief/alternative {sum(1 for r in recs.values() if r.get('score') is None)}", ""]
    by_origin = defaultdict(dict)
    for rid, r in rows.items():
        by_origin[r["origin"]][r["kind"]] = rid
    def auc(ids):
        pairs = [(rows[i]["label"], recs[i]["score"]) for i in ids if i in recs and recs[i].get("score") is not None]
        y = [a for a, _ in pairs]; s = [b for _, b in pairs]
        return (float(roc_auc_score(y, s)) if len(set(y)) == 2 else None), len(pairs)
    sets = {
        "natural (FPQ vs NFP)": [i for i, r in rows.items() if r["kind"] == "natural"],
        "twins (FPQ vs true twin)": [x for d in by_origin.values() if "twin" in d and "natural" in d and rows[d["natural"]]["set"] == "fpq" for x in (d["natural"], d["twin"])],
        "edited (false para vs true twin)": [x for d in by_origin.values() if "twin" in d and "fpara" in d for x in (d["fpara"], d["twin"])],
        "para (FPQ vs NFP, one writer)": [i for i, r in rows.items() if r["kind"] == "para"],
    }
    lines += ["| eval set | AUROC | n |", "|---|---:|---:|"]
    for name, ids in sets.items():
        a, n = auc(ids)
        lines.append(f"| {name} | {a:.3f} | {n} |" if a is not None else f"| {name} | NA | {n} |")
    # belief quality on natural FPQ
    fpq = [i for i, r in rows.items() if r["kind"] == "natural" and r["set"] == "fpq" and r.get("premise_text")]
    jac = [jaccard(recs[i]["belief"], rows[i]["premise_text"]) for i in fpq if i in recs]
    lines += ["", f"Belief vs annotated myth (natural FPQ, content-word Jaccard): median {np.median(jac):.2f}; share >= 0.25: {np.mean(np.asarray(jac) >= 0.25):.1%}; share >= 0.5: {np.mean(np.asarray(jac) >= 0.5):.1%}", ""]
    # routing with judged answers
    def well_scores(d):
        d = Path(d)
        if not (d / "judge_plan.json").exists():
            return {}
        plan = json.load(open(d / "judge_plan.json"))["plan"]
        sc = {}
        for e in read_jsonl(d / "attempts.jsonl"):
            if e.get("status") == "finished" and e.get("valid"):
                sc[e["id"]] = e["score"]
        return {m: {q: sc.get(j) for q, j in jobs.items()} for m, jobs in plan["mapping"].items()}
    scores = well_scores(S / "well_judge_claude"); scores.update(well_scores(S / "well_judge_claude_fpu"))
    plain, fpu = scores.get("plain", {}), scores.get("fp_unconditional", {})
    nat = [i for i, r in rows.items() if r["kind"] == "natural" and i in recs and recs[i].get("score") is not None]
    nfp_scores = sorted(recs[i]["score"] for i in nat if rows[i]["set"] == "nfp")
    def route(th):
        fq = [i for i in nat if rows[i]["set"] == "fpq" and plain.get(i) is not None and fpu.get(i) is not None]
        nq = [i for i in nat if rows[i]["set"] == "nfp" and plain.get(i) is not None and fpu.get(i) is not None]
        clear = sum((fpu[i] if recs[i]["score"] >= th else plain[i]) >= 4 for i in fq)
        natural5 = sum((fpu[i] if recs[i]["score"] >= th else plain[i]) == 5 for i in nq)
        on = sum(recs[i]["score"] >= th for i in nat)
        return clear, len(fq), natural5, len(nq), on
    lines += ["## Routing with judged answers (gate on -> unconditional correction answer, off -> Plain answer)", "",
              "| operating point | threshold | routed to correction | FPQ clear (>=4) | NFP natural (5) |", "|---|---:|---:|---:|---:|"]
    if nfp_scores:
        for label, q in (("NFP FPR 5%", 0.95), ("NFP FPR 10%", 0.90), ("NFP FPR 20%", 0.80), ("NFP FPR 50%", 0.50)):
            th = float(np.quantile(nfp_scores, q))
            c, nf, n5, nn, on = route(th)
            lines.append(f"| {label} | {th:.3f} | {on}/{len(nat)} | {c}/{nf} ({c / max(nf, 1):.1%}) | {n5}/{nn} ({n5 / max(nn, 1):.1%}) |")
        c, nf, n5, nn, on = route(-1e9); lines.append(f"| always correct (unconditional) | – | {on}/{len(nat)} | {c}/{nf} ({c / max(nf, 1):.1%}) | {n5}/{nn} ({n5 / max(nn, 1):.1%}) |")
        c, nf, n5, nn, on = route(1e9); lines.append(f"| never (Plain) | – | 0/{len(nat)} | {c}/{nf} ({c / max(nf, 1):.1%}) | {n5}/{nn} ({n5 / max(nn, 1):.1%}) |")
        fq = [i for i in nat if rows[i]["set"] == "fpq" and fpu.get(i) is not None]; nq = [i for i in nat if rows[i]["set"] == "nfp" and plain.get(i) is not None]
        lines.append(f"| oracle (FPQ on, NFP off) | – | {len(fq)}/{len(nat)} | {sum(fpu[i] >= 4 for i in fq)}/{len(fq)} ({sum(fpu[i] >= 4 for i in fq) / max(len(fq), 1):.1%}) | {sum(plain[i] == 5 for i in nq)}/{len(nq)} ({sum(plain[i] == 5 for i in nq) / max(len(nq), 1):.1%}) |")
    lines += ["", "## Examples (natural FPQ then NFP: belief | alternative | score)", ""]
    shown = 0
    for i, r in rows.items():
        if r["kind"] == "natural" and i in recs and shown < 12:
            e = recs[i]; shown += 1
            lines.append(f"- {i} ({r['set']}) myth={str(r.get('premise_text') or '')[:70]!r} | belief={e['belief'][:110]!r} | alt={e['alternative'][:90]!r} | score={e.get('score')}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
