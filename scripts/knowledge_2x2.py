"""Knowledge x outcome tables (docs/31 §4). CPU, judge calls 0.

    python scripts/knowledge_2x2.py --suite-dir $SUITE_DIR [--controls v2] [--name qwen25_7b]

Reads knowledge/<name>/labels.jsonl (knows_fc from the forced choice),
Well scores from well_judge_claude/ (+ well_judge_claude_fpu/ if present),
and style_controls/<controls>/result.json. Writes knowledge/<name>/2x2.md:
  1. knows_fc x Plain outcome (fail = Well <= 3, clear correction = >= 4)
  2. per method: clear corrections among Plain failures, split by knows_fc
  3. gate AUROC recomputed on origins with knows_fc == yes vs the rest
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl


def well_scores(judge_dir):
    judge_dir = Path(judge_dir)
    if not (judge_dir / "judge_plan.json").exists():
        return {}
    plan = json.load(open(judge_dir / "judge_plan.json"))["plan"]
    score = {}
    if (judge_dir / "attempts.jsonl").exists():
        for d in read_jsonl(judge_dir / "attempts.jsonl"):
            if d.get("status") == "finished" and d.get("valid"):
                score[d["id"]] = d["score"]
    return {m: {q: score.get(j) for q, j in jobs.items()} for m, jobs in plan["mapping"].items()}


def auroc(pairs):
    from sklearn.metrics import roc_auc_score
    y = [p[0] for p in pairs]; s = [p[1] for p in pairs]
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--name", default="qwen25_7b")
    ap.add_argument("--controls", default="v2")
    ap.add_argument("--clear", type=int, default=4, help="Well score at/above which an FPQ answer counts as a clear correction")
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    labels = {l["id"]: l for l in read_jsonl(S / "knowledge" / args.name / "labels.jsonl")}
    know = {q: (l.get("knows_fc") or "not run") for q, l in labels.items()}
    scores = well_scores(S / "well_judge_claude")
    scores.update(well_scores(S / "well_judge_claude_fpu"))
    questions = {q["id"]: q for q in read_jsonl(S / "questions.jsonl")}
    fpq = [q for q in questions if questions[q]["set"] == "fpq"]
    lines = [f"# Knowledge x outcome ({args.name}; clear correction = Well >= {args.clear})", ""]

    # 1. 2x2 on Plain
    plain = scores.get("plain", {})
    cells = Counter()
    for q in fpq:
        s = plain.get(q)
        if s is None:
            continue
        cells[(know.get(q, "not run"), "clear" if s >= args.clear else "fail")] += 1
    lines += ["## 1. knows_fc x Plain outcome (FPQ with a Plain score)", "", "| knows_fc | Plain fail (<= 3) | Plain clear (>= 4) | total |", "|---|---:|---:|---:|"]
    for k in ("yes", "unsure", "no", "not run"):
        f, c = cells[(k, "fail")], cells[(k, "clear")]
        if f + c:
            lines.append(f"| {k} | {f} | {c} | {f + c} |")
    nf = sum(v for (k, o), v in cells.items() if o == "fail")
    ny = cells[("yes", "fail")]
    lines += ["", f"Plain failures {nf}; of these the model knows the myth in isolation (knows_fc=yes) for {ny} ({ny / max(nf, 1):.1%}). "
              "That share is the deference/recognition failure; the rest is knowledge gap or unlabeled.", ""]

    # 2. per method rescue by knowledge cell
    lines += [f"## 2. Clear corrections (>= {args.clear}) among Plain failures, by knows_fc", "",
              "| method | knows=yes (n) | knows=unsure (n) | knows=no (n) | all Plain failures |", "|---|---|---|---|---|"]
    fails = [q for q in fpq if plain.get(q) is not None and plain[q] < args.clear]
    for m in sorted(scores):
        if m == "plain":
            continue
        row = {}
        for k in ("yes", "unsure", "no"):
            ids = [q for q in fails if know.get(q) == k]
            got = [q for q in ids if scores[m].get(q) is not None and scores[m][q] >= args.clear]
            row[k] = f"{len(got)}/{len(ids)} ({len(got) / max(len(ids), 1):.1%})"
        allg = [q for q in fails if scores[m].get(q) is not None and scores[m][q] >= args.clear]
        lines.append(f"| {m} | {row['yes']} | {row['unsure']} | {row['no']} | {len(allg)}/{len(fails)} ({len(allg) / max(len(fails), 1):.1%}) |")
    lines.append("")

    # 3. gate AUROC by knowledge subset
    res_path = S / "style_controls" / args.controls / "result.json"
    if res_path.exists():
        preds = json.load(open(res_path))["predictions"]
        lines += [f"## 3. Gate AUROC restricted by the origin FPQ's knows_fc (style_controls/{args.controls})", "",
                  "Positives whose origin FPQ is knows_fc=yes vs the rest; negatives (NFP / true twins / paraphrased NFP) kept in both. "
                  "If hidden reads 'the model knows but is not applying it', its AUROC should be higher in the yes subset than text's.", "",
                  "| condition | signal | AUROC yes-subset (n) | AUROC rest-subset (n) | all |", "|---|---|---|---|---|"]
        by = defaultdict(list)
        for p in preds:
            by[(p["condition"], p["signal"])].append(p)
        for (cond, sig), ps in sorted(by.items()):
            if cond not in ("natural", "twins", "edited", "para", "natural->twins", "natural->edited", "natural->para"):
                continue
            def subset(keep):
                out = []
                for p in ps:
                    origin = p["origin"]
                    if p["label"] == 1:
                        k = know.get(origin, "not run")
                        if (k == "yes") != keep:
                            continue
                    out.append((p["label"], p["score"]))
                return out
            a_yes, a_rest, a_all = subset(True), subset(False), [(p["label"], p["score"]) for p in ps]
            fmt = lambda v, n: f"{v:.3f} ({n})" if v is not None else f"NA ({n})"
            lines.append(f"| {cond} | {sig} | {fmt(auroc(a_yes), sum(1 for l, _ in a_yes if l == 1))} | {fmt(auroc(a_rest), sum(1 for l, _ in a_rest if l == 1))} | {fmt(auroc(a_all), len(a_all))} |")
    else:
        lines.append(f"(style_controls/{args.controls}/result.json not found; section 3 skipped)")
    out = S / "knowledge" / args.name / "2x2.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Written to {out}. Judge calls: 0.")


if __name__ == "__main__":
    main()
