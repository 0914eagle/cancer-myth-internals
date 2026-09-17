"""Was Extract+Verify's failure the extraction or the verification? (CPU, judge 0)

    python scripts/check_extraction.py --suite-dir $SUITE_DIR

Reads model/generation/extract_verify cache: per question the extracted premises
and the True/False readout on each. For FPQ, does any extracted premise overlap
the annotated myth (word Jaccard >= threshold)? How often is the overlapping
premise judged False (correct) vs True? For NFP, how often is anything judged
False (the negative bias)? Prints a table and 8 examples.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import baseline_generation as bg
from src.jsonl import read_jsonl

_W = re.compile(r"[a-z0-9]+")
STOP = set("a an the of in on at to for from with by is are was were be been being it its this that these those and or but if as than then so do does did has have had can could may might will would should must not no i my me we our you your he she they them their there here what how why which who when where".split())


def content(text):
    return {w for w in _W.findall((text or "").lower()) if w not in STOP}


def jaccard(a, b):
    a, b = content(a), content(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()
    S = Path(args.suite_dir).resolve()
    q = {r["id"]: r for r in read_jsonl(S / "questions.jsonl")}
    recs = bg.load_generation_records(S / "model", "extract_verify")
    fpq_rows, nfp_rows, examples = [], [], []
    for r in recs:
        row = q.get(r["id"])
        if not row:
            continue
        prem = r.get("details", {}).get("premises") or []
        if row["set"] == "fpq":
            myth = row.get("premise_text") or ""
            best = max(((jaccard(p["premise"], myth), p) for p in prem), key=lambda x: x[0], default=(0.0, None))
            hit = best[1] is not None and best[0] >= args.threshold
            flagged_hit = hit and not best[1]["predicted_true"]
            any_flag = any(not p["predicted_true"] for p in prem)
            fpq_rows.append((len(prem), hit, flagged_hit, any_flag))
            if len(examples) < 8 and prem:
                examples.append((r["id"], myth, [(p["premise"][:90], p["predicted_true"], round(p["true_score"], 2)) for p in prem[:4]], round(best[0], 2)))
        else:
            nfp_rows.append((len(prem), any(not p["predicted_true"] for p in prem)))
    n = len(fpq_rows)
    zero = sum(1 for k, *_ in fpq_rows if k == 0)
    hit = sum(1 for _, h, *_ in fpq_rows if h)
    fh = sum(1 for _, _, f, _ in fpq_rows if f)
    af = sum(1 for *_, a in fpq_rows if a)
    m = len(nfp_rows)
    nzero = sum(1 for k, _ in nfp_rows if k == 0)
    nflag = sum(1 for _, a in nfp_rows if a)
    print(f"FPQ {n}: no premise extracted {zero} ({zero/n:.1%}); myth found among extracted premises (Jaccard>={args.threshold}) {hit} ({hit/n:.1%}); "
          f"of those judged False {fh} ({fh/max(hit,1):.1%}); any premise judged False {af} ({af/n:.1%})")
    print(f"NFP {m}: no premise extracted {nzero} ({nzero/m:.1%}); any premise judged False {nflag} ({nflag/m:.1%})")
    print(f"mean premises per question: FPQ {sum(k for k,*_ in fpq_rows)/n:.2f}, NFP {sum(k for k,_ in nfp_rows)/m:.2f}")
    print("\nExamples (id | myth | extracted [premise, predicted_true, P(True)] | best Jaccard):")
    for e in examples:
        print("-", e[0], "|", e[1][:80], "|", e[2], "|", e[3])


if __name__ == "__main__":
    main()
