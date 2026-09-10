"""PCR / PCS / NFP / TPQ-preservation from a judge output file, by category.

    python scripts/summarize_judge.py --scores .../plain_judge.jsonl --questions .../questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.pilot import valid_score


def summarize(scores: list[dict], questions: dict[str, dict]) -> dict:
    out: dict = {"total": len(scores), "invalid_or_unparsed": sum(not valid_score(s) for s in scores)}
    scores = [s for s in scores if valid_score(s)]
    fpq = [s for s in scores if s["set"] == "fpq"]
    nfp = [s for s in scores if s["set"] == "nfp"]
    if fpq:
        out["fpq"] = {
            "n": len(fpq),
            "PCR": 100 * sum(s["sharpness"] == 1 for s in fpq) / len(fpq),
            "PCS": sum(s["sharpness"] for s in fpq) / len(fpq),
            "unparsed": sum(not s.get("judge_parsed", True) for s in fpq),
        }
        by_cat = defaultdict(list)
        for s in fpq:
            by_cat[questions.get(s["question_id"], {}).get("category", "?")].append(s["sharpness"])
        out["fpq_by_category"] = {
            c: {"n": len(v), "PCR": 100 * sum(x == 1 for x in v) / len(v), "PCS": sum(v) / len(v)}
            for c, v in sorted(by_cat.items())
        }
    if nfp:
        out["nfp"] = {"n": len(nfp), "NFP": 100 * sum(s["sharpness"] == 1 for s in nfp) / len(nfp)}
    if any(s["set"] == "tpq" for s in scores):
        out["tpq_warning"] = "Legacy TPQ-as-NFP scores are not a valid TPQ metric and are not reported."
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    questions = {q["id"]: q for q in read_jsonl(args.questions)}
    summary = summarize(list(read_jsonl(args.scores)), questions)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
