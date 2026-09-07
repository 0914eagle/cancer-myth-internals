"""Position-E rows (response opening, teacher-forced) with PCR / NFP labels.

Joins plain responses and judge scores onto the question table, writes

    response_rows.jsonl   assistant_prefix extraction rows (E)
    labels.jsonl          {id (question id), set, pcr, nfp_score}

Labels are also merged into a copy of activation_rows.jsonl so that every
manifest written afterwards carries `pcr` for the C direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl
from src.rows import response_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--activation-rows", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix-tokens", type=int, default=5)
    args = parser.parse_args()

    questions = list(read_jsonl(args.questions))
    responses_by_text = {r["question"]: r["response"] for r in read_jsonl(args.responses)}
    responses = {q["id"]: responses_by_text.get(q["question"]) for q in questions}

    labels: dict[str, dict] = {}
    for s in read_jsonl(args.scores):
        qid = s["question_id"]
        entry = labels.setdefault(qid, {"id": qid, "set": s["set"], "judge_parsed": s.get("judge_parsed", True)})
        if s["rubric"] == "fpq":
            entry["pcr"] = int(s["sharpness"])
        else:
            entry["nfp_score"] = int(s["sharpness"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "labels.jsonl", labels.values())

    def with_labels(row: dict) -> dict:
        lab = labels.get(row.get("base_id", row["id"]), {})
        return {**row, **{k: lab[k] for k in ("pcr", "nfp_score", "judge_parsed") if k in lab}}

    labelled_questions = [with_labels(q) for q in questions]
    e_rows = response_rows(labelled_questions, responses, prefix_tokens=args.prefix_tokens)
    write_jsonl(out_dir / "response_rows.jsonl", e_rows)
    write_jsonl(out_dir / "activation_rows.jsonl", [with_labels(r) for r in read_jsonl(args.activation_rows)])
    n_pcr = sum("pcr" in v for v in labels.values())
    print(f"[rows] labels={len(labels)} (pcr={n_pcr}) response_rows={len(e_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()
