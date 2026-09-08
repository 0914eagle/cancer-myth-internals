"""Rows for the paired C construction (E1 stage 8).

For every fpq question take, from Cancer-Myth's all_data.json, one reference
answer GPT-4o scored +1 (corrects the premise) and one it scored -1 (follows
it), plus the model's own Plain response. All three are teacher-forced behind
the same prompt, so the corr-minus-follow difference at the response opening
is the correcting-vs-following direction with the question's content
cancelled. 232 of 585 questions have both a +1 and a -1 reference.

    python scripts/make_paired_rows.py --config configs/llama31_8b.yaml \
        --questions $ROWS/questions.jsonl --responses $RES/plain_responses.jsonl \
        --labels $RES/rows/labels.jsonl --output $RES/rows/paired_rows.jsonl

Reference authors are chosen greedily so the author mix on the +1 side
matches the mix on the -1 side as far as the data allows (otherwise corr is
mostly Gemini and follow mostly MDAgents, and the difference carries style).
One question never pairs two answers of the same author.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import load_json, read_jsonl, write_jsonl
from src.rows import paired_rows


def pick_references(all_data: list[dict]) -> tuple[dict[str, dict], Counter]:
    """question text -> {"corr": (author, text), "follow": (author, text)}

    Authors are chosen greedily so that the author mix of the +1 side matches
    the author mix of the -1 side as closely as the data allows; otherwise
    corr would be mostly Gemini and follow mostly MDAgents, and corr minus
    follow would carry those two styles along with the disposition.
    """
    used_corr: Counter = Counter()
    used_follow: Counter = Counter()
    refs, authors = {}, Counter()
    for r in all_data:
        scores, answers = r.get("scores", {}), r.get("answers", {})
        corr_opts = [a for a, s in scores.items() if s == 1 and answers.get(a)]
        follow_opts = [a for a, s in scores.items() if s == -1 and answers.get(a)]
        entry = {}
        if corr_opts:
            a = max(corr_opts, key=lambda x: (used_follow[x] - used_corr[x], x))
            entry["corr"] = (a, answers[a])
        if follow_opts:
            avoid = entry["corr"][0] if "corr" in entry else None
            opts = [b for b in follow_opts if b != avoid] or follow_opts
            b = max(opts, key=lambda x: (used_corr[x] - used_follow[x], x))
            entry["follow"] = (b, answers[b])
        if "corr" in entry:
            used_corr[entry["corr"][0]] += 1
            authors[f"corr:{entry['corr'][0]}"] += 1
        if "follow" in entry:
            used_follow[entry["follow"][0]] += 1
            authors[f"follow:{entry['follow'][0]}"] += 1
        if entry:
            refs[r["example_question"].strip()] = entry
    return refs, authors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[5, 32])
    args = parser.parse_args()

    cfg = load_config(args.config)
    all_data = load_json(Path(cfg["data"]["cancer_myth_repo"]) / "data" / "all_data.json")
    by_text, authors = pick_references(all_data)

    questions = list(read_jsonl(args.questions))
    labels = {r["id"]: r for r in read_jsonl(args.labels)}
    for q in questions:
        lab = labels.get(q["id"], {})
        for k in ("pcr", "nfp_score", "judge_parsed"):
            if k in lab:
                q[k] = lab[k]
    references = {q["id"]: by_text[q["question"].strip()] for q in questions if q["question"].strip() in by_text}
    responses_by_text = {r["question"]: r["response"] for r in read_jsonl(args.responses)}
    own = {q["id"]: responses_by_text.get(q["question"]) for q in questions}

    rows = paired_rows(questions, references, own, prefix_tokens=tuple(args.prefix_tokens))
    write_jsonl(args.output, rows)
    n_fpq = sum(q.get("set") == "fpq" for q in questions)
    n_both = sum(1 for q in questions if len(references.get(q["id"], {})) == 2)
    roles = Counter(r["pair_role"] for r in rows)
    print(f"[pair] fpq={n_fpq} matched={len(references)} with both corr+follow={n_both}; rows={len(rows)} {dict(roles)}")
    print(f"[pair] authors: {dict(authors)}")
    print(f"[pair] -> {args.output}")


if __name__ == "__main__":
    main()
