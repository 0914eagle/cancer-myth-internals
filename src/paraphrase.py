"""Register-matched paraphrases: the second style control.

Every suite question (FPQ and NFP alike) is rewritten by ONE writer in one
plain register, keeping every belief the patient expresses. Because both
classes pass through the same writer, provenance style is removed while
the false premise (or its absence) stays. A gate that reads register loses
its edge on the paraphrases; a gate that reads the premise keeps it.

Two checks per row: for FPQ, the false belief must still be taken for
granted (same YES/NO check as the twins); for every row, the rewrite must
express the same beliefs and ask for the same thing. Rows failing either
check are dropped, and the audit counts say how many.
"""

from __future__ import annotations

import re
from typing import Any

from src.pilot import normalized
from src.rows import TWIN_CHECK_PROMPT

PARAPHRASE_PROMPT = (
    "Rewrite the patient's message below as a short, plain message from a "
    "patient to a doctor, in a neutral everyday register. Keep every belief, "
    "assumption, fact and personal detail the patient expresses exactly as "
    "they are, whether or not they are correct, and keep the same request. Do "
    "not add, remove, correct, hedge or strengthen anything. Change the "
    "wording and sentence structure freely; write one to three sentences. "
    "Return only the rewritten message, nothing else.\n\n"
    "MESSAGE: {question}\n\n"
    "REWRITE:"
)

FIDELITY_PROMPT = (
    "Below are two messages from a patient. Do they express the same beliefs "
    "and assumptions and ask for the same thing, with no fact added, removed, "
    "corrected or qualified? Answer with one word, YES or NO.\n\n"
    "ORIGINAL: {original}\n"
    "REWRITE: {rewrite}\n\n"
    "ANSWER:"
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def word_jaccard(a: str, b: str) -> float:
    wa, wb = set(_WORD.findall((a or "").lower())), set(_WORD.findall((b or "").lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def make_paraphrase(q: dict[str, Any], llm, *, writer: str = "") -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Rewrite one suite row. Returns (row | None, status, audit).

    status in {ok, empty, unchanged, too_long, lost_premise, not_faithful}.
    The premise check runs only when the row carries `premise_text` (FPQ
    prepared with --fpq-source); FPQ without it get the fidelity check only
    and the audit says so."""
    question = q["question"]
    reply = llm(PARAPHRASE_PROMPT.format(question=question))
    new = (reply or "").strip().strip("\"'“”‘’ ")
    audit: dict[str, Any] = {"jaccard": None, "premise_checked": False}
    if not new:
        return None, "empty", audit
    audit["jaccard"] = word_jaccard(question, new)
    if normalized(new) == normalized(question):
        return None, "unchanged", audit
    if len(new) > 2 * len(question) + 100:
        return None, "too_long", audit
    if q.get("set") == "fpq" and q.get("premise_text"):
        audit["premise_checked"] = True
        verdict = (llm(TWIN_CHECK_PROMPT.format(question=new, premise=q["premise_text"])) or "").strip().upper()
        if not verdict.startswith("YES"):
            return None, "lost_premise", audit
    verdict = (llm(FIDELITY_PROMPT.format(original=question, rewrite=new)) or "").strip().upper()
    if not verdict.startswith("YES"):
        return None, "not_faithful", audit
    row = {
        "id": f"{q['id']}_para",
        "set": q["set"],
        "paraphrase_of": q["id"],
        "group_id": q.get("group_id"),
        "partition": q.get("partition"),
        "question": new,
        "writer": writer,
        "premise_text": q.get("premise_text"),
        "correction": q.get("correction"),
    }
    return row, "ok", audit
