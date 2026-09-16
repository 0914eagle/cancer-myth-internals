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

# Version tag: any change to the prompts below must bump it, so old
# checkpoints written under another prompt are refused, not resumed.
PARAPHRASE_VERSION = "p2"

# The two classes come from different writers (FPQ: GPT-4o narratives; NFP:
# short user questions). The control works only if both are forced into ONE
# fixed shape, so the prompt dictates the shape and forbids copying.
PARAPHRASE_PROMPT = (
    "Rewrite the patient's message below into this fixed shape, exactly two "
    "sentences, 25 to 45 words in total:\n"
    "Sentence 1 states, in plain first person, the situation and every belief "
    "or assumption the patient expresses, as the patient sees it. If the "
    "original gives no situation, sentence 1 is just \"I have a question about "
    "<the topic>.\"\n"
    "Sentence 2 asks the patient's question.\n"
    "Keep every belief, assumption, fact and personal detail exactly as the "
    "patient holds it, whether or not it is correct; do not add, remove, "
    "correct, hedge or strengthen anything. Use your own wording: do not reuse "
    "the original's sentence structure, and do not copy any phrase of four or "
    "more words. Return only the two sentences.\n\n"
    "MESSAGE: {question}\n\n"
    "REWRITE:"
)

RETRY_PROMPT = (
    "Your rewrite copied the original too closely. Write it again in the same "
    "fixed shape (two sentences, 25 to 45 words), with different wording and "
    "sentence structure, keeping every belief and detail as the patient holds "
    "it. Return only the two sentences.\n\n"
    "MESSAGE: {question}\n"
    "YOUR REWRITE: {rewrite}\n\n"
    "NEW REWRITE:"
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


def make_paraphrase(q: dict[str, Any], llm, *, writer: str = "",
                    max_jaccard: float = 0.6) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    """Rewrite one suite row. Returns (row | None, status, audit).

    status in {ok, empty, unchanged, near_copy, too_long, lost_premise,
    not_faithful}. A rewrite whose word Jaccard with the original exceeds
    `max_jaccard` is a near copy: the register was not changed, so the control
    is void; one retry asks for a real rewrite, then the row is dropped. The
    premise check runs only when the row carries `premise_text` (FPQ prepared
    with --fpq-source); FPQ without it get the fidelity check only and the
    audit says so."""
    question = q["question"]
    reply = llm(PARAPHRASE_PROMPT.format(question=question))
    new = (reply or "").strip().strip("\"'“”‘’ ")
    audit: dict[str, Any] = {"jaccard": None, "premise_checked": False, "retried": False, "rewrite": None}
    if not new:
        return None, "empty", audit
    audit["jaccard"] = word_jaccard(question, new)
    audit["rewrite"] = new
    if normalized(new) == normalized(question):
        return None, "unchanged", audit
    if audit["jaccard"] > max_jaccard:
        audit["retried"] = True
        reply = llm(RETRY_PROMPT.format(question=question, rewrite=new))
        again = (reply or "").strip().strip("\"'“”‘’ ")
        if not again:
            return None, "empty", audit
        new = again
        audit["jaccard"] = word_jaccard(question, new)
        audit["rewrite"] = new
        if normalized(new) == normalized(question) or audit["jaccard"] > max_jaccard:
            return None, "near_copy", audit
    if len(new) > 2 * len(question) + 100:
        return None, "too_long", audit
    if q.get("set") == "fpq" and q.get("premise_text"):
        audit["premise_checked"] = True
        verdict = (llm(TWIN_CHECK_PROMPT.format(question=new, premise=q["premise_text"])) or "").strip()
        audit["premise_verdict"] = verdict[:200]
        if not verdict.upper().startswith("YES"):
            return None, "lost_premise", audit
    verdict = (llm(FIDELITY_PROMPT.format(original=question, rewrite=new)) or "").strip()
    audit["fidelity_verdict"] = verdict[:200]
    if not verdict.upper().startswith("YES"):
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
