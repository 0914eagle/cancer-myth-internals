"""Build the E1 row set: Cancer-Myth FPQ 585, NFP 150, Well TPQ, with premise spans.

Three question sets, one row format:

    id                   fpq_<QID> / nfp_<QID> / tpq_<id>
    set                  fpq | nfp | tpq
    label_false_premise  1 for fpq, 0 for nfp and tpq
    question             the patient question, verbatim
    premise_text         the presupposition as a standalone statement
                         (fpq: HF `source_myth` or GitHub `source_info.myth`;
                          tpq: Well's annotated true presupposition; nfp: none)
    correction           fpq only: physician-validated explanation (judge input)
    hallucination_text   nfp only: the presupposition the LLM wrongly saw
                         (validate_nfp.py's "Possible hallucination")
    premise_span         [start, end) character span of the premise *inside the
                         question*, or null when no alignment was found
    align_score, align_method

The premise span is what position A (premise last token) is read at. The
premise statement is a paraphrase of what the question presupposes, never a
verbatim substring, so alignment is a judgement: an LLM is asked for the exact
substring that carries the presupposition (verified by exact match), with a
content-word-overlap heuristic as the offline fallback. Both record their
method, so a downstream table can be restricted to high-confidence spans.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .jsonl import load_json

STOPWORDS = set(
    """
    a an the and or but if then so of to in on at for from by with without about as
    is are was were be been being am do does did done have has had having will would
    can could should may might must shall this that these those it its i my me we our
    you your he she his her they them their there here what which who whom whose when
    where why how not no yes any some all every each both either neither very more most
    much many few little less least own same other another such only just also too
    into onto over under again further once than because while during before after
    above below between through up down out off since until although though
    """.split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def content_words(text: str) -> set[str]:
    words = set()
    for match in _WORD.finditer(text.lower()):
        w = match.group(0).strip("-'")
        if len(w) < 3 or w in STOPWORDS:
            continue
        words.add(stem(w))
    return words


def stem(word: str) -> str:
    """A very small stemmer: enough to match 'treatments' with 'treatment'."""
    for suffix in ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: -len(suffix)]
            if suffix == "ies":
                base += "y"
            return base
    return word


_CLAUSE_SPLIT = re.compile(r"(?<=[.?!;:])\s+|,\s+(?=(?:and|but|so|because|which|who|that)\b)")


def clause_spans(question: str) -> list[tuple[int, int]]:
    """Character spans of clause-sized windows, plus sentence-sized ones.

    The premise usually lives in one clause ("he believes surgery is the only
    way to treat it"), sometimes in a whole sentence. Both granularities are
    candidates; the best-scoring one wins.
    """
    spans: list[tuple[int, int]] = []
    # Sentences.
    start = 0
    for match in re.finditer(r"[.?!]\s+", question):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(question):
        spans.append((start, len(question)))
    # Clauses within the whole question.
    start = 0
    for match in _CLAUSE_SPLIT.finditer(question):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(question):
        spans.append((start, len(question)))
    # Trim whitespace/punctuation at both ends, drop empties, dedupe.
    cleaned: list[tuple[int, int]] = []
    seen = set()
    for s, e in spans:
        while s < e and question[s] in " \t\n,;:":
            s += 1
        while e > s and question[e - 1] in " \t\n":
            e -= 1
        if e - s >= 8 and (s, e) not in seen:
            seen.add((s, e))
            cleaned.append((s, e))
    return cleaned


def align_premise_heuristic(question: str, premise: str) -> tuple[int, int, float] | None:
    """Best clause by content-word overlap with the premise statement.

    Score = |premise words ∩ window words| / |premise words|, with a length
    penalty so a whole sentence does not always beat the clause inside it.
    """
    target = content_words(premise)
    if not target:
        return None
    best: tuple[float, int, int] | None = None
    for s, e in clause_spans(question):
        window = content_words(question[s:e])
        if not window:
            continue
        overlap = len(target & window) / len(target)
        penalty = 0.02 * max(0, len(window) - len(target))
        score = overlap - penalty
        if best is None or score > best[0]:
            best = (score, s, e)
    if best is None or best[0] < 0.25:
        return None
    score, s, e = best
    return s, e, round(score, 3)


def find_verbatim(question: str, snippet: str) -> tuple[int, int] | None:
    """Locate an LLM-returned snippet in the question, tolerant of quotes and
    whitespace differences, never of a paraphrase."""
    snippet = snippet.strip().strip("\"'“”‘’ ")
    if not snippet:
        return None
    idx = question.find(snippet)
    if idx >= 0:
        return idx, idx + len(snippet)
    idx = question.lower().find(snippet.lower())
    if idx >= 0:
        return idx, idx + len(snippet)
    pattern = r"\s+".join(re.escape(w) for w in snippet.split())
    match = re.search(pattern, question, flags=re.IGNORECASE)
    if match:
        return match.start(), match.end()
    return None


LLM_ALIGN_PROMPT = (
    "A patient question contains a presupposition, stated separately below. "
    "Return the shortest contiguous substring of the QUESTION, copied exactly "
    "character for character, that expresses that presupposition. Do not "
    "paraphrase, do not add quotes, do not explain. If no substring expresses "
    "it, return NONE.\n\n"
    "QUESTION: {question}\n"
    "PRESUPPOSITION: {premise}\n\n"
    "SUBSTRING:"
)


def align_premise(
    question: str, premise: str | None, *, llm=None
) -> tuple[list[int] | None, float | None, str]:
    """Returns (span, score, method). method in {llm, heuristic, none}."""
    if not premise:
        return None, None, "none"
    if llm is not None:
        reply = llm(LLM_ALIGN_PROMPT.format(question=question, premise=premise))
        if reply and reply.strip().upper() != "NONE":
            found = find_verbatim(question, reply)
            if found is not None:
                return [found[0], found[1]], 1.0, "llm"
    result = align_premise_heuristic(question, premise)
    if result is None:
        return None, None, "none"
    s, e, score = result
    return [s, e], score, "heuristic"


# --- loaders -----------------------------------------------------------------


def _usable_myth(text: str | None) -> bool:
    return bool(text) and len(text) >= 15 and "physician" not in text.lower()


def load_fpq(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cancer-Myth 585. HF copy first (has `source_myth`), GitHub JSON as fallback."""
    rows: list[dict[str, Any]] = []
    try:
        import datasets

        ds = datasets.load_dataset(cfg["data"]["hf_cancer_myth"], split="validation")
        for i, item in enumerate(ds):
            qid = item.get("QID", i)
            rows.append(
                {
                    "id": f"fpq_{qid}",
                    "set": "fpq",
                    "label_false_premise": 1,
                    "question": str(item["question"]).strip(),
                    "premise_text": item.get("source_myth"),
                    "correction": item.get("presupposition_correction"),
                    "category": item.get("category"),
                    "cancer": item.get("cancer"),
                    "from_model": item.get("from_model"),
                    "source": "hf",
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001 - fall back to the GitHub copy
        print(f"[fpq] HF dataset unavailable ({exc!r}); using GitHub all_data.json", flush=True)
    path = Path(cfg["data"]["cancer_myth_repo"]) / "data" / "all_data.json"
    for item in load_json(path):
        myth = (item.get("source_info") or {}).get("myth")
        rows.append(
            {
                "id": f"fpq_{item['QID']}",
                "set": "fpq",
                "label_false_premise": 1,
                "question": str(item["example_question"]).strip(),
                "premise_text": myth if _usable_myth(myth) else None,
                "correction": item.get("example_assumption"),
                "category": item.get("category"),
                "cancer": (item.get("source_info") or {}).get("cancer"),
                "from_model": item.get("from_model"),
                "source": "github",
            }
        )
    return rows


def load_nfp(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cancer-Myth-NFP 150 from the GitHub copy (nfp.json)."""
    path = Path(cfg["data"]["cancer_myth_repo"]) / "data" / "nfp.json"
    rows = []
    for item in load_json(path):
        rows.append(
            {
                "id": f"nfp_{item['QID']}",
                "set": "nfp",
                "label_false_premise": 0,
                "question": str(item["example_question"]).strip(),
                "premise_text": None,
                "hallucination_text": item.get("example_assumption"),
                "cancer": item.get("source_cancer"),
                "from_model": item.get("from_model"),
                "source": "github",
            }
        )
    return rows


def _first_presupposition(item: dict[str, Any]) -> str | None:
    for key in ("presuppositions", "example_presuppositions", "presupposition"):
        value = item.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
        if isinstance(value, str) and value.strip():
            return value
    return None


def load_tpq(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Well's TPQ: NFP questions with a human-annotated *true* presupposition.

    Field names are read defensively because the HF card is not reachable
    from the machine this was written on; `scripts/make_rows.py` prints the
    columns it found so the mapping can be checked once on the server.
    """
    rows: list[dict[str, Any]] = []
    try:
        import datasets

        name = cfg["data"]["hf_tpq"]
        dsd = datasets.load_dataset(name)
        splits = list(dsd.keys())
        print(f"[tpq] {name} splits={splits} columns={dsd[splits[0]].column_names}", flush=True)
        for split in splits:
            for i, item in enumerate(dsd[split]):
                premise = _first_presupposition(item)
                if not premise:
                    continue
                qid = item.get("id", item.get("QID", f"{split}{i}"))
                rows.append(
                    {
                        "id": f"tpq_{qid}",
                        "set": "tpq",
                        "label_false_premise": 0,
                        "question": str(item.get("question") or item.get("example_question")).strip(),
                        "premise_text": premise,
                        "cancer": item.get("cancer"),
                        "tpq_split": split,
                        "source": "hf",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[tpq] HF dataset unavailable ({exc!r}); TPQ rows skipped", flush=True)
    return rows


# --- activation rows ---------------------------------------------------------

PASSTHROUGH = [
    "set",
    "label_false_premise",
    "category",
    "cancer",
    "from_model",
    "premise_text",
    "premise_span",
    "align_score",
    "align_method",
    "tpq_split",
]


def activation_rows(question_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One question -> up to three extraction rows sharing one forward pass.

    __last  position D  last prompt token (assistant turn about to start)
    __qend  position B  last token of the question text
    __prem  position A  last token of the premise span, plus its mean (C-mean)
    """
    out: list[dict[str, Any]] = []
    for q in question_rows:
        common = {k: q.get(k) for k in PASSTHROUGH}
        base = q["id"]
        out.append(
            {
                "id": f"{base}__last",
                "base_id": base,
                "prompt": q["question"],
                "position_mode": "last_token",
                "position_family": "D_last_prompt_token",
                **common,
            }
        )
        out.append(
            {
                "id": f"{base}__qend",
                "base_id": base,
                "prompt": q["question"],
                "position_mode": "target_text",
                "target_text": q["question"],
                "target_text_strategy": "last_subtoken",
                "position_family": "B_question_end",
                **common,
            }
        )
        span = q.get("premise_span")
        if span:
            s, e = int(span[0]), int(span[1])
            out.append(
                {
                    "id": f"{base}__prem",
                    "base_id": base,
                    "prompt": q["question"],
                    "position_mode": "target_text",
                    "target_text": q["question"][s:e],
                    "target_text_strategy": "last_subtoken",
                    "position_family": "A_premise",
                    **common,
                }
            )
    return out


def response_rows(
    question_rows: list[dict[str, Any]], responses: dict[str, str], *, prefix_tokens: int = 5
) -> list[dict[str, Any]]:
    """Position E: the first `prefix_tokens` of the model's own response,
    teacher-forced, plus the response's last token for a later H4 check."""
    out = []
    for q in question_rows:
        resp = responses.get(q["id"])
        if not resp:
            continue
        common = {k: q.get(k) for k in PASSTHROUGH}
        messages = [
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": resp},
        ]
        out.append(
            {
                "id": f"{q['id']}__resp{prefix_tokens}",
                "base_id": q["id"],
                "chat_messages": messages,
                "position_mode": "assistant_prefix",
                "prefix_tokens": prefix_tokens,
                "position_family": f"E_response_first{prefix_tokens}",
                **common,
            }
        )
    return out
