"""Content-blind style readouts: the floor that text and hidden gates must clear.

FPQ were written by GPT-4o from myths; NFP are questions on which LLMs
false-alarmed. The two sets can differ in register (length, question form,
first-person framing, function-word habits) without any difference in the
belief they carry. A gate that reads that register is a benchmark artifact.

Two readouts see no content words at all:

* `style_vector`   dense counts: length, punctuation, sentence shape,
                   person, opening form, and per-word rates of a fixed
                   function-word list.
* `mask_content`   the question with every content token replaced by `_`
                   (digits by `0`), so a bag of n-grams over it captures
                   function-word patterns and sentence shape only.

Absolutes (only, always, never, cure, ...) are deliberately NOT function
words here: a corrected premise loses them, so they are premise signal, not
register. They are masked with the content.
"""

from __future__ import annotations

import re

import numpy as np

# Determiners, pronouns, prepositions, conjunctions, auxiliaries, modals,
# wh-words, hedges. No absolutes, no medical or topical words.
FUNCTION_WORDS: tuple[str, ...] = (
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "either", "neither",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    "of", "in", "on", "at", "to", "for", "from", "with", "without", "by", "about", "as", "into",
    "over", "under", "after", "before", "between", "through", "during", "since", "until", "than",
    "and", "or", "but", "so", "if", "because", "while", "although", "though", "whether", "unless",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had", "having",
    "can", "could", "may", "might", "will", "would", "shall", "should", "must",
    "not", "no", "nor", "n't",
    "there", "here", "then", "now", "also", "just", "still", "even", "again", "too", "very",
    "really", "quite", "rather", "maybe", "perhaps", "probably", "likely", "usually", "often",
    "sometimes", "much", "many", "more", "most", "less", "few", "other", "another", "such",
    "own", "same", "both", "already", "yet", "ever", "else", "please", "thanks", "hi", "hello",
)
_FUNCTION_SET = frozenset(FUNCTION_WORDS)

OPENERS_WH = ("what", "why", "how", "when", "where", "which", "who")
OPENERS_AUX = ("is", "are", "can", "could", "do", "does", "did", "should", "will", "would", "was", "were", "has", "have")
OPENERS_FIRST = ("i", "my", "i'm", "i've", "i'd", "we", "our")

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)
_SENTENCE = re.compile(r"[.?!]+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text or "")


def _words(toks: list[str]) -> list[str]:
    return [t.lower() for t in toks if t[0].isalpha()]


def style_feature_names() -> list[str]:
    base = [
        "n_chars", "n_words", "n_sentences", "mean_word_len", "mean_sentence_len", "type_token_ratio",
        "frac_long_words", "n_question_marks", "n_exclaim", "n_commas", "n_digits", "n_parens", "n_quotes",
        "frac_capitalized_words", "first_person_rate", "second_person_rate", "third_person_rate",
        "function_word_rate", "opens_wh", "opens_aux", "opens_first_person", "ends_with_question",
        "last_sentence_is_question", "n_clause_marks",
    ]
    return base + [f"fw_{w}" for w in FUNCTION_WORDS]


def style_vector(text: str) -> np.ndarray:
    """Dense, content-blind description of one question. Same length as
    `style_feature_names()`; every entry finite."""
    text = text or ""
    toks = tokens(text)
    words = _words(toks)
    n_words = max(len(words), 1)
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    n_sent = max(len(sentences), 1)
    lower = [w.lower() for w in words]
    first = lower[0] if lower else ""
    stripped = text.rstrip()
    last_sentence = sentences[-1] if sentences else ""
    last_mark = re.findall(r"[.?!]+", text)
    counts = {w: 0 for w in FUNCTION_WORDS}
    for w in lower:
        if w in counts:
            counts[w] += 1
        elif w.endswith("n't"):
            counts["n't"] += 1
    base = [
        len(text),
        len(words),
        len(sentences),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        len(words) / n_sent,
        len(set(lower)) / n_words,
        sum(len(w) >= 7 for w in words) / n_words,
        text.count("?"),
        text.count("!"),
        text.count(","),
        sum(t.isdigit() for t in toks),
        text.count("(") + text.count(")"),
        text.count('"') + text.count("“") + text.count("”"),
        sum(w[0].isupper() for w in words) / n_words,
        sum(w in ("i", "me", "my", "mine", "myself", "we", "us", "our", "i'm", "i've", "i'd") for w in lower) / n_words,
        sum(w in ("you", "your", "yours") for w in lower) / n_words,
        sum(w in ("he", "him", "his", "she", "her", "hers", "they", "them", "their") for w in lower) / n_words,
        sum(w in _FUNCTION_SET for w in lower) / n_words,
        float(first in OPENERS_WH),
        float(first in OPENERS_AUX),
        float(first in OPENERS_FIRST),
        float(stripped.endswith("?")),
        float("?" in last_sentence or (bool(last_mark) and "?" in last_mark[-1])),
        sum(w in ("and", "but", "or", "because", "so", "if", "while", "although") for w in lower) + text.count(";"),
    ]
    rates = [counts[w] / n_words for w in FUNCTION_WORDS]
    vec = np.asarray(base + rates, dtype=np.float64)
    if not np.isfinite(vec).all():
        raise ValueError("Nonfinite style feature")
    return vec


def mask_content(text: str) -> str:
    """Keep function words and punctuation; replace content words with `_`
    and digits with `0`. Case is dropped."""
    out = []
    for t in tokens(text or ""):
        low = t.lower()
        if t[0].isalpha():
            if low in _FUNCTION_SET:
                out.append(low)
            elif low.endswith("n't"):
                out.append("_n't")
            else:
                out.append("_")
        elif t[0].isdigit():
            out.append("0")
        else:
            out.append(t)
    return " ".join(out)
