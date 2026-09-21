"""Truth x expression 2x2 control (9/21 review): does a FIXED gate move with the
medical truth of the premise span, with its expression (assertive vs hedged),
or with neither (topic only)?

Cells, all spliced into the same question at the aligned premise span:
  FA  false belief, stated as fact        FH  false belief, hedged (hearsay / tentative)
  TA  correct belief, stated as fact      TH  correct belief, hedged
Hedging must change the speaker's stance only, never the claim: "sugar always
feeds tumours" -> "I've read that sugar feeds tumours" keeps the false claim;
"sugar may or may not feed tumours" does not and is rejected by the check.

Pure helpers here (no LLM, no torch); scripts/truth_form_2x2.py drives them.
"""

from __future__ import annotations

import json
import re

import numpy as np

CELLS = ("FA", "FH", "TA", "TH")          # v1 free-form design (docs/35: not a clean truth x expression test)
TRUTH = {"FA": "FALSE", "FH": "FALSE", "TA": "TRUE", "TH": "TRUE"}
STANCE = {"FA": "ASSERTIVE", "FH": "HEDGED", "TA": "ASSERTIVE", "TH": "HEDGED"}

# Assembled design (docs/35 §4): one target proposition pair F/T written first, then the SAME
# source phrase prepended to both; the claim string is identical inside a truth level.
#   FD false / direct     FS false / source phrase     TD true / direct     TS true / source phrase
DESIGNS = {
    "freeform": {"cells": ("FA", "FH", "TA", "TH"), "factor2": ("assertive", "hedged"), "label": "truth x stance (free-form spans, v1)"},
    "assembled": {"cells": ("FD", "FS", "TD", "TS"), "factor2": ("direct", "source-phrase"), "label": "truth x source-phrase (fixed propositions, v2)"},
}
SOURCE_PHRASE = "I've read that "
FRAME_PREFIX = "Given that "

PROP_PROMPT = (
    "A patient's question is given with one stretch marked between [[ and ]] that carries a FALSE medical belief. "
    "Write a JSON object with three strings:\n"
    "F: the false belief as ONE short declarative clause that can follow 'Given that ' (e.g. 'bronchial tumors affect only older people'). "
    "Use the belief exactly as the patient holds it; no hedging, no explanation.\n"
    "T: the corrected belief as ONE clause on the SAME target claim, changed as little as possible (ideally by negation or the "
    "minimal factual substitution), e.g. 'bronchial tumors do not affect only older people'. Do not add drug names, mechanisms, "
    "or advice; do not change which claim is at stake.\n"
    "TAIL: the rest of the patient's question as a single request that does NOT depend on the belief being true or false: keep the "
    "patient's own situation (diagnosis, family, treatment already given) and the kind of help they ask for exactly as in the "
    "original; remove any leftover wording that presupposes the false belief. It must read naturally after 'Given that <F>, ' "
    "and after 'Given that <T>, ' alike. Start it with a lowercase letter and end with a question mark.\n\n"
    "Return ONLY the JSON object with keys F, T, TAIL.\n\n"
    "QUESTION: {marked}\n"
    "FALSE BELIEF (reference): {premise}\n"
    "CORRECT BELIEF (reference): {correction}\n\n"
    "JSON:"
)

PAIR_CHECK_PROMPT = (
    "Two clauses and a question tail are given. Judge:\n"
    "SAME_CLAIM: YES if F and T are about the same target claim and differ only in its truth (negation or a minimal factual "
    "substitution), NO if they change the subject, scope, or add other content.\n"
    "F_FALSE: YES if F states the false belief below, NO otherwise.\n"
    "T_TRUE: YES if T states the correct belief below (and nothing beyond it), NO otherwise.\n"
    "TAIL_NEUTRAL: YES if the tail keeps the patient's situation and request and does not itself presuppose the false belief "
    "or the correct belief, NO otherwise.\n\n"
    "FALSE BELIEF (reference): {premise}\n"
    "CORRECT BELIEF (reference): {correction}\n"
    "F: {f}\nT: {t}\nTAIL: {tail}\n\n"
    "Answer exactly: SAME_CLAIM=<YES|NO>; F_FALSE=<YES|NO>; T_TRUE=<YES|NO>; TAIL_NEUTRAL=<YES|NO>"
)

GEN_PROMPT = (
    "A patient's question is given with one stretch marked between [[ and ]]. That stretch carries a belief "
    "about a medical matter. Write FOUR replacement texts for the bracketed stretch only. Everything outside the "
    "brackets stays exactly as it is, so each replacement must fit grammatically in that slot, in the same voice, "
    "about the same situation, with similar length.\n\n"
    "FA: the patient states the FALSE belief below as a plain fact (certain, no hedging).\n"
    "FH: the patient conveys the SAME FALSE belief, but as something heard, read, or assumed with some uncertainty "
    "(e.g. 'I've read that ...', 'I was told ...', 'as far as I know ...'). The content of the belief must stay the "
    "false one; do not soften the claim itself into something that could be true.\n"
    "TA: the patient states the CORRECT belief below as a plain fact.\n"
    "TH: the patient conveys the SAME CORRECT belief as something heard, read, or assumed with some uncertainty.\n\n"
    "Return ONLY a JSON object with keys FA, FH, TA, TH and the four replacement strings, nothing else.\n\n"
    "QUESTION: {marked}\n"
    "FALSE BELIEF: {premise}\n"
    "CORRECT BELIEF: {correction}\n\n"
    "JSON:"
)

CHECK_PROMPT = (
    "Read the patient question and judge two things about the belief it conveys regarding the matter below.\n\n"
    "CLAIM: FALSE if the question states or takes for granted the false belief; TRUE if it conveys the correct "
    "belief; NEITHER if it conveys neither, is ambiguous, or has been weakened so that it no longer commits to "
    "either.\n"
    "STANCE: ASSERTIVE if the belief is stated as a plain fact or certainty; HEDGED if it is presented as hearsay, "
    "a possibility, an assumption, or with uncertainty.\n\n"
    "FALSE BELIEF: {premise}\n"
    "CORRECT BELIEF: {correction}\n"
    "QUESTION: {question}\n\n"
    "Answer exactly in the form CLAIM=<FALSE|TRUE|NEITHER>; STANCE=<ASSERTIVE|HEDGED>"
)


def parse_variants(reply):
    """JSON object with the four cells -> dict, or None. Tolerates code fences and prose around it."""
    if not isinstance(reply, str):
        return None
    text = reply.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or set(CELLS) - set(data):
        return None
    out = {c: str(data[c]).strip().strip("\"'“”‘’[] ") for c in CELLS}
    return out if all(out.values()) else None


def parse_verdict(reply):
    """'CLAIM=FALSE; STANCE=HEDGED' -> ('FALSE', 'HEDGED'); None when not parseable."""
    if not isinstance(reply, str):
        return None
    claim = re.search(r"CLAIM\s*=\s*(FALSE|TRUE|NEITHER)", reply, re.I)
    stance = re.search(r"STANCE\s*=\s*(ASSERTIVE|HEDGED)", reply, re.I)
    if not claim or not stance:
        return None
    return claim.group(1).upper(), stance.group(1).upper()


def cell_ok(cell, verdict):
    return verdict is not None and verdict == (TRUTH[cell], STANCE[cell])


def parse_props(reply):
    """JSON with F, T, TAIL -> dict or None."""
    if not isinstance(reply, str):
        return None
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or {"F", "T", "TAIL"} - set(data):
        return None
    out = {k: str(data[k]).strip().strip("\"'“”‘’[] ").rstrip(".") for k in ("F", "T")}
    out["TAIL"] = str(data["TAIL"]).strip().strip("\"'“”‘’[] ")
    if not all(out.values()) or out["F"].lower() == out["T"].lower():
        return None
    return out


def parse_pair_verdict(reply):
    """'SAME_CLAIM=YES; F_FALSE=YES; T_TRUE=YES; TAIL_NEUTRAL=NO' -> dict of bools, or None."""
    if not isinstance(reply, str):
        return None
    out = {}
    for key in ("SAME_CLAIM", "F_FALSE", "T_TRUE", "TAIL_NEUTRAL"):
        m = re.search(key + r"\s*=\s*(YES|NO)", reply, re.I)
        if not m:
            return None
        out[key] = m.group(1).upper() == "YES"
    return out


def assemble(props, source=SOURCE_PHRASE, prefix=FRAME_PREFIX):
    """Four questions from one proposition pair and a fixed tail. The claim string is
    identical within a truth level; the source phrase is the only difference D vs S."""
    tail = props["TAIL"]
    tail = tail[0].lower() + tail[1:] if tail else tail

    def lower(c):
        return c[0].lower() + c[1:]
    return {"FD": f"{prefix}{lower(props['F'])}, {tail}", "FS": f"{prefix}{source}{lower(props['F'])}, {tail}",
            "TD": f"{prefix}{lower(props['T'])}, {tail}", "TS": f"{prefix}{source}{lower(props['T'])}, {tail}"}


def splice(question, span, replacement):
    s, e = int(span[0]), int(span[1])
    return question[:s] + replacement + question[e:], [s, s + len(replacement)]


def two_by_two(scores, cells=CELLS):
    """scores: {origin: {cell: value}} with all four cells, ordered (F1, F2, T1, T2) where 1/2 is
    the second factor (stance or source phrase). Returns
      truth effect      = mean over origins of mean(F cells) - mean(T cells)
      expression effect = mean over origins of mean(*1 cells) - mean(*2 cells)
      AUROC truth | 1 : F1 vs T1 ; truth | 2 : F2 vs T2 ; expression | F : F1 vs F2 ; expression | T : T1 vs T2
      variance shares   : SS_truth, SS_expr, SS_interaction over the within-origin deviations. With four cells
                          these three contrasts are saturated (residual is identically 0), so the shares describe
                          how the within-origin variation splits, not how completely it is explained."""
    from sklearn.metrics import roc_auc_score
    origins = [o for o, v in scores.items() if all(c in v and np.isfinite(v[c]) for c in cells)]
    if len(origins) < 3:
        raise ValueError("Need at least three complete origins")
    M = np.array([[scores[o][c] for c in cells] for o in origins], dtype=float)  # (n, 4) F1 F2 T1 T2
    truth = (M[:, 0] + M[:, 1]) / 2 - (M[:, 2] + M[:, 3]) / 2
    expr = (M[:, 0] + M[:, 2]) / 2 - (M[:, 1] + M[:, 3]) / 2

    def auc(pos, neg):
        return float(roc_auc_score([1] * len(pos) + [0] * len(neg), np.concatenate([pos, neg])))
    aurocs = {"truth|assertive": auc(M[:, 0], M[:, 2]), "truth|hedged": auc(M[:, 1], M[:, 3]),
              "expression|false": auc(M[:, 0], M[:, 1]), "expression|true": auc(M[:, 2], M[:, 3])}
    # paired (within-origin) AUROC: share of origins where the ordering holds (ties count half)
    paired = {"truth|assertive": float(np.mean(np.sign(M[:, 0] - M[:, 2]) * 0.5 + 0.5)),
              "truth|hedged": float(np.mean(np.sign(M[:, 1] - M[:, 3]) * 0.5 + 0.5)),
              "expression|false": float(np.mean(np.sign(M[:, 0] - M[:, 1]) * 0.5 + 0.5)),
              "expression|true": float(np.mean(np.sign(M[:, 2] - M[:, 3]) * 0.5 + 0.5))}
    W = M - M.mean(1, keepdims=True)  # remove origin main effect
    t_sign = np.array([1, 1, -1, -1]) / 2; e_sign = np.array([1, -1, 1, -1]) / 2; i_sign = np.array([1, -1, -1, 1]) / 2
    ss = {"truth": float(np.sum((W @ t_sign) ** 2)), "expression": float(np.sum((W @ e_sign) ** 2)),
          "interaction": float(np.sum((W @ i_sign) ** 2))}
    denom = sum(ss.values()) or 1.0
    shares = {k: v / denom for k, v in ss.items()}
    rng = np.random.default_rng(17)
    boot_t, boot_e = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(origins), len(origins))
        boot_t.append(truth[idx].mean()); boot_e.append(expr[idx].mean())
    return {"n": len(origins), "cells": list(cells), "cell_means": {c: float(M[:, i].mean()) for i, c in enumerate(cells)},
            "truth_effect": float(truth.mean()), "truth_ci": [float(np.percentile(boot_t, 2.5)), float(np.percentile(boot_t, 97.5))],
            "expression_effect": float(expr.mean()), "expression_ci": [float(np.percentile(boot_e, 2.5)), float(np.percentile(boot_e, 97.5))],
            "auroc": aurocs, "paired_order": paired, "variance_share": shares}


def report_lines(name, r):
    def f(x):
        return f"{x:+.3f}"
    return [f"| {name} | {r['n']} | " + " / ".join(f"{r['cell_means'][c]:+.2f}" for c in r.get("cells", CELLS))
            + f" | {f(r['truth_effect'])} [{f(r['truth_ci'][0])}, {f(r['truth_ci'][1])}]"
            + f" | {f(r['expression_effect'])} [{f(r['expression_ci'][0])}, {f(r['expression_ci'][1])}]"
            + f" | {r['auroc']['truth|assertive']:.2f} / {r['auroc']['truth|hedged']:.2f}"
            + f" | {r['auroc']['expression|false']:.2f} / {r['auroc']['expression|true']:.2f}"
            + f" | {r['variance_share']['truth']:.2f} / {r['variance_share']['expression']:.2f} / {r['variance_share']['interaction']:.2f} |"]
