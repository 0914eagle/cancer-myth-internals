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

CELLS = ("FA", "FH", "TA", "TH")
TRUTH = {"FA": "FALSE", "FH": "FALSE", "TA": "TRUE", "TH": "TRUE"}
STANCE = {"FA": "ASSERTIVE", "FH": "HEDGED", "TA": "ASSERTIVE", "TH": "HEDGED"}

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


def splice(question, span, replacement):
    s, e = int(span[0]), int(span[1])
    return question[:s] + replacement + question[e:], [s, s + len(replacement)]


def two_by_two(scores):
    """scores: {origin: {cell: value}} with all four cells. Returns effects, within-condition
    AUROCs and a per-origin two-way variance decomposition.
      truth effect      = mean over origins of mean(F cells) - mean(T cells)
      expression effect = mean over origins of mean(A cells) - mean(H cells)
      AUROC truth | assertive : FA vs TA ; truth | hedged : FH vs TH
      AUROC expression | false : FA vs FH ; expression | true : TA vs TH
      variance shares   : SS_truth, SS_expr, SS_interaction, SS_residual over the within-origin
                          deviations (origin main effect removed), as fractions of their sum."""
    from sklearn.metrics import roc_auc_score
    origins = [o for o, v in scores.items() if all(c in v and np.isfinite(v[c]) for c in CELLS)]
    if len(origins) < 3:
        raise ValueError("Need at least three complete origins")
    M = np.array([[scores[o][c] for c in CELLS] for o in origins], dtype=float)  # (n, 4) FA FH TA TH
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
    total = float(np.sum(W ** 2))
    ss["residual"] = max(0.0, total - sum(ss.values()))
    denom = sum(ss.values()) or 1.0
    shares = {k: v / denom for k, v in ss.items()}
    rng = np.random.default_rng(17)
    boot_t, boot_e = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(origins), len(origins))
        boot_t.append(truth[idx].mean()); boot_e.append(expr[idx].mean())
    return {"n": len(origins), "cell_means": {c: float(M[:, i].mean()) for i, c in enumerate(CELLS)},
            "truth_effect": float(truth.mean()), "truth_ci": [float(np.percentile(boot_t, 2.5)), float(np.percentile(boot_t, 97.5))],
            "expression_effect": float(expr.mean()), "expression_ci": [float(np.percentile(boot_e, 2.5)), float(np.percentile(boot_e, 97.5))],
            "auroc": aurocs, "paired_order": paired, "variance_share": shares}


def report_lines(name, r):
    def f(x):
        return f"{x:+.3f}"
    return [f"| {name} | {r['n']} | " + " / ".join(f"{r['cell_means'][c]:+.2f}" for c in CELLS)
            + f" | {f(r['truth_effect'])} [{f(r['truth_ci'][0])}, {f(r['truth_ci'][1])}]"
            + f" | {f(r['expression_effect'])} [{f(r['expression_ci'][0])}, {f(r['expression_ci'][1])}]"
            + f" | {r['auroc']['truth|assertive']:.2f} / {r['auroc']['truth|hedged']:.2f}"
            + f" | {r['auroc']['expression|false']:.2f} / {r['auroc']['expression|true']:.2f}"
            + f" | {r['variance_share']['truth']:.2f} / {r['variance_share']['expression']:.2f} / {r['variance_share']['interaction']:.2f} / {r['variance_share']['residual']:.2f} |"]
