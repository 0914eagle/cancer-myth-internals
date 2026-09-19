"""Where does the premise-truth signal go between the premise span and the answer
position? Pure helpers for scripts/signal_flow.py (no torch).

Directions: per layer, a truth direction d_l fitted on premise-span means
(false: FPQ + false paraphrase; true: twin), 2-fold by origin so every row is
projected with a direction it did not train. Projection p = (h - center) @ d.
Heatmap: mean projection per token bin (pre-question template, question before
the span, span, three thirds of the question after the span, post-question
template, last token) -> AUROC FPQ vs twin per layer x bin.
Patching: add (twin span mean - FPQ span mean) at layer L on the FPQ's span
tokens, read the last-token projection at later layers and the Yes/No readout;
transfer fraction = (patched - fpq) / (twin - fpq).
"""

from __future__ import annotations

import hashlib

import numpy as np

BINS = ("pre", "q_before", "span", "q_after_1", "q_after_2", "q_after_3", "post", "last")
ROLE_PRE, ROLE_Q_BEFORE, ROLE_SPAN, ROLE_Q_AFTER, ROLE_POST, ROLE_LAST = range(6)


def content_positions(rendered, text):
    start = rendered.find(text)
    if start < 0:
        raise ValueError("Rendered prompt does not contain the question verbatim")
    return start, start + len(text)


def token_positions(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if a != b and a < end and b > start]


def token_roles(offsets, q_start, q_end, span_abs=None):
    """One role per token. span_abs: absolute char span of the premise or None."""
    roles = []
    for i, (a, b) in enumerate(offsets):
        if a == b:  # special/empty token
            roles.append(ROLE_PRE if b <= q_start else ROLE_POST)
            continue
        if b <= q_start:
            roles.append(ROLE_PRE)
        elif a >= q_end:
            roles.append(ROLE_POST)
        elif span_abs and a < span_abs[1] and b > span_abs[0]:
            roles.append(ROLE_SPAN)
        elif span_abs and a >= span_abs[1]:
            roles.append(ROLE_Q_AFTER)
        else:
            roles.append(ROLE_Q_BEFORE)
    if roles:
        roles[-1] = ROLE_LAST
    return roles


def bin_indices(roles):
    """{bin: [token idx]}; q_after is split into thirds by order. Missing bins are absent."""
    out = {}
    for name, role in (("pre", ROLE_PRE), ("q_before", ROLE_Q_BEFORE), ("span", ROLE_SPAN), ("post", ROLE_POST), ("last", ROLE_LAST)):
        idx = [i for i, r in enumerate(roles) if r == role]
        if idx:
            out[name] = idx
    after = [i for i, r in enumerate(roles) if r == ROLE_Q_AFTER]
    if after:
        for k, part in enumerate(np.array_split(np.asarray(after), min(3, len(after))), 1):
            out[f"q_after_{k}"] = [int(i) for i in part]
    return out


def fold_of(origin, n_folds=2):
    return int(hashlib.sha256(str(origin).encode()).hexdigest(), 16) % n_folds


def fit_direction(X_pos, X_neg, C=0.01):
    """Logistic direction in raw space (unit norm) and the training mean as center."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([X_pos, X_neg]).astype(np.float32)
    y = np.array([1] * len(X_pos) + [0] * len(X_neg))
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    d = clf.coef_[0] / sc.scale_
    d = (d / (np.linalg.norm(d) + 1e-12)).astype(np.float32)
    return d, X.mean(0).astype(np.float32)


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() < 2 or len(set(labels[ok])) < 2:
        return None
    return float(roc_auc_score(labels[ok], scores[ok]))


def bin_means(proj, bins):
    """proj: (L, T) projections; bins: {name: [idx]} -> {name: (L,) mean}."""
    return {name: proj[:, idx].mean(1) for name, idx in bins.items()}


def transfer_fraction(patched, base_fpq, base_twin, eps=1e-6):
    """Elementwise (patched - fpq) / (twin - fpq); nan where the pair barely differs."""
    denom = base_twin - base_fpq
    out = (patched - base_fpq) / np.where(np.abs(denom) < eps, np.nan, denom)
    return out


def heatmap_lines(layer_names, bin_names, table, title):
    lines = [f"## {title}", "", "| layer | " + " | ".join(bin_names) + " |", "|---|" + "---:|" * len(bin_names)]
    for li, name in enumerate(layer_names):
        cells = []
        for b in bin_names:
            v = table.get((li, b))
            cells.append("NA" if v is None else f"{v:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines
