"""K3 forced-choice result x actual answer behavior (doc 62 ledger). Pure helpers, no model.

K3 shows the myth and the correction together and asks which is accurate (two orders).
Group names describe that task only; they are not knowledge labels:
  stable_choice   picked the correction in both orders
  unsure          picked it in one order
  wrong           picked the myth in both orders
  not_run         id present in labels but K3 not run
  excluded        id absent from labels (statement-format filter in probe_knowledge.py)
Behavior buckets are doc 62's mutually exclusive FPQ alignment codes.
"""

from __future__ import annotations

import math
from collections import Counter

K3_GROUPS = ("stable_choice", "unsure", "wrong", "not_run", "excluded")
K3_NAMES = {
    "stable_choice": "선택지가 있으면 두 순서 모두 교정문 선택",
    "unsure": "순서에 따라 선택이 바뀜",
    "wrong": "두 순서 모두 통념 선택",
    "not_run": "K3 미실행",
    "excluded": "검사 제외(형식 필터)",
}
BUCKETS = ("misses_target", "endorses_target_error", "related_without_correction", "corrects_target", "unclear")
BUCKET_NAMES = {
    "misses_target": "목표 오류 미언급",
    "endorses_target_error": "목표 오류 수용",
    "related_without_correction": "관련 설명만",
    "corrects_target": "교정 코드",
    "unclear": "불명확",
}


def k3_group(label):
    """label: one labels.jsonl record or None (id absent)."""
    if label is None:
        return "excluded"
    v = label.get("knows_fc")
    return {"yes": "stable_choice", "unsure": "unsure", "no": "wrong"}.get(v, "not_run")


def bucket(row):
    a = row["alignment"]
    return a if a in BUCKETS else "unclear"


def crosstab(ledger_rows, labels_by_id, model, side, threshold=4, score_key="original_score"):
    """{group: {"n", "high", "low", "low_buckets": Counter, "ids": [...]}} for FPQ rows of one model/side."""
    out = {g: {"n": 0, "high": 0, "low": 0, "low_buckets": Counter(), "high_buckets": Counter()} for g in K3_GROUPS}
    for r in ledger_rows:
        if r["model"] != model or r["dataset"] != "fpq" or r["side"] != side:
            continue
        g = k3_group(labels_by_id.get(r["id"]))
        cell = out[g]
        cell["n"] += 1
        if float(r[score_key]) >= threshold:
            cell["high"] += 1
            cell["high_buckets"][bucket(r)] += 1
        else:
            cell["low"] += 1
            cell["low_buckets"][bucket(r)] += 1
    return out


def transitions(ledger_rows, labels_by_id, model, threshold=4, score_key="original_score"):
    """Plain -> alternative score transitions per K3 group: {group: Counter("low->high", ...)}."""
    by = {}
    for r in ledger_rows:
        if r["model"] == model and r["dataset"] == "fpq":
            by.setdefault(r["id"], {})[r["side"]] = float(r[score_key]) >= threshold
    out = {g: Counter() for g in K3_GROUPS}
    for qid, s in by.items():
        if "plain" in s and "alternative" in s:
            key = f"{'high' if s['plain'] else 'low'}->{'high' if s['alternative'] else 'low'}"
            out[k3_group(labels_by_id.get(qid))][key] += 1
    return out


def chi2_independence(table):
    """Pearson chi-square on a list of count rows; drops all-zero rows/cols. Returns (stat, dof, min_expected)."""
    rows = [r for r in table if sum(r) > 0]
    if len(rows) < 2:
        return None
    cols = [j for j in range(len(rows[0])) if sum(r[j] for r in rows) > 0]
    rows = [[r[j] for j in cols] for r in rows]
    if len(cols) < 2:
        return None
    total = sum(map(sum, rows))
    rs = [sum(r) for r in rows]
    cs = [sum(r[j] for r in rows) for j in range(len(cols))]
    stat, min_e = 0.0, math.inf
    for i, r in enumerate(rows):
        for j, o in enumerate(r):
            e = rs[i] * cs[j] / total
            min_e = min(min_e, e)
            stat += (o - e) ** 2 / e
    return stat, (len(rows) - 1) * (len(cols) - 1), min_e


def permutation_p(groups_a, groups_b, n_perm=5000, seed=0):
    """Permutation p-value for chi-square between two lists of bucket labels."""
    import random
    def stat(a, b):
        keys = sorted(set(a) | set(b))
        res = chi2_independence([[a.count(k) for k in keys], [b.count(k) for k in keys]])
        return res[0] if res else 0.0
    obs = stat(groups_a, groups_b)
    pool = list(groups_a) + list(groups_b)
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if stat(pool[:len(groups_a)], pool[len(groups_a):]) >= obs - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)
