"""Linear readouts of stored activations: logistic probe and difference-of-means.

Two readers, both cross-validated, following Two Axes (Wagner 2026) which
found the difference-of-means direction beat the logistic readout on CREPE
(0.74-0.78 vs 0.69-0.73). Both are reported at every (layer, position).

Manifest rows come from `src.extract_activations`; each names one .pt tensor
of shape (d_model,). Grouping is by `base_id` so a fold never sees the same
question in train and test through two of its positions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .jsonl import read_jsonl


def load_matrix(
    manifest: str | Path,
    *,
    label_key: str = "label_false_premise",
    keep: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """X (n, d) float32, y (n,), base ids, and the manifest rows, in manifest order."""
    import torch

    keep_set = set(keep) if keep is not None else None
    xs, ys, ids, rows = [], [], [], []
    for row in read_jsonl(manifest):
        if keep_set is not None and str(row.get("set")) not in keep_set:
            continue
        label = row.get(label_key)
        if label is None:
            continue
        tensor = torch.load(row["activation_path"], map_location="cpu", weights_only=True)
        if tensor.dim() != 1:
            tensor = tensor.reshape(-1)
        xs.append(tensor.to(torch.float32).numpy())
        ys.append(int(label))
        ids.append(str(row.get("base_id", row["id"])))
        rows.append(row)
    if not xs:
        raise ValueError(f"no labelled rows in {manifest}")
    return np.stack(xs), np.asarray(ys), ids, rows


def _folds(groups: list[str], y: np.ndarray, n_splits: int, seed: int):
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def cv_auroc_logistic(
    X: np.ndarray, y: np.ndarray, groups: list[str], *, n_splits: int = 5, seed: int = 17, C: float = 1.0
) -> tuple[float, np.ndarray]:
    """Standardize, L2 logistic, class-balanced. Returns (AUROC, out-of-fold scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros(len(y), dtype=np.float64)
    for train, test in _folds(groups, y, n_splits, seed):
        scaler = StandardScaler().fit(X[train])
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(X[train]), y[train])
        oof[test] = clf.decision_function(scaler.transform(X[test]))
    return float(roc_auc_score(y, oof)), oof


def diff_means_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)
    norm = np.linalg.norm(d)
    return d / norm if norm > 0 else d


def cv_auroc_diffmeans(
    X: np.ndarray, y: np.ndarray, groups: list[str], *, n_splits: int = 5, seed: int = 17
) -> tuple[float, np.ndarray]:
    from sklearn.metrics import roc_auc_score

    oof = np.zeros(len(y), dtype=np.float64)
    for train, test in _folds(groups, y, n_splits, seed):
        d = diff_means_direction(X[train], y[train])
        oof[test] = X[test] @ d
    return float(roc_auc_score(y, oof)), oof


def fit_probe(X: np.ndarray, y: np.ndarray, *, C: float = 1.0) -> dict[str, np.ndarray]:
    """Full-data logistic probe as raw arrays, for use inside a steering gate
    without importing sklearn at generation time: score = (x - mu) / sd @ w + b."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X), y
    )
    return {
        "mu": scaler.mean_.astype(np.float32),
        "sd": scaler.scale_.astype(np.float32),
        "w": clf.coef_[0].astype(np.float32),
        "b": np.asarray([clf.intercept_[0]], dtype=np.float32),
    }


def probe_score(probe: dict[str, np.ndarray], x: np.ndarray) -> float:
    return float(((x - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"][0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def bootstrap_auroc_ci(
    y: np.ndarray, scores: np.ndarray, *, n_boot: int = 1000, seed: int = 17
) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        values.append(roc_auc_score(y[idx], scores[idx]))
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
