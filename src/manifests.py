"""Read extraction-run manifests (medical_nla layout) into row lists and matrices."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .jsonl import read_jsonl


def layer_of(manifest: Path) -> int:
    return int(re.fullmatch(r"layer(\d+)", manifest.parent.parent.name).group(1))


def load_family(run_dir: Path, family: str, selection: str) -> dict[int, list[dict]]:
    """layer -> manifest rows of one position family / reduction."""
    out: dict[int, list[dict]] = {}
    for manifest in sorted(Path(run_dir).glob(f"layer*/{selection}/manifest.jsonl")):
        rows = [r for r in read_jsonl(manifest) if r.get("position_family") == family]
        if rows:
            out[layer_of(manifest)] = rows
    return out


def matrix(rows: list[dict]) -> np.ndarray:
    import torch

    return np.stack(
        [torch.load(r["activation_path"], map_location="cpu", weights_only=True).reshape(-1).float().numpy() for r in rows]
    )


def dedupe_negatives(rows: list[dict]) -> list[dict]:
    """nfp and tpq share question text; keep one row per (text, position)."""
    seen, out = set(), []
    for r in rows:
        if str(r.get("set")) in {"nfp", "tpq"}:
            key = (r.get("chat_text"), str(r.get("position")))
            if key in seen:
                continue
            seen.add(key)
        out.append(r)
    return out
