"""Config loading with ${VAR} substitution, following medical_nla/src/config.py.

One variable per machine (`CANCER_MYTH_DATA_ROOT`) replaces per-disk config
duplicates. Model configs may name a `_base:` file; the base is loaded first
and the model file's top-level keys replace the base's, so a model file only
states what differs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULTS = {
    "CANCER_MYTH_DATA_ROOT": "/data1/heejae",
    "CANCER_MYTH_CODE_ROOT": str(REPO_ROOT),
}

_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = os.environ.get(name, _DEFAULTS.get(name))
        if value is None:
            raise KeyError(
                f"Config refers to ${{{name}}}, which is neither set in the "
                f"environment nor given a default. Known defaults: {sorted(_DEFAULTS)}."
            )
        return value

    return os.path.expanduser(_PLACEHOLDER.sub(replace, text))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return _substitute(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a config, resolve `_base:`, then resolve ${VAR} everywhere.

    Top-level keys in the model file replace the base's whole section, so a
    model file that states `activation:` must state all of it. An unresolved
    placeholder raises rather than surviving into a filename.
    """
    path = Path(path)
    raw = _read_yaml(path)
    base_name = raw.pop("_base", None)
    if base_name:
        base = _read_yaml(path.parent / base_name)
        base.pop("_base", None)
        base.update(raw)
        raw = base
    cfg = _expand(raw)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create {p}. A path starting at / usually means an empty "
            "shell variable -- $ART or $DATA in a shell that has not run:\n"
            "    source scripts/env.sh"
        ) from exc
    return p


def torch_dtype(name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def model_short_name(cfg: dict[str, Any]) -> str:
    src = cfg["source_model"]
    return str(src.get("short_name") or src["model_id"].split("/")[-1].lower())
