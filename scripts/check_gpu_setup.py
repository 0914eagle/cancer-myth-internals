"""Report whether the configured backbone would land on GPU (from medical_nla).

device_map="auto" falls back to CPU without raising when no GPU is visible.
This checks driver visibility, device count and free memory before a long
job, and exits non-zero from --require-free-gb so a queue stops at the first
occupied card instead of reporting every run as failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--require-free-gb", type=float)
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    if not torch.cuda.is_available():
        print("\nNo GPU visible to torch. device_map='auto' would place the model on CPU.")
        raise SystemExit(1 if args.require_free_gb else 0)

    total_free, short = 0.0, []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        total_free += free / 1e9
        print(f"  [{index}] {torch.cuda.get_device_name(index)}: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB")
        if args.require_free_gb and free / 1e9 < args.require_free_gb:
            short.append((index, free / 1e9))
    if short:
        for index, free in short:
            print(f"\n[!] GPU {index} has {free:.1f} GB free, {args.require_free_gb:.1f} GB required.")
        print("Another process is most likely still holding it:")
        print("  nvidia-smi --query-compute-apps=pid,used_memory --format=csv")
        raise SystemExit(1)

    src = cfg["source_model"]
    budget = src.get("max_memory") or {}
    print(f"\nsource_model: {src['model_id']} dtype={src.get('dtype')} device_map={src.get('device_map')}")
    print(f"  max_memory={budget} (needs {len(budget)} visible card(s))")
    if len(budget) > torch.cuda.device_count():
        print(f"[!] config names {len(budget)} cards but only {torch.cuda.device_count()} are visible")
        raise SystemExit(1)
    print(f"\nTotal free across visible GPUs: {total_free:.1f} GB")


if __name__ == "__main__":
    main()
