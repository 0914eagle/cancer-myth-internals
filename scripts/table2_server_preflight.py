"""Read-only server inventory for the Table 2 handoff; never launches experiments.

Uses only the standard library. Package presence is not runtime compatibility.
An output report is a preparation record, not permission to evaluate test data.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/reviews/paper_tables_2026-10-06"
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "4090",
    "Qwen/Qwen3-Embedding-0.6B": "4090",
    "google/gemma-4-12B-it": "a6000",
    "Qwen/Qwen3.8-27B-FP8": "a6000",
}


def command(args):
    try:
        result = subprocess.run(args, cwd=str(ROOT), capture_output=True,
                                text=True, timeout=30)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inventory(cache_root, server):
    found = {}
    for model, owner in MODELS.items():
        if owner != server:
            continue
        snapshots = cache_root / ("models--" + model.replace("/", "--")) / "snapshots"
        found[model] = [{"revision": p.name,
                         "config_present": (p / "config.json").is_file(),
                         "tokenizer_present": (p / "tokenizer.json").is_file(),
                         "weight_file_count": sum(1 for q in p.glob("*.safetensors") if q.is_file())}
                        for p in sorted(snapshots.glob("*")) if p.is_dir()]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", choices=("4090", "a6000"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Local HF hub cache directory; do not copy another host's path")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; use a new dated report path")
    gpu = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits"])
    gpu_rows = []
    if gpu["returncode"] == 0:
        for row in csv.reader(io.StringIO(gpu["stdout"]), skipinitialspace=True):
            if len(row) != 6:
                raise ValueError("Unexpected nvidia-smi CSV schema")
            gpu_rows.append(dict(zip(("index", "uuid", "name", "memory_total_mib",
                                      "memory_used_mib", "utilization_pct"), row)))
    packages = {}
    for package in ("torch", "transformers", "sentence-transformers", "scikit-learn", "gepa", "vllm"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    expected = "4090" if args.server == "4090" else "A6000"
    hardware_matches = bool(gpu_rows) and all(expected in row["name"] for row in gpu_rows)
    inputs = ["table2_run_settings_2026-10-08.json", "table2_prompt_registry_2026-10-07.json",
              "table2_server_allocation_2026-10-08.json"]
    report = {
        "server_role": args.server,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git_commit": command(["git", "rev-parse", "HEAD"])["stdout"],
        "working_tree_dirty": bool(command(["git", "status", "--porcelain"])["stdout"]),
        "preflight_script_sha256": fingerprint(Path(__file__)),
        "protocol_hashes": {name: fingerprint(DOCS / name) for name in inputs},
        "gpu_inventory": gpu_rows,
        "nvidia_smi_error": gpu["stderr"] if gpu["returncode"] != 0 else None,
        "expected_hardware_matches": hardware_matches,
        "packages_in_this_python": packages,
        "model_cache_metadata": inventory(args.cache_root, args.server),
        "split_manifest_present": (ROOT / "results/table2_v2/protocol/split_manifest.json").is_file(),
        "launch_ready": False,
        "launch_note": "Inventory only. Frozen manifests, weight completeness, model-load/kernel smoke, message parity and unified runner release must be checked separately.",
        "model_calls": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(args.out), "gpu_count": len(gpu_rows),
                      "expected_hardware_matches": hardware_matches,
                      "launch_ready": False, "model_calls": 0}, ensure_ascii=False))
    return 0 if hardware_matches else 2


if __name__ == "__main__":
    sys.exit(main())
