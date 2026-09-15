"""Download original annotated FPQ references only. No model or judge calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jsonl import read_jsonl
from src.pilot import file_digest, frozen_json, output_lock
from src.baseline_suite import write_frozen_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    out = Path(args.output)
    meta_path = Path(str(out) + ".source.json")
    with output_lock(out):
        if out.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if file_digest(out) != meta["sha256"] or len(list(read_jsonl(out))) != meta["rows"]:
                raise ValueError("Reference download changed")
            if args.revision != meta["requested_revision"]:
                raise ValueError("Revision differs; use a new output filename")
            print(f"Using verified frozen references: {out}; no download or model calls")
            return
        if out.exists() or meta_path.exists():
            raise ValueError("Incomplete reference artifact; use a new path or inspect interrupted write")
        from datasets import load_dataset
        from huggingface_hub import HfApi
        name = "Cancer-Myth/Cancer-Myth"
        revision = HfApi().dataset_info(name, revision=args.revision).sha
        ds = load_dataset(name, revision=revision, split="validation")
        rows = []
        for i, q in enumerate(ds):
            if any(not isinstance(q.get(k), str) or not q[k].strip()
                   for k in ("question", "source_myth", "presupposition_correction")):
                raise ValueError(f"Missing source annotation at row {i}")
            rows.append({"source_index": i, **{k: q[k] for k in
                         ("question", "source_myth", "presupposition_correction")}})
        write_frozen_jsonl(out, rows)
        frozen_json(meta_path, {"dataset": name, "resolved_revision": revision,
                               "requested_revision": args.revision, "split": "validation",
                               "rows": len(rows), "sha256": file_digest(out)})
        print(f"Saved {len(rows)} original reference records at {revision}. GPU/GPT calls: 0.")


if __name__ == "__main__":
    main()
