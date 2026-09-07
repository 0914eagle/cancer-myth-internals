"""Add judge labels (pcr, nfp_score) to every manifest of an extraction run.

The A/B/D activations are extracted before the judge has run, so their
manifests lack the PCR label the C direction needs. Rather than re-running
the backbone, rewrite each manifest.jsonl in place, joining labels.jsonl on
base_id. Tensors are untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()

    labels = {r["id"]: r for r in read_jsonl(args.labels)}
    n_files, n_rows, n_hit = 0, 0, 0
    for manifest in sorted(Path(args.run_dir).glob("layer*/*/manifest.jsonl")):
        rows = list(read_jsonl(manifest))
        for row in rows:
            lab = labels.get(str(row.get("base_id", row["id"])))
            n_rows += 1
            if lab:
                n_hit += 1
                for key in ("pcr", "nfp_score", "judge_parsed"):
                    if key in lab:
                        row[key] = lab[key]
        write_jsonl(manifest, rows)
        n_files += 1
    print(f"[merge] {n_files} manifests, {n_hit}/{n_rows} rows labelled")


if __name__ == "__main__":
    main()
