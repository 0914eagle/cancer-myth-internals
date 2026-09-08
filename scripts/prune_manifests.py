"""Drop manifest rows that are no longer in the current row file.

After a re-alignment the A rows get new ids (the span hash is part of the id),
so a resumed extraction adds the new rows and the old `__prem_*` rows would
stay in every manifest next to them. This removes every manifest row whose id
is not in --rows. Tensor files are left in place (harmless, and cheap to keep
until the run dir is rebuilt).

    python scripts/prune_manifests.py --run-dir $ACT_AD --rows $ROWS/activation_rows.jsonl
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
    parser.add_argument("--rows", required=True)
    args = parser.parse_args()

    keep = {r["id"] for r in read_jsonl(args.rows)}
    n_files, n_rows, n_dropped = 0, 0, 0
    for manifest in sorted(Path(args.run_dir).glob("layer*/*/manifest.jsonl")):
        rows = list(read_jsonl(manifest))
        kept = [r for r in rows if r["id"] in keep]
        n_rows += len(rows)
        n_dropped += len(rows) - len(kept)
        if len(kept) != len(rows):
            write_jsonl(manifest, kept)
        n_files += 1
    print(f"[prune] {n_files} manifests, {n_dropped}/{n_rows} rows dropped (ids not in {args.rows})")


if __name__ == "__main__":
    main()
