"""E1 step 1: layer x position AUROC for "the premise is false" (A readout).

For every (hidden-state index, position family, reduction) manifest under an
extraction run, fit the two cross-validated readouts from src.probes on
positive = fpq rows vs negative = nfp + tpq rows, and write

    sweep.jsonl / sweep.csv    one row per cell, both readouts, CIs on the best
    heatmap.png                logistic and diff-of-means panels
    summary.md                 best cells, and the 0.70 reference line
                               (Two Axes' CREPE last-token readout)

NFP and TPQ rows share question text, so for positions that do not depend on
the premise span (B, D) they are the same vector twice; negatives are deduped
by (chat_text, position) before fitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.probes import bootstrap_auroc_ci, cv_auroc_diffmeans, cv_auroc_logistic

REFERENCE_AUROC = 0.70


def manifests(run_dir: Path):
    for manifest in sorted(run_dir.glob("layer*/*/manifest.jsonl")):
        layer = int(re.fullmatch(r"layer(\d+)", manifest.parent.parent.name).group(1))
        yield layer, manifest.parent.name, manifest


def pair_key(row: dict) -> str:
    """fpq rows are their own pair; a twin carries pair_id = its fpq id.
    Falls back to base_id for fpq rows extracted before pair_id existed."""
    return str(row.get("pair_id") or (row.get("base_id") if str(row.get("set")) == "fpq" else None) or row.get("id"))


def load_cell(manifest: Path, positive: set[str], negative: set[str], paired_only: bool = False):
    """Returns {family: (X, y, groups)} for the rows in one manifest.

    paired_only keeps only rows whose pair_id occurs on both sides (fpq and
    its true-premise twin), so the comparison is a set of minimal pairs.
    """
    import torch

    per_family: dict[str, list] = defaultdict(list)
    seen_neg: set[tuple[str, str]] = set()
    for row in read_jsonl(manifest):
        s = str(row.get("set"))
        if s in positive:
            y = 1
        elif s in negative:
            key = (row.get("chat_text", ""), str(row.get("position")))
            if key in seen_neg:
                continue
            seen_neg.add(key)
            y = 0
        else:
            continue
        per_family[str(row.get("position_family"))].append((row, y))
    out = {}
    for family, items in per_family.items():
        if paired_only:
            pos_ids = {pair_key(r) for r, y in items if y == 1}
            neg_ids = {pair_key(r) for r, y in items if y == 0}
            both = pos_ids & neg_ids
            items = [(r, y) for r, y in items if pair_key(r) in both]
        xs, ys, groups = [], [], []
        for row, y in items:
            t = torch.load(row["activation_path"], map_location="cpu", weights_only=True)
            xs.append(t.reshape(-1).to(torch.float32).numpy())
            ys.append(y)
            # Group by pair (fpq and its twin) when present, else by question
            # text so nfp/tpq twins never straddle a fold.
            groups.append(pair_key(row) if row.get("pair_id") or str(row.get("set")) == "fpq" else str(row.get("prompt") or row.get("chat_text")))
        y_arr = np.asarray(ys)
        if len(set(ys)) < 2 or min((y_arr == 1).sum(), (y_arr == 0).sum()) < 10:
            continue
        out[family] = (np.stack(xs), y_arr, groups)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="extraction run directory (has layerNN/)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive", nargs="+", default=["fpq"])
    parser.add_argument("--negative", nargs="+", default=["nfp", "tpq"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--layers", nargs="+", type=int, default=None, help="subset of hidden-state indices")
    parser.add_argument("--paired-only", action="store_true", help="keep only pair_ids present on both sides (minimal pairs)")
    args = parser.parse_args()

    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    positive, negative = set(args.positive), set(args.negative)

    results = []
    for layer, selection, manifest in manifests(run_dir):
        if args.layers and layer not in set(args.layers):
            continue
        for family, (X, y, groups) in load_cell(manifest, positive, negative, args.paired_only).items():
            auc_lr, oof_lr = cv_auroc_logistic(X, y, groups, n_splits=args.folds, seed=args.seed)
            auc_dm, oof_dm = cv_auroc_diffmeans(X, y, groups, n_splits=args.folds, seed=args.seed)
            results.append(
                {
                    "layer": layer,
                    "selection": selection,
                    "position_family": family,
                    "n_pos": int((y == 1).sum()),
                    "n_neg": int((y == 0).sum()),
                    "auroc_logistic": round(auc_lr, 4),
                    "auroc_diffmeans": round(auc_dm, 4),
                    "_oof": (y, oof_lr, oof_dm),
                }
            )
            print(f"[sweep] L{layer:02d} {family:<22} {selection:<13} lr={auc_lr:.3f} dm={auc_dm:.3f}", flush=True)
    if not results:
        raise SystemExit(f"no usable cells under {run_dir}")

    # CIs on the best cell per readout.
    for key in ("auroc_logistic", "auroc_diffmeans"):
        best = max(results, key=lambda r: r[key])
        y, oof_lr, oof_dm = best["_oof"]
        lo, hi = bootstrap_auroc_ci(y, oof_lr if key.endswith("logistic") else oof_dm, seed=args.seed)
        best[f"{key}_ci95"] = [round(lo, 4), round(hi, 4)]
    for r in results:
        r.pop("_oof")

    with (out_dir / "sweep.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        writer.writeheader()
        writer.writerows(results)

    # Heatmap: rows = (family, selection), cols = layer.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cells = sorted({(r["position_family"], r["selection"]) for r in results})
        layers = sorted({r["layer"] for r in results})
        fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.3 * len(layers) + 3), 0.5 * len(cells) + 2))
        for ax, key in zip(axes, ("auroc_logistic", "auroc_diffmeans")):
            grid = np.full((len(cells), len(layers)), np.nan)
            for r in results:
                grid[cells.index((r["position_family"], r["selection"])), layers.index(r["layer"])] = r[key]
            im = ax.imshow(grid, vmin=0.5, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels(layers, fontsize=7)
            ax.set_yticks(range(len(cells)))
            ax.set_yticklabels([f"{f} / {s}" for f, s in cells], fontsize=7)
            ax.set_title(key)
            ax.set_xlabel("hidden-state index")
        fig.colorbar(im, ax=axes, shrink=0.8, label="AUROC (5-fold, grouped)")
        fig.savefig(out_dir / "heatmap.png", dpi=150, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"[sweep] heatmap skipped: {exc!r}")

    lines = ["# A readout sweep", "", f"run: `{run_dir}`", f"positive={sorted(positive)} negative={sorted(negative)}", ""]
    lines += ["| readout | best cell | AUROC | 95% CI |", "|---|---|---:|---|"]
    for key in ("auroc_logistic", "auroc_diffmeans"):
        best = max(results, key=lambda r: r[key])
        lines.append(
            f"| {key} | L{best['layer']} {best['position_family']} {best['selection']} | "
            f"{best[key]:.3f} | {best.get(f'{key}_ci95')} |"
        )
    lines += ["", f"Reference: Two Axes CREPE last-token readout 0.69-0.78 (line at {REFERENCE_AUROC}).", ""]
    lines += ["## Best layer per position family", "", "| family | selection | best L | logistic | diff-means |", "|---|---|---:|---:|---:|"]
    for family, selection in sorted({(r["position_family"], r["selection"]) for r in results}):
        sub = [r for r in results if r["position_family"] == family and r["selection"] == selection]
        b = max(sub, key=lambda r: r["auroc_diffmeans"])
        lines.append(f"| {family} | {selection} | {b['layer']} | {b['auroc_logistic']:.3f} | {b['auroc_diffmeans']:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
