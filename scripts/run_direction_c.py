"""E1 step 2: the disposition direction C, its overlap with A, and PCR prediction.

Inputs are two extraction runs of the same model:

    --run-ad   positions A/B/D (from activation_rows.jsonl, with pcr merged in)
    --run-e    position E (from response_rows.jsonl; the response opening)

Per hidden-state index:

    C direction   diff-of-means over fpq rows: PCR +1 (corrects) minus PCR -1
                  (follows), at E (span_mean) and at D (last_token)
    A direction   diff-of-means fpq minus nfp/tpq at D and at A (premise)
    cos(A, C)     at D (same position) and A_prem vs C_E (the design's pair)
    PCR AUROC     grouped CV: can the D-position activation predict, before
                  generation, whether the response will correct (+1) or
                  follow (-1)? Both a D-fit probe and the E-learned C
                  direction projected onto D.

Writes directions.npz (per layer: c_e, c_d, a_d, a_prem), table.csv/.md and
a plot. The .npz is what run_steer.py consumes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.probes import (
    cosine,
    cv_auroc_diffmeans,
    cv_auroc_logistic,
    diff_means_direction,
    fit_probe,
)


def _layer_of(manifest: Path) -> int:
    return int(re.fullmatch(r"layer(\d+)", manifest.parent.parent.name).group(1))


def load_family(run_dir: Path, family: str, selection: str) -> dict[int, list[dict]]:
    """layer -> manifest rows of one position family / reduction."""
    out: dict[int, list[dict]] = {}
    for manifest in sorted(run_dir.glob(f"layer*/{selection}/manifest.jsonl")):
        rows = [r for r in read_jsonl(manifest) if r.get("position_family") == family]
        if rows:
            out[_layer_of(manifest)] = rows
    return out


def matrix(rows: list[dict]):
    import torch

    X = np.stack(
        [torch.load(r["activation_path"], map_location="cpu", weights_only=True).reshape(-1).float().numpy() for r in rows]
    )
    return X


def dedupe_negatives(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if str(r.get("set")) in {"nfp", "tpq"}:
            key = (r.get("chat_text"), str(r.get("position")))
            if key in seen:
                continue
            seen.add(key)
        out.append(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ad", required=True)
    parser.add_argument("--run-e", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--e-family", default="E_response_first5")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    run_ad, run_e, out_dir = Path(args.run_ad), Path(args.run_e), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_rows = load_family(run_ad, "D_last_prompt_token", "last_token")
    a_rows = load_family(run_ad, "A_premise", "last_subtoken")
    e_rows = load_family(run_e, args.e_family, "span_mean")
    layers = sorted(set(d_rows) & set(e_rows))
    if not layers:
        raise SystemExit("no common layers between the A/D run and the E run")

    table, store = [], {}
    for layer in layers:
        rec = {"layer": layer}
        # --- C at E and at D, from fpq rows with a definite PCR label ---
        def pcr_rows(rows):
            return [r for r in rows if r.get("set") == "fpq" and r.get("pcr") in (1, -1)]

        e_fpq = pcr_rows(e_rows[layer])
        d_all = dedupe_negatives(d_rows[layer])
        d_fpq = pcr_rows(d_all)
        if len(e_fpq) < 20 or len(d_fpq) < 20:
            print(f"[C] L{layer}: too few PCR-labelled fpq rows ({len(e_fpq)}/{len(d_fpq)})")
            continue
        XE, yE = matrix(e_fpq), np.asarray([1 if r["pcr"] == 1 else 0 for r in e_fpq])
        XD, yD = matrix(d_fpq), np.asarray([1 if r["pcr"] == 1 else 0 for r in d_fpq])
        gE = [str(r.get("base_id")) for r in e_fpq]
        gD = [str(r.get("base_id")) for r in d_fpq]
        c_e = diff_means_direction(XE, yE)
        c_d = diff_means_direction(XD, yD)
        rec["n_correct"], rec["n_follow"] = int(yE.sum()), int((1 - yE).sum())
        # --- PCR predictability before generation (position D) ---
        rec["pcr_auroc_D_logistic"], _ = cv_auroc_logistic(XD, yD, gD, n_splits=args.folds, seed=args.seed)
        rec["pcr_auroc_D_diffmeans"], _ = cv_auroc_diffmeans(XD, yD, gD, n_splits=args.folds, seed=args.seed)
        from sklearn.metrics import roc_auc_score

        rec["pcr_auroc_D_via_cE"] = float(roc_auc_score(yD, XD @ c_e))
        rec["pcr_auroc_E_diffmeans"], _ = cv_auroc_diffmeans(XE, yE, gE, n_splits=args.folds, seed=args.seed)
        # --- A at D and at the premise position ---
        XDa = matrix(d_all)
        yDa = np.asarray([int(r["label_false_premise"]) for r in d_all])
        a_d = diff_means_direction(XDa, yDa)
        rec["cos_A_C_at_D"] = cosine(a_d, c_d)
        rec["cos_Ad_Ce"] = cosine(a_d, c_e)
        a_prem = None
        if layer in a_rows:
            ar = dedupe_negatives(a_rows[layer])
            ya = np.asarray([int(r["label_false_premise"]) for r in ar])
            if len(set(ya.tolist())) == 2:
                a_prem = diff_means_direction(matrix(ar), ya)
                rec["cos_Aprem_Ce"] = cosine(a_prem, c_e)
        store[layer] = {"c_e": c_e, "c_d": c_d, "a_d": a_d, "a_prem": a_prem}
        # A probe at D for gating, as raw arrays.
        probe = fit_probe(XDa, yDa)
        for k, v in probe.items():
            store[layer][f"a_probe_D_{k}"] = v
        table.append(rec)
        print(
            f"[C] L{layer:02d} n=+{rec['n_correct']}/-{rec['n_follow']} "
            f"pcr@D lr={rec['pcr_auroc_D_logistic']:.3f} dm={rec['pcr_auroc_D_diffmeans']:.3f} "
            f"viaCe={rec['pcr_auroc_D_via_cE']:.3f} | cos(A,C)@D={rec['cos_A_C_at_D']:.2f} "
            f"cos(Aprem,Ce)={rec.get('cos_Aprem_Ce', float('nan')):.2f}",
            flush=True,
        )

    if not table:
        raise SystemExit("no layers produced a C direction")
    np.savez(
        out_dir / "directions.npz",
        layers=np.asarray(sorted(store)),
        **{f"L{layer}_{k}": v for layer, d in store.items() for k, v in d.items() if v is not None},
    )
    import csv

    keys = sorted({k for r in table for k in r})
    with (out_dir / "table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    md = ["# C direction and A/C overlap", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in table:
        md.append("| " + " | ".join(f"{r.get(k, ''):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + " |")
    best = max(table, key=lambda r: r["pcr_auroc_D_diffmeans"])
    md += ["", f"Best pre-generation PCR readout at D: L{best['layer']} diff-means {best['pcr_auroc_D_diffmeans']:.3f}.",
           "Pandey reports cos(truth, sycophancy) 0.44-0.73 in the residual stream; compare cos_A_C_at_D.",
           "Two Axes' CREPE readout is 0.69-0.78; the gate probe lives in directions.npz as a_probe_D_*."]
    (out_dir / "table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ls = [r["layer"] for r in table]
        for key in ("pcr_auroc_D_diffmeans", "pcr_auroc_D_via_cE", "pcr_auroc_E_diffmeans"):
            ax.plot(ls, [r[key] for r in table], marker="o", ms=3, label=key)
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_ylabel("AUROC (PCR +1 vs -1)")
        ax2 = ax.twinx()
        ax2.plot(ls, [r["cos_A_C_at_D"] for r in table], color="k", ls="--", label="cos(A,C)@D")
        ax2.set_ylabel("cosine")
        ax.set_xlabel("hidden-state index")
        ax.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        fig.savefig(out_dir / "c_direction.png", dpi=150, bbox_inches="tight")
    except Exception as exc:  # noqa: BLE001
        print(f"[C] plot skipped: {exc!r}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
