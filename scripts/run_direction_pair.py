"""E1 stage 8: the paired C direction and how it relates to A and to c_e.

Inputs: the paired E run (scripts/make_paired_rows.py -> src.extract_activations)
and the A/B/D run with pcr merged in, plus direction_c/directions.npz to
extend.

Per hidden-state index and prefix length (5 / 32 response tokens):

    c_pair        mean over questions of E(corr) - E(follow), the same prompt
                  behind both, so the question's content cancels exactly
    own check     the model's own responses projected on c_pair: does the
                  direction separate its own +1 from its own -1 (AUROC,
                  leave-question-out through the corr/follow folds)?
    D transfer    c_pair learned on training questions, scored on held-out
                  questions' pre-generation D vectors against own PCR
    cos           c_pair vs a_d (truth at D), a_prem (truth at the premise),
                  c_e (the natural-response direction), c_d

Writes table_pair.md/.csv and adds L{layer}_c_pair{K} to directions.npz so
run_steer.py can use --direction-key c_pair5 / c_pair32.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.manifests import dedupe_negatives, load_family, matrix
from src.probes import cosine

MIN_PAIRS = 30


def _folds(n: int, n_splits: int, seed: int):
    from sklearn.model_selection import KFold

    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(n)))


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-pair", required=True)
    parser.add_argument("--run-ad", required=True)
    parser.add_argument("--directions", required=True, help="direction_c/directions.npz to extend in place")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[5, 32])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    run_pair, run_ad, out_dir = Path(args.run_pair), Path(args.run_ad), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = dict(np.load(args.directions))
    d_rows = load_family(run_ad, "D_last_prompt_token", "last_token")

    table = []
    for n in args.prefix_tokens:
        fam = load_family(run_pair, f"E_pair_first{n}", "span_mean")
        for layer in sorted(set(fam) & set(d_rows)):
            rows = fam[layer]
            by_q: dict[str, dict[str, dict]] = {}
            for r in rows:
                by_q.setdefault(str(r.get("base_id")), {})[r.get("pair_role")] = r
            pairs = [(q, v) for q, v in by_q.items() if "corr" in v and "follow" in v]
            if len(pairs) < MIN_PAIRS:
                print(f"[pair] L{layer} first{n}: only {len(pairs)} corr/follow pairs")
                continue
            qids = [q for q, _ in pairs]
            Xc = matrix([v["corr"] for _, v in pairs])
            Xf = matrix([v["follow"] for _, v in pairs])
            diff = Xc - Xf
            c_pair = diff.mean(axis=0)
            c_pair = c_pair / np.linalg.norm(c_pair)
            rec = {"prefix": n, "layer": layer, "n_pairs": len(pairs)}
            # consistency of the pair difference: share of questions whose own
            # difference points with the mean direction
            rec["pair_sign_agreement"] = float(((diff @ c_pair) > 0).mean())

            # own responses and D vectors for the same questions
            own = {q: v["own"] for q, v in by_q.items() if "own" in v and v["own"].get("pcr") in (1, -1)}
            d_by_q = {str(r.get("base_id")): r for r in dedupe_negatives(d_rows[layer]) if r.get("set") == "fpq" and r.get("pcr") in (1, -1)}
            own_q = [q for q in own if q in d_by_q]
            if len(own_q) >= 20:
                Xo, XD = matrix([own[q] for q in own_q]), matrix([d_by_q[q] for q in own_q])
                yo = np.asarray([1 if own[q]["pcr"] == 1 else 0 for q in own_q])
                rec["n_own_correct"], rec["n_own_follow"] = int(yo.sum()), int(len(yo) - yo.sum())
                # leave-question-out: c_pair from pairs whose question is not
                # among the scored own/D items of that fold
                pair_idx = {q: i for i, q in enumerate(qids)}
                oof_own, oof_D = np.zeros(len(own_q)), np.zeros(len(own_q))
                for tr, te in _folds(len(own_q), args.folds, args.seed):
                    held = {own_q[i] for i in te}
                    keep = [pair_idx[q] for q in qids if q not in held]
                    d = diff[keep].mean(axis=0)
                    d = d / np.linalg.norm(d)
                    oof_own[te] = Xo[te] @ d
                    oof_D[te] = XD[te] @ d
                rec["own_pcr_auroc_E_via_cpair"] = _auroc(yo, oof_own)
                rec["own_pcr_auroc_D_via_cpair"] = _auroc(yo, oof_D)
            for key, name in (("a_d", "cos_cpair_Ad"), ("a_prem", "cos_cpair_Aprem"), ("c_e", "cos_cpair_Ce"), ("c_d", "cos_cpair_Cd")):
                vec = npz.get(f"L{layer}_{key}")
                if vec is not None:
                    rec[name] = cosine(c_pair, vec)
            npz[f"L{layer}_c_pair{n}"] = c_pair.astype(np.float32)
            table.append(rec)
            print(
                f"[pair] L{layer:02d} first{n} pairs={len(pairs)} agree={rec['pair_sign_agreement']:.2f} "
                f"own@E={rec.get('own_pcr_auroc_E_via_cpair', float('nan')):.3f} "
                f"own@D={rec.get('own_pcr_auroc_D_via_cpair', float('nan')):.3f} "
                f"cos(Ad)={rec.get('cos_cpair_Ad', float('nan')):.2f} cos(Ce)={rec.get('cos_cpair_Ce', float('nan')):.2f}",
                flush=True,
            )

    if not table:
        raise SystemExit("no layer produced a paired direction")
    np.savez(args.directions, **npz)
    keys = sorted({k for r in table for k in r})
    with (out_dir / "table_pair.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)
    md = ["# Paired C direction (corr minus follow, same prompt)", "", "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in table:
        md.append("| " + " | ".join(f"{r.get(k, ''):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + " |")
    best = max((r for r in table if "own_pcr_auroc_D_via_cpair" in r), key=lambda r: r["own_pcr_auroc_D_via_cpair"], default=None)
    if best:
        md += ["", f"Best transfer to the model's own pre-generation PCR: L{best['layer']} first{best['prefix']} "
               f"AUROC {best['own_pcr_auroc_D_via_cpair']:.3f} (own +1 = {best['n_own_correct']}, -1 = {best['n_own_follow']})."]
    md += ["", "Columns: pair_sign_agreement = share of questions whose corr-follow difference points with c_pair "
           "(1.0 = every question agrees); own_* = the model's own responses scored with a c_pair built without that "
           "question; cos_* = cosine with the directions in directions.npz.",
           f"directions.npz now carries L{{layer}}_c_pair{{K}} for K in {args.prefix_tokens}; run_steer.py --direction-key c_pair{args.prefix_tokens[0]}."]
    (out_dir / "table_pair.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
