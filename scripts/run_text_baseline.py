"""Surface-form ceiling for the A readout (Two Axes' bag-of-words control).

The probe separates fpq from nfp/tpq at AUROC ~0.8. That is only evidence of
a truth representation if the same split is not already readable from the
question text: fpq were written by GPT-4o from myths, NFP are questions on
which LLMs false-alarmed, and the two may differ in style. This fits a
TF-IDF logistic classifier on the question text (and, for position A, on
the premise span text alone) with the same grouped folds, and reports the
AUROC next to the probe's. Read: probe well above text = the hidden state
carries more than style; probe at or below text = the readout may be
surface form and the minimal-pair control is required.

    python scripts/run_text_baseline.py --questions $ROWS/questions.jsonl --out-dir $RES/probe_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.probes import bootstrap_auroc_ci


def cv_text_auroc(texts: list[str], y: np.ndarray, groups: list[str], *, n_splits: int, seed: int, analyzer: str) -> tuple[float, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    oof = np.zeros(len(y))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if analyzer == "word":
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        else:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
        Xtr = vec.fit_transform([texts[i] for i in train])
        Xte = vec.transform([texts[i] for i in test])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[train])
        oof[test] = clf.decision_function(Xte)
    return float(roc_auc_score(y, oof)), oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--positive", nargs="+", default=["fpq"])
    parser.add_argument("--negative", nargs="+", default=["nfp", "tpq"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = list(read_jsonl(args.questions))
    pos, neg = set(args.positive), set(args.negative)

    # B/D-equivalent: whole question, nfp/tpq twins collapsed to one text.
    seen, q_texts, q_y, q_groups = set(), [], [], []
    for r in rows:
        s = r.get("set")
        if s in pos:
            y = 1
        elif s in neg:
            if r["question"] in seen:
                continue
            seen.add(r["question"])
            y = 0
        else:
            continue
        q_texts.append(r["question"])
        q_y.append(y)
        q_groups.append(r["question"])
    # A-equivalent: the premise span text only, rows that have one.
    a_texts, a_y, a_groups = [], [], []
    for r in rows:
        s = r.get("set")
        span = r.get("premise_span")
        if not span or s not in pos | neg:
            continue
        a_texts.append(r["question"][int(span[0]):int(span[1])])
        a_y.append(1 if s in pos else 0)
        a_groups.append(r["question"])

    out = {}
    for name, texts, y, groups in (("question_text", q_texts, np.asarray(q_y), q_groups), ("premise_span_text", a_texts, np.asarray(a_y), a_groups)):
        if len(set(y.tolist())) < 2:
            continue
        rec = {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        for analyzer in ("word", "char"):
            auc, oof = cv_text_auroc(texts, y, groups, n_splits=args.folds, seed=args.seed, analyzer=analyzer)
            lo, hi = bootstrap_auroc_ci(y, oof, seed=args.seed)
            rec[f"auroc_{analyzer}"] = auc
            rec[f"ci_{analyzer}"] = [lo, hi]
            print(f"[text] {name:<18} {analyzer:<5} n=+{rec['n_pos']}/-{rec['n_neg']} AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
        # length alone
        lengths = np.asarray([len(t) for t in texts], dtype=float)
        from sklearn.metrics import roc_auc_score

        rec["auroc_length_only"] = float(max(roc_auc_score(y, lengths), 1 - roc_auc_score(y, lengths)))
        out[name] = rec

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text_baseline.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = ["# Text-only baseline (surface-form ceiling for the A readout)", "",
          "| input | n | word TF-IDF | char TF-IDF | length only |", "|---|---|---|---|---|"]
    for name, rec in out.items():
        md.append(f"| {name} | +{rec['n_pos']}/-{rec['n_neg']} | {rec['auroc_word']:.3f} [{rec['ci_word'][0]:.2f}, {rec['ci_word'][1]:.2f}] "
                  f"| {rec['auroc_char']:.3f} | {rec['auroc_length_only']:.3f} |")
    md += ["", "Compare with probe_sweep/summary.md: question_text vs B/D probes, premise_span_text vs A probes. "
           "A probe that does not clearly exceed its text row may be reading surface form (Two Axes' bag-of-words control)."]
    (out_dir / "text_baseline.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {out_dir / 'text_baseline.md'}")


if __name__ == "__main__":
    main()
