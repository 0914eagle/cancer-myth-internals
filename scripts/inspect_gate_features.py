"""What does the natural FPQ-vs-NFP text/style gate actually weight? (CPU, judge 0)

    python scripts/inspect_gate_features.py --suite-dir $SUITE_DIR

Fits the same TF-IDF logistic classifier and the style classifier on all 732
natural questions and prints the top +/- features, the length/shape stats per
set, and a few NFP and FPQ examples. Diagnostic only (fit on all data).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.jsonl import read_jsonl
from src.style_features import style_feature_names, style_vector


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rows = list(read_jsonl(Path(args.suite_dir) / "questions.jsonl"))
    y = np.array([1 if r["set"] == "fpq" else 0 for r in rows])
    texts = [r["question"] for r in rows]

    # shape stats
    for name, mask in (("FPQ", y == 1), ("NFP", y == 0)):
        t = [texts[i] for i in np.where(mask)[0]]
        words = [len(x.split()) for x in t]
        sents = [max(1, len(re.findall(r"[.!?]+", x))) for x in t]
        qmarks = [x.count("?") for x in t]
        first = [bool(re.search(r"\b(I|my|me|we|our)\b", x)) for x in t]
        print(f"{name}: n={len(t)} words median {np.median(words):.0f} (p10 {np.percentile(words,10):.0f}, p90 {np.percentile(words,90):.0f}); "
              f"sentences median {np.median(sents):.0f}; '?' per question {np.mean(qmarks):.2f}; first-person {np.mean(first):.0%}")

    # TF-IDF logistic
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=1)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000).fit(X, y)
    names = np.asarray(vec.get_feature_names_out())
    w = clf.coef_[0]
    order = np.argsort(w)
    print("\nTF-IDF top FPQ-side features:", ", ".join(f"{names[i]} ({w[i]:+.2f})" for i in order[::-1][:args.top]))
    print("TF-IDF top NFP-side features:", ", ".join(f"{names[i]} ({w[i]:+.2f})" for i in order[:args.top]))

    # style logistic
    S = np.stack([style_vector(t) for t in texts])
    sc = StandardScaler().fit(S)
    clf2 = LogisticRegression(C=0.01, class_weight="balanced", max_iter=5000).fit(sc.transform(S), y)
    fn = np.asarray(style_feature_names())
    w2 = clf2.coef_[0]
    o2 = np.argsort(w2)
    print("\nStyle top FPQ-side:", ", ".join(f"{fn[i]} ({w2[i]:+.2f})" for i in o2[::-1][:args.top]))
    print("Style top NFP-side:", ", ".join(f"{fn[i]} ({w2[i]:+.2f})" for i in o2[:args.top]))
    # mean raw values for the top style features
    print("\nMean raw value of the top 8 style features per side:")
    for i in list(o2[::-1][:8]) + list(o2[:8]):
        print(f"  {fn[i]:<28} FPQ {S[y==1, i].mean():.3f}  NFP {S[y==0, i].mean():.3f}")

    print("\nNFP examples (first 5):")
    for r in [r for r in rows if r["set"] == "nfp"][:5]:
        print("-", r["id"], "|", r["question"][:300])
    print("\nFPQ examples (first 3):")
    for r in [r for r in rows if r["set"] == "fpq"][:3]:
        print("-", r["id"], "|", r["question"][:300])


if __name__ == "__main__":
    main()
