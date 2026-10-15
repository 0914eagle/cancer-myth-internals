"""Conditional permutation tests for saved K3/behavior association; no model calls."""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/k3_behavior_scope_2026-10-04'
SOURCE = OUT / 'k3_qwen_answer_join.csv'

def main():
    rows = list(csv.DictReader(SOURCE.open()))
    rng = np.random.default_rng(20261004)
    results = []
    for side in ('plain', 'alternative'):
        for subset in ('all', 'well_below4'):
            rr = [r for r in rows if r['side'] == side and r['k3'] != 'choice_missing'
                  and (subset == 'all' or int(r['score']) < 4)]
            categories = sorted({r['alignment'] for r in rr})
            a = np.array([sum(r['alignment'] == c and r['k3'] == 'choice_both_correct' for r in rr) for c in categories])
            b = np.array([sum(r['alignment'] == c and r['k3'] != 'choice_both_correct' for r in rr) for c in categories])
            total = a + b
            ea, eb = total * a.sum() / total.sum(), total * b.sum() / total.sum()
            def stat(x):
                return (((x - ea)**2 / ea) + ((total - x - eb)**2 / eb)).sum(axis=-1)
            observed = float(stat(a))
            samples = rng.multivariate_hypergeometric(total, int(a.sum()), size=100000)
            p = float((1 + (stat(samples) >= observed - 1e-12).sum()) / 100001)
            results.append(dict(side=side, subset=subset, categories=categories,
                stable_counts=a.tolist(), other_counts=b.tolist(), stable_n=int(a.sum()),
                other_n=int(b.sum()), statistic=observed, permutation_p=p,
                cramers_v=float(np.sqrt(observed / total.sum()))))
    order = sorted(range(len(results)), key=lambda i: results[i]['permutation_p'])
    prev = 0
    for rank, i in enumerate(order):
        prev = max(prev, min(1., (len(results)-rank)*results[i]['permutation_p']))
        results[i]['holm_p_four_tests'] = prev
    payload = dict(seed=20261004, permutations=100000,
        input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        null='Behavior distribution independent of saved K3 category, conditional on margins.',
        comparison='Both orders choose correction vs order-disagreement or both-wrong. Missing excluded.',
        limitations=['Exploratory association, not a test of latent knowledge or causation.',
          'Well-below4 selection conditions on outcome; primary result uses all available answers.',
          'Behavior labels are existing AI codes, not new independent human annotations.',
          'Two methods reuse the same question IDs; do not interpret tests as independent replications.'],
        results=results)
    (OUT/'k3_permutation_tests.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
