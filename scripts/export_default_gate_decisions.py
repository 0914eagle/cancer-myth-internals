"""Recount saved held-out gate scores with default binary decisions; no inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dim_boundaries(suite, gates):
    """Use the midpoint of training centroids, equivalent to nearest Euclidean mean."""
    features, hashes = {}, {}
    for path in sorted((suite / 'model/features/cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            qid = str(data['id'])
            features[qid] = {int(k[1:]): data[k].copy() for k in data.files if k.startswith('L')}
            hashes[qid] = sha(path)
    labels = {r['id']: r['set'] for r in gates['predictions']}
    result = {}
    for selection in gates['selections']:
        if selection['signal'] != 'mean':
            continue
        layer = selection['best']['layer']
        ids = selection['train_ids']
        positive = np.stack([features[i][layer] for i in ids if labels[i] == 'fpq']).mean(0)
        negative = np.stack([features[i][layer] for i in ids if labels[i] == 'nfp']).mean(0)
        direction = positive - negative
        direction /= np.linalg.norm(direction)
        midpoint = (positive + negative) / 2
        boundary = float(midpoint @ direction)
        held = [r for r in gates['predictions'] if r['signal'] == 'mean' and r['fold'] == selection['fold']]
        matrix = np.stack([features[r['id']][layer] for r in held])
        scores = matrix @ direction
        assert np.allclose(scores, [r['score'] for r in held], rtol=1e-5, atol=1e-4)
        # Independent check: compare distances to both centroids in float64.
        fp_distance = ((matrix.astype(float) - positive.astype(float)) ** 2).sum(1)
        nf_distance = ((matrix.astype(float) - negative.astype(float)) ** 2).sum(1)
        assert np.array_equal(fp_distance < nf_distance, np.array([r['score'] > boundary for r in held]))
        result[selection['fold']] = dict(layer=layer, boundary=boundary,
                                        train_fpq=sum(labels[i] == 'fpq' for i in ids),
                                        train_nfp=sum(labels[i] == 'nfp' for i in ids))
    return result, hashes


def export(suite, luna, out):
    source = suite / 'gates/crossfit_v2_style/result.json'
    gates = json.loads(source.read_text())
    detections = {r['id']: r for p in (suite / 'model/detection/cache').glob('*.json')
                  for r in [json.loads(p.read_text())]}
    predictions = gates['predictions']
    groups = {r['id']: r['group_id'] for r in predictions}
    selections = {(r['signal'], r['fold']): r for r in gates['selections']}
    dim, feature_hashes = dim_boundaries(suite, gates)
    decisions = []
    for r in predictions:
        selection = selections[r['signal'], r['fold']]
        assert r['id'] in selection['evaluation_ids']
        assert not set(selection['evaluation_ids']) & set(selection['train_ids'])
        assert not set(selection['evaluation_ids']) & set(selection['calibration_ids'])
        for role in ('train_ids', 'calibration_ids'):
            assert not {groups[i] for i in selection[role]} & {
                groups[i] for i in selection['evaluation_ids']}
        signal = r['signal']
        boundary = 0.5 if signal in ('direct', 'review') else 0.0
        if signal == 'mean':
            boundary = dim[r['fold']]['boundary']
        if signal in ('direct', 'review'):
            record = detections[r['id']]
            detail = record['details']['direct' if signal == 'direct' else 'after_review']
            assert record[signal + '_score'] == r['score']
            assert (r['score'] > boundary) == (detail['positive_logit'] > detail['negative_logit'])
        decisions.append(dict(method=signal, id=r['id'], set=r['set'],
                              group_id=r['group_id'], fold=r['fold'], score=r['score'],
                              boundary=boundary, tie=int(r['score'] == boundary),
                              predicted_fpq=int(r['score'] > boundary)))
    luna_sources = {}
    for method in ('direct', 'cot'):
        for path in sorted((luna / method).glob('*/record.json')):
            r = json.loads(path.read_text())
            assert r['status'] == 'complete'
            result = r['result']
            assert result['has_false_premise'] == (result['false_probability'] >= 0.5)
            label = next(p['set'] for p in predictions if p['id'] == r['id'])
            decisions.append(dict(method='luna_' + method, id=r['id'], set=label,
                                  group_id=groups[r['id']], fold='', score='', boundary='',
                                  tie=int(result['false_probability'] == 0.5),
                                  predicted_fpq=int(result['has_false_premise'])))
            luna_sources[str(path)] = sha(path)
    summary = {}
    for method in dict.fromkeys(r['method'] for r in decisions):
        rows = [r for r in decisions if r['method'] == method]
        assert len(rows) == len({r['id'] for r in rows}) == 732
        pos = [r for r in rows if r['set'] == 'fpq']
        neg = [r for r in rows if r['set'] == 'nfp']
        assert len(pos) == 583 and len(neg) == 149
        tp, fp = sum(r['predicted_fpq'] for r in pos), sum(r['predicted_fpq'] for r in neg)
        summary[method] = dict(tp=tp, fn=583-tp, fp=fp, tn=149-fp,
                               tpr=tp/583, fpr=fp/149,
                               ties=[r['id'] for r in rows if r['tie']])
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'decisions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(decisions[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(decisions)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    provenance = dict(source=str(source), source_sha256=sha(source),
                      script_sha256=sha(Path(__file__)),
                      decisions_sha256=sha(out / 'decisions.csv'),
                      qwen_detection_dir=str(suite / 'model/detection/cache'),
                      qwen_rule='Yes logit > No logit; ties -> NFP. Not free-form generation.',
                      classifier_rule='decision_function > 0; ties -> NFP; equivalent to binary LogisticRegression.predict.',
                      luna_rule='Use saved has_false_premise verbatim; original prompt requires probability >= 0.5.',
                      dim_rule='Nearest training centroid in Euclidean distance; equivalently projection above training-centroid midpoint. New explicit binary rule; original scores/layers unchanged.',
                      dim_folds=dim, feature_sha256=feature_hashes,
                      no_new_training_or_inference=True, luna_record_sha256=luna_sources)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def export_routing(decision_dir, old_scores):
    decisions = list(csv.DictReader((decision_dir / 'decisions.csv').open()))
    old = list(csv.DictReader(old_scores.open()))
    base = {r['id']: r for r in old if r['method'] == 'plain'}
    assert len(base) == 731
    flags = {method: {r['id']: bool(int(r['predicted_fpq'])) for r in decisions if r['method'] == method}
             for method in ('text', 'hidden')}
    for r in old:
        assert (r['plain'], r['corrected'], r['set']) == (
            base[r['id']]['plain'], base[r['id']]['corrected'], base[r['id']]['set'])
    rows, summary = [], {}
    for method in ('plain', 'always_correct', 'text', 'hidden', 'label_oracle'):
        current = []
        for qid, r in sorted(base.items()):
            on = (method == 'always_correct' or
                  (method == 'label_oracle' and r['set'] == 'fpq') or
                  (method in flags and flags[method][qid]))
            plain, corrected = int(r['plain']), int(r['corrected'])
            current.append(dict(method=method, id=qid, set=r['set'], gate_on=int(on),
                                plain=plain, corrected=corrected, score=corrected if on else plain))
        rows.extend(current)
        summary[method] = dict(interventions=sum(r['gate_on'] for r in current))
        for subset in ('fpq', 'nfp'):
            sr = [r for r in current if r['set'] == subset]
            summary[method][subset] = dict(n=len(sr), ge4=sum(r['score'] >= 4 for r in sr),
                s5=sum(r['score'] == 5 for r in sr),
                rescue_ge4=sum(r['plain'] < 4 <= r['score'] for r in sr),
                harm_ge4=sum(r['score'] < 4 <= r['plain'] for r in sr))
    for method in flags:
        k = summary[method]['interventions']
        rate = k / len(base)
        expected = dict(interventions=k, n=len(base))
        for subset in ('fpq', 'nfp'):
            sr = [r for r in base.values() if r['set'] == subset]
            expected[subset] = {name: sum((1-rate)*(int(r['plain']) >= cutoff) + rate*(int(r['corrected']) >= cutoff)
                                         for r in sr) / len(sr)
                                for name, cutoff in [('ge4_fraction', 4), ('s5_fraction', 5)]}
        summary['random_expected_' + method] = expected
    with (decision_dir / 'routing_scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    output = dict(common_n=len(base), excluded_ids=['fpq_389'], results=summary,
                  original_score_file=str(old_scores), original_score_sha256=sha(old_scores),
                  decisions_sha256=sha(decision_dir / 'decisions.csv'),
                  note='Retrospective selection of unchanged stored answers and Well scores using default classifier decisions.')
    (decision_dir / 'routing_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Routing:', json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--luna', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--routing-scores', type=Path)
    args = parser.parse_args()
    export(args.suite, args.luna, args.out)
    if args.routing_scores:
        export_routing(args.out, args.routing_scores)
