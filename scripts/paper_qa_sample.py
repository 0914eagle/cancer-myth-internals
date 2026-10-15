"""Frozen Table 3 evaluation subset; never changes training or FPQA identities."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'results/paper_baselines_v4/protocol/table3_sample500.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def make_manifest(bundle, cap=500, seed=20261014):
    if cap < 1:
        raise ValueError('QA sample cap must be positive')
    groups = defaultdict(list)
    for row in bundle['qa']:
        groups[row['dataset']].append(row)
    datasets = {}
    for dataset, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r['id'])
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate QA IDs')
        chosen = random.Random(f'{seed}:{dataset}').sample(rows, min(cap, len(rows)))
        datasets[dataset] = {'population': len(rows), 'selected': len(chosen),
                             'ids': sorted(r['id'] for r in chosen)}
    result = {'version': 1, 'seed': seed, 'cap_per_dataset': cap,
              'sampling': 'Uniform random without replacement per dataset; shared across all models and methods',
              'source_bundle_sha256': bundle['content_sha256'],
              'qa_population_sha256': digest(bundle['qa']), 'datasets': datasets}
    result['content_sha256'] = digest(result)
    return result


def select_rows(bundle, manifest):
    body = {k: v for k, v in manifest.items() if k != 'content_sha256'}
    if digest(body) != manifest['content_sha256']:
        raise ValueError('QA sample manifest hash mismatch')
    if manifest['source_bundle_sha256'] != bundle['content_sha256'] or manifest['qa_population_sha256'] != digest(bundle['qa']):
        raise ValueError('QA sample population mismatch')
    expected = make_manifest(bundle, manifest['cap_per_dataset'], manifest['seed'])
    if expected != manifest:
        raise ValueError('QA sample does not match frozen sampling rule')
    ids = {(d, i) for d, x in manifest['datasets'].items() for i in x['ids']}
    return [r for r in bundle['qa'] if (r['dataset'], r['id']) in ids]


def install_qa_subset(runner):
    """Apply only after worker initialization; retain frozen scientific caches."""
    original = runner.V4Worker.execute
    if getattr(original, '_qa_subset_installed', False):
        return
    def execute(worker):
        if MANIFEST.exists():
            manifest = json.loads(MANIFEST.read_text())
            source = json.loads(Path(worker.args.bundle).read_text())
            selected = select_rows(source, manifest)
            allowed = {(r['dataset'], r['id']) for r in selected}
            # Table2-only wrappers intentionally set qa=[]; do not restore it.
            qa = [r for r in worker.bundle['qa'] if (r['dataset'], r['id']) in allowed]
            worker.bundle = {**worker.bundle, 'qa': qa}
            policy = {'manifest': str(MANIFEST), 'manifest_sha256': manifest['content_sha256'],
                      'qa_rows_this_run': len(qa), 'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'scope': 'Table3 subset only; original bundle, FPQA folds, prompts and cached answers unchanged'}
            from src.paper_baselines import write_json
            write_json(worker.out/'execution_policies/table3_sample500.json', policy)
        return original(worker)
    execute._qa_subset_installed = True
    runner.V4Worker.execute = execute


def main():
    """Explicit offline preparation; never starts generation or grading."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=MANIFEST)
    parser.add_argument('--cap', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    if digest({k: v for k, v in bundle.items() if k != 'content_sha256'}) != bundle['content_sha256']:
        raise ValueError('Source bundle hash mismatch')
    manifest = make_manifest(bundle, args.cap, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text()) != manifest:
            raise ValueError('Refusing to replace an existing different sample')
    else:
        with args.output.open('x') as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v['selected'] for k, v in manifest['datasets'].items()}))


if __name__ == '__main__':
    main()
