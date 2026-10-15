"""Local-only matched Plain audit, including explicit pending/reuse provenance."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(config):
    out = Path(config['artifact_root'])
    original = Path(config['original_artifact_root'])
    nla = {r['id']: r for r in (json.loads(s) for s in (original / 'gemma27/av_results.jsonl').read_text().splitlines())}
    ao = {r['activation_path']: r for r in (json.loads(s) for s in (original / 'gemma27/ao_results.jsonl').read_text().splitlines())
          if r['requested_window_size'] == 20}
    requests = json.loads((out / 'native/cohort_request_ids.json').read_text())
    records = []
    for ident, sample in zip(requests, config['cohort']):
        path = out / 'native/results' / (ident + '.json')
        if not path.exists(): continue
        native = json.loads(path.read_text())
        assert native['row_id'] == sample['id']
        readers = {}
        for kind in ['nla', 'ao']:
            provenance = native['reader_reuse'].get(kind, {})
            if provenance.get('bitwise_equal'):
                saved = nla[provenance['original_activation_id']] if kind == 'nla' else ao[provenance['original_activation_path']]
                readers[kind] = dict(state='bytewise_verified_reuse', explanation=saved['nla_output' if kind == 'nla' else 'ao_output'],
                    original_output_id=saved['id'], original_activation_path=provenance['original_activation_path'])
            else: readers[kind] = dict(state='needs_fresh_verbalization', explanation=None)
        records.append(dict(id=sample['id'], dataset=sample['dataset'], label=sample['label'],
            question=sample['question'], answer=native['answer'], completed=native['completed'],
            output_tokens=native['output_tokens'], native_id=ident, nla=readers['nla'], ao=readers['ao']))
    target = out / 'local_cohort'; target.mkdir(exist_ok=True)
    (target / 'matched_records.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in records))
    with (target / 'matched_records.csv').open('w', newline='') as stream:
        keys = ['id', 'dataset', 'label', 'question', 'answer', 'completed', 'nla_explanation', 'ao_explanation', 'nla_state', 'ao_state']
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in records:
            flat = {k: row[k] for k in keys[:6]}
            flat.update(nla_explanation=row['nla']['explanation'], ao_explanation=row['ao']['explanation'],
                        nla_state=row['nla']['state'], ao_state=row['ao']['state'])
            writer.writerow(flat)
    summary = dict(state='complete' if len(records) == 100 else 'running', questions_available=len(records), expected=100,
        completed_native_answers=sum(x['completed'] for x in records),
        verified_nla_pairs=sum(x['nla']['state'] == 'bytewise_verified_reuse' for x in records),
        verified_ao_pairs=sum(x['ao']['state'] == 'bytewise_verified_reuse' for x in records),
        scope='Local matching only; no external calls and no correction/over-rebuttal scores inferred', timestamp_utc=time.time())
    temp = target / 'summary.tmp'; temp.write_text(json.dumps(summary, indent=2) + '\n'); temp.replace(target / 'summary.json')
    return summary


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--watch', action='store_true')
    a = p.parse_args(); config = json.loads(a.config.read_text())
    while True:
        summary = export(config)
        print(json.dumps(summary), flush=True)
        if not a.watch or summary['state'] == 'complete': break
        time.sleep(30)


if __name__ == '__main__': main()
