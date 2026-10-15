"""Audited recovery of the Luna run after a successfully recovered WS warning.

Keeps original experiment code, prompts and manifests frozen. The only runtime
change is accepting reconnect/fallback notices IF the full successful turn and
final-answer checks still pass. Failed attempts are archived before retry.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_frontier_luna as base
from src.pilot import output_lock

ORIGINAL_PARSE = base.parse_events


def recovered_connection_notice(event):
    message = event.get('message', '')
    if event.get('type') == 'error':
        return message.startswith('Reconnecting...') and '403 Forbidden' in message and 'wss://chatgpt.com/backend-api/codex/responses' in message
    item = event.get('item', {})
    return (event.get('type') == 'item.completed' and item.get('type') == 'error'
            and item.get('message', '').startswith('Falling back from WebSockets to HTTPS transport.')
            and '403 Forbidden' in item['message'])


def parse_events(stdout, final):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    filtered = [e for e in events if not recovered_connection_notice(e)]
    ORIGINAL_PARSE('\n'.join(json.dumps(e) for e in filtered), final)
    return events  # Preserve warnings in native provenance; no model output is rewritten.


def archive_and_recover(out):
    """Explicit one-time operator review, not an automatic retry loop."""
    with output_lock(out / 'pipeline'):
        entries = []
        for method in base.PROMPTS:
            spec = json.loads((out / method / 'plan.json').read_text())
            for path in sorted((out / method / 'records').glob('*.json')):
                record = json.loads(path.read_text())
                if record['status'] == 'complete':
                    continue
                row = {k: record[k] for k in ('id', 'question')}
                assert record['signature'] == base.tr.digest([spec, row])
                native = out / method / 'native' / record['id']
                archive = out / 'recovery/first_failures' / method / record['id']
                if archive.exists():
                    raise ValueError(f'Failure was already reviewed: {path}')
                archive.mkdir(parents=True)
                shutil.copy2(path, archive / 'record.json')
                if native.exists():
                    shutil.copytree(native, archive / 'native')
                events_path = native / 'events.jsonl'
                stdout = events_path.read_text() if events_path.exists() else ''
                events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
                completed = [e for e in events if e.get('type') == 'turn.completed']
                messages = [e['item']['text'] for e in events if e.get('type') == 'item.completed'
                            and e.get('item', {}).get('type') == 'agent_message']
                if completed:
                    final = messages[-1] if messages else ''
                    answer = native / 'answer.txt'
                    if answer.exists() and answer.read_text().strip() != final.strip():
                        raise ValueError('Native output disagrees with final answer')
                    parse_events(stdout, final)
                    record.update(status='complete', response=final.strip(), model_used=base.MODEL,
                                  usage=completed[0]['usage'],
                                  thread_id=next(e['thread_id'] for e in events if e['type'] == 'thread.started'),
                                  recovery={'reason': 'Native completed turn survives transport warning/process interruption',
                                            'archived_attempt': str(archive), 'at_unix': time.time()})
                    record.pop('error', None)
                    base.tr.write(path, record)
                    action = 'recovered_from_native_without_model_call'
                else:
                    # Only retries incomplete failed attempts, preserving all originals.
                    path.unlink()
                    action = 'archived_incomplete_attempt_for_retry'
                entries.append({'method': method, 'id': row['id'], 'action': action})
        base.tr.write(out / 'recovery/audit.json', {'entries': entries, 'at_unix': time.time()}, frozen=True)
        print(json.dumps({'reviewed': len(entries), 'recovered': sum(e['action'].startswith('recovered') for e in entries)}), flush=True)


def main():
    out = Path(sys.argv[sys.argv.index('--out-dir') + 1]).resolve()
    manifest = {'implementation_sha256': base.tr.sha(Path(__file__)),
                'original_manifest_sha256': base.tr.sha(out / 'manifest.json'),
                'change': 'Allow only recognized recovered WebSocket 403 notices; require normal final completion and no tools.',
                'generation_workers': 5, 'new_automatic_retry': False}
    base.tr.write(out / 'recovery/manifest.json', manifest, frozen=True)
    if '--recover-failed' in sys.argv:
        sys.argv.remove('--recover-failed')
        archive_and_recover(out)
    base.parse_events = parse_events
    base.main()


if __name__ == '__main__':
    main()
