#!/usr/bin/env python3
"""Preserve transport diagnostics and accept explicitly recovered reconnect warnings.

Prompts/model options stay fixed. Only completed turns with transient reconnect
warnings may pass; fatal errors and incomplete turns still fail closed.
"""
import json
import runpy
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import src.fpqa_cli_backend as backend
import src.fpqa_prompting as prompting

target=Path(sys.argv[1]).resolve()
if target.parent!=ROOT/'scripts':raise ValueError('Only repository scripts allowed')
args=sys.argv[2:]
out=Path(args[args.index('--out')+1]).resolve()/'transport_errors'
original=backend.parse_codex
original_freeze=prompting.freeze
def append_failure(path,value):
    path=Path(path)
    if path.name=='FAILED.json' and path.exists() and json.loads(path.read_text())!=value:
        return original_freeze(path.parent/'failure_history'/f'{uuid.uuid4().hex}.json',value)
    return original_freeze(path,value)
prompting.freeze=append_failure
def capture(stdout,final):
    events=[json.loads(line) for line in stdout.splitlines() if line.strip()]
    errors=[e for e in events if e.get('type')=='error']
    completed=[i for i,e in enumerate(events) if e.get('type')=='turn.completed']
    try:
        if (errors and len(completed)==1 and not any(e.get('type')=='turn.failed' for e in events)
            and all(e.get('message','').startswith('Reconnecting...') for e in errors)
            and all(i<completed[0] for i,e in enumerate(events) if e.get('type')=='error')):
            cleaned='\n'.join(json.dumps(e) for e in events if e.get('type')!='error')
            result=original(cleaned,final)  # Still enforces final text, tools and single completed turn.
            out.mkdir(parents=True,exist_ok=True)
            (out/f'recovered_{uuid.uuid4().hex}.jsonl').write_text(stdout)
            print('RECOVERED_RECONNECT_WARNING',flush=True)
            return result[0],events
        return original(stdout,final)
    except Exception:
        out.mkdir(parents=True,exist_ok=True)
        name=uuid.uuid4().hex
        (out/f'{name}.jsonl').write_text(stdout)
        events=[json.loads(line) for line in stdout.splitlines() if line.strip()]
        errs=[e for e in events if e.get('type') in {'error','turn.failed'}
              or e.get('item',{}).get('type')=='error']
        print('CAPTURED_CODEX_ERRORS '+json.dumps(errs,ensure_ascii=False),flush=True)
        raise
backend.parse_codex=capture
sys.argv=[str(target),*args]
runpy.run_path(str(target),run_name='__main__')
