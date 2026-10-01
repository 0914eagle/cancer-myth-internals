"""Continue frozen audit; one quote-only repair, bounded transport failure handling.

Original protocol and successful judgments stay immutable. Repair may alter ONLY
quote fields; all categorical judgments and explanations must remain identical.
"""
import argparse
import json
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import audit_correction_boundaries_20261001 as audit
from scripts.run_frontier_luna import schedule
from scripts.run_luna_premise_gate import decode, parse_luna_events
from src.pilot import output_lock

tr=audit.tr
QUOTE_FIELDS={'question_quote','answer_quote','later_quote'}


def call(folder,out,plan,prompt,suffix):
    final=folder/(suffix+'.txt')
    raw=tr.invoke(['codex','exec']+plan['options']+['--json','--output-last-message',str(final),'-'],prompt,out/'isolated_cwd',240)
    (folder/(suffix+'_events.jsonl')).write_text(raw)
    response=final.read_text();parse_luna_events(raw,response)
    return response


def invoke(out,plan,job):
    folder=out/'records'/job['job'];folder.mkdir(parents=True,exist_ok=True)
    prompt=audit.PROMPT+json.dumps(job['items'],ensure_ascii=False)
    rec=dict(job=job['job'],signature=tr.digest([plan,job]),status='started',prompt=prompt,started_unix=time.time())
    tr.write(folder/'record.json',rec)
    try:
        response=call(folder,out,plan,prompt,'answer')
        try:result=audit.validate(response,job['items'])
        except (AssertionError,KeyError,ValueError) as exc:
            before=decode(response)
            repair=(prompt+'\n\nThe previous response failed exact-quote validation. Repair ONLY question_quote, answer_quote, '
                'and later_quote. Copy exact contiguous substrings without ellipses or normalization; '
                'use empty strings if no evidence exists, except a yes later_reacceptance requires supporting later_quote. '
                'Keep ALL other fields, judgments, explanations, item order and keys EXACTLY unchanged. '
                'Do not change a judgment to make validation pass. Return the corrected JSON only. PREVIOUS RESPONSE:\n'+response)
            (folder/'repair_prompt.txt').write_text(repair)
            fixed=call(folder,out,plan,repair,'quote_repair');after=decode(fixed)
            strip=lambda v:[{k:x for k,x in r.items() if k not in QUOTE_FIELDS} for r in v['items']]
            assert strip(before)==strip(after),'Repair changed judgment or explanation'
            result=audit.validate(fixed,job['items'])
            rec.update(quote_repaired=True,initial_validation_error=str(exc))
        rec.update(status='complete',result=result)
    except Exception as exc:
        rec.update(status='failed',error=f'{type(exc).__name__}: {str(exc)[:1000]}')
        # Content/schema failures do not cancel unrelated requests. Rate-limit/auth
        # errors do stop dispatch; no polling or automatic unlimited retries.
        message=str(exc).lower()
        if any(s in message for s in ['usage limit','rate limit','quota','oauth','unauthorized','authentication','429']):
            tr.terminate_active()
    tr.write(folder/'record.json',rec)
    return rec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--retry-failed',action='store_true')
    args=ap.parse_args();out=args.out_dir.resolve();assert 1<=args.workers<=8
    signal.signal(signal.SIGTERM,tr.terminate_active);signal.signal(signal.SIGINT,tr.terminate_active)
    with output_lock(out):
        plan=json.loads((out/'plan.json').read_text())
        jobs=json.loads((out/'jobs.json').read_text())
        assert plan['prompt']==audit.PROMPT and plan['system']==audit.SYSTEM
        for source,expected in plan['source_sha256'].items():
            assert tr.sha(Path(source))==expected,source
        if (out/'STOP').exists():raise RuntimeError('STOP exists')
        tr.write(out/'continuation_protocol.json',dict(script_sha256=tr.sha(Path(__file__)),
            same_original_prompt=True,quote_only_repair_max=1,judgments_preserved_on_repair=True,
            invalid_records_excluded=True),frozen=True)
        records={};pending=[]
        for j in jobs:
            p=out/'records'/j['job']/'record.json'
            if not p.exists():pending.append(j);continue
            r=json.loads(p.read_text());assert r['signature']==tr.digest([plan,j])
            if r['status']=='complete':records[j['job']]=r
            elif args.retry_failed:
                archive=out/'recovery_initial_validation'/j['job'];archive.parent.mkdir(exist_ok=True)
                assert not archive.exists(),'Only one initial failed-call retry is allowed'
                p.parent.rename(archive);pending.append(j)
            else:records[j['job']]=r
        def report(state):
            tr.write(out/'status.json',dict(state=state,updated_unix=time.time(),expected=len(jobs),
                complete=sum(r['status']=='complete' for r in records.values()),
                failed=sum(r['status']=='failed' for r in records.values()),
                quote_repairs=sum(r.get('quote_repaired',False) for r in records.values()),workers=args.workers))
        def consume(job,future):
            r=future.result();records[job['job']]=r;report('running')
            print(json.dumps(dict(job=job['job'],status=r['status'],processed=len(records),error=r.get('error'))),flush=True)
        report('running');schedule(out,pending,lambda j:invoke(out,plan,j),consume,args.workers)
        ok=len(records)==len(jobs) and all(r['status']=='complete' for r in records.values())
        report('complete' if ok else 'incomplete_validation_or_transport')
        if ok:
            from scripts.report_correction_boundaries_20261001 import main as summarize
            summarize()


if __name__=='__main__':main()
