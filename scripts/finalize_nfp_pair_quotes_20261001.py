"""Restore literal acronyms in two source quotes, leaving all pair judgments intact."""
import json
from pathlib import Path
import time
from scripts import audit_nfp_plain_pairs_20261001 as audit
from src.pilot import output_lock

RUN=audit.ROOT/'results/audits/nfp_plain_pairs_20261001_v1'
FIXES={
    'pair_010':{'a':{'answer_quote':'Acute Myeloid Leukemia (AML) is NOT primarily a childhood cancer.'}},
    'pair_051':{'b':{'answer_quote':'Acute myeloid leukemia (AML) is not primarily a childhood cancer; it is actually much more common in adults.'}},
}


def main():
    with output_lock(RUN):
        state=json.loads((RUN/'status.json').read_text());assert state['state']!='running'
        jobs=json.loads((RUN/'jobs.json').read_text());proposed=[]
        for j in jobs:
            p=RUN/'records'/j['job']/'record.json';r=json.loads(p.read_text())
            if r['status']=='complete':continue
            assert r['status']=='failed' and j['job'] in FIXES,j['job']
            original=json.loads((p.parent/'answer.txt').read_text())
            fixed=json.loads((p.parent/'quote_repair.txt').read_text())
            for side,patch in FIXES[j['job']].items():fixed[side].update(patch)
            assert audit.strip_quotes(original)==audit.strip_quotes(fixed)
            result=audit.validate(json.dumps(fixed),j);proposed.append((p,r,result))
        archive=RUN/'final_quote_recovery';archive.mkdir(exist_ok=True)
        for p,r,result in proposed:
            audit.tr.write(archive/(r['job']+'.json'),r,frozen=True)
            recovered=dict(r,status='complete',result=result,
                final_quote_recovery=FIXES[r['job']],judgment_unchanged=True)
            recovered.pop('error',None);audit.tr.write(p,recovered)
        for j in jobs:
            r=json.loads((RUN/'records'/j['job']/'record.json').read_text());assert r['status']=='complete'
            audit.validate(json.dumps(r['result']),j)
        audit.tr.write(archive/'manifest.json',dict(fixes=FIXES,new_model_calls=0,
            judgment_unchanged=True,script_sha256=audit.tr.sha(Path(__file__).resolve())),frozen=True)
        audit.tr.write(RUN/'status.json',dict(state='complete',expected=121,complete=121,failed=0,updated_unix=time.time()))
    print(json.dumps(audit.report(RUN),ensure_ascii=False))


if __name__=='__main__':main()
