"""Join the full audits when ready; local-only dependent batch processing."""
from collections import Counter
import csv
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCOPE=ROOT/'results/audits/question_scope_full_20260930_v1'
AUDITS=[ROOT/'results/audits'/n for n in ['answer_behavior_full_20260929_v2','answer_behavior_extension_20260930_v1']]
OUT=ROOT/'results/analysis/full_behavior_completion_20260930_v1'


def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def scope_link():
    from scripts.audit_question_scope_full import validate
    inputs=json.loads((SCOPE/'inputs.json').read_text());codes={}
    for p in sorted((SCOPE/'batches').glob('*/result.json')):
        i=int(p.parent.name);packet=json.loads(p.read_text())
        valid,_=validate(packet['items'],inputs[i*3:i*3+3])
        for c in valid:
            assert c['key'] not in codes;codes[c['key']]=c
    assert len(codes)==732
    manifest=json.loads((SCOPE/'manifest.json').read_text())
    context=json.loads((ROOT/'results/analysis/full_behavior_census_20260930_v2/context_all_items.json').read_text())
    byid={r['id']:r for r in context}; rows=[]
    for m in manifest:
        c=codes[m['key']];r=byid[m['id']]
        claims=[x for x in c['claims'] if x['role']!='queried']
        features={
            'has_individual_claim':any(x['scope']=='individual' for x in claims),
            'has_general_claim':any(x['scope']=='general' for x in claims),
            'has_ambiguous_claim':any(x['scope']=='ambiguous' for x in claims),
            'has_clinician_reported_individual_claim':any(x['scope']=='individual' and x['role']=='reported' and x['attribution']=='clinician' for x in claims),
            'has_explicit_diagnosis_to_outcome_inference':any(x['diagnosis_to_outcome_inference']=='explicit' for x in claims),
            'has_queried_claim':any(x['role']=='queried' for x in c['claims'])}
        rows.append(dict(id=m['id'],dataset=m['dataset'],transition=r['transition'],features=features,coding=c))
    comparisons=[]
    for ds in ['fpq','nfp']:
        for feature in rows[0]['features']:
            for present in [False,True]:
                rr=[r for r in rows if r['dataset']==ds and r['features'][feature]==present]
                counts=Counter(r['transition'] for r in rr)
                before_yes=counts['1->0']+counts['1->1']
                comparisons.append(dict(dataset=ds,feature=feature,present=present,n=len(rr),
                    transitions=counts,before_yes=before_yes,lost_yes=counts['1->0'],
                    lost_yes_fraction=counts['1->0']/before_yes if before_yes else None,
                    ids=[r['id'] for r in rr]))
    write('question_scope_items.json',rows)
    write('question_scope_associations.json',dict(n=732,comparisons=comparisons,
        interpretation='Descriptive association only. Not Direct reasoning, medical adjudication, or proof of causation. Features overlap; never sum their counts as disjoint categories.'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not (OUT/'STOP').exists():
            states={}
            try:
                for folder in [SCOPE]+AUDITS:
                    states[folder.name]=json.loads((folder/'status.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):
                time.sleep(10);continue
            if states[SCOPE.name].get('complete') and not (OUT/'question_scope_associations.json').exists():
                scope_link()
            if all(s.get('complete') for s in states.values()):
                final=OUT/'final_census'
                run=subprocess.run([sys.executable,'-m','scripts.analyze_full_behavior_census','--out-dir',str(final)],cwd=ROOT,capture_output=True,text=True)
                (OUT/'final_census.log').write_text(run.stdout+run.stderr)
                if run.returncode:
                    write('status.json',dict(state='final_aggregation_failed',returncode=run.returncode));return
                summary=json.loads((final/'summary.json').read_text())
                assert summary['semantic_coded_answers']==summary['individual_answers']==13906
                with (final/'all_pairs.csv').open() as f:
                    assert all(x['semantic_complete']=='True' for x in csv.DictReader(f))
                write('status.json',dict(state='complete',answers=13906,pairs=10244,questions=732,at=time.time()));return
            write('status.json',dict(state='waiting_audits',at=time.time(),dependencies={k:{f:v.get(f) for f in ['state','valid','expected']} for k,v in states.items()}))
            time.sleep(60)
        write('status.json',dict(state='stopped_by_user',at=time.time()))


if __name__=='__main__':main()
