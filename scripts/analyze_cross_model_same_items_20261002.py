"""Align all saved primary answer pairs by question, retaining code/score distinction."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT=ROOT/'docs/reviews/cross_model_same_items_2026-10-02'
PRIMARY=['qwen25','gemma','luna','qwen38_off']
SENSITIVITY=['qwen25','gemma','luna','qwen38_on']
EXAMPLES=['nfp_1002','nfp_1023','nfp_1048','nfp_1083','fpq_2','fpq_30','fpq_56','fpq_144']


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def csvwrite(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def correction(answer):
    return answer['coding']['reference_alignment']['status']=='corrects_target'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in SOURCE.open()]
    ix={(r['model'],r['id']):r for r in rows}
    assert len(ix)==len(rows)==3659
    for r in rows:
        assert r['plain']['question']==r['alternative']['question']
    model_rows=[]
    for model in PRIMARY+['qwen38_on']:
        for label in ['fpq','nfp']:
            sub=[r for r in rows if r['model']==model and r['dataset']==label]
            a=sum(correction(r['plain']) for r in sub) if label=='fpq' else None
            b=sum(correction(r['alternative']) for r in sub) if label=='fpq' else None
            model_rows.append(dict(model=model,alternative_method=sub[0]['method'],label=label,n=len(sub),
                plain_target_correction=a,alternative_target_correction=b,
                gained_correction=sum(not correction(r['plain']) and correction(r['alternative']) for r in sub) if label=='fpq' else None,
                lost_correction=sum(correction(r['plain']) and not correction(r['alternative']) for r in sub) if label=='fpq' else None,
                plain_challenge=sum(r['plain']['coding']['stance']=='challenge' for r in sub),
                alternative_challenge=sum(r['alternative']['coding']['stance']=='challenge' for r in sub),
                plain_low_well=sum(r['plain']['score']<4 for r in sub),
                alternative_low_well=sum(r['alternative']['score']<4 for r in sub)))
    csvwrite('model_condition_counts.csv',model_rows)
    summaries={};wide=[];shared_fpq=[];nfp_multiple=[]
    for panel,models in [('qwen38_off',PRIMARY),('qwen38_on',SENSITIVITY)]:
        ids=sorted(set.intersection(*[{i for m,i in ix if m==model} for model in models]))
        assert len(ids)==731
        assert Counter(ix[models[0],i]['dataset'] for i in ids)=={'fpq':582,'nfp':149}
        summary=dict(models=models,n_common=len(ids),excluded=['fpq_389'],by_label={})
        for label in ['fpq','nfp']:
            ii=[i for i in ids if ix[models[0],i]['dataset']==label]
            stats=dict(n=len(ii))
            for side in ['plain','alternative']:
                funcs=dict(low_well=lambda r:r['score']<4,
                           challenge=lambda r:r['coding']['stance']=='challenge')
                if label=='fpq':funcs['no_target_correction']=lambda r:not correction(r)
                for name,fn in funcs.items():
                    votes={i:sum(fn(ix[m,i][side]) for m in models) for i in ii}
                    stats[side+'_'+name]=dict(vote_distribution=dict(Counter(votes.values())),
                        all_four_ids=[i for i,k in votes.items() if k==4],
                        none_ids=[i for i,k in votes.items() if k==0])
            summary['by_label'][label]=stats
        summaries[panel]=summary
        if panel!='qwen38_off':continue
        for i in ids:
            first=ix[models[0],i];label=first['dataset']
            assert len({ix[m,i]['plain']['question'] for m in models})==1
            item=dict(id=i,label=label)
            for m in models:
                r=ix[m,i]
                for side in ['plain','alternative']:
                    a=r[side];prefix=m+'_'+side
                    item[prefix+'_score']=a['score']
                    item[prefix+'_stance']=a['coding']['stance']
                    item[prefix+'_target']=a['coding']['reference_alignment']['status']
            wide.append(item)
            if label=='fpq' and all(not correction(ix[m,i]['alternative']) for m in models):
                shared_fpq.append(dict(id=i,question=first['plain']['question'],reference=first['plain']['reference'],
                    evidence={m:dict(score=ix[m,i]['alternative']['score'],
                        target=ix[m,i]['alternative']['coding']['reference_alignment'],
                        response=ix[m,i]['alternative']['coding']['response'],
                        response_quote=ix[m,i]['alternative']['coding']['response_quote']) for m in models}))
            if label=='nfp' and sum(ix[m,i]['alternative']['coding']['stance']=='challenge' for m in models)>=3:
                nfp_multiple.append(dict(id=i,question=first['plain']['question'],
                    evidence={m:dict(score=ix[m,i]['alternative']['score'],method=ix[m,i]['method'],
                        target=ix[m,i]['alternative']['coding']['target_ko'],
                        stance=ix[m,i]['alternative']['coding']['stance'],
                        quote=ix[m,i]['alternative']['coding']['stance_quote']) for m in models}))
    csvwrite('all_731_same_item_patterns.csv',wide)
    dump('overlap_summary.json',summaries)
    dump('all_73_shared_uncorrected_fpq.json',shared_fpq)
    dump('all_20_nfp_challenged_by_three_or_four.json',nfp_multiple)
    examples=[dict(id=i,question=ix[PRIMARY[0],i]['plain']['question'],
                   settings={m:ix[m,i] for m in PRIMARY}) for i in EXAMPLES]
    dump('eight_case_full_answers.json',examples)
    measures=dict(shared_uncorrected_fpq=len(shared_fpq),
        answer_count=len(shared_fpq)*4,
        response=Counter(e['response'] for r in shared_fpq for e in r['evidence'].values()),
        target=Counter(e['target']['status'] for r in shared_fpq for e in r['evidence'].values()),
        high_well_answers=[dict(id=r['id'],model=m,score=e['score']) for r in shared_fpq for m,e in r['evidence'].items() if e['score']>=4])
    dump('shared_uncorrected_details.json',measures)
    dump('manifest.json',dict(source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_model_calls=0,
        scope='Existing 3659 pairs / 7318 answers; main matched comparison 731 questions across four model families. Qwen3.8 ON replaces OFF in sensitivity analysis, not a fifth independent vote.',
        limitation='Existing Sonnet behavior codes are provisional, not medical truth; a negative answer to an inquiry is not automatically false-premise challenge. Alternative prompts differ across models. No causal model ranking.',
        reading='All 73 shared-uncorrected questions and reference premises inspected; all 20 multi-model NFP target/quote comparisons exported. Eight case bundles retained for full-answer checking; no claim of fresh independent coding of 7318 answers.'))
    assert len(shared_fpq)==73 and len(nfp_multiple)==20
    print(json.dumps(measures,ensure_ascii=False,indent=2))
    for panel,s in summaries.items():
        print(panel)
        for label,ss in s['by_label'].items():
            print(label,{k:len(v['all_four_ids']) for k,v in ss.items() if isinstance(v,dict)})


if __name__=='__main__':main()
