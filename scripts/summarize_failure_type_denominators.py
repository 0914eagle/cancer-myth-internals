"""Count existing per-answer codes against every success/failure denominator.

No generation, judging or new semantic coding. Never treat missing optional tags
as absence of the newly proposed claim-reconstruction failure.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT = ROOT / 'docs/reviews/failure_type_denominators_2026-10-02'
MODELS = ['qwen25','gemma','luna','qwen38_off','qwen38_on']
FPQ = ['misses_target','endorses_target_error','related_without_correction','corrects_target','unclear']
NFP = ['challenge','qualification','none','unclear']


def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def cell(k,n):
    return f'{k} ({100*k/n:.1f}%)' if n else '—'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=[json.loads(l) for l in SRC.open()]
    assert len(pairs)==3659
    flat=[]
    for r in pairs:
        for cond in ['plain','alternative']:
            a=r[cond];c=a['coding']
            flat.append(dict(model=r['model'],id=r['id'],dataset=r['dataset'],condition=cond,method=a['method'],
                score=a['score'],well_group='below4' if a['score']<4 else 'atleast4',
                existing_alignment=c['reference_alignment']['status'],existing_stance=c['stance'],
                existing_grounding=c['target_grounding'],reference_quality=c['reference_alignment']['reference_quality'],
                target_quote=c['reference_alignment']['answer_quote'],stance_quote=c['stance_quote'],
                reconstruction_audit='not_population_recoded'))
    assert len(flat)==7318 and len({(x['model'],x['id'],x['condition']) for x in flat})==7318
    write_csv('all_answers_existing_codes.csv',flat)
    distributions=[];conditions=[];transitions=[]
    lines=['# 기존 코드 전수 집계','',
           '단위는 답변. 비율 분모는 각 행의 Well 점수 집합이며, 원인 확정 비율이 아니다.','']
    for ds,types,field in [('fpq',FPQ,'existing_alignment'),('nfp',NFP,'existing_stance')]:
        for group in ['below4','atleast4']:
            lines += [f'## {ds.upper()} / {group}','',
                      '| 모델 | 조건 | 분모 | '+' | '.join(types)+' |',
                      '|---|---|---:|'+'---:|'*len(types)]
            for m in MODELS:
                for cond in ['plain','alternative']:
                    allrows=[r for r in flat if r['dataset']==ds and r['model']==m and r['condition']==cond]
                    rows=[r for r in allrows if r['well_group']==group];n=len(rows)
                    counts=Counter(r[field] for r in rows)
                    assert set(counts)<=set(types) and sum(counts.values())==n
                    for t in types:
                        distributions.append(dict(dataset=ds,model=m,condition=cond,well_group=group,
                            type=t,n=counts[t],denominator=n,percent=100*counts[t]/n if n else ''))
                    lines.append('| '+m+' | '+cond+' | '+str(n)+' | '+' | '.join(cell(counts[t],n) for t in types)+' |')
                    if group=='below4':
                        conditions.append(dict(dataset=ds,model=m,condition=cond,total=len(allrows),
                            below4=n,below4_percent=100*n/len(allrows),atleast4=len(allrows)-n))
            lines+=['']
    for ds in ['fpq','nfp']:
        for m in MODELS:
            rr=[r for r in pairs if r['dataset']==ds and r['model']==m]
            counter=Counter((r['plain']['score']<4,r['alternative']['score']<4,
                r['plain']['coding']['reference_alignment']['status'] if ds=='fpq' else r['plain']['coding']['stance'],
                r['alternative']['coding']['reference_alignment']['status'] if ds=='fpq' else r['alternative']['coding']['stance']) for r in rr)
            for (pl,al,pc,ac),n in counter.items():
                transitions.append(dict(dataset=ds,model=m,plain_low=pl,alternative_low=al,plain_code=pc,alternative_code=ac,n=n))
            assert sum(counter.values())==len(rr)
    write_csv('distributions.csv',distributions)
    write_csv('condition_denominators.csv',conditions)
    write_csv('paired_score_code_transitions.csv',transitions)
    challenge_changes=[]
    for m in MODELS:
        rr=[x for x in pairs if x['model']==m and x['dataset']=='nfp']
        pc=lambda x:x['plain']['coding']['stance']=='challenge'
        ac=lambda x:x['alternative']['coding']['stance']=='challenge'
        challenge_changes.append(dict(model=m,n=len(rr),plain_challenge=sum(pc(x) for x in rr),
            alternative_challenge=sum(ac(x) for x in rr),
            new_challenge=sum(not pc(x) and ac(x) for x in rr),
            new_challenge_alternative_low=sum(not pc(x) and ac(x) and x['alternative']['score']<4 for x in rr),
            ceased_challenge=sum(pc(x) and not ac(x) for x in rr)))
    write_csv('nfp_challenge_transitions.csv',challenge_changes)
    (OUT/'tables.md').write_text('\n'.join(lines).rstrip()+'\n')
    # Record optional tags separately; these are NOT the new causal categories.
    flags=[]
    for r in pairs:
        for cond in ['plain','alternative']:
            a=r[cond]
            for f in a['coding']['flags']:
                flags.append(dict(model=r['model'],id=r['id'],dataset=r['dataset'],condition=cond,
                    score=a['score'],tag=f['tag'],question_quote=f['question_quote'],answer_quote=f['answer_quote'],note=f['reason_ko']))
    write_csv('optional_flags_NOT_cause_rates.csv',flags)
    meta=dict(source=str(SRC.relative_to(ROOT)),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
        pairs=len(pairs),answers=len(flat),fpq_low=sum(r['dataset']=='fpq' and r['well_group']=='below4' for r in flat),
        nfp_low=sum(r['dataset']=='nfp' and r['well_group']=='below4' for r in flat),
        new_model_calls=0,new_semantic_population_audit=False,
        source_coder='Existing reference-assisted Sonnet per-answer audit; labels implied by references; model/condition/Well scores hidden during coding.',
        limitations=['Well<4 is an outcome grouping, not a proof of medical error or gate failure.',
                    'corrects_target with Well<4 is an evaluation disagreement, not proof of adequate correction.',
                    'Optional flags are overlapping and incomplete for the new proposed taxonomy.',
                    'Different alternative instructions prevent isolating model effects.',
                    'Qwen2.5 lacks fpq_389; retain original per-model denominators.',
                    'Qwen3.8 OFF and ON are two settings of one model family.'])
    (OUT/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(meta,ensure_ascii=False))


if __name__=='__main__': main()
