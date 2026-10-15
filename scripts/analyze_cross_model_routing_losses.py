"""Audit saved-answer routing across models; no model calls or score changes."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import well_eval as we
from scripts.analyze_detection_plain_link import table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/detection_plain_link_2026-09-28/cross_model_routing'
LEDGER = ROOT / 'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
Q25 = Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1')
G = ROOT / 'results/gemma4/full_20260926_v2'
L = ROOT / 'results/frontier_cli'
CONFIG = {
    'qwen25': (Q25/'well_judge_claude', Q25/'well_judge_claude_fpu'),
    'gemma': (G/'well/plain', G/'well/fp_unconditional'),
    'luna': (L/'luna_plain_20260922_v1/well_judge_sonnet_v1',
             L/'luna_corrections_20260922_v1/well_judge_sonnet_v1'),
}
NAMES = {'qwen25':'Qwen2.5', 'gemma':'Gemma', 'luna':'Luna'}
SOURCES = {}


def read(p, lines=False):
    data=p.read_bytes(); SOURCES[str(p)]=hashlib.sha256(data).hexdigest()
    return [json.loads(s) for s in data.decode().splitlines() if s.strip()] if lines else json.loads(data)


def load(well, method, questions, rubric):
    read(well/'judge_plan.json'); read(well/'attempts.jsonl', True)
    plan, ph=we.load_plan(well); _, scores=we.read_ledger(well, plan, ph)
    assert plan['model']=='claude-sonnet-5'
    assert rubric==plan['sources']['templates']['derived_templates']
    answers={}
    for src in plan['sources']['answers']:
        p=Path(src['path'])
        if p.name!=method+'.jsonl' and not (p.name=='answers_eval732.jsonl' and method in p.parts):
            continue
        data=read(p,True); assert SOURCES[str(p)]==src['sha256']
        for r in data:
            assert r['id'] not in answers
            answers[r['id']]=r
    assert answers
    out={}
    for q,r in answers.items():
        assert r['question']==questions[q]['question']==plan['questions'][q]['question']
        assert questions[q]['set']==plan['questions'][q]['set']
        assert we.digest(r)==plan['answer_hashes'][method][q]
        j=plan['mapping'][method][q]
        if j in scores:
            out[q]={'score':scores[j], 'answer':r['response'], 'job':j, 'well_dir':str(well)}
    return out


def summarize(rows):
    result={}
    for model in NAMES:
        result[model]={}
        for cl in ['fpq','nfp']:
            rr=[r for r in rows if r['model']==model and r['set']==cl]
            s={'n':len(rr)}
            for key in ['plain','corrected','direct_route','review_route']:
                s[key]=[r['id'] for r in rr if r[key]>=4]
            for gate in ['direct','review']:
                cells={}
                for g in [0,1]:
                    for p in [0,1]:
                        for c in [0,1]:
                            cells[f'{g}{p}{c}']=[r['id'] for r in rr if
                                (int(r[gate]),int(r['plain']>=4),int(r['corrected']>=4))==(g,p,c)]
                assert sum(map(len,cells.values()))==len(rr)
                s[gate+'_cells']=cells
            s['direct_to_review']={f'{a}{b}':[r['id'] for r in rr if
                (int(r['direct_route']>=4),int(r['review_route']>=4))==(a,b)]
                for a in [0,1] for b in [0,1]}
            result[model][cl]=s
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ledger=read(LEDGER,True); ix={(r['model'],r['id']):r for r in ledger}
    assert len(ix)==len(ledger)==3660
    qs={r['id']:r for r in ledger if r['model']=='gemma'}
    plan,_=we.load_plan(G/'well/plain'); rubric=plan['sources']['templates']['derived_templates']
    rows=[]; missing={}; cache={}
    for m,(pw,cw) in CONFIG.items():
        plain=load(pw,'plain',qs,rubric); corr=load(cw,'fp_unconditional',qs,rubric)
        cache[m]={'plain':plain,'corrected':corr}
        valid=set(plain)&set(corr); missing[m]=sorted(set(qs)-valid)
        for q in sorted(valid):
            r=ix[m,q]; p=plain[q]['score']; c=corr[q]['score']
            assert p==r['plain_score'] and plain[q]['answer']==r['plain']
            rows.append(dict(model=m,id=q,set=r['set'],plain=p,corrected=c,
                direct=r['direct'],review=r['review_gate'],
                direct_route=c if r['direct'] else p,review_route=c if r['review_gate'] else p,
                plain_job=plain[q]['job'],correction_job=corr[q]['job']))
    assert missing=={'qwen25':['fpq_389'],'gemma':[],'luna':[]}
    common=sorted(set.intersection(*[{r['id'] for r in rows if r['model']==m} for m in NAMES]))
    full=summarize(rows); paired=summarize([r for r in rows if r['id'] in common])
    # Previously inspected cases, selected for illustration, not prevalence.
    case_ids=['fpq_155','nfp_1149']
    evidence=['# 같은 문항에서 달라지는 실패 지점','',
              '이전에 읽었던 두 문항의 세 모델 교정 답변을 추가 연결했다. 목적 표본이며 의학적 진위 재판정이 아니다.','']
    for q in case_ids:
        evidence += [f'## {q}','',qs[q]['question'],'',
                     '참조: '+json.dumps({k:qs[q][k] for k in ['premise','correction','tpq_premises']},ensure_ascii=False),'']
        for m in NAMES:
            r=ix[m,q]; c=cache[m]['corrected'][q]
            evidence += [f'### {NAMES[m]}','',
                f"Review gate={r['review_gate']}; Plain={r['plain_score']}; 교정={c['score']}; 선택={c['score'] if r['review_gate'] else r['plain_score']}",'',
                '**검토문**','',r['review'],'','**무조건 교정 답변**','',c['answer'],'']
    quotes=[
        ('qwen25','fpq_155','The false premise in the question is the assumption that the wife has been diagnosed with ALL'),
        ('luna','fpq_155','necessarily means your wife will be unable to participate in daily activities'),
        ('qwen25','nfp_1149','the assumption that hair loss is caused by radiation rather than chemotherapy'),
        ('luna','nfp_1149','support groups are generally **not organized specifically around Wilms tumor–related hair loss**'),
    ]
    for m,q,quote in quotes:
        assert quote in cache[m]['corrected'][q]['answer']
    (OUT/'cases.md').write_text('\n'.join(evidence))
    availability={}
    for mode in ['thinking_off','thinking_on']:
        base=ROOT/'results/qwen38/full_transformers_20260926_v1'/mode
        availability[mode]={}
        for method in ['plain','zero_shot_cot_one_step','premise_review_answer','fp_unconditional']:
            counts=Counter(json.loads(p.read_text()).get('status') for p in (base/'records'/method).glob('*.json'))
            availability[mode][method]=dict(counts)
    summary=dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),full=full,common=paired,
                 common_ids=common,missing=missing,qwen38_record_status=availability,
                 illustrative_ids=case_ids,verified_correction_quotes=quotes,
                 source_sha256=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'links.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def rate(s,k): return f'{len(s[k])}/{s["n"]} ({100*len(s[k])/s["n"]:.1f}%)'
    doc=['# 세 모델의 gate 선택 손실 전수 비교','',
         '질문별 저장된 Plain·무조건 교정 답변과 기존 Well 점수를 연결했다. Direct 또는 Review gate가 Yes면 교정 답변, No면 Plain을 고른다. 검토문이나 gate의 구체적 지적 내용을 교정 답변에 전달한 실험은 아니다. 새 생성·채점·문턱 조정 없음.','',
         '## 같은 질문 731개에서 비교','',
         'FPQ 582/NFP 149. Qwen2.5 Plain이 없는 fpq_389를 세 모델 모두에서 제외한 표다. 두 집단 모두 Well ≥4 비율이 높을수록 좋다.','',
         table(['모델','집단','Plain','무조건 교정','Direct gate 선택','Review gate 선택'],[
             [NAMES[m],cl]+[rate(paired[m][cl],k) for k in ['plain','corrected','direct_route','review_route']]
             for m in NAMES for cl in ['fpq','nfp']]),'',
         '## 각 모델 전체 유효 문항: Review gate의 손실','',
         'Gemma·Luna는 FPQ 583/NFP 149, Qwen2.5는 FPQ 582/NFP 149. 위 공통 표와 분모가 다르다.','',
         table(['모델','놓쳤지만 교정 답변은 ≥4인 FPQ','탐지했지만 두 답변 모두 <4인 FPQ','두 답변 모두 <4인 FPQ 전체','오탐해 ≥4→<4가 된 NFP','NFP 오탐 전체'],[
             [NAMES[m],len(full[m]['fpq']['review_cells']['001']),len(full[m]['fpq']['review_cells']['100']),
              len(full[m]['fpq']['review_cells']['000'])+len(full[m]['fpq']['review_cells']['100']),
              len(full[m]['nfp']['review_cells']['110']),sum(len(v) for k,v in full[m]['nfp']['review_cells'].items() if k[0]=='1')]
             for m in NAMES]),'',
         '첫 수치 열은 Plain <4·gate No·교정 ≥4인 문항이다. 이미 Plain이 성공한 탐지 누락은 포함하지 않는다.',
         '이는 원인별 의미 감사가 아니라 저장 답변 선택의 기회·손실 집계다. 현재 두 답변 모두 <4인 문항은 그 둘 중 하나를 고르는 gate만 바꿔서는 ≥4가 되지 않는다. 새 답변 생성 방법의 가능성까지 부정하지 않는다.','',
         '## Direct에서 Review로 바꿀 때: 공통 문항의 실제 답변 선택 변화','',
         table(['모델','집단','<4→≥4','≥4→<4','순증'],[
             [NAMES[m],cl,len(paired[m][cl]['direct_to_review']['01']),len(paired[m][cl]['direct_to_review']['10']),
              len(paired[m][cl]['direct_to_review']['01'])-len(paired[m][cl]['direct_to_review']['10'])]
             for m in NAMES for cl in ['fpq','nfp']]),'',
         '## 범위와 제한','',
         '- Qwen3.8 OFF/ON은 무조건 교정 답변이 없어 같은 routing 집계에 넣지 않았다. 생성 파일 상태는 summary.json의 확인 시각과 함께 기록했다. 작업 프로세스의 생존 여부를 확인한 것은 아니다.',
         '- Qwen2.5는 Yes>No 토큰 점수, Gemma·Luna는 저장 JSON boolean 판정이다. 모델별 프롬프트·생성 조건 차이가 남아 있어 순수 모델 순위나 내부 원인으로 해석하지 않는다.',
         '- 무조건 교정은 질문에 오류가 있다고 단정하는 기존 지시다. 모든 교정 방법의 효과로 일반화하지 않는다.',
         '- 질문 일치, 답변 파일 SHA256, 개별 답변의 judge hash, 동일 Sonnet 판정기와 Well 템플릿 해시를 확인했다. Well 통과가 의학적 정확성·질문 의미의 완전한 보존을 인증하지 않는다.',
         '- 라벨 oracle도 가능한 모든 답변 방법의 상한이 아니다. FPQ에서도 Plain이 성공하고 교정이 실패한 경우가 있다.',
         '- 이 집계로 게이트가 잘못 판정한 이유를 알아낸 것은 아니다. 각 손실 집단의 검토문과 답변을 대조할 대상을 정한 것이다.','',
         '[전체 셀·문항 ID·원본 해시](summary.json) · [질문별 판정·점수](links.csv) · [해석](findings.md)','',
         '재현: `python3 -m scripts.analyze_cross_model_routing_losses`','']
    (OUT/'README.md').write_text('\n'.join(doc))
    print('\n'.join(doc[4:19]))
    print('Qwen3.8 availability:',availability)


if __name__=='__main__':main()
