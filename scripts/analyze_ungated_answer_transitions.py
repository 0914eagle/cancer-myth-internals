"""Pair saved ungated answers with Plain; keep protocol differences explicit."""
import csv
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

from src import well_eval as we
from scripts import analyze_cross_model_routing_losses as source
from scripts.analyze_detection_plain_link import table

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'ungated_answer_transitions'
DATA=ROOT/'results/analysis/ungated_answer_transitions_20260929_v1'
Q=source.Q25;G=source.G;L=source.L
Q38=ROOT/'results/qwen38/full_transformers_20260926_v1'
CONFIG={
 'qwen25':{'plain':Q/'well_judge_claude','zero_shot_cot':Q/'well_judge_claude',
           'premise_review':Q/'well_judge_claude','fp_unconditional':Q/'well_judge_claude_fpu'},
 'gemma':{m:G/'well'/m for m in ['plain','zero_shot_cot_one_step','premise_review_answer','fp_unconditional']},
 'luna':{'plain':L/'luna_plain_20260922_v1/well_judge_sonnet_v1',
         'balanced':L/'luna_corrections_20260922_v1/well_judge_sonnet_v1',
         'fp_unconditional':L/'luna_corrections_20260922_v1/well_judge_sonnet_v1'},
 'qwen38_off':{m:Q38/'thinking_off/well'/m for m in ['plain','zero_shot_cot_one_step']},
 'qwen38_on':{m:Q38/'thinking_on/well'/m for m in ['plain','zero_shot_cot_one_step']},
}
NAMES={'qwen25':'Qwen2.5','gemma':'Gemma','luna':'Luna','qwen38_off':'Qwen3.8 OFF','qwen38_on':'Qwen3.8 ON'}
METHODS={'zero_shot_cot':'CoT (추론 후 별도 답변)','zero_shot_cot_one_step':'One-step CoT',
         'premise_review':'검토문 → 답변','premise_review_answer':'검토문 → 답변',
         'balanced':'균형 지시','fp_unconditional':'무조건 교정'}


def main():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    original=source.read(source.LEDGER,True);ix={(r['model'],r['id']):r for r in original}
    questions={r['id']:r for r in original if r['model']=='gemma'}
    plan,_=we.load_plan(G/'well/plain');rubric=plan['sources']['templates']['derived_templates']
    allrows=[];summary={};answer_counts={}
    for model,cfg in CONFIG.items():
        answers={method:source.load(well,method,questions,rubric) for method,well in cfg.items()}
        answer_counts[model]={k:len(v) for k,v in answers.items()};summary[model]={}
        for method,alt in answers.items():
            if method=='plain':continue
            common=set(answers['plain'])&set(alt);summary[model][method]={}
            rr=[]
            for q in sorted(common):
                r=ix[model,q];p=answers['plain'][q];a=alt[q]
                assert p['answer']==r['plain'] and p['score']==r['plain_score']
                rr.append(dict(model=model,method=method,id=q,set=r['set'],question=r['question'],
                  premise=r['premise'],correction=r['correction'],tpq_premises=r['tpq_premises'],
                  plain_score=p['score'],answer_score=a['score'],plain=p['answer'],answer=a['answer'],
                  plain_job=p['job'],answer_job=a['job'],answer_well=a['well_dir'],
                  direct_gate=r['direct'],review_gate=r['review_gate']))
            allrows+=rr
            for cl in ['fpq','nfp']:
                subset=[r for r in rr if r['set']==cl];n=len(subset)
                cells={f'{a}{b}':[r['id'] for r in subset if (int(r['plain_score']>=4),int(r['answer_score']>=4))==(a,b)]
                       for a in [0,1] for b in [0,1]}
                matrix=[[sum(r['plain_score']==a and r['answer_score']==b for r in subset) for b in range(6)] for a in range(6)]
                assert sum(map(len,cells.values()))==sum(map(sum,matrix))==n
                summary[model][method][cl]=dict(n=n,plain_ge4=sum(r['plain_score']>=4 for r in subset),
                    answer_ge4=sum(r['answer_score']>=4 for r in subset),cells=cells,score_matrix=matrix,
                    missing=[q for q,r in questions.items() if r['set']==cl and q not in common])
    (DATA/'pairs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in allrows))
    (OUT/'summary.json').write_text(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
        models=summary,answer_counts=answer_counts,sources=source.SOURCES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    with (OUT/'pairs.csv').open('w') as f:
        keys=['model','method','id','set','plain_score','answer_score','direct_gate','review_gate']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in allrows)
    tbl=[]
    for m,methods in summary.items():
        for method,sets in methods.items():
            for cl,s in sets.items():
                n=s['n'];tbl.append([NAMES[m],METHODS[method],cl,n,
                    f"{100*s['plain_ge4']/n:.1f}%",f"{100*s['answer_ge4']/n:.1f}%",
                    len(s['cells']['01']),len(s['cells']['10'])])
    doc=['# Gate 없이 답했을 때: Plain 대비 문항별 변화','',
         '실제 생성된 두 답변을 비교한다. gate에 따른 선택·정답 라벨에 따른 선택을 하지 않았다. gate 결정은 사후 진단 자료로만 연결했으며 답변 생성에 투입된 값이 아니다. 새 생성·채점 없음.','',
         table(['모델','대안 조건','집단','공통 문항','Plain ≥4','대안 ≥4','<4→≥4','≥4→<4'],tbl),'',
         'FPQ·NFP 모두 ≥4가 높을수록 좋다. <4→≥4는 점수 기준 개선, ≥4→<4는 점수 기준 악화다. 실제 교정·과잉 반박 여부는 원문 독해로 따로 확인해야 한다. 같은 통과 여부 안의 점수 변화도 summary.json의 6×6 행렬에 보존했다.','',
         '## 비교 범위','',
         '- Qwen2.5 Plain은 fpq_389, CoT는 fpq_632가 각각 빠져 Plain–CoT 공통 비교는 FPQ 581/NFP 149다. 다른 Qwen2.5 비교는 FPQ 582/NFP 149, 다른 모델조건은 FPQ 583/NFP 149다. 누락을 실패로 채우지 않았다.',
         '- Qwen2.5 CoT는 추론 생성 후 별도 답변, Gemma·Qwen3.8 CoT는 한 번의 답변 생성이다. Luna는 완료된 균형 지시와 무조건 교정을 사용했으며 이를 CoT라고 부르지 않는다.',
         '- Qwen3.8은 완료·채점된 Plain 및 One-step CoT만 포함했다. 검토문→답변·무조건 교정은 이 표에 없다. 진행 중 작업을 변경하지 않았다.',
         '- 시스템·프롬프트·추론 설정·출력 제한이 완전히 같지 않다. 모델 순위 또는 CoT의 보편적 효과를 주장하지 않는다.',
         '- 같은 질문, 원본 답변 파일·개별 답변 해시, 판정 작업, 동일 Sonnet 및 Well 템플릿을 확인했다. 자동 평가의 타당성 한계는 남는다.','',
         '[문항별 판정·점수](pairs.csv) · [전체 전이 ID와 점수 행렬](summary.json) · [독해 결과](findings.md)','',
         '재현: `python3 -m scripts.analyze_ungated_answer_transitions`','']
    (OUT/'README.md').write_text('\n'.join(doc))
    print(table(['모델','조건','집단','n','Plain','대안','개선','악화'],tbl))


if __name__=='__main__':main()
