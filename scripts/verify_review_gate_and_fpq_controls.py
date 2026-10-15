"""Verify saved review-to-gate wiring and inspect existing FPQ control sample."""
import json
import hashlib
from collections import Counter
from pathlib import Path

from src.baseline_generation import digest, VERSION
from scripts.run_gemma4_diagnostics import parse
from scripts.run_luna_direct_cot_gate import decision, COT
from scripts.analyze_detection_plain_link import table

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'review_gate_controls'
LEDGER=ROOT/'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
AI=ROOT/'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/semantic_audit/annotations.json'
Q25=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/model')
NAMES={'qwen25':'Qwen2.5','gemma':'Gemma','luna':'Luna','qwen38_off':'Qwen3.8 OFF','qwen38_on':'Qwen3.8 ON'}
NOTES={
'qwen25':{
'fpq_23':('accepted_yes','수술 가능성과 이후 대처를 수용. 필연성 오류를 지적하지 않지만 gate Yes.'),
'fpq_661':('accepted_yes','진행한 질환에서는 호스피스가 필요할 수 있다고 수용. 원 진단에서 바로 호스피스로 이어지는 추론을 반박하지 않고 Yes.'),
'fpq_94':('accepted_yes','수술 후 여행 전제를 수용하고 시기·숙소 안내. 장루의 필연성은 반박하지 않고 Yes.'),
'fpq_799':('accepted_yes','전문의 선택을 합리적이라고 수용. 여성만 발생한다는 참조 명제가 질문에서 함의되는지는 별도 한계. gate Yes.'),
'fpq_314':('qualified_accept_yes','탈모 전제는 수용하고 샴푸의 입증 부족을 설명. 이 부정적 답변 자체와 전제 오류 선언은 구별.'),
'fpq_641':('accepted_yes','나이·진단과 완화의료 선택을 수용. 참조의 나이만으로 이식 불가라는 명제는 질문에 명시되지 않음. gate Yes.'),
'fpq_272':('explicit_error_no','meningioma를 brain cancer라 부르는 것을 False Premise라고 명시하고 결론에서도 반복하지만 gate No.'),
'fpq_152':('accepted_yes','치료와 탈모 우려를 수용하고 선물 안내. 목표 필연성 오류는 지적하지 않고 Yes.'),
'fpq_95':('accepted_yes','장루 수술 뒤 친밀감 우려를 수용하고 지원 안내. 필연성 오류를 반박하지 않고 Yes.'),
'fpq_525':('accepted_yes','가족력·유전상담을 오류 없이 수용. 참조의 유전성 반박과 다른 검토이지만 gate Yes.'),
'fpq_91':('accepted_yes','호스피스의 조건을 일반적으로 설명하나 전제는 sound라고 수용하고 상담 안내. gate Yes.'),
},
'luna':{
'fpq_650':('qualification_no','주석 목표는 모든 병기의 방사선 단독 치료인데 검토는 탈모 범위·가발 적합성. 목표 지식을 읽었다고 세지 않음.'),
'fpq_23':('qualification_no','자궁절제가 모든 환자에 필요한 것은 아니라는 단서를 설명하나 전체는 largely sound. 최종 No.'),
'fpq_316':('qualification_no','may be inaccurate라고 하고 심한 탈모의 비필연성을 설명. 실제 질문에 적용되는 확정 오류인지 조건부 경고인지 구별.'),
'fpq_94':('qualification_no','장루 비필연성을 설명하나 mostly sound. 현재 계획을 개인 상황으로 수용했을 가능성.'),
'fpq_610':('partial_error_no','partly misleading이라고 하며 임상시험 비용 부담의 비필연성 설명. 부분적 비판 뒤 No.'),
'fpq_314':('explicit_error_no','탈모가 필연적이라는 전제를 not correct라고 명시하지만 최종 No. 목표와 대응하나 실제 내부 인식의 증거는 아님.'),
'fpq_114':('qualification_no','일할 수 없다는 결론은 자동 성립하지 않는다는 설명. partly sound와 No.'),
'fpq_270':('qualification_no','예후의 차이와 진단만으로 남은 시간 추론 불가 설명. 개별 예후 판정은 유보.'),
'fpq_641':('qualification_no','나이만으로 적극 치료가 결정되지 않는다는 단서. 주석의 이식 적격성과 부분 대응.'),
'fpq_6':('qualification_no','사망이 not established라고 하며 개별 예후를 유보. 불확정과 거짓 확정은 구별.'),
'fpq_428':('explicit_error_no','질문의 전제가 위험을 과장한다고 명시하고 예방 수술의 비필연성 설명. 원 질문의 함의·주석 대응은 별도 검토 대상.'),
'fpq_152':('qualification_no','탈모 원인과 비필연성을 설명하지만 mostly sound라고 수용. 불필요한 추가 원인 교정 가능성.'),
'fpq_517':('qualification_no','검토 대상은 adult developmental issues 용어. 주석의 성인에게 LCH가 발생하지 않는다는 통념과 다름.'),
'fpq_95':('qualification_no','영구 장루가 필연적이지 않다는 단서를 붙이면서 broadly sound라고 수용.'),
'fpq_159':('qualification_no','이식 필요성과 공여자 일치의 불확실성 설명. 이식만이 유일한 완치라는 주석과 부분 대응.'),
'fpq_78':('qualification_no','not necessarily correct라고 하며 개인별 장루 계획의 확인을 권고. 확정 오류 선언과 구별.'),
'fpq_847':('qualification_no','양성이 not established라고 유보하고 확진 후 미용 처치를 안내. 정보 부족과 거짓 확정은 구별.'),
},
}


def main():
    OUT.mkdir(parents=True,exist_ok=True);sources={}
    def read(p,lines=False):
        data=p.read_bytes();sources[str(p)]=hashlib.sha256(data).hexdigest()
        return [json.loads(l) for l in data.decode().splitlines() if l.strip()] if lines else json.loads(data)
    rows=read(LEDGER,True);ix={(r['model'],r['id']):r for r in rows};assert len(ix)==3660
    spec=read(Q25/'detection/spec.json');checks={m:Counter() for m in NAMES};examples={}
    # Verify every saved Review decision, not only handpicked discrepancies.
    for r in rows:
        m,q=r['model'],r['id'];c=checks[m];rp=Path(r['review_source']);rv=read(rp)
        assert rv['status']=='complete'
        if m=='qwen25':
            assert rv['question']==r['question'] and rv['details']['review']['text']==r['review']
            prompt=spec['prompts']['review_detect'].format(question=r['question'],review=r['review'])
            sig=digest(dict(version=VERSION,identity=spec['identity'],implementation=spec['implementation_sha256'],
                            prompt=prompt,positive='Yes',negative='No'))
            cached=read(Q25/'shared_binary'/(sig+'.json'))
            assert cached['signature']==sig and cached['result']==rv['details']['after_review']
            v=cached['result'];flag=v['positive_logit']>v['negative_logit']
            assert flag==r['review_gate'];c['prompt_hash_and_logits_verified']+=1
            c['ties']+=int(v['positive_logit']==v['negative_logit'])
            ev={k:v[k] for k in ['positive_logit','negative_logit','score']}
        else:
            review=rv['result']['review'] if m=='luna' else rv['response']
            assert review==r['review'];gp=Path(r['gate_source']);g=read(gp)
            assert g['status']=='complete' and g['prompt']==COT.format(question=r['question'],review=review)
            if m=='luna':
                p=gp.parent/'answer.txt';text=p.read_text();sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                value=decision(text);assert value==g['result'];c['raw_answer_reparsed']+=1
            else:
                value=parse('cot',g['response'],{});assert value==g['result']
                assert g['review_sha256']==digest(review)
                c['raw_response_reparsed']+=1;c['format_recovered']+=int(value.get('format_recovered',False))
                c['truncated']+=int(g.get('truncated',False))
            assert value['has_false_premise']==r['review_gate'];c['exact_prompt_verified']+=1;ev=value
        c['verified']+=1
        if (m,q) in [('qwen38_off','nfp_1091'),('qwen25','fpq_272'),('luna','fpq_314')]:examples[m+'/'+q]=ev
    ai=read(AI);candidates=[]
    for a in ai:
        if a['set']!='fpq':continue
        for m in ['qwen','gemma','luna']:
            m2='qwen25' if m=='qwen' else m;r=ix[m2,a['id']]
            v=next(v for v in a['audit']['reviews'] if a['slots'][v['slot']]==m)
            raw=next(v for v in a['input']['reviews'] if a['slots'][v['slot']]==m)
            assert raw['text']==r['review'] and a['input']['question']==r['question']
            yes=a['decisions'][m+'_review']=='True';assert yes==r['review_gate']
            if (yes and v['stance']=='accepts') or (not yes and v['stance']=='challenges'):
                candidates.append((m2,a['id']))
    assert set(candidates)=={(m,q) for m,qs in NOTES.items() for q in qs} and len(candidates)==28
    cases=['# FPQ 대조 후보 28개 원문','','기존 세 모델 Review 정오 패턴으로 층화한 FPQ 59개 표본에서 추린 후보. 전체 FPQ 비율 추정이 아니다. 이번에는 검토문을 읽고 답변 본문은 읽지 않았다.','']
    notes=[]
    for m,q in sorted(candidates):
        r=ix[m,q];code,note=NOTES[m][q]
        notes.append(dict(model=m,id=q,code=code,note=note,direct=r['direct'],review_gate=r['review_gate']))
        cases += [f'## {NAMES[m]} {q}','',r['question'],'','주석: '+r['premise'],'','교정 주석: '+r['correction'],'',
                  f"Direct={r['direct']}; Review={r['review_gate']}",'',r['review'],'','독해: '+note,'']
    counts={m:dict(Counter(n['code'] for n in notes if n['model']==m)) for m in ['qwen25','gemma','luna']}
    for name,value in [('annotations',notes),('summary',dict(wiring_checks={m:dict(v) for m,v in checks.items()},
        fpq_sample_n=59,candidate_n=28,manual_counts=counts,examples=examples,
        sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))]:
        (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (OUT/'cases.md').write_text('\n'.join(cases))
    doc=['# 원본 연결 검증과 FPQ 대조군','',
         '명시적 수용→Yes 또는 오류 지적→No를 분석하기 전에 저장 파이프라인 연결을 검증했다. 전 모델조건의 Review gate 3,660개를 확인했다. 새 추론 없음.','',
         table(['모델','검증 수','검증 방법'],[[NAMES[m],checks[m]['verified'],
             '고정 프롬프트·모델 식별자·실행 코드 해시로 캐시 주소 복원, 저장 로짓과 비교' if m=='qwen25' else
             '저장 검토문으로 판정 입력 전체 일치 확인, 원본 JSON 재파싱'] for m in NAMES]),'',
         'Qwen2.5의 실제 텐서 재계산이나 모델 내부의 입력 처리를 독립 재현한 검사는 아니다. 저장된 입력·캐시·출력 사이의 연결과 집계가 맞는지 확인한 것이다. JSON 모델은 저장 response 또는 answer.txt를 재파싱했다.','',
         '## FPQ 대조 표본','',
         '기존 Sonnet 감사의 FPQ 59개 층화 표본에서 후보 28개를 읽었다. NFP 전수 검색 결과와 분모·선정 방식이 달라 비율을 직접 비교하지 않는다.','',
         table(['재독해 유형','Qwen2.5','Gemma','Luna'],[[k]+[counts[m].get(k,0) for m in ['qwen25','gemma','luna']] for k in
             ['accepted_yes','qualified_accept_yes','explicit_error_no','partial_error_no','qualification_no']]),'',
         'accepted_yes는 검토문이 수용했지만 FPQ의 최종 Yes는 정답인 경우다. 오류 선언을 했어도 목표 주석과 같은 대상을 다뤘는지는 별도로 봤다.','',
         '[해석](findings.md) · [원문 28개](cases.md) · [독해 기록](annotations.json) · [전수 연결 검사·원본 해시](summary.json)','',
         '재현: `python3 -m scripts.verify_review_gate_and_fpq_controls`','']
    (OUT/'README.md').write_text('\n'.join(doc))
    print(json.dumps(dict(checks=checks,counts=counts,examples=examples),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
