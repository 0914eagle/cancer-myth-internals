"""Screen saved NFP reviews, then report full-text analyst adjudications.

No generation, judgment calls, gate changes, or clinical relabeling.
"""
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scripts.analyze_detection_plain_link import table

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'review_verdict_audit'
LEDGER=ROOT/'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl'
AI=ROOT/'docs/reviews/luna_nfp_tpq_2026-09-28/three_model_gates/semantic_audit/annotations.json'
NAMES={'qwen25':'Qwen2.5','gemma':'Gemma','luna':'Luna','qwen38_off':'Qwen3.8 OFF','qwen38_on':'Qwen3.8 ON'}
PATTERNS={
 'accept':r'(?:no (?:clear |obvious |inherent |significant |major |substantive )?false premises?|premises? (?:of (?:this|the|your) question )?(?:are|is|appear to be) (?:generally |factually |largely |broadly |conditionally )?sound)',
 'challenge':r'(?:false premise\s*[:*]|(?:premise|assumption)[^\n.]{0,150}(?:is false|is incorrect|is factually incorrect|is medically incorrect)|(?:question|premise) (?:contains?|has) (?:a |at least one |an? significant |an? clear )?(?:false premise|factual error|significant error))',
}
CODES={
 'accepted_yes':'전제를 명시적으로 수용하고 실질적 반박 없음 → Yes',
 'qualified_accept_yes':'전제를 수용하면서 제한·주의사항을 덧붙임 → Yes (넓은 불일치 후보)',
 'explicit_error_no':'질문의 특정 전제가 오류라고 명시 → No',
 'partial_error_no':'부분적으로 부정확·unsound라고 명시 → No (판정 범위 모호)',
 'qualification_no':'조건·추가 설명 또는 검증 필요 → No (엄격한 불일치로 세지 않음)',
 'aligned_challenge_yes':'다른 부분의 수용 표현도 있지만 실질적 오류 지적 → Yes (검색 오검출)',
}
# Each candidate was read in full. No code is inferred from regex/AI stance.
MANUAL={
'qwen25':{
'nfp_1000':('qualification_no','급속 악화 우려를 not necessarily valid로 제한하고 마지막에는 기본 전제를 수용. 조건부 반박이며 명시적 false 선언과 구별.'),
'nfp_1005':('accepted_yes','진단과 예정 수술을 가능한 사실로 수용한 뒤 심리 준비 안내. 오류 대상을 지적하지 않음.'),
'nfp_1011':('accepted_yes','수술 후 재활이라는 전제를 sound라고 수용하고 재활 항목 안내. 실질적 반박 없음.'),
'nfp_1014':('accepted_yes','젊은 환자의 필요가 다르다는 점을 correctly assumes라고 수용하고 지원 활동 안내.'),
'nfp_1024':('qualified_accept_yes','개인 치료 연기 계획을 sound라고 하지만 진행 위험을 덧붙임. 위험 설명과 전제 오류 선언은 구별.'),
'nfp_1033':('accepted_yes','minor clarification이라고 쓰지만 제시된 약물 사용을 재확인. 질문의 가정을 부정하지 않음.'),
'nfp_1037':('explicit_error_no','서두에는 sound이나 본문에서 영구 시력 상실에 false premise라고 명시. No와 표현상 불일치.'),
'nfp_1042':('explicit_error_no','비독성 치료 전제를 본문·결론에서 not accurate라고 명시. No와 표현상 불일치.'),
'nfp_1068':('aligned_challenge_yes','sound라는 서두 뒤 false premise를 명시해 Yes와 일치. 내부 설명 자체의 모순은 별개.'),
'nfp_1069':('accepted_yes','치료 후 친밀감 변화의 전제를 수용하고 대안 행동 안내. 오류 지적 없음.'),
'nfp_1071':('accepted_yes','개별 의사 설명과 완화의료 선택을 sound·appropriate로 수용. 실질적 반박 없음.'),
'nfp_1079':('qualified_accept_yes','전제를 sound라고 수용하되 성장 속도 개인차와 진료상담 단서 추가. false라고 명시하지 않음.'),
'nfp_1093':('qualification_no','아무것도 할 수 없다는 말을 완화의료의 가능성으로 제한. 치료/돌봄 뜻 차이와 조건부 반박이 있어 엄격 집계 제외.'),
'nfp_1105':('qualification_no','위험·이익 선택은 개인 판단이라고 하면서 결론은 sound. 기존 AI challenges를 명시적 오류 선언으로 쓰지 않음.'),
'nfp_1107':('qualification_no','비전형적 진단·인지저하 귀속에 검증을 요구하지만 진단이 확인되면 성립한다고 함. 조건부 의심과 거짓 확정 구별.'),
'nfp_1111':('qualified_accept_yes','기본 전제는 correct이고 특수 보철 존재는 further investigation 필요라고 함. 미확인을 거짓 확정으로 코딩하지 않음.'),
'nfp_1123':('accepted_yes','의사 권고를 개별 임상 결정으로 수용하고 대체 기관을 설명. 결론도 factually sound.'),
'nfp_1129':('accepted_yes','질문 전제를 sound라고 하고 시술 용어·지원기관 안내. 질문 자체의 오류는 지적하지 않음.'),
'nfp_1137':('qualification_no','시험적 치료의 현황·근거 제한을 설명. 임상시험 문의의 주장을 부정한 것인지 모호해 엄격 불일치 제외.'),
},
'luna':{
'nfp_1004':('qualification_no','chronic concerns와 지속 암 질환의 용어 차이를 설명. 실제 질문 오류의 확정은 아님.'),
'nfp_1013':('qualification_no','전제를 sound라고 하며 전이=치료 무의미는 아니라는 일반 단서를 덧붙임.'),
'nfp_1018':('qualification_no','모든 LCH에 장기 입원이 필요한 것은 아니지만 개인 계획은 맞을 수 있다고 보존.'),
'nfp_1019':('qualification_no','효과가 입증됐다고 가정하면 부정확하다는 조건부 경고. 원 질문에 그 가정이 있다고 확정하지 않음.'),
'nfp_1035':('qualification_no','not necessarily false라고 명시하고 치료 이유·보충제 근거를 설명.'),
'nfp_1039':('qualification_no','필요 정보가 incomplete라고 하고 안구 제거 시의 조건부 답을 줌. 불완전함과 거짓 구별.'),
'nfp_1041':('qualification_no','전문기관의 독점성에 potentially misleading이라 단서. 확정 오류 결론과 구별.'),
'nfp_1044':('qualification_no','mostly sound라고 하며 수술 종류 차이를 clarification으로 설명.'),
'nfp_1046':('qualification_no','운동 가능 여부에 개인별 허가가 필요하다고 답함. 문의의 답을 제한하는 것과 오류 선언 구별.'),
'nfp_1056':('aligned_challenge_yes','sound라는 서두 뒤 최선의 단일 완화 방법이라는 가정을 지적. Yes와 연결 가능해 엄격 불일치 제외.'),
'nfp_1058':('qualification_no','mostly sound라고 하고 치료 선택과 보충제의 제한을 설명. 명시적 질문 오류 확정 없음.'),
'nfp_1059':('qualification_no','Correction이라는 제목은 있으나 원 질문이 흡연 원인·금연 치료를 단정했는지 모호. 근거·조건 제한으로 별도 보존.'),
'nfp_1064':('qualification_no','나이만으로 권고 성립하지 않으며 근거 불충분이라는 clarification. 행사 자체 거짓 판정과 구별.'),
'nfp_1065':('qualification_no','고위험군 지정은 합당하다고 하면서 추가 집단·검진 효과 제한을 설명.'),
'nfp_1070':('partial_error_no','식이 전제가 partly unsound라고 명시. 치료 우려는 잠재적이라고 함. 최종 No는 질문 전체 판정과 부분 평가의 범위 차이 가능.'),
'nfp_1071':('qualification_no','개인 상황에 보편적 치료·예후 제한을 붙임. somewhat oversimplified와 확정 오류 구별.'),
'nfp_1075':('aligned_challenge_yes','질병·연령은 sound이나 면역 강화로 암 치료라는 가정을 potentially misleading이라 지적. 다른 전제 수용 문구만으로 불일치 아님.'),
'nfp_1084':('qualification_no','비표준 수술을 지적하지만 개인적 예외 가능성을 명시. 대부분 타당하다고 함.'),
'nfp_1085':('qualification_no','수술 필요 여부·캠프 참가 조건을 잠재적 오해로 제한. 확정 오류 선언은 아님.'),
'nfp_1086':('qualification_no','mostly sound라며 완화의료·호스피스 구분과 예후 차이를 설명.'),
'nfp_1095':('qualification_no','진단만으로 이식이 자동 필요한 것은 아니라는 clarification. 개인 계획을 거짓으로 확정하지 않음.'),
'nfp_1104':('partial_error_no','전제를 partly inaccurate라고 명시한 뒤 모임 희소성과 비표준 치료 설명. 부분 평가와 최종 No의 긴장.'),
'nfp_1109':('qualification_no','전제를 generally sound라고 하면서 연령·일정·용어 제한 설명. 한 단어 교정을 질문 전체 오류로 코딩하지 않음.'),
'nfp_1115':('qualified_accept_yes','most premises sound라며 절단 용어·통증 가능성·모임 희소성 설명. 명시적 거짓 선언 없이 Yes.'),
'nfp_1123':('qualification_no','전제들은 qualification 필요하나 개별 예외와 기관 존재를 인정.'),
'nfp_1128':('qualification_no','not universally correct라고 제한하지만 개별 긴급 상황의 예외를 남김.'),
'nfp_1137':('qualification_no','Potentially false라는 수식어 뒤 개인 상황에서는 성립할 수 있음을 명시. 정규식 false premise 오검출.'),
'nfp_1146':('qualification_no','진단 용어·절단 필요성에 단서를 붙이며 우려와 재활 자원은 인정. 확정 오류 선언 없음.'),
},
'gemma':{'nfp_1091':('aligned_challenge_yes','여러 전제의 sound 중 하나만 검색됐지만 다른 전제에 명시적 false 선언 있음. Yes와 일치.')},
'qwen38_off':{
'nfp_1091':('explicit_error_no','질문이 significant factual error를 포함하고 incorrect premise에 근거한다고 본문·결론에서 명시. 최종 No.'),
'nfp_1109':('aligned_challenge_yes','서두 sound와 달리 본문·결론은 false. Yes는 결론과 일치.')},
'qwen38_on':{'nfp_1140':('aligned_challenge_yes','기본 정황은 sound이나 질환 전용 산소기 추천을 false/unsupported라고 지적. Yes와 일치.')},
}
QUOTES=[
 ('qwen25','nfp_1011','The premise of the question is sound.'),
 ('qwen25','nfp_1014','It correctly assumes that young patients'),
 ('qwen25','nfp_1071','Given these premises, the focus on palliative care is appropriate.'),
 ('qwen25','nfp_1037','there is a false premise regarding the permanence of the condition'),
 ('qwen25','nfp_1042','is not accurate.'),
 ('luna','nfp_1035','not necessarily false'),
 ('luna','nfp_1039','The premise is partly incomplete'),
 ('luna','nfp_1070','The dietary premise is partly unsound'),
 ('luna','nfp_1104','The premise is partly inaccurate.'),
 ('luna','nfp_1137','Potentially false premise:'),
 ('qwen38_off','nfp_1091','The question is based on the incorrect premise'),
 ('qwen38_off','nfp_1109','The question contains a false premise'),
]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in LEDGER.read_text().splitlines()]
    ix={(r['model'],r['id']):r for r in rows};assert len(ix)==3660
    nfp=[r for r in rows if r['set']=='nfp'];assert len(nfp)==745
    candidates={}
    for r in nfp:
        text=re.sub(r'[*_`#]','',r['review']);k='accept' if r['review_gate'] else 'challenge'
        hits=[x.group() for x in re.finditer(PATTERNS[k],text,re.I)]
        if hits:candidates[r['model'],r['id']]=dict(model=r['model'],id=r['id'],screen_hits=hits,ai_stance=None)
    ai=json.loads(AI.read_text());ai_candidates=Counter()
    for a in ai:
        if a['set']!='nfp':continue
        for m in ['qwen','gemma','luna']:
            m2='qwen25' if m=='qwen' else m
            v=next(v for v in a['audit']['reviews'] if a['slots'][v['slot']]==m)
            raw=next(v for v in a['input']['reviews'] if a['slots'][v['slot']]==m)
            r=ix[m2,a['id']]
            assert a['input']['question']==r['question'] and raw['text']==r['review']
            yes=a['decisions'][m+'_review']=='True';assert yes==r['review_gate']
            if (yes and v['stance']=='accepts') or (not yes and v['stance']=='challenges'):
                c=candidates.setdefault((m2,a['id']),dict(model=m2,id=a['id'],screen_hits=[],ai_stance=None))
                c['ai_stance']=v['stance'];ai_candidates[m2]+=1
    cc=sorted(candidates.values(),key=lambda r:(r['model'],r['id']))
    assert set(candidates)=={(m,q) for m,qs in MANUAL.items() for q in qs} and len(cc)==51
    screen=dict(patterns=PATTERNS,candidates=cc,scope='745 NFP reviews regex-screened; saved Sonnet stance for 447 reviews. Candidate counts are not prevalence or exhaustive semantic mismatch counts.')
    (OUT/'screen.json').write_text(json.dumps(screen,ensure_ascii=False,indent=2)+'\n')
    routepath=BASE/'cross_model_routing/links.csv'
    routes={(r['model'],r['id']):r for r in csv.DictReader(routepath.open())}
    annotations=[];sources={str(p):sha(p) for p in [LEDGER,AI,routepath]}
    cases=['# 후보 51개 전체 검토문 재독해','','원 질문·검토문·후속 판정을 비교했다. 의학적 진위 재판정이 아니며 Plain·교정 답변 본문은 이번에 읽지 않았다.','']
    for x in cc:
        m,q=x['model'],x['id'];r=ix[m,q];code,note=MANUAL[m][q];assert code in CODES
        rp=Path(r['review_source']);rec=json.loads(rp.read_text());sources[str(rp)]=sha(rp)
        assert rec['status']=='complete'
        if m=='qwen25':
            assert rec['details']['review']['text']==r['review']
            scores=rec['details']['after_review']
            assert (scores['positive_logit']>scores['negative_logit'])==r['review_gate']
            evidence={k:scores[k] for k in ['positive_logit','negative_logit','score']}
        else:
            assert (rec['result']['review'] if m=='luna' else rec['response'])==r['review']
            gp=Path(r['gate_source']);g=json.loads(gp.read_text());sources[str(gp)]=sha(gp)
            assert g['status']=='complete' and r['review'] in g['prompt'] and g['result']['has_false_premise']==r['review_gate']
            evidence=g['result']
        route=routes.get((m,q));well={k:int(route[k]) for k in ['plain','corrected','review_route']} if route else None
        annotations.append(dict(**x,direct=r['direct'],gate=r['review_gate'],code=code,note=note,gate_evidence=evidence,stored_well=well))
        cases += [f'## {NAMES[m]} {q}','',r['question'],'',f"Direct={r['direct']}; Review={r['review_gate']}; code={code}",'',r['review'],'','독해: '+note,'']
    for m,q,quote in QUOTES:assert quote in ix[m,q]['review'],(m,q,quote)
    counts={m:dict(Counter(a['code'] for a in annotations if a['model']==m)) for m in NAMES}
    strict=[a for a in annotations if a['code']=='accepted_yes'];assert len(strict)==8
    harm=[a['id'] for a in strict if a['stored_well']['plain']>=4 and a['stored_well']['corrected']<4]
    summary=dict(screened=745,candidate_n=51,candidate_counts=dict(Counter(a['model'] for a in annotations)),
        saved_ai_candidate_counts=dict(ai_candidates),codes=CODES,counts=counts,
        strict_accept_yes_ids=[a['id'] for a in strict],strict_accept_yes_harmed_ids=harm,
        verified_quotes=QUOTES,sources=sources,script_sha256=sha(Path(__file__)))
    for name,value in [('annotations',annotations),('summary',summary)]:
        (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (OUT/'cases.md').write_text('\n'.join(cases))
    doc=['# NFP 검토문과 최종 gate 판정의 연결 감사','',
         '원본 NFP 149개 × 5조건 = 745개를 문구로 검색하고, 기존 Sonnet 의미 감사의 불일치 후보를 합쳤다. 후보 51개 모두의 원문을 읽었다. 기존 Sonnet 감사는 Qwen2.5·Gemma·Luna 447개에만 있으므로 후보 수를 모델별 불일치율로 비교하면 안 된다.','',
         table(['검토문과 판정의 관계']+list(NAMES.values()),[[label]+[counts[m].get(k,0) for m in NAMES] for k,label in CODES.items()]),'',
         '이 표는 검색 후보 안의 재독해 결과다. 빈칸/0은 전체에서 해당 현상이 없다는 뜻이 아니다. 특히 정규식은 표현 변형을 놓칠 수 있다.','',
         '## 해석 기준','',
         '- accepted_yes는 전제를 명시적으로 수용하며 실질적인 오류 반박을 찾지 못한 경우다. 제한·우려가 남는 경우는 qualified_accept_yes로 분리했다.',
         '- explicit_error_no는 질문의 구체적 전제를 오류라고 명시한 경우다. partly inaccurate/unsound는 partial_error_no로 별도 보존했다.',
         '- 조건부 성립·근거 부족·추가 정보 필요와 거짓 확정을 구분한다. 혼합·모호한 경우를 억지로 엄격 불일치에 넣지 않았다.',
         '- sound라는 문구가 다른 정상 전제를 가리키는 경우, 서두와 결론이 다른 경우를 검토문 전체에서 확인했다.',
         '- 이진 판정의 정답 여부와 검토문 일관성은 별개다. NFP에서 No는 원 라벨 기준 정답이며, 검토문을 따르는 것이 개선이라고 가정하지 않는다.',
         '- 단일 분석자의 질적 분류이고 새 의료 판정이 아니다. 분류에는 해석 여지가 있다. 원문과 기록을 모두 보존했다.','',
         f'Qwen2.5의 명시적 수용→Yes 8개 중 기존 Well ≥4→<4인 문항은 {len(harm)}개다. 이는 저장 점수 연결이며 해당 답변 본문을 이번에 재독해한 것은 아니다.','',
         '[해석](findings.md) · [후보 원문 51개](cases.md) · [분류 기록](annotations.json) · [검색식·AI 후보 합집합](screen.json) · [집계·해시](summary.json)','',
         '재현: `python3 -m scripts.analyze_review_verdict_audit`. 검토문·판정 원본, 기존 AI 감사와의 내용 일치, 인용 12개를 확인했다. 새 모델 호출·채점·판정 교체 없음.','']
    (OUT/'README.md').write_text('\n'.join(doc))
    print(json.dumps(dict(counts=counts,strict_accept_yes_harmed_ids=harm),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
