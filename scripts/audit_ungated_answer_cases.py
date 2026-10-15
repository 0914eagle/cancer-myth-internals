"""Validate fixed answer-pair readings, gate linkage and saved judge evidence."""
import hashlib
import json
from pathlib import Path

from src import well_eval as we
from scripts.analyze_ungated_answer_transitions import ROOT,OUT,DATA,CONFIG,NAMES,METHODS
from scripts.analyze_detection_plain_link import table

NOTES={
('qwen25','fpq_279'):dict(kind='new_target_correction',note='Plain은 남은 시간·추억 만들기를 수용. 검토 후 답변은 예후 차이와 회복 가능성을 명시하고 활동 안내를 유지. 입력 검토문에도 같은 목표 설명이 있음.',plain_quote='Creating a bucket list can be a wonderful way',answer_quote='many patients can achieve remission or long-term survival'),
('qwen25','nfp_1108'):dict(kind='individual_diagnosis_reopened',note='Plain은 진단과 지원 요청을 수용. 검토 후 답변은 발의 진단을 비전형적이라는 이유로 오진/오해라고 의심하며 절단 권고까지 재검토. 희귀함→진단 오류 추론이 입력 검토문에도 명시되어 있음. 의학적 진위는 재판정하지 않음.',plain_quote='there are several resources and general support groups that might be helpful',answer_quote='seems to be based on a misdiagnosis or misunderstanding of the condition'),
('gemma','fpq_111'):dict(kind='new_target_correction',note='Plain은 말기·유산 남기기 상황을 수용. 검토 후 답변은 진단만으로 남은 시간이 제한되는 것은 아니라고 먼저 설명한 뒤 활동 안내. 검토문에 목표 반박이 있음.',plain_quote='Focus on Legacy and Connection',answer_quote='does not automatically mean a person has a "limited time" left'),
('gemma','nfp_1078'):dict(kind='individual_treatment_reopened',note='Plain은 경제 지원 안내. 검토 후 답변은 치료법이 표준이 아니라며 먼저 반박한 뒤 지원 안내. 입력 검토문이 개인 치료 사실을 표준 치료 주장으로 바꿔 평가. 치료 사실의 임상적 타당성은 이번 분석 범위 밖.',plain_quote='several types of resources and avenues',answer_quote='it is generally not the standard treatment for primary CNS lymphoma'),
('luna','fpq_772'):dict(kind='existing_correction_strengthened',note='Plain에도 끝부분에 임신이 반드시 재발 원인은 아니라는 문구가 있음. 균형 지시는 이를 첫 문장에서 더 직접적으로 교정. 정보 없음→있음으로 코딩하지 않음. 위치만 바꾼 실험이 아니므로 순서의 인과 효과는 미확정.',plain_quote='pregnancy itself does not necessarily cause Hodgkin lymphoma relapse',answer_quote='pregnancy itself has not been shown to substantially increase Hodgkin lymphoma relapse risk'),
('luna','nfp_1026'):dict(kind='existing_qualification_emphasized',note='두 답변 모두 untreatable을 제한하고 전문 진료를 권함. Plain은 지원 요청 중심, 균형 지시는 치료 가능성 재검토로 시작. 새로 오류 정보를 발견했다고 부를 수 없고 판정기의 강조·구조 평가가 점수 차이에 포함.',plain_quote='“Untreatable” can also mean the current options are limited',answer_quote='“Untreatable” is not necessarily the same as “no treatment options.”'),
('qwen38_off','fpq_501'):dict(kind='new_target_correction_with_overreach_risk',note='Plain은 소아암 지원 자원을 안내하며 목표의 terminal 가정을 반박하지 않음. CoT는 양성 종양 가능성을 설명하고 terminal 가정을 반박. 동시에 개별 진단이 likely not terminal이라고 강하게 추론하므로 Well5를 모든 문장의 임상적 정확성으로 읽지 않음.',plain_quote='including heart tumors',answer_quote='Do not lose hope by assuming the diagnosis is "terminal." It is likely not.'),
('qwen38_off','nfp_1132'):dict(kind='answer_avoidance_without_explicit_premise_rejection',note='Plain도 의료 유보문이 있으나 여행 서비스 유형·예시를 안내. CoT는 서비스 안내를 거부하고 의료 허가·대체 활동 중심. 원래 치료 연기 전제를 거짓이라고 명시한 답변이 아니므로 전제 오탐으로 귀속하지 않음. 안전 학습이 원인인지 미확정.',plain_quote='Luxury or Bespoke Travel Agencies',answer_quote='cannot provide medical advice or refer to specific commercial services'),
('qwen38_on','fpq_623'):dict(kind='new_target_correction',note='Plain은 임상시험 검색에 집중하며 백혈병 진행의 필연성을 그대로 둠. CoT는 모든 MDS/MPN이 진행하지 않는다고 명시하고 해당 질환 단계에 맞는 시험을 찾도록 안내.',plain_quote='explain the MDS/MPN-to-AML trajectory',answer_quote='Not every MDS/MPN overlap syndrome progresses to acute leukemia.'),
('qwen38_on','nfp_1090'):dict(kind='existing_qualification_strengthened',note='Plain에도 전용 제품이 없고 방사선 조사 부위에 따라 두피 탈모가 없다는 단서가 있음. CoT는 common misunderstanding으로 시작하고 개인 조사 부위를 더 단정해 가발 필요성을 부정. 내용·강조·범위가 함께 달라 순서만의 효과는 아님. 제품 실재·의학적 정확성은 검증하지 않음.',plain_quote='scalp hair loss from the radiation itself is unlikely',answer_quote='some of her worries may be based on a common misunderstanding'),
}


def main():
    sel=json.loads((OUT/'selection.json').read_text());p=Path(sel['source'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sel['sha256']
    rows=[json.loads(l) for l in p.read_text().splitlines()];ix={(r['model'],r['method'],r['id']):r for r in rows}
    old=[json.loads(l) for l in (ROOT/'results/analysis/detection_plain_link_20260928_v1/ledger.jsonl').read_text().splitlines()]
    ledger={(r['model'],r['id']):r for r in old};assert len(sel['strata'])==10
    records=[];cases=['# 고정한 실제 답변 대조 10쌍','','Plain과 대안 답변 20개 전체를 읽었다. 점수로 층화한 목적 표본으로, 유형의 빈도를 추정하지 않는다. 아래 모델 출력의 의학적 주장을 승인하지 않는다.','']
    judges=[];cache={};sources={};t=[]
    for s in sel['strata']:
        m,method,cl=s['model'],s['method'],s['set'];q=s['selected_id']
        eligible=[r['id'] for r in rows if r['model']==m and r['method']==method and r['set']==cl and
                  ((r['plain_score']<4 and r['answer_score']>=4) if cl=='fpq' else (r['plain_score']>=4 and r['answer_score']<4))]
        eligible.sort(key=lambda q:hashlib.sha256((sel['seed']+m+':'+method+':'+cl+':'+q).encode()).hexdigest())
        assert eligible==s['eligible_ids'] and eligible[0]==q
        r=ix[m,method,q];note=NOTES[m,q]
        assert note['plain_quote'] in r['plain'] and note['answer_quote'] in r['answer']
        record=dict(model=m,method=method,id=q,set=cl,plain_score=r['plain_score'],answer_score=r['answer_score'],
                    direct_gate=r['direct_gate'],review_gate=r['review_gate'],**note)
        if (m,q)==('qwen38_off','nfp_1132'):
            review=ledger[m,q]['review']
            assert 'treatment *must* be delayed until after birth' in review
            record['separate_review_target']='별도 Review gate는 출산까지 치료를 미뤄야 한다는 전제를 문제 삼음. CoT 답변 저점은 여행 서비스 안내 회피이며 동일한 오류 대상을 반박한 것이 아님.'
            record['separate_review_quote']='treatment *must* be delayed until after birth'
        # Preserve actual intermediate review when that answer received one.
        supplied_review=None
        if m=='qwen25':
            f=Path('/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1/answers/premise_review.jsonl')
            ans=next(x for x in map(json.loads,f.read_text().splitlines()) if x['id']==q)
            assert ans['response']==r['answer'];supplied_review=ans['details']['review']['text']
            sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        elif m=='gemma':
            f=ROOT/'results/gemma4/full_20260926_v2/records/premise_review_answer'/(q+'.json');ans=json.loads(f.read_text())
            supplied_review=ledger[m,q]['review']
            assert supplied_review in ans['prompt'] and ans['review_sha256']==we.digest(supplied_review)
            assert ans['response']==r['answer'];sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
        if supplied_review is not None:
            record['supplied_review_matches_gate_review']=supplied_review==ledger[m,q]['review']
            record['supplied_review']=supplied_review
        for condition,job,score in [('plain',r['plain_job'],r['plain_score']),(method,r['answer_job'],r['answer_score'])]:
            well=CONFIG[m][condition]
            if well not in cache:
                plan,ph=we.load_plan(well);_,scores=we.read_ledger(well,plan,ph)
                f=well/'attempts.jsonl';events=[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();cache[well]=(scores,events)
            scores,events=cache[well];assert scores[job]==score
            ev=[v for v in events if v.get('id')==job and v.get('valid') and v.get('score')==score];assert ev
            judges.append(dict(model=m,id=q,method=condition,score=score,job=job,raw=ev[-1]['raw']))
        records.append(record);t.append([NAMES[m],q,r['plain_score'],r['answer_score'],r['review_gate'],note['kind']])
        cases += [f'## {NAMES[m]} {q} — {METHODS[method]}','',r['question'],'',
                  '주석: '+json.dumps({k:r[k] for k in ['premise','correction','tpq_premises']},ensure_ascii=False),'',
                  f"Plain={r['plain_score']}, 대안={r['answer_score']}; 별도 Direct gate={r['direct_gate']}, Review gate={r['review_gate']}",'',
                  '**Plain**','',r['plain'],'','**대안 답변**','',r['answer'],'','독해: '+note['note'],'']
        if supplied_review is not None:cases+=['**실제 입력된 검토문**','',supplied_review,'']
    assert len(records)==len(NOTES)==10
    for name,v in [('annotations',records),('judge_explanations',judges),('audit_sources',sources)]:
        (OUT/(name+'.json')).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    (OUT/'cases.md').write_text('\n'.join(cases))
    (OUT/'case_table.md').write_text(table(['모델','문항','Plain','대안','별도 Review gate','행동 변화'],t)+'\n')
    print('Validated 10 frozen pairs, 20 answer quotes, 20 saved scores and judge explanations.')


if __name__=='__main__':main()
