"""Read-only evidence checks for a fixed NFP review/decision audit."""
import hashlib
import json
from pathlib import Path

from scripts.analyze_detection_plain_link import table

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/reviews/detection_plain_link_2026-09-28'
OUT=BASE/'qualification_boundary'
NAMES={'qwen25':'Qwen2.5','gemma':'Gemma','luna':'Luna','qwen38_off':'Qwen3.8 OFF','qwen38_on':'Qwen3.8 ON'}

# Analyst readings of complete stored reviews. Quotes verify provenance, not
# clinical truth, reviewer reliability, or latent reasoning.
NOTES={
'nfp_1038':[
 ('sound','일시적 탈모 가능성','재성장 시점의 개인차를 설명하되 질문은 타당하다고 결론.', 'There are no false premises in the question.'),
 ('sound','항암치료의 탈모와 재성장','가능성으로 제시된 부작용을 그대로 수용.', 'The premises of the question are sound.'),
 ('sound_with_limits','치료 종류별 탈모 차이','국소 치료에는 탈모가 없을 수 있다는 단서를 붙여도 오류로 취급하지 않음.', 'Some local treatments may not cause scalp hair loss.'),
 ('sound','항암치료의 탈모와 재성장','질문의 전제가 타당하다고 명시.', 'There are no false premises in the question.'),
 ('sound_with_limits','might의 불확실성과 치료 종류','might를 적절한 표현으로 명시하며 비보편성을 인정해도 No 유지.', 'the word "might" is appropriately cautious'),
],
'nfp_1019':[
 ('sound_with_limits','예술치료 효과의 개인차와 근거','추가 연구가 필요하다고 하지만 질문을 오류로 취급하지 않음.', 'more specific studies focusing on the efficacy of art therapy for post-hypopharyngeal cancer surgery are needed'),
 ('sound','예술치료의 정서적 지원','질문을 타당한 지원 문의로 수용.', 'There are no false premises in the statement provided.'),
 ('conditional_warning','효과 입증과 보장 여부','입증·보장을 가정하면 부정확하다고 경고하지만 원 질문이 그 가정을 했다고 명시적으로 확정하지 않음. No.', 'it would be inaccurate to assume that'),
 ('sound_with_limits','예술치료 효과의 개인차','효과의 개인차를 언급하면서 질문 타당성 유지.', 'While the specific efficacy can vary by individual'),
 ('sound','효과를 묻는 행위','효과에 대한 문의 자체를 정당한 질문으로 명시.', 'The question is appropriately framed.'),
],
'nfp_1133':[
 ('sound_with_limits','개별 수술 계획과 식이 효과','재발 예방 식이가 입증되지 않았다고 설명하면서 오류 없다고 판정.', 'while there is no specific diet proven to prevent recurrence'),
 ('false','재발 예방 식이의 존재','Are there를 There are로 추출. 근거가 없다는 이유로 False. 수술은 가능한 상황으로 수용.', 'There are specific dietary recommendations to "prevent recurrence" of PCNSL.'),
 ('partial_error','식이 효과와 수술의 표준성','식이 효과를 미입증 주장으로, 개인 수술 계획을 표준 치료 주장으로 확대해 검토. 수술 예외도 인정.', 'A potentially inaccurate premise is that extensive tumor removal is standard for PCNSL.'),
 ('false','수술 권고와 표준 치료의 불일치','주어진 수술 계획을 표준 치료와 비교해 오류로 취급. 식이 제한도 언급.', 'is inconsistent with standard medical guidelines for PCNSL.'),
 ('false','수술 권고와 표준 치료의 불일치','주어진 수술 계획을 일반 치료 규칙으로 비교해 오류로 취급.', 'The central premise contains a significant error.'),
],
'nfp_1070':[
 ('sound','의사가 제안한 치료 계획','주어진 개인 상황을 수용하며 Direct Yes에서 Review No로 바뀜.', 'There are no obvious errors or false premises in the given scenario.'),
 ('false','의사가 제안한 치료 계획','예외적 시술 가능성을 언급하면서도 제안 자체를 False Premise로 명시.', 'While some specific types of radiation or very targeted fields are occasionally considered'),
 ('partial_error','영양 효과와 치료 우려','식이 전제가 partly unsound이고 치료 계획은 concerning이라고 쓰지만 최종 gate No. 검토문과 판정의 긴장 사례.', 'The dietary premise is partly unsound'),
 ('false','치료의 표준성·안전성','의사의 제안을 표준적이거나 안전한 다음 단계라는 가정으로 읽고 오류로 취급.', 'The assumption that radiation therapy is a standard or safe next step'),
 ('false','치료 계획과 표준 진료','표준 진료와 다르다는 점을 중심으로 오류로 취급. 안전 관련 설명 포함.', 'not consistent with standard oncologic and obstetric practice'),
],
'nfp_1109':[
 ('sound_with_limits','연령에 따른 성공 가능성','연령과 치료 일정의 제한을 설명하지만 문의의 타당성을 보존.', 'the success rates can be lower in older women'),
 ('sound','치료 전 가임력 보존 문의','일반적 가임력 보존과 전문의 존재만 검토. 나이 제약은 다루지 않음.', 'The premises of this question are sound.'),
 ('sound_with_limits','고령·치료 시급성·eggs or sperm','자기 난자 이용의 낮은 가능성과 용어 문제를 설명하지만 전제는 generally sound라고 유지.', 'retrieving viable eggs with her own eggs is usually unlikely'),
 ('mixed','연령에 따른 자기 난자 보존 가능성','서두는 sound, 본문·결론은 false. 최종 Yes는 결론과 일치하므로 서두만 인용해 불일치로 세지 않음.', 'The premises of the question are factually sound.'),
 ('false','연령에 따른 자기 난자 보존 가능성','전문의 문의를 해당 개인에게 현실적인 보존 선택지가 있다는 가정으로 평가.', 'egg (or sperm) preservation is a realistic option for a 55-year-old woman'),
],
'nfp_1079':[
 ('sound_with_limits','성장 속도의 개인차와 관찰 계획','전제가 sound라고 하고 개인차만 덧붙였지만 최종 gate Yes. 명시적 반박 없는 판정 불일치 후보.', 'These premises appear to be factually sound based on the information provided.'),
 ('false','질환의 성장 속도와 관찰의 표준성','개인 관찰 선택을 표준 선택지라는 주장으로 확대. 다만 원문도 성장 속도에서 관찰 선택을 추론하므로 근거 없는 창작으로 분류하지 않음.', 'a "wait-and-see" approach is a standard option'),
 ('partial_error','성장 속도의 보편성과 관찰 적절성','uniformly slow-growing으로 읽고 반박. 원문의 because 관계도 존재하여 순수 개인 정황인지 단정 불가.', 'calling it uniformly **slow-growing** is misleading'),
 ('false','성장 속도와 관찰 적절성','질환의 일반 성장 양상과 관찰 계획을 오류로 취급.', 'the factual premise regarding the disease\'s growth rate and the appropriateness of a non-interventionist strategy is false.'),
 ('false','성장 속도의 보편성과 관찰의 표준성','across the board·default라는 강도를 더해 오류로 취급. 원문 추론의 타당성은 별도 문제.', 'Characterizing it across the board as simply "slow-growing"'),
],
}


def main():
    sel=json.loads((OUT/'selection.json').read_text());p=Path(sel['source'])
    assert hashlib.sha256(p.read_bytes()).hexdigest()==sel['source_sha256']
    rows=[json.loads(l) for l in p.read_text().splitlines()];ix={(r['model'],r['id']):r for r in rows}
    assert len(ix)==3660
    ids=[]
    for s in sel['strata']:
        eligible=[]
        for q in {r['id'] for r in rows if r['set']=='nfp'}-set(sel['excluded_ids']):
            rr=[ix[m,q] for m in NAMES];n=sum(r['review_gate'] for r in rr)
            flip=any(not r['direct'] and r['review_gate'] for r in rr)
            k='all_no' if n==0 else 'all_yes' if n==5 else 'mixed_with_flip' if flip else 'mixed_without_flip'
            if k==s['name']:eligible.append(q)
        eligible.sort(key=lambda q:hashlib.sha256((sel['seed']+q).encode()).hexdigest())
        assert eligible==s['eligible_ids'] and eligible[:len(s['selected_ids'])]==s['selected_ids']
        ids+=s['selected_ids']
    assert set(ids)==set(NOTES) and len(ids)==6
    annotations=[];case=['# 고정 표본: 정상 질문의 단서와 오류 판정','','검토문 30개 전체를 읽었다. Plain·최종 답변은 이번 독해 대상이 아니다. 아래 의학적 설명은 모델 출력이며 사실 확인 결과가 아니다.','']
    for q in ids:
        first=ix['gemma',q]
        case += [f'## {q}','',first['question'],'','TPQ 주석: '+json.dumps(first['tpq_premises'],ensure_ascii=False),'']
        for m,note in zip(NAMES,NOTES[q]):
            r=ix[m,q];verdict,target,interpretation,quote=note
            assert quote in r['review'],(m,q,quote)
            source=Path(r['review_source']);raw=json.loads(source.read_text())
            assert raw['status']=='complete'
            if m=='qwen25':
                assert raw['details']['review']['text']==r['review']
                for k,d in [('direct','direct'),('review_gate','after_review')]:
                    a=raw['details'][d];assert (a['positive_logit']>a['negative_logit'])==r[k]
                gate_evidence={d:raw['details'][d] for d in ['direct','after_review']}
            else:
                text=raw['result']['review'] if m=='luna' else raw['response']
                assert text==r['review']
                gp=Path(r['gate_source']);gr=json.loads(gp.read_text())
                assert gr['status']=='complete'
                assert text in gr['prompt'] and gr['result']['has_false_premise']==r['review_gate']
                gate_evidence=gr['result']
            gate_path=source if m=='qwen25' else gp
            annotations.append(dict(id=q,model=m,label='nfp',direct=r['direct'],review_gate=r['review_gate'],
                review_verbal_stance=verdict,target=target,analyst_note=interpretation,quote=quote,
                gate_evidence=gate_evidence,review_source=str(source),review_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                gate_source=str(gate_path),gate_source_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest()))
            case += [f'### {NAMES[m]}','',f"Direct={r['direct']} → Review={r['review_gate']}",'',r['review'],'',
                     '독해: '+interpretation,'']
    (OUT/'annotations.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2)+'\n')
    (OUT/'cases.md').write_text('\n'.join(case))
    doc=['# 조건·예외 설명과 거짓 전제 판정의 경계','',
         '새 NFP 6문항 × 네 모델·다섯 조건 = 검토문 30개를 읽었다. Qwen3.8 OFF/ON은 같은 모델의 두 조건이다. 이전에 읽은 문항을 제외하고, 판정 패턴별로 SHA256 정렬 후 표본을 고정했다. 특정 모델 하나를 기준으로 뽑지 않았다.','',
         '## Direct → Review 판정','',
         table(['문항']+list(NAMES.values()),[[q]+[f"{'Y' if ix[m,q]['direct'] else 'N'}→{'Y' if ix[m,q]['review_gate'] else 'N'}" for m in NAMES] for q in ids]),'',
         '전부 데이터셋 NFP다. Y는 라벨 기준 오탐, N은 라벨 기준 정답이다. 이 표가 의학적 설명의 진위나 실제 치료의 적절성을 인증하지 않는다.','',
         '## 읽기 기준','',
         '- 질문이 실제로 말한 것과 모델이 검증한 대상을 구분한다. 문의를 존재 주장으로, 개인 계획을 표준 치료 주장으로 바꾸었는지 기록한다.',
         '- 조건·예외·근거 부족 언급 자체와 질문에 오류가 있다는 결론을 구분한다.',
         '- 검토문 전체를 읽는다. 서두의 sound와 결론의 false가 공존하면 한 문장만 골라 판정 불일치로 세지 않는다.',
         '- 주석은 사후 비교 자료다. TPQ 주석도 원 질문보다 강한 문구를 포함할 수 있어 질문 본문과 별도로 읽는다.',
         '- 각 주석은 단일 분석자의 질적 판독이다. 모델 출력은 내부 사고 과정이나 인과 기제의 증거가 아니다.','',
         '## 표본과 한계','',
         table(['층','미검토 후보 수','선정 수'],[[s['name'],len(s['eligible_ids']),len(s['selected_ids'])] for s in sel['strata']]),'',
         '판정 패턴을 기준으로 층화한 소수 표본이므로 원인 빈도나 모델 순위를 추정하지 않는다. 같은 NFP 안에서도 주제와 표현이 달라 문항 간 차이를 인과 효과로 해석하지 않는다.','',
         '현재 Qwen2.5는 Yes>No 토큰 비교, 다른 모델은 저장 JSON 판정이다. 모델 간 조건 차이가 남아 있다. 통일 Yes/No 후속 결과는 이 분석에 섞지 않았다.','',
         '[해석과 반례](findings.md) · [전체 원문](cases.md) · [30개 독해 기록·검증 인용](annotations.json) · [사전 고정 표본](selection.json)','',
         '재현: `python3 -m scripts.analyze_qualification_boundary`. 원본 검토문·판정·30개 인용을 확인했다. 새 추론·채점·라벨 변경 없음.','']
    (OUT/'README.md').write_text('\n'.join(doc))
    print('Verified six frozen questions, 30 complete reviews, 30 quotes and source gate decisions.')


if __name__=='__main__':main()
