"""Report completed blinded AI coding separately from saved Well scores."""
from collections import Counter
import csv
import hashlib
import json

from scripts.audit_ungated_failure_census import OUT, DOC, PRIMARY, prepare, validate, write
from scripts.analyze_ungated_answer_transitions import NAMES, METHODS
from scripts.analyze_detection_plain_link import table


def main():
    inputs,manifest=prepare()
    annotations=json.loads((DOC/'annotations.json').read_text())
    assert len(annotations)==len(inputs)==234
    bykey={r['key']:r for r in annotations}
    assert len(bykey)==234
    sources={};raw_models=Counter();repairs=0
    for i in range(0,len(inputs),3):
        folder=OUT/'batches'/f'{i//3:03d}'
        items,changes=validate(json.loads((folder/'result.json').read_text())['items'],inputs[i:i+3])
        assert not changes
        for x in items:assert bykey[x['key']]['audit']==x
        raw=json.loads((folder/'stdout.json').read_text())
        assert raw.get('subtype')=='success' and not raw.get('is_error') and raw.get('stop_reason')=='end_turn'
        assert not raw.get('permission_denials') and not any(raw.get('usage',{}).get('server_tool_use',{}).values())
        raw_models.update(raw.get('modelUsage',{}).keys())
        repairs+=len(json.loads((folder/'quote_repairs.json').read_text()))
        for f in [folder/'result.json',folder/'stdout.json']:
            sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    for f in [OUT/'plan.json',OUT/'manifest.json',OUT/'inputs.json',DOC/'annotations.json']:
        sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    write(DOC/'completed_audit_sources.json',dict(sources=sources,models=dict(raw_models),
          final_quote_normalizations=repairs,archived_attempts=len(list((OUT/'retry_archive').glob('*/*'))),
          valid_pairs=234,valid_answer_codings=468,limits='Mechanical validation is not semantic or medical certification.'))
    group_order=['new_loss','plain_low_persistent','plain_low_recovered','stable_nfp_control','fpq_gain_control']
    groups={g:[r for r in annotations if r['group']==g] for g in group_order}
    def challenge(r,side):return r['coded'][side]['stance']=='challenge'
    def context(r,side):return any(f['tag']=='personal_context_reopened' for f in r['coded'][side]['flags'])
    def count(rr,fn):return sum(fn(r) for r in rr)
    counts={g:dict(n=len(rr),new_challenge=count(rr,lambda r:not challenge(r,'plain') and challenge(r,'answer')),
                  persistent_challenge=count(rr,lambda r:challenge(r,'plain') and challenge(r,'answer')),
                  shared=count(rr,lambda r:r['audit']['relation']=='shared_caveat_or_challenge'),
                  substantive_both=count(rr,lambda r:all(r['coded'][s]['response']=='substantive' for s in ['plain','answer'])),
                  context_answer=count(rr,lambda r:context(r,'answer'))) for g,rr in groups.items()}
    assert counts['new_loss']['new_challenge']==76 and counts['new_loss']['persistent_challenge']==37
    assert counts['new_loss']['substantive_both']==105
    write(DOC/'ai_interpretation_counts.json',counts)
    rows=[]
    for m in PRIMARY:
        rr=[r for r in groups['new_loss'] if r['model']==m]
        rows.append([NAMES[m],METHODS[PRIMARY[m]],len(rr),count(rr,lambda r:not challenge(r,'plain') and challenge(r,'answer')),
                     count(rr,lambda r:challenge(r,'plain') and challenge(r,'answer')),
                     count(rr,lambda r:not challenge(r,'answer'))])
    with (DOC/'ai_codings.csv').open('w') as f:
        fields=['model','method','id','group','plain_score','answer_score','plain_stance','answer_stance',
                'plain_response','answer_response','relation','plain_flags','answer_flags','difference_ko']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in sorted(annotations,key=lambda r:(r['model'],r['group'],r['id'])):
            row={k:r[k] for k in fields[:6]}
            for side in ['plain','answer']:
                for field in ['stance','response']:row[side+'_'+field]=r['coded'][side][field]
                row[side+'_flags']=';'.join(f['tag'] for f in r['coded'][side]['flags'])
            relation=r['audit']['relation']
            if relation in ['only_A','only_B']:relation='only_'+r['slots'][relation[-1]]
            row.update(relation=relation,difference_ko=r['audit']['difference_ko']);w.writerow(row)
    old=json.loads((DOC.parent/'ungated_answer_transitions/annotations.json').read_text())
    manual=json.loads((DOC/'manual_readings.json').read_text())
    overlap={(r['model'],r['id']) for r in annotations}&{(r['model'],r['id']) for r in old+manual}
    assert len(overlap)==14
    write(DOC/'manual_ai_crosscheck.json',dict(n=14,keys=sorted(overlap),
        scope='Compared AI codes against prior direct readings; not an independent second medical review or an accuracy estimate.',
        consistent_examples=['qwen25/nfp_1108: new diagnosis challenge','gemma/nfp_1078: new treatment challenge',
          'qwen38_off/nfp_1132: requested information reduced without explicit premise rejection',
          'luna/nfp_1021 and qwen38_off/nfp_1000: same correction retained across score recovery'],
        boundary_examples=['luna/nfp_1082: short context qualification called challenge by AI',
          'luna/nfp_1026: minor Plain qualification called challenge by AI',
          'qwen38_on/nfp_1033: softer alternative contextual note still called challenge'],
        annotation_caveat='Some target_ko fields describe the model\'s disputed proposition as if it were the question\'s actual premise. Standard treatment or universal need must not be inferred from a reported individual treatment/plan. No raw AI codes or Well scores were overwritten.'))
    doc='''# 완료된 Claude 행동 코딩: Plain 저점과 지시 후 새 저점

**234쌍/468답변의 AI 코딩과 인용·형식 검증이 완료됐다. 새 저점 NFP의 다수는 더 명시적인 반박으로 분류됐지만, 이미 있던 반박이 남은 채 점수만 내려간 경우와 반박 없이 정보 제공이 줄어든 경우도 있다.** 아래 수치는 Claude의 잠정 의미 분류이며, 직접 확정한 원인 빈도가 아니다.

## 자료와 검증 범위

- 새 저점 NFP 121쌍, Plain 저점 NFP 63쌍 전수. 같은 모델조건 안에서 두 집단은 겹치지 않는다. 고유 질문 92개이며 모델 간 문항 반복이 있다.
- 대조군은 두 답변 모두 ≥4인 NFP 25쌍과 <4→≥4인 FPQ 25쌍. 각 모델조건 5개씩 해시로 정한 작은 표본이다. 전체의 대표 비율로 해석하지 않는다.
- 모델명·조건명·라벨·점수·주석·gate 결정을 숨기고 질문과 답변 두 개를 제공했다. A/B 순서도 섞었다. 문체로 조건을 추정할 가능성까지 차단한 것은 아니다.
- 모든 최종 78개 묶음에서 근거 인용과 원문 대응을 확인했다. 검증 실패 응답은 보존하고 동일 입력·지시로 재요청했다. 기계적 검증 통과는 의미 분류의 정확성 인증이 아니다.
- 앞서 직접 읽었던 사례와 겹치는 14개 모델조건–문항의 분류를 대조했다. 서로 다른 기준의 독해이므로 이를 독립 평가 정확도로 계산하지 않았다. 경계 사례의 원 AI 분류와 Well 점수는 수정하지 않았다.

## 1. 새로 저점이 된 NFP 121쌍

MODEL_TABLE

‘새 반박’은 Claude가 Plain을 `none/qualification`, 대안을 `challenge`로 코딩한 경우다. 질문에서 새로운 내용을 발견했다는 뜻이 아니라 **출력에서 명시적으로 반박하는 행동으로 분류가 바뀌었다**는 뜻이다. 마지막 열은 대안에서도 `challenge`로 분류되지 않은 경우다.

- **76/121:** 반박 없음 또는 단서 수준 → 명시적 반박. 이 중 단서→반박 46건, 없음→반박 30건.
- **37/121:** 두 답변 모두 이미 반박으로 분류됨. 새롭게 전제를 탐지했다는 설명만으로는 이 집단을 설명할 수 없다.
- **8/121:** 대안에서도 단서 수준으로 분류됨. 낮은 점수만으로 전제 오탐이라고 세면 안 된다.
- **105/121:** 두 답변 모두 요청에 실질적으로 답하는 것으로 분류됨. 적어도 이 코딩에서는 새 저점의 대부분이 전면 답변 거부는 아니다. 도움을 주면서 개인 상황도 반박하는 답변이 포함된다.
- **64/121:** 양쪽에 같은 핵심 단서·반박 내용이 존재한다고 분류됨. 이는 위의 행동 변화 구분과 겹치는 별도 축이다. 같은 내용이 더 강한 반박으로 표현될 수도 있다.

모델조건의 차이도 있다. Qwen2.5 새 저점 8건 중 6건, Gemma 32건 중 27건은 ‘새 반박’이었다. Luna는 새 저점 54건 중 21건이 이미 Plain에도 반박이 있는 것으로 분류됐다. Qwen3.8 ON의 새 저점은 4건뿐이며 그중 3건이 양쪽 반박이다. **모든 모델의 변화가 같은 형태라고 할 수 없다.** 프롬프트와 추론 절차가 달라 모델 자체의 차이로만 해석하지 않는다.

## 2. Plain부터 저점인 NFP 63쌍

- 지시 후에도 저점인 **46쌍 전부**에서 Claude는 양쪽을 반박으로, 같은 핵심 내용을 유지한 것으로 분류했다.
- 지시 후 ≥4로 회복한 **17쌍 중 7쌍**은 대안에서도 반박으로 분류됐다. 나머지는 반박→단서 5, 반박→없음 2, 단서→단서 2, 없음→없음 1이다.

따라서 **반박 행동이 없어야만 NFP ≥4가 되는 것은 아니며, ≥4로 회복했다고 오류 판정이 사라졌다고 볼 수도 없다.** 실제 강도·내용·도움의 양과 판정기의 기준이 함께 들어 있다.

직접 읽었던 Luna `nfp_1021`(2→4), Qwen3.8 OFF `nfp_1000`(2→5)은 AI도 양쪽에 동일한 교정 내용이 남아 있다고 분류했다. 반면 Qwen3.8 ON `nfp_1033`의 대안은 직접 독해에서 완화된 맥락 설명으로 보았지만 AI는 여전히 반박으로 분류했다. 이 경계의 모호함 때문에 세부 빈도를 확정 수치처럼 쓰면 안 된다.

## 3. FPQ 개선 대조와 함께 보면

FPQ 개선 표본 25쌍에서도 16쌍이 새 반박, 7쌍이 양쪽 반박, 2쌍이 대안에서도 단서로 분류됐다. 개인의 진단·예후·치료 계획을 재검토한다는 넓은 표지는 FPQ 개선 대안 **17/25**, NFP 새 저점 대안 **92/121**에 있었다.

이는 유용한 단서지만 원인 증명은 아니다. **‘개인 정황을 재검토한다’는 행동만으로 유익한 교정과 불필요한 반박을 가를 수 없다.** FPQ 개선 표본에도 같은 표지가 있으므로 이 행동을 일괄 금지하는 규칙을 곧바로 해법으로 삼을 근거는 없다. 또한 FPQ ≥4를 해당 답변의 모든 주장에 대한 의학적 인증으로 해석하지 않는다.

보존 NFP 대조 25쌍에서도 대안 7쌍이 반박으로 분류됐다. 단순히 반박 문구의 존재 여부를 Well 통과 여부와 동일시하지 않아야 한다.

## 4. 원문 대조에서 확인한 AI 코딩의 한계

- Luna `nfp_1082`의 짧은 수술·장루 종류 확인, `nfp_1026`의 Plain 끝 단서를 AI가 모두 반박으로 코딩했다. 직접 독해에서 중요하게 본 강도·배치 차이가 `challenge` 하나에 합쳐졌다.
- Gemma `nfp_1078` 등에서 AI 설명은 ‘그 치료가 표준이라는 전제’라고 썼다. 질문은 개인이 치료받았다고 말했으므로 **모델이 지적한 대상을 질문이 실제로 주장한 내용으로 바꿔 적으면 안 된다.** 모델 출력의 대상과 질문의 실제 주장 범위는 별도로 대조해야 한다.
- Qwen3.8 OFF `nfp_1132`는 양쪽 모두 단서로, 대안의 정보 제공만 감소한 것으로 분류됐다. 앞선 직접 독해와 일치하며, 이 저점을 전제 오탐으로 단정하는 해석을 막아준다.
- `personal_context_reopened`는 정황 재검토의 존재만 뜻한다. 정황을 잘못 일반화했다는 판정도, 그 반박이 의학적으로 틀렸다는 판정도 아니다. 이번 AI 코딩은 ‘범위를 일반화함’과 ‘개인 정황을 반박함’을 동일시할 근거를 주지 않는다.

## 현재 결론

**Plain부터 개인 정황을 반박하는 사례가 있고, 지시 후에는 새 반박 또는 기존 단서의 강화가 늘어난다. 그러나 동일한 종류의 개입이 FPQ 개선과 NFP 저점 양쪽에 나타나며, 점수 변화와 반박 유무도 일대일 대응하지 않는다.**

따라서 다음 원인 검증은 ‘반박하는가’만 볼 것이 아니라 **질문의 어떤 명제를 대상으로 삼았는가, 질문의 주장 범위를 유지했는가, 그 명제에 어떤 근거로 반박했는가**를 맞춰 비교해야 한다. 내부 분석에서도 점수 저점 전체를 한 라벨로 묶기보다, 대상·범위가 확인된 같은 행동 사례를 대조하는 것이 필요하다. 아직 새로운 gate의 성능이나 내부 인과 기제를 증명한 결과는 아니다.

[원 점수 집계와 직접 독해](README.md) · [문항별 AI 분류 CSV](ai_codings.csv) · [원문·AI 근거 전체](annotations.json) · [14개 사례 대조 메모](manual_ai_crosscheck.json) · [완료 결과 출처 검증](completed_audit_sources.json)
'''
    doc=doc.replace('MODEL_TABLE',table(['모델조건','대안 조건','새 저점 전체','새 반박','양쪽 반박','대안 반박 아님'],rows))
    (DOC/'ai_findings.md').write_text(doc)
    print('Validated 234 pairs, 468 answer codings; reported 14 prior-reading overlaps.')


if __name__=='__main__':main()
