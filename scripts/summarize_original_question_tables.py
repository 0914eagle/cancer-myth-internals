"""Assemble existing original-question results; no model calls or re-scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/reviews/luna_nfp_tpq_2026-09-28'
SOURCES = {}


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def cell(k, n):
    assert 0 <= k <= n and n > 0
    return f'{k}/{n} ({100*k/n:.1f}%)'


def main():
    gates = read('docs/reviews/default_gate_decisions_2026-09-23/summary.json')
    answers = read('docs/reviews/advisor_presentation_2026-09-23/answer_summary.json')
    routes = read('docs/reviews/default_gate_decisions_2026-09-23/routing_summary.json')['results']
    gemma = read('results/gemma4/full_20260926_v2/analysis/summary.json')
    luna_ev = read('docs/reviews/luna_premise_gate_2026-09-23/table1_summary.json')
    annotated = read('docs/reviews/advisor_presentation_2026-09-23/annotated_summary.json')['groups']['paired_all']
    context = read('docs/reviews/luna_nfp_tpq_2026-09-28/context_direct_summary.json')['luna']
    audit = read('docs/reviews/luna_nfp_tpq_2026-09-28/summary.json')
    lost = read('docs/reviews/luna_nfp_tpq_2026-09-28/lost_fpq/summary.json')
    presentation = ROOT / 'docs/37_advisor_presentation_2026-09-23.md'
    SOURCES[str(presentation.relative_to(ROOT))] = hashlib.sha256(presentation.read_bytes()).hexdigest()
    t1 = []
    for key, name, auc in [
        ('text','TF-IDF text',0.755),('style','Style 특징',0.687),
        ('hidden','Qwen2.5 hidden logistic',0.693),('mean','Qwen2.5 DiM',0.681),
        ('direct','Qwen2.5 Direct',0.466),('review','Qwen2.5 Review → gate',0.587),
        ('luna_direct','Luna Direct',None),('luna_cot','Luna Review/CoT → gate',None)]:
        g = gates[key]
        assert g['tp'] + g['fn'] == 583 and g['fp'] + g['tn'] == 149
        t1.append(dict(method=name,auroc=auc,fpq_n=583,nfp_n=149,tp=g['tp'],fp=g['fp']))
    for key,name in [('direct','Gemma Direct'),('cot','Gemma Review/CoT → gate')]:
        g=gemma['gate'][key]
        t1.append(dict(method=name,auroc=None,fpq_n=g['fpq']['valid'],nfp_n=g['nfp']['valid'],tp=g['fpq']['positive'],fp=g['nfp']['positive']))
    t2=[]
    def add(model,name,f,n,kind):
        t2.append(dict(model=model,method=name,kind=kind,fpq=f,nfp=n))
    for model,conditions in [
        ('qwen',[('plain','Plain'),('zero_shot_cot','CoT (추론 후 별도 답변)'),('fp_identification','Self-gated FP identification'),('premise_review','Review → answer'),('fp_unconditional','무조건 교정')]),
        ('sonnet',[('plain','Plain'),('balanced','균형 지시'),('fp_unconditional','무조건 교정')]),
        ('luna',[('plain','Plain'),('balanced','균형 지시'),('fp_unconditional','무조건 교정')])]:
        for key,name in conditions:
            a=answers[f'{model}/{key}']
            add({'qwen':'Qwen2.5-7B','sonnet':'Sonnet','luna':'Luna'}[model],name,
                {k:a['fpq'][k] for k in ['valid','ge4','s5']},{k:a['nfp'][k] for k in ['valid','ge4','s5']},'generated')
        if model=='qwen':
            for key,name in [('text','Text gate → 답변 선택'),('hidden','Hidden gate → 답변 선택'),('label_oracle','Label oracle (참고)')]:
                a=routes[key]
                add('Qwen2.5-7B',name,
                    dict(valid=a['fpq']['n'],ge4=a['fpq']['ge4'],s5=a['fpq']['s5']),
                    dict(valid=a['nfp']['n'],ge4=a['nfp']['ge4'],s5=a['nfp']['s5']),'stored_answer_selection')
    for key,name in [('plain','Plain'),('zero_shot_cot_one_step','One-step CoT'),('premise_review_answer','Review → answer'),('fp_unconditional','무조건 교정'),('direct_routed','Direct gate → 답변 선택'),('cot_routed','Review/CoT gate → 답변 선택')]:
        a=gemma['answers'][key]
        vals=[]
        for split in ['fpq','nfp']:
            x=a[split]
            vals.append(dict(valid=x['valid'],ge4=x['at_least_4_count'],s5=x['counts_0_to_5']['5']))
        add('Gemma',name,*vals,'stored_answer_selection' if key.endswith('_routed') else 'generated')
    lines=['# 원본 FPQ/NFP: 표 1·표 2와 9월 28일 원인 분석','',
        '저장 결과를 재집계한 문서다. 의역·전제 수정·쌍둥이·Masked·SFT 및 CREPE는 제외했다. 원 질문 본문을 유지한 Direct·Review·교정 과제 지시는 포함한다. Extract→Verify는 기존 본표 제외 요청을 유지하고 부록으로 분리했다. 새 생성·채점·학습은 하지 않았다.','',
        '## 표 1. 원 질문의 거짓 전제 탐지','',
        'FPQ 탐지는 높을수록, NFP 오탐은 낮을수록 좋다. 두 비율의 분모는 각각 FPQ와 NFP다.','',
        '| 방법 | AUROC | FPQ 탐지 | NFP 오탐 |','|---|---:|---:|---:|']
    for x in t1:
        auc='' if x['auroc'] is None else f"{x['auroc']:.3f}"
        lines.append(f"| {x['method']} | {auc} | {cell(x['tp'],x['fpq_n'])} | {cell(x['fp'],x['nfp_n'])} |")
    lines += ['',
        'Hidden probe는 원 질문만 넣은 입력의 마지막 chat-prefill 토큰 위치에서 L11·17·22 중 학습 CV로 선택한 층을 읽는다. 마지막 토큰 위치는 마지막 층이라는 뜻이 아니다. Direct는 오류 탐지 지시를 포함한 다른 입력의 최종 LM head에서 Yes/No를 읽으므로 동일 hidden의 두 판독 비교가 아니다. 학습 분류기는 source-group crossfit의 기본 판정이다. Qwen Direct/Review는 Yes 토큰 점수가 No보다 클 때 양성이다. Luna·Gemma는 생성 JSON의 boolean 판정이므로 출력 방식이 다르다. Luna·Gemma 자기보고 확률 기반 AUROC는 보고하지 않는다. 나머지 AUROC는 기존 발표 문서의 fold 가중값을 보존했다. 오탐 억제용으로 별도 선택했던 과거 문턱의 결과가 아니다.',
        'Review/CoT → gate는 검토문을 만든 뒤 질문과 검토문으로 FPQ/NFP를 판정한다. 표 2의 Review → answer는 검토문을 참고해 최종 답변을 만드는 다른 측정이다. 모델 간 프롬프트·판정 방식이 완전히 같지 않아 순수 모델 능력 순위로 해석하지 않는다.','',
        '## 표 2. 최종 답변의 Well 평가','',
        '두 집단 모두 4점 이상 비율이 높을수록 좋다. S5는 5점 비율이다. Well은 0–5 척도이며 이 점수를 독립적인 의료 사실 정확도로 해석하지 않는다.','',
        '| 모델 | 조건 | FPQ ≥4 | NFP ≥4 | FPQ S5 | NFP S5 |','|---|---|---:|---:|---:|---:|']
    for x in t2:
        f,n=x['fpq'],x['nfp']
        lines.append(f"| {x['model']} | {x['method']} | {cell(f['ge4'],f['valid'])} | {cell(n['ge4'],n['valid'])} | {100*f['s5']/f['valid']:.1f}% | {100*n['s5']/n['valid']:.1f}% |")
    lines += ['',
        'Qwen Plain·CoT·Self-gated는 FPQ 582/NFP 149, Review·무조건 교정은 FPQ 583/NFP 149다. Qwen Text·Hidden·oracle은 두 저장 답변 점수가 모두 있는 FPQ 582/NFP 149이며 fpq_389가 빠진다. 나머지 본표 행은 FPQ 583/NFP 149다. 누락을 실패로 채우지 않았다. 표 전체가 완전히 동일한 문항의 paired 비교는 아니다.',
        'Qwen Self-gated FP identification과 Gemma Direct gate 답변 선택은 직접 판정 후 Plain/교정을 고르는 같은 종류의 정책이나 모델·판정 방식·실행 방식은 다르다. Review → answer는 검토문을 답변 입력으로 전달하며, Review/CoT gate 답변 선택은 검토 결과로 답변 종류만 고르고 기존 교정 답변에는 그 검토문을 전달하지 않는다. 답변 선택 행은 gate가 켜지면 기존 무조건 교정 답변, 꺼지면 기존 Plain 답변을 선택하고 저장 점수를 사용한다. Label oracle은 정답 라벨로 그 두 답변을 선택한 참고점이며 가능한 모든 방법의 상한이 아니다. Qwen 공통 582개에서 무조건 교정은 332/582=57.0%로 oracle과 같다.',
        '무조건 교정은 NFP에도 거짓 전제가 있다고 알려 주는 지시다. 균형 지시는 틀리면 교정하고 타당하면 직접 답하라고 한다. 둘은 같은 프롬프트가 아니다. Qwen Self-gated의 원 실행은 동점도 양성(>=)이라 표 1의 엄격한 Yes>No와 동점 처리 차이가 남는다.',
        'Qwen CoT는 추론 생성 후 별도 답변 호출, Gemma One-step CoT는 한 호출이므로 같은 조건이 아니다. Qwen·프론티어·Gemma의 생성 설정은 완전히 같지 않다. 판정기는 Sonnet이며 자신의 답변도 채점하는 한계가 있다. Sonnet/Luna Direct gate에 각각 최종 답변을 연결한 새 표 2 실험은 이 본표에 없다. 진행 중인 Qwen3.8 전체 suite는 완료 표에 넣지 않았다.','',
        '## 9월 28일 원인 분석','',
        '| 분석 | 대상·결과 | 확인한 범위 |','|---|---|---|',
        '| NFP 오탐 의미 감사 | Luna CoT 오탐 97개. 사실성 반박 51, 개인 정황 재검토 28, 질문 변형 7, 조건·예외를 오류로 연결 8, 검토/판정 불일치 3 | Sonnet의 대표 텍스트 패턴. 원인별 확정 비율 또는 의학적 라벨 재판정이 아님 |',
        '| 개인 정황 일반화 방지 지시 | Luna Direct 732개. FPQ 탐지 62.6→38.6%, NFP 오탐 39.6→15.4% | 지시 변화에 따라 양성 판정이 감소. 해석 변화와 일반적인 판정 성향 변화는 아직 분리 못 함 |',
        '| 놓친 FPQ의 주석·Review 대조 | Yes→No FPQ 147개: 주석과 같은 주요 대상 136, 부분 4, 다른 대상 7 | 별도 저장 Review의 대응이며 Direct의 실제 이유는 아님 |',
        '| 문항·주석의 해석 경계 확인 | 개인 예후·치료 계획을 일반 통념으로 읽는 FPQ와 이를 개인 정황으로 보존한 NFP 사례 비교 | 교정 정보의 타당성과 질문이 실제 그 오류를 함의하는지는 구분해야 함. 경계 사례의 원인 기여율은 미측정 |',
        '| 내부 원인 및 관련 연구 정리 | 기존 probe·개입의 증거 범위와 활성화/활용·거절·맥락 충돌 가설을 정리 | 오늘 이 분석에서 새 내부 개입 실험을 수행한 것은 아님 |','',
        '| Luna Direct 조건 | FPQ 탐지 | NFP 오탐 |','|---|---:|---:|']
    for key,name in [('baseline','기존'),('context_rule','개인 정황 지시 추가')]:
        x=context['conditions'][key]
        lines.append(f"| {name} | {cell(x['fpq']['yes'],x['fpq']['valid'])} | {cell(x['nfp']['yes'],x['nfp']['valid'])} |")
    lines += ['',
        'FPQ Yes→No 147, No→Yes 7, NFP Yes→No 36, No→Yes 0이다. 두 집단 평균 정답률은 61.5→61.6%로 거의 같다. 이것만으로 내부 문턱 변화나 유용성 개선을 확정하지 않는다. 기본 결과는 이전 호출을 재사용했고 새 지시 조건은 나중에 호출했으므로 시점 차이도 있다.',
        '아직 하지 않은 원인 확인: FPQ 새 탐지와 NFP 새 오탐을 같은 틀로 전수 비교, 개인 정황 지시에서 해소된 NFP 36개와 놓친 FPQ 147개 및 유지 집단의 체계적인 공통 기준 비교, 진위에 선택적인 내부 신호가 행동을 바꾸는지의 검증. 136개 대응은 기존 Direct의 실제 이유나 주석 해석의 타당성을 인증하지 않는다.',
        '[97개 감사](report.md) · [지시 추가 결과](context_direct_results.md) · [147개 감사](lost_fpq/README.md) · [원인 분석 현황](causal_status.md) · [관련 내부 연구](mechanistic_related_work.md)','',
        '## 부록. Extract→Verify 및 주석 입력 진단','',
        '실행한 기록을 남기되 본표에는 복귀시키지 않는다. 주석 조건은 일반 사용 시 없는 정보를 제공하므로 자동 gate와 구분한다. 아래는 각 조건의 유효 문항이며 Luna·Gemma 자동/주석 행은 분모가 달라 그대로 paired 비교하지 않는다. Luna 공통 비교는 별도 JSON의 common_questions에 있다.','',
        '| 모델 | 전제 입력 | FPQ 탐지 | NFP 오탐 |','|---|---|---:|---:|']
    for key,name in [('original','Qwen 자체 추출 (공통 집합)'),('annotated','기존 주석 (같은 공통 집합)')]:
        x=annotated['arms'][key];n=annotated['counts']
        lines.append(f"| Qwen2.5 | {name} | {cell(x['fpq_detected'],n['fpq'])} | {cell(x['nfp_false_positive'],n['nfp'])} |")
    for key,name in [('verify_extracted','Luna 자체 추출'),('verify_annotated','기존 주석')]:
        x=luna_ev['all_valid'][key]
        lines.append(f"| Luna | {name} | {cell(x['fpq']['gate_on'],x['fpq']['n'])} | {cell(x['nfp']['gate_on'],x['nfp']['n'])} |")
    for key,name in [('verify_extracted','Gemma 자체 추출'),('verify_annotated','기존 주석')]:
        x=gemma['gate'][key]
        lines.append(f"| Gemma | {name} | {cell(x['fpq']['positive'],x['fpq']['valid'])} | {cell(x['nfp']['positive'],x['nfp']['valid'])} |")
    a=answers['qwen/extract_verify']; g=gemma['answers']['verify_extracted_routed']
    lines += ['',
        f"최종 답변의 기존 Qwen Extract+Verify는 FPQ ≥4 {cell(a['fpq']['ge4'],a['fpq']['valid'])}, NFP ≥4 {cell(a['nfp']['ge4'],a['nfp']['valid'])}다. Gemma의 추출·검증 gate로 저장 답변을 선택한 조건은 {cell(g['fpq']['at_least_4_count'],g['fpq']['valid'])}, {cell(g['nfp']['at_least_4_count'],g['nfp']['valid'])}다. 두 구현은 동일한 답변 절차가 아니다.",
        'Luna 추출은 732개, 추출 전제 검증은 730개 유효/2개 실패, 주석 검증은 주석 있는 730개 완료, 추출 감사는 730개 유효/2개 실패다. Gemma 추출 전제 검증은 725개 유효/7개 무효, 주석 검증은 729개 유효/1개 무효다. 실패를 NFP 판정으로 처리하지 않았다.',
        '기존 Qwen verifier에 Sonnet이 보완한 전제 목록을 준 별도 725개 비교도 실행했다: FPQ 108/577→199/577(18.7→34.5%), NFP 20/148→24/148(13.5→16.2%). 이는 원 질문 수정이 아니라 검증 입력 목록의 변경이다. 주석 조건의 공통 집합과 분모가 다르다.','',
        '## 재현','',
        '생성: `python3 scripts/summarize_original_question_tables.py`. 원 집계와 해시는 `original_question_tables.json`에 보존한다. 본문 AUROC는 기존 발표 문서의 반올림 값을 사용했고 새 AUROC를 계산하지 않았다.']
    (OUT/'original_question_tables.md').write_text('\n'.join(lines)+'\n')
    (OUT/'original_question_tables.json').write_text(json.dumps(dict(table1=t1,table2=t2,today=dict(context=context,audit=audit,lost=lost),sources=SOURCES),ensure_ascii=False,indent=2)+'\n')
    print(f'Wrote {len(t1)} gate rows and {len(t2)} answer rows with counts and provenance.')


if __name__ == '__main__':
    main()
