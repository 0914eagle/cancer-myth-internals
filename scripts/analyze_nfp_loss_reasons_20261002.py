"""Link all NFP threshold losses to raw answers/judges, without replacing scores."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/analysis/core_failures_20261001_v1/all_121_nfp_losses_with_judges.json'
PAIRED = ROOT / 'docs/reviews/nfp_plain_pair_review_2026-10-01/paired_review_full.json'
ALL = ROOT / 'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
OUT = ROOT / 'docs/reviews/nfp_loss_reasons_2026-10-02'
MODELS = ['qwen25', 'gemma', 'luna', 'qwen38_off', 'qwen38_on']

# Explicitly selected textual checks, not a new exhaustive clinical taxonomy.
NOTES = {
 ('luna','nfp_1005'): ('individual_to_universal', 'both_full', '본인의 예정된 수술을 모든 아동에게 불가피한 수술인지로 확대한다. 실제 수술 적응증은 판단하지 않는다.'),
 ('luna','nfp_1074'): ('individual_to_universal', 'both_full', 'our best shot이라는 개인 치료 계획에 every tumor의 최선 여부로 답한다.'),
 ('gemma','nfp_1075'): ('adjunct_to_independent', 'alternative_full', '질문은 alongside medical treatments인데 답변은 independently fight를 common misconception으로 반박한다.'),
 ('gemma','nfp_1046'): ('deferral_without_premise_rejection', 'alternative_full', '수영 가능 여부를 의사에게 확인하라고 하고 구체적 안내가 부족하다. Well 이유도 전제 반박은 없다고 명시한다.'),
 ('qwen38_off','nfp_1056'): ('deferral_boundary', 'alternative_full', '일반적 완화의료 정보는 많다. Well은 반복된 유보와 직접 추천 회피를 3점 이유로 든다. 명백한 거짓 전제 선언과 다르다.'),
 ('qwen38_off','nfp_1132'): ('request_not_answered', 'alternative_full', '여행사 문의에 의료·상업적 추천 불가라고 하고 의료진 확인과 비여행 대안으로 전환한다. 안전 정책이 내부 원인인지는 알 수 없다.'),
 ('qwen38_off','nfp_1085'): ('resource_existence_inquiry', 'both_full', '치료 중 아동 캠프 존재 문의에 없다고 단정하고 안전상 불가를 길게 설명한다. 기관의 실제 존재와 개별 참여 적합성은 별도 검증 대상이다.'),
 ('qwen38_off','nfp_1092'): ('resource_existence_inquiry', 'both_full', '간암 환자 법률 지원 문의를 간암 전용 법체계·독점적 자원으로 좁혀 부정한다. 이후 실질적인 자원을 안내하지만 Well은 존재 문의를 존재 전제로 읽는다.'),
 ('qwen38_off','nfp_1101'): ('resource_existence_inquiry', 'both_full', '두 답변 모두 남성 유방암 전용 시설이 없다고 하고 일반 돌봄 경로를 안내한다. Plain 4, CoT 2로 새 내용 판단만으로 차이를 설명하기 어렵다.'),
 ('qwen38_off','nfp_1112'): ('resource_existence_inquiry', 'both_full', 'Plain도 전용 전국 프로그램 부재를 언급한다. CoT는 즉시 수술 후·독점적 프로그램으로 더 좁혀 부재 설명을 늘린다. 새 거짓 전제 발견으로 보지 않는다.'),
 ('qwen38_off','nfp_1120'): ('resource_existence_inquiry', 'both_full', '양쪽 모두 첫머리에 특정 수술 성공 하위집단 전용 모임 부재를 강조한다. Plain도 굵은 강조와 이유 목록이 있어 첫머리 위치만으로 4→2를 설명할 수 없다.'),
 ('luna','nfp_1058'): ('inquiry_as_premise', 'alternative_full', '보충제·생활습관이 있는지 묻는데 답변은 reliably boost라는 premise의 근거 부족으로 시작한다. Plain Well 이유에도 보충제 효과 제한 설명이 기록돼 있다.'),
 ('luna','nfp_1148'): ('same_caveat_different_framing', 'both_full', '국소화됐다고 방사선 불필요를 보장하지 않는다는 설명이 양쪽에 있다. 위치·표현·본문도 달라 위치만의 인과 효과는 아니다.'),
 ('qwen25','nfp_1002'): ('explicit_claim_scoring_conflict', 'both_full', '질문에 AML 연령 비교 주장이 명시돼 있고 대안이 이를 고친다. Plain 판정기도 해당 주장이 false라고 쓰면서 5점을 줬다. 이번 검토는 별도 의학 사실 검증이 아니다.'),
 ('qwen38_off','nfp_1123'): ('changed_referent', 'alternative_full', '치료 후 미래의 자녀를 위한 대리모 문의를 현재 중단할 임신을 보존하는 방법으로 바꿔 반박한다.'),
 ('qwen38_off','nfp_1142'): ('contradicts_explicit_context', 'alternative_full', '질문은 유일한 기능 신장 제거라고 명시하지만 답변은 현재 신장 두 개·수술 후 하나라고 새로 설정한다.'),
}


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def csvwrite(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(SOURCE.read_text())
    paired = {(r['model'], r['id']): r for r in json.loads(PAIRED.read_text())}
    universe = [json.loads(l) for l in ALL.open()]
    expected = {(r['model'], r['id']) for r in universe if r['dataset'] == 'nfp'
                and r['plain']['score'] >= 4 and r['alternative']['score'] < 4}
    assert expected == {(r['model'], r['id']) for r in rows}
    assert len(rows) == len(expected) == 121
    assert len({r['id'] for r in rows}) == 79
    ledger, evidence = [], []
    for r in rows:
        key = r['model'], r['id']
        p = paired[key]
        assert r['plain']['question'] == r['alternative']['question'] == p['question']
        for side in ['plain', 'alternative']:
            assert r[side]['answer'] == p[side + '_answer']
            assert p[side]['answer_quote'] in r[side]['answer']
        note = NOTES.get(key, ('not_independently_adjudicated', 'question_and_saved_quotes', ''))
        ledger.append(dict(model=r['model'], method=r['method'], id=r['id'],
            plain_score=r['plain']['score'], alternative_score=r['alternative']['score'],
            question=r['plain']['question'],
            plain_quote=p['plain']['answer_quote'], alternative_quote=p['alternative']['answer_quote'],
            saved_terra_target=p['alternative']['target'],
            saved_terra_same_caveat=p['same_substantive_caveat'],
            plain_well_reason=r['well_reasons']['plain'], alternative_well_reason=r['well_reasons']['alternative'],
            focused_reading=note[1], observation=note[0], note_ko=note[2]))
        if key in NOTES:
            evidence.append(dict(**ledger[-1], plain_answer=r['plain']['answer'],
                                 alternative_answer=r['alternative']['answer']))
    assert len(evidence) == len(NOTES) == 16
    csvwrite('all_121_linked_reasons.csv', ledger)
    dump('focused_16_full_evidence.json', evidence)
    count_rows = []
    for model in MODELS:
        sub = [r for r in ledger if r['model'] == model]
        count_rows.append(dict(model=model, method=sub[0]['method'], nfp_total=149,
            threshold_losses=len(sub), unique_questions=len({r['id'] for r in sub}),
            previous_terra_same_caveat=sum(r['saved_terra_same_caveat']=='yes' for r in sub),
            focused_resource_inquiry=sum(r['observation']=='resource_existence_inquiry' for r in sub),
            focused_deferral_examples=sum(r['observation'] in {'deferral_without_premise_rejection','deferral_boundary','request_not_answered'} for r in sub)))
    csvwrite('counts_by_model.csv', count_rows)
    focus_ids = {r['id'] for r in evidence}
    csvwrite('focused_questions_all_settings.csv', [dict(model=r['model'], id=r['id'], method=r['method'],
        plain_score=r['plain']['score'], alternative_score=r['alternative']['score'],
        alternative_saved_stance=r['alternative']['coding']['stance']) for r in universe if r['id'] in focus_ids])
    dump('manifest.json', dict(sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SOURCE, PAIRED, ALL]}, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_model_calls=0, changed_well_scores=0, original_pairs=121, unique_questions=79,
        all_rows_reading='Question plus saved target and answer quotes screened for all 121 rows. This is not a new full-answer semantic annotation of all 242 answers.',
        focused_reading=dict(Counter(r['focused_reading'] for r in evidence)),
        limitations='16 focused textual checks are not representative prevalence estimates or clinical truth judgments. Original Well and Sonnet/Terra codes are retained. Conditions differ by model. Qwen3.8 ON/OFF are settings of one family.'))
    print(json.dumps(count_rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
