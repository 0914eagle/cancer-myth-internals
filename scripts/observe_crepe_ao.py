"""Export saved CREPE AO observations; no model, network, or GPU calls."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/ao_behavior_comparison_crepe_20261014'
QUERY = ROOT / 'results/ao_query_comparison_20261014'
PILOT = ROOT / 'external/paper_baselines_b81ace2/results/nla_ao_parallel_20261012/gepa_reader_pilot'
OUT = ROOT / 'results/crepe_ao_observation_20261014'

# Qualitative case notes written after reading the saved question, reference,
# all three AO descriptions, and the complete corresponding source answer.
# Cases illustrate failure modes and are not a randomly sampled accuracy panel.
NOTES = {
    'crepe:2018-00499': dict(
        outcome='target_missing_from_reader_and_answer',
        target_ko='젖은 손가락은 얼어붙은 기둥에 붙지 않는다는 가정',
        reader_field='stance_text',
        reader_quote='a misunderstanding of the scientific principles involved',
        answer_quote="With a wet finger, the water doesn't have as much surface area",
        note_ko='AO는 일반적인 misunderstanding만 말하고 젖은 손가락도 붙을 수 있다는 주석의 대안을 말하지 않는다. 실제 답변도 차이를 설명하는 데 머문다. 이 사례에서 답변을 보완할 구체적 교정 단서는 확인하지 못했다.'),
    'crepe:2018-15077': dict(
        outcome='annotation_target_missing_from_reader_and_answer',
        target_ko='데이터 주석의 목표는 전위가 절대적이라는 가정',
        reader_field='behavior_text',
        reader_quote='which is a standard setup for increasing voltage',
        answer_quote='current requires a closed loop',
        note_ko='AO는 직렬 연결을 설명할 것이라고 하고 실제 답변은 닫힌 회로를 설명한다. 전위의 상대성이라는 주석의 교정 내용은 AO에 없다. 주석 기준의 비교이며 질문의 물리적 정답을 새로 검증한 것은 아니다.'),
    'crepe:2018-03248': dict(
        outcome='relevant_alternative_shared_with_answer',
        target_ko='작가는 자기 이야기가 기존 작품과 겹치지 않는다고 확실히 알 수 있다는 가정',
        reader_field='stance_text',
        reader_quote='authors may not always be aware of the originality of their ideas',
        answer_quote="Authors don't have a foolproof way to *know for sure*",
        note_ko='stance 설명에는 확실히 알 수 없다는 방향의 단서가 있고 실제 답변도 이미 이를 말한다. behavior 설명의 verify originality는 같은 한계를 명시하지 않는다. 답변에 없는 추가 정보로 세지 않는다.'),
    'crepe:2018-02290': dict(
        outcome='relevant_alternative_shared_with_answer',
        target_ko='수분을 포함한 나무는 불탈 수 없다는 가정',
        reader_field='stance_text',
        reader_quote='trees can burn even when they are full of moisture',
        answer_quote="water *does* inhibit burning, it doesn't prevent it entirely",
        note_ko='stance 설명은 목표 전제의 대안을 담지만 실제 답변에도 이미 있다. behavior 설명은 수분을 포함한다는 주제를 반복하는 데 가깝다.'),
    'crepe:2018-22025': dict(
        outcome='behavior_description_matches_existing_correction',
        target_ko='사람들이 동일한 색 경험을 한다는 확신',
        reader_field='behavior_text',
        reader_quote='question the assumption that colors are perceived identically by everyone',
        answer_quote='we *don\'t* truly know if your experience of "red" is the same as mine',
        note_ko='behavior 설명은 실제 답변과 대응한다. 반면 stance 설명은 동일한 색 경험을 수용한다고 한다. 같은 활성의 질문 문구 민감도 사례이며 추가 정보 사례는 아니다. 기존 자동 검토는 질문-주석 관계를 ambiguous로 분류했다.'),
    'crepe:2018-01728': dict(
        outcome='more_explicit_diagnostic_candidate_not_unique_information',
        target_ko='숙성주는 모두 최소 10년 숙성해야 한다는 가정',
        reader_field='behavior_text',
        reader_quote='question the assumption that all aged spirits require a minimum of 10 years of aging',
        answer_quote="even if the *majority* isn't 10+ years old",
        note_ko='AO는 최소 10년이라는 가정을 실제 답변보다 직접적으로 지목한다. 실제 답변에도 어린 술 언급이 있어 정보의 부재가 아니라 명료성 차이일 수 있다. 주석과 질문 대응도 기존 검토에서 ambiguous여서 성공 수에 넣지 않는다. 법규나 주류 지식의 정확성을 이 분석에서 새로 검증하지 않았다.'),
    'crepe:2018-01138': dict(
        outcome='nfp_direct_answer_description_matches',
        target_ko='정상적인 버터밀크의 용도와 일반 우유와의 차이 질문',
        reader_field='behavior_text',
        reader_quote='providing information on its unique properties and how it differs from regular milk',
        answer_quote='Buttermilk is used in many cakes and breads',
        note_ko='AO는 차이를 설명할 것이라고 하고 실제 답변도 그렇게 한다. 대응하는 정상 답변의 사례이지 내부에 명시적인 NFP 판정기가 있다는 증거는 아니다.'),
    'crepe:2018-01918': dict(
        outcome='nfp_reader_invents_universal_assumption',
        target_ko='질문은 일부 음식의 조리 후 영양 손실을 묻는다',
        reader_field='behavior_text',
        reader_quote='cooking always reduces nutritional value',
        answer_quote='Cooking can indeed reduce the nutritional value of some foods',
        note_ko='AO가 some을 always라는 다른 가정으로 확대한다. 실제 답변은 일부 음식의 영양 손실을 설명하고 예외도 보충한다. 예외 보충 자체를 과잉 반박으로 세지 않는다.'),
    'crepe:2018-06974': dict(
        outcome='hold_annotation_and_interpretation_conflict',
        target_ko='음의 무게, 부력, 용기의 구조적 제약이 혼재된 진공 풍선 질문',
        reader_field='behavior_text',
        reader_quote="it will discuss how this property affects the balloon's buoyancy",
        answer_quote="A vacuum balloon wouldn't float; it would likely implode",
        note_ko='과거 자동 검토에서 추가 정보 1건으로 표시한 문항이다. 같은 검토에서 질문-주석 충돌도 표시됐다. AO가 부력을 언급한다는 사실만으로 관련 정보가 정확하다고 검증되지 않으므로 성공 사례에서 보류한다.'),
}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    rows = read_rows(SOURCE / 'crepe50.jsonl')
    query = {r['question_id']: r for r in read_rows(QUERY / 'crepe50.jsonl')}
    audits = {r['row_id']: r for r in read_rows(SOURCE / 'content_audits.jsonl')}
    runtime = json.loads((PILOT / 'runtime.json').read_text())
    references = {r['id']: r for r in json.loads(Path(runtime['bundle']).read_text())['rows']}
    assert len(rows) == len({r['question_id'] for r in rows}) == 50
    assert Counter(r['question_type'] for r in rows) == {'FPQ': 25, 'NFP': 25}
    OUT.mkdir(parents=True, exist_ok=True)
    exports = []
    for r in rows:
        q = query[r['question_id']]
        # All AO query variants must correspond to the same activation and answer.
        assert r['activation_sha256'] == q['activation_sha256']
        assert r['actual_answer'] == q['actual_answer']
        assert r['source_native_id'] == q['source_native_id']
        assert q['exact_activation_match_verified'] is True
        assert r['source_text_given_to_ao'] is False
        ref = references[r['question_id']]
        a = audits[r['question_id']]
        note = NOTES.get(r['question_id'])
        if note:
            assert note['reader_quote'] in r[note['reader_field']]
            assert note['answer_quote'] in r['actual_answer']
        exports.append(dict(**r, presuppositions=ref['presuppositions'],
            reference_answer=ref['reference_answer'],
            prior_automated_audit=a['audit'], prior_audit_model=a['model'],
            qualitative_review=note,
            review_scope='Retrospective assistant review of selected cases; not independent human validation.'))
    (OUT / 'crepe50_observation.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exports))
    fields = ['question_id', 'question_type', 'question', 'presuppositions', 'reference_answer',
              'neutral_text', 'stance_text', 'behavior_text', 'actual_answer', 'well_score',
              'prior_annotation_alignment', 'review_outcome', 'review_note_ko']
    with (OUT / 'crepe50_observation.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in exports:
            writer.writerow({**{k: json.dumps(r[k], ensure_ascii=False) if k == 'presuppositions' else r[k]
                                for k in fields[:10]},
                'prior_annotation_alignment': r['prior_automated_audit']['annotation_alignment'],
                'review_outcome': (r['qualitative_review'] or {}).get('outcome', 'not_selected_for_detailed_review'),
                'review_note_ko': (r['qualitative_review'] or {}).get('note_ko', '')})
    write_json(OUT / 'reviewed_cases.json', [r for r in exports if r['qualitative_review']])
    summary = dict(
        source_questions=50, FPQ=25, NFP=25, ao_queries_per_question=3, ao_descriptions=150,
        distinct_native_answers=50, exact_activation_and_answer_match_verified=50,
        reader_input='Same Gemma3-27B Plain prefill activation; layer31, last up to20 question tokens. No question/answer/reference text supplied to AO.',
        source_model='google/gemma-3-27b-it',
        new_external_model_calls=0, new_gpu_calls=0,
        detailed_qualitative_cases=len(NOTES),
        previous_audit_provenance='Existing historical gpt-6.1-sol content audit, read only; no new Sol or Luna requests made by this script.',
        prior_automated_results=json.loads((SOURCE / 'comparison_summary.json').read_text()),
        source_well_distributions={label: dict(Counter(str(r['well_score']) for r in rows if r['question_type'] == label))
                                  for label in ['FPQ', 'NFP']},
        prior_additional_information_flagged_ids=[r['question_id'] for r in exports
            if r['prior_automated_audit']['current']['new_information_beyond_answer']],
        conclusion_ko='목표 전제를 설명할 수 있는 사례와 실패 사례가 공존한다. 실제 답변에 없는 정확한 교정 정보의 확정적 성공 사례는 이번 정성 검토에서 확보하지 못했다. 숙성주 사례는 설명의 명료성 향상 후보로 보존한다.',
        limits=['Previously examined train50, not independent test or a random accuracy estimate.',
                'Prior automated topic/target judgments are not newly validated ground truth.',
                'Specific premise mention, attributed stance, correct alternative, response prediction, and added information are separate observations.',
                'No claim about scientific/medical/legal factual correctness beyond the saved benchmark alignment.',
                'Useful feedback may reorganize information already present in the answer; uniqueness is not necessary for GEPA benefit.'])
    write_json(OUT / 'observation_summary.json', summary)
    write_json(OUT / 'input_checksums.json', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SOURCE / 'crepe50.jsonl', QUERY / 'crepe50.jsonl', SOURCE / 'content_audits.jsonl', SOURCE / 'protocol.json']})
    (OUT / 'README.txt').write_text(
        'CREPE AO 관찰 자료\n\n'
        'crepe50_observation.csv: 질문, 데이터 전제 주석, reference, AO 세 조건, 같은 생성의 답변, 기존 Well 점수를 나란히 배치.\n'
        'crepe50_observation.jsonl: 기존 자동 분석과 새 정성 검토를 구분해 보존.\n'
        'reviewed_cases.json: 내용 검토한 9개 사례와 정확한 인용, 보류 사유. 선택 사례는 정확도 분모로 쓰지 않음.\n'
        'observation_summary.json: 실험 범위, 기존 자동 분석의 출처와 한계.\n\n'
        '새 모델 호출, GPU 실행, GEPA 설정 변경 없음. 과거 Sol 자동 판정은 참고용으로만 보존했고 재호출하지 않음.\n'
        '재생성: python3 scripts/observe_crepe_ao.py\n')
    print(json.dumps({'output': str(OUT), 'questions': 50, 'AO_descriptions': 150,
                      'detailed_cases': len(NOTES), 'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
