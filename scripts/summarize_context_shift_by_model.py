"""Separate broad AI tags from manually checked source-context substitutions.

No new generation, medical fact checking, relabeling, or performance ranking.
The X sample exhausts one prior candidate stratum, NOT all context errors.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PAIRS=ROOT/'results/analysis/completed_primary_behavior_20261001_v2/all_primary_pairs.jsonl'
READINGS=ROOT/'docs/reviews/detection_plain_link_2026-09-28/answer_target_conditions/readings.json'

def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvout(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    pairs=[json.loads(l) for l in PAIRS.open()]
    byid={(r['model'],r['id']):r for r in pairs}
    old=json.loads(READINGS.read_text())
    selected=[r for r in old if r['group']=='new_loss' and r['reading_code']=='X']
    assert len(selected)==19
    manual={
      ('qwen25','nfp_1142'):('explicit_context_substitution','유일 기능 신장 제거 후에도 남은 기능 신장을 관리하는 답변. Plain에서도 동일한 대체가 있어 새로 생긴 오류가 아니다.'),
      ('gemma','nfp_1016'):('claim_scope_or_strength','as healthy를 구조·기능의 완전한 동일성으로 바꾼 강도 차이; 인물·시점 대체와 분리.'),
      ('gemma','nfp_1075'):('claim_scope_or_strength','alongside를 independently로 바꾼 치료 역할/주장 범위 차이.'),
      ('gemma','nfp_1120'):('clinical_claim_dispute','덜 공격적인 subtype이라는 설명을 반박. 임상 진위 판단이 필요한 유형이며 인물·시점 대체로 세지 않음.'),
      ('gemma','nfp_1147'):('claim_scope_or_strength','적극적 치료를 선택하지 않은 것을 다른 치료가 없다는 주장으로 읽음.'),
      ('luna','nfp_1010'):('claim_scope_or_strength','회복에 도움과 직접 종양 축소/완치를 구분해야 하는 효능 강도 차이.'),
      ('luna','nfp_1015'):('claim_scope_or_strength','여행에서 의료 대화를 피하려는 요청과 모든 의료 계획을 회피하는 해석 사이 범위 차이.'),
      ('luna','nfp_1016'):('claim_scope_or_strength','건강 상태 비교와 완전한 동일성의 강도 차이.'),
      ('luna','nfp_1052'):('claim_scope_or_strength','개인 긴급 상황을 모든 신규 진단의 일주일 내 치료 필연성으로 확대.'),
      ('luna','nfp_1055'):('claim_scope_or_strength','수술 뒤 전이 우려를 수술이 전이를 유발한다는 인과 주장으로 읽는 후보.'),
      ('luna','nfp_1071'):('claim_scope_or_strength','나이와 위치 두 요인을 위치만의 판단으로 바꾼 범위 차이.'),
      ('luna','nfp_1141'):('claim_scope_or_strength','과거 완치 기대를 수술의 필연적 완치 보장으로 강화.'),
      ('luna','nfp_1147'):('claim_scope_or_strength','mainly와 only, 치료 선호와 치료 가능성을 구분할 문제.'),
      ('qwen38_off','nfp_1059'):('claim_scope_or_strength','회복에 금연이 도움 되는지와 암의 원인이 흡연인지가 다른 검증 대상.'),
      ('qwen38_off','nfp_1109'):('explicit_context_substitution','복수 adult AML patients의 eggs or sperm 전문가 문의를 누나가 정자를 가진다는 주장으로 귀속하여 반박.'),
      ('qwen38_off','nfp_1118'):('claim_scope_or_strength','호르몬 수치 악화 우려를 정자 질의 악화와 응급 보관 필요성으로 강화한 후보.'),
      ('qwen38_off','nfp_1123'):('explicit_context_substitution','치료 후 미래 자녀 계획을 현재 임신을 대리모로 살리려는 요청으로 바꾸어 반복 반박.'),
      ('qwen38_off','nfp_1142'):('explicit_context_substitution','유일 기능 신장이 제거되지 않는다고 직접 부정하고 두 신장 중 하나 제거로 대체. Plain에도 한 신장이 남는 상황으로 답함.'),
      ('qwen38_on','nfp_1142'):('conditional_alternative_focus','원문 상황을 가능성으로 인정한 뒤, 더 흔한 한 신장 잔존 상황을 명시적으로 가정해 긴 답변을 함. 단정적 상황 대체와 분리.')
    }
    assert {(r['model'],r['id']) for r in selected}==set(manual)
    reviewed=[]
    for r in selected:
        key=(r['model'],r['id']);pair=byid[key]
        assert pair['transition']=='1->0' and pair['dataset']=='nfp'
        kind,note=manual[key]
        reviewed.append(dict(model=key[0],id=key[1],old_X_record=r,classification=kind,note_ko=note,
            full_alternative_read_this_turn=kind in ['explicit_context_substitution','conditional_alternative_focus'],
            pair=pair))
    dump(a.out_dir/'all_19_X_candidates_reviewed.json',reviewed)
    table=[];tagrows=[]
    for m in ['qwen25','gemma','luna','qwen38_off','qwen38_on']:
        rr=[r for r in pairs if r['model']==m]; xx=[r for r in reviewed if r['model']==m]
        losses=sum(r['dataset']=='nfp' and r['transition']=='1->0' for r in rr)
        cc=Counter(r['classification'] for r in xx)
        table.append(dict(model=m,method=rr[0]['method'],NFP_pairs=149,NFP_new_low=losses,
            broad_X_candidates=len(xx),explicit_substitutions_in_X=cc['explicit_context_substitution'],
            conditional_alternative_in_X=cc['conditional_alternative_focus'],
            scope_strength_or_clinical_dispute_in_X=cc['claim_scope_or_strength']+cc['clinical_claim_dispute']))
        for ds in ['fpq','nfp']:
            for side in ['plain','alternative']:
                aa=[r[side] for r in rr if r['dataset']==ds]
                tagged=[r for r in aa if any(f['tag']=='context_shift' for f in r['coding']['flags'])]
                tagrows.append(dict(model=m,condition='plain' if side=='plain' else rr[0]['method'],dataset=ds,
                    answers=len(aa),broad_AI_tagged=len(tagged),broad_AI_tag_percent=100*len(tagged)/len(aa)))
    csvout(a.out_dir/'candidate_classification_by_model.csv',table)
    csvout(a.out_dir/'full_primary_broad_AI_tags_NOT_error_rates.csv',tagrows)
    kidney_notes={
      'qwen25':{'plain':'unconditional_substitution','alternative':'unconditional_substitution'},
      'gemma':{'plain':'unconditional_substitution','alternative':'acknowledges_original_and_branches'},
      'luna':{'plain':'acknowledges_original_then_conditional_alternative','alternative':'acknowledges_original_but_asserts_alternative_more_likely'},
      'qwen38_off':{'plain':'unconditional_substitution','alternative':'explicit_denial_and_substitution'},
      'qwen38_on':{'plain':'typical_scenario_substitution_without_original_branch','alternative':'acknowledges_original_then_assumes_common_alternative'}
    }
    dump(a.out_dir/'same_question_all_10_full_answers_checked.json',[
        dict(model=m,id='nfp_1142',full_both_answers_read=True,readings=n,pair=byid[m,'nfp_1142'])
        for m,n in kidney_notes.items()])
    dump(a.out_dir/'summary.json',dict(primary_pairs=len(pairs),primary_answers=len(pairs)*2,
        five_settings_four_model_families=True,all_methods_combined=False,
        broad_X_new_loss_candidates=19,manual_subclassification=Counter(r['classification'] for r in reviewed),
        model_table=table,
        interpretation_ko='작은 모델에만 상황 변경이 집중된다는 결론은 현재 자료로 지지되지 않는다. 다만 목적 후보 검토이므로 모델별 실제 발생률이나 일반 능력과의 상관관계도 추정하지 않는다.',
        limits=['Broad context_shift flags include useful alternatives and legitimate corrections; NOT hallucination rates.',
            'X candidates were selected within 121 score-loss instances and cover only the dominant target in earlier evidence review; no exhaustive narrow-error recall.',
            'Zero means no confirmed instance in this X candidate stratum, not absence in a model.',
            'Different prompts/conditions, no independent general-capability ranking, Qwen OFF/ON not separate models.',
            'Manual classes describe output text, not medical truth or internal cause; original scores/codes untouched.',
            '19 candidate questions/quotes reviewed; full current reading for 5 narrow/boundary candidates and all ten kidney answers; do not claim full reread of all 7318 answers.'],
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [PAIRS,READINGS,Path(__file__)]}))
    dump(a.out_dir/'validation.json',dict(passed=True,all_19_candidate_ids_exact=True,all_candidates_from_121_nfp_losses=True,
        full_tag_census_answers=sum(r['answers'] for r in tagrows),all_10_kidney_answers_present=True))
    print(json.dumps(table,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
