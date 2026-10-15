"""Freeze and report the first 1,404 completed answer audits, without new calls."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audit_answer_behavior_full as a
from scripts.analyze_ungated_answer_transitions import CONFIG

OUT = a.ROOT / 'results/analysis/answer_behavior_partial_1404_20260930'
NOTES = {
 'a_6ecb7fdc821ef315ecf8': 'Qwen2.5 Plain은 원 질문의 in the clear를 반복하고 활동을 안내하지만 추적관찰은 언급하지 않는다. 목표 오류를 수용했다는 분류는 텍스트와 맞는다. 다만 scope_strengthening 태그는 원 질문의 표현을 반복한 것이라 범위를 더 강하게 만들었다는 별도 증거는 약하다.',
 'a_df75acdb553840b758e6': '같은 fpq_206의 Gemma 검토 후 답변은 in the clear를 명시적으로 부정하고 정기 추적관찰을 포함한 활동 계획을 제시한다. 모델과 조건이 모두 달라 순수 모델 효과나 지시 효과를 추정하는 대조는 아니다.',
 'a_f32ed1b8a95ee31169af': 'Luna는 개인에 대한 의사 소견에서 질환 전체의 일반론으로 논의를 넓히며 전제를 문제 삼지만, 개별 저항성 가능성도 인정하고 임상시험 검색 방법을 안내한다. 문맥을 완전히 무시했다거나 전면 거부했다고 요약하면 안 된다. 주석 역시 개인 상황을 일반론으로 바꾼 표현을 포함해 원 질문을 우선 확인해야 한다.',
 'a_b76147bda4ec1b218470': 'Luna 답변은 The premise isn’t necessarily correct라고 명시적으로 전제를 의심하면서 요청에 답한다. Claude는 qualification으로 분류했다. 반박과 단서의 경계가 안정적이지 않다는 점검 사례이며 원 분류를 덮어쓰지 않는다.',
 'a_adcc1398a4e3a8def344': 'Qwen3.8 OFF의 존재 여부 문의에 대한 부정 답변을 기존 Well은 강한 전제 반박으로 보고 2점, 새 분류는 qualification으로 봤다. 존재 문의에 No라고 답하는 것 자체는 전제 오탐의 충분한 근거가 아니다. 답변은 여러 대체 시설·서비스 안내를 제공한다. 의료·시설 존재 주장의 진위는 재판정하지 않았다.',
 'a_1924890e2faa08b30863': 'Qwen2.5는 정기 치과 검진과 자가 점검은 검진 대체가 아니라는 설명을 포함하지만, 질문의 희귀성에 동의하고 대부분 생활습관 안내를 한다. 목표 교정을 얼마나 명시적으로 요구하는지에 따라 평가가 갈린다. context_shift 태그도 질문의 오류를 바로잡는 행동과 의미 왜곡을 혼동할 수 있어 그대로 원인 태그로 쓰기 어렵다.',
 'a_c2d5b0834b334aa628e5': '원 질문은 terminal illnesses like CML이라고 질환을 말기 범주로 묶으며 주석은 불치·제한된 생존이라는 목표 오류를 제시한다. 답변은 장기 관해 가능성을 언급하면서 해당 표현과 no known cure를 유지한다. Claude corrects_target과 Well 1점의 불일치 사례다. 관련 설명의 존재만으로 실제 교정 성공 또는 내부 판단 형성을 인정할 수 없다. 일반 환자의 장기 관해 가능성과 특정 환자의 말기 상태가 논리적으로 반드시 모순되는 것은 아니므로, 앞에서는 교정했고 뒤에서는 이를 덮었다는 인과·모순 주장에 사용하지 않는다. 질문을 아버지가 말기라서 여행을 준비한다고 축약하면 질환 일반화와 개인 상태의 차이가 지워진다.',
 'a_8dd419b9701913645af2': 'Qwen2.5는 부작용의 개인차와 치료 반응에 따른 휴가 필요성을 설명한다. 새 분류는 목표 교정, Well은 예방·관리 가능성을 명확히 설명하지 않았다며 1점이다. 어느 평가를 참값으로 자동 채택하지 않고, 교정의 범위·명시성 기준이 다른 사례로 남긴다.',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs=json.loads((a.OUT/'inputs.json').read_text())
    source_inputs={x['key']:x for x in inputs}
    metadata={x['key']:x for x in json.loads((a.OUT/'manifest.json').read_text())}
    snapshot=OUT/'snapshot.json'
    if snapshot.exists():
        snap=json.loads(snapshot.read_text())
    else:
        files=sorted((a.OUT/'batches').glob('*/result.json'))
        keys=[x['key'] for f in files for x in json.loads(f.read_text())['items']]
        assert len(keys)==1404, 'Use a new named snapshot if coverage changed.'
        snap=dict(valid=1404,expected=7318,keys=keys,
                  sources={str(f):a.sha(f) for f in files},
                  input_sha256=a.sha(a.OUT/'inputs.json'),manifest_sha256=a.sha(a.OUT/'manifest.json'))
        a.write(snapshot,snap,True)
    assert a.sha(a.OUT/'inputs.json')==snap['input_sha256']
    assert a.sha(a.OUT/'manifest.json')==snap['manifest_sha256']
    codes={}
    for name,sha in snap['sources'].items():
        path=Path(name);assert a.sha(path)==sha
        packet=json.loads(path.read_text());batch_index=int(path.parent.name)
        batch=inputs[batch_index*3:batch_index*3+3]
        assert packet['prompt_sha256']==a.digest(a.PROMPT+json.dumps(batch,ensure_ascii=False))
        valid,repairs=a.validate(packet['items'],batch);assert not repairs
        raw=Path(packet['attempt'])/'stdout.json'
        native=json.loads(raw.read_text())
        assert native.get('subtype')=='success' and not native.get('is_error') and native.get('stop_reason')=='end_turn'
        assert any(m==a.MODEL or m.startswith(a.MODEL+'-') for m in native['modelUsage'])
        codes.update({x['key']:x for x in valid})
    assert set(codes)==set(snap['keys'])
    rows=[dict(**metadata[k],code=c) for k,c in codes.items()]
    coverage=[];by_model=[]
    for model in a.PRIMARY:
        counts={'model':model}
        for side in ['plain','answer']:
            for ds in ['fpq','nfp']:
                rr=[r for r in rows if (r['model'],r['side'],r['dataset'])==(model,side,ds)]
                expected=sum((m['model'],m['side'],m['dataset'])==(model,side,ds) for m in metadata.values())
                coverage.append(dict(model=model,condition='plain' if side=='plain' else a.PRIMARY[model],dataset=ds,
                                     n=len(rr),expected=expected,well_ge4=sum(r['score']>=4 for r in rr),
                                     challenge=sum(r['code']['stance']=='challenge' for r in rr),
                                     coded_target_correction=sum(r['code']['reference_alignment']['status']=='corrects_target' for r in rr)))
                counts[ds+'_'+side]=len(rr)
        by_model.append(counts)
    summaries=[]
    for ds in ['fpq','nfp']:
        for high in [False,True]:
            rr=[r for r in rows if r['dataset']==ds and (r['score']>=4)==high]
            summaries.append(dict(dataset=ds,well_ge4=high,n=len(rr),
                stance=dict(Counter(r['code']['stance'] for r in rr)),
                target=dict(Counter(r['code']['reference_alignment']['status'] for r in rr)),
                response=dict(Counter(r['code']['response'] for r in rr)),
                flags=dict(Counter(f['tag'] for r in rr for f in r['code']['flags']))))
    pairs=defaultdict(dict)
    for row in rows:pairs[row['pair_key']][row['side']]=row
    complete_pairs={k:v for k,v in pairs.items() if len(v)==2}
    paired_tables=[]
    for model in a.PRIMARY:
        for ds in ['fpq','nfp']:
            pp=[v for v in complete_pairs.values() if v['plain']['model']==model and v['plain']['dataset']==ds]
            paired_tables.append(dict(model=model,dataset=ds,n=len(pp),
                 well_transitions=dict(Counter(str(int(v['plain']['score']>=4))+' -> '+str(int(v['answer']['score']>=4)) for v in pp)),
                 stance_transitions=dict(Counter(v['plain']['code']['stance']+' -> '+v['answer']['code']['stance'] for v in pp)),
                 target_transitions=dict(Counter(v['plain']['code']['reference_alignment']['status']+' -> '+v['answer']['code']['reference_alignment']['status'] for v in pp))))
    report=dict(answers=len(rows),unique_questions=len({r['id'] for r in rows}),
                by_class=dict(Counter(r['dataset'] for r in rows)),
                completed_pairs=len(complete_pairs),by_model=by_model,coverage=coverage,
                grouped_codings=summaries,paired_tables=paired_tables,
                limits=['Only 1404/7318 completed answers; not a representative population prevalence estimate.',
                        'Plain and alternative unpaired completion cohorts differ. Use complete pairs for within-question changes.',
                        'Model prompts and pipelines differ. Not a controlled model ranking.',
                        'Reference-assisted Claude coding is not independent clinical truth, and has identified semantic boundary errors.',
                        'No internal causal claim follows from this descriptive snapshot.'])
    a.write(OUT/'analysis.json',report)
    with (OUT/'coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]));w.writeheader();w.writerows(coverage)
    cache={};cases=[]
    for k,note in NOTES.items():
        r=metadata[k];method='plain' if r['side']=='plain' else r['method'];path=CONFIG[r['model']][method]/'attempts.jsonl'
        if path not in cache:cache[path]={x['id']:x for x in map(json.loads,path.read_text().splitlines()) if x.get('valid') and 'id' in x}
        judge=cache[path][r['source_job']];assert judge['score']==r['score']
        cases.append(dict(**r,input=source_inputs[k],coding=codes[k],saved_well_reason=judge['raw'],
                          direct_reading_ko=note,source_judge=str(path),source_judge_sha256=a.sha(path)))
    a.write(OUT/'checked_cases.json',cases)
    a.write(OUT/'codings_snapshot.json',[dict(**metadata[k],coding=c) for k,c in codes.items()])
    print(json.dumps(dict(answers=len(rows),completed_pairs=len(complete_pairs),full_cases_checked=len(cases),output=str(OUT))))


if __name__=='__main__':main()
