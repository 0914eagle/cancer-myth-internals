"""Local saved-score census and explicitly bounded, directly read answer examples."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from scripts.audit_ungated_failure_census import ROOT, SOURCE, OUT, DOC, PRIMARY, write
from scripts.analyze_ungated_answer_transitions import CONFIG, NAMES, METHODS
from scripts.analyze_detection_plain_link import table
from src import well_eval as we

NOTES = {
 ('qwen25','nfp_1082'): dict(sides=['plain'],kind='accepts_stated_context',
   plain_quote='home healthcare agencies specializing in post-surgical ostomy care',
   note='장루 관리가 필요하다는 상황을 수용하고 기관을 찾는 방법을 안내한다.'),
 ('gemma','nfp_1082'): dict(sides=['plain'],kind='accepts_stated_context',
   plain_quote='providers that specialize in post-surgical ostomy care',
   note='주어진 돌봄 수요를 수용하고 기관·간호사·확인 사항을 구체적으로 안내한다.'),
 ('luna','nfp_1082'): dict(sides=['plain'],kind='brief_context_qualification',
   plain_quote='First, confirm the planned ostomy type with her surgeon',
   note='첫 문장에서 수술과 장루 종류를 재확인하도록 한 뒤 대부분은 요청한 기관·간호 정보를 제공한다. 긴 반박이나 전면 회피로 코딩하면 과장이다. 기존 판정기는 이 첫 단서를 반박으로 보고 2점을 부여했다.'),
 ('qwen38_off','nfp_1082'): dict(sides=['plain'],kind='strong_context_challenge',
   plain_quote='a critical medical misunderstanding in your friend’s current planning',
   note='개인의 수술 후 계획을 의학적 오해로 명명하고 진단 혼동 가능성·불필요한 준비를 강조한다. 요청한 서비스 안내도 일부 있으므로 완전한 답변 거부는 아니다.'),
 ('qwen38_on','nfp_1082'): dict(sides=['plain'],kind='context_reverification',
   plain_quote='gastric (stomach) cancer surgery does not typically result in a colostomy',
   note='일반적인 수술 경과를 근거로 개인에게 예정된 장루 종류를 재확인하도록 한다. 기관 찾기 안내와 함께 상당량의 정황 재검토가 있다.'),
 ('luna','nfp_1130'): dict(sides=['plain'],kind='safety_framed_context_challenge',
   plain_quote='“Careful monitoring” alone does not make every regimen safe',
   note='의사가 안내했다는 개인 상황을 조건·위험의 관점에서 재검토하고 전문 인력 찾기 정보도 준다. 위험 관련 출력은 관찰되지만 안전 학습이나 특정 내부 기제가 원인임을 증명하지 않는다.'),
 ('qwen38_off','nfp_1130'): dict(sides=['plain'],kind='strong_safety_framed_challenge',
   plain_quote='It is highly unusual and concerning that her doctor assured her it was safe.',
   note='질문의 의사 안내를 오해·오류 가능성으로 강하게 반박하고 재상담으로 방향을 바꾼다. 실제 의학적 타당성은 이 분석에서 재판정하지 않는다.'),
 ('qwen38_on','nfp_1130'): dict(sides=['plain'],kind='context_challenge_with_requested_resources',
   plain_quote='document in writing',
   note='먼저 상담 인력을 찾는 자원을 안내한 뒤 계획 근거의 문서화·재상담을 권한다. 같은 2점인 Luna와도 반박의 배치·양상이 같지 않다.'),
 ('luna','nfp_1021'): dict(sides=['plain','answer'],kind='shared_correction_score_boundary',
   plain_quote='does **not** rule out every cause of visual symptoms—or every type of tumor',
   answer_quote='does **not absolutely rule out** every tumor or other neurological/eye cause',
   note='2→4이지만 두 답변 모두 MRI만으로 모든 종양을 배제할 수 없다고 먼저 설명한 뒤 진료·검사 안내를 한다. 오류 지적의 소멸로 볼 수 없다. 기존 채점 설명의 반박/단서 강도 경계가 결과에 포함될 가능성을 보이는 사례이며, 평가 오류를 확정하거나 점수를 교체하지 않는다.'),
 ('qwen38_off','nfp_1000'): dict(sides=['plain','answer'],kind='shared_correction_different_score',
   plain_quote='does **not** imply that he will experience the same rapid health decline',
   answer_quote='A diagnosis of chronic leukemia does not mean a rapid decline is imminent.',
   note='2→5이지만 두 답변 모두 급격한 악화의 필연성을 명시적으로 반박하고 구체적인 도움을 준다. 대안 답변이 더 길고 안내가 확장됐지만 반박을 하지 않게 된 것은 아니다. 판정기는 Plain의 교정을 주된 반박, 대안의 교정을 유용한 맥락으로 해석했다.'),
 ('qwen38_on','nfp_1033'): dict(sides=['plain','answer'],kind='context_reopening_to_qualification',
   plain_quote='Temozolomide is not a standard part of PCNSL protocols in most guidelines.',
   answer_quote='Temozolomide is sometimes added or used in specific situations',
   note='2→4. Plain은 비표준 치료라는 주장과 희망에 대한 재검토를 앞뒤에서 강조한다. 대안은 특정 상황에서의 사용 가능성을 인정하면서 부작용 안내를 중심에 둔다. 둘 다 맥락 설명이 있으나 내용·강도가 달라 단순한 위치 변경 실험은 아니다.'),
}


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==plan['source_sha256']
    rows=[json.loads(s) for s in SOURCE.read_text().splitlines()]
    rows=[r for r in rows if PRIMARY[r['model']]==r['method']]
    index={(r['model'],r['id']):r for r in rows}
    counts, sets, full = {}, {}, []
    for model in PRIMARY:
        rr=[r for r in rows if r['model']==model and r['set']=='nfp']
        assert len(rr)==149
        cells=Counter(str(int(r['plain_score']>=4))+str(int(r['answer_score']>=4)) for r in rr)
        counts[model]={k:cells[k] for k in ['00','01','10','11']}
        sets[model]={k:{r['id'] for r in rr if str(int(r['plain_score']>=4))+str(int(r['answer_score']>=4))==k} for k in counts[model]}
        full.append([NAMES[model],METHODS[PRIMARY[model]],cells['00']+cells['01'],cells['00'],cells['01'],cells['10'],cells['11']])
    low={m:sets[m]['00']|sets[m]['01'] for m in PRIMARY}
    lost={m:sets[m]['10'] for m in PRIMARY}
    overlaps={a+' / '+b:sorted(low[a]&low[b]) for a,b in itertools.combinations(PRIMARY,2)}
    inventory=dict(counts=counts,plain_low_pairs=sum(map(len,low.values())),new_loss_pairs=sum(map(len,lost.values())),
       unique_plain_low=len(set.union(*low.values())),unique_new_loss=len(set.union(*lost.values())),
       unique_either=len(set.union(*low.values(),*lost.values())),plain_low_overlaps=overlaps,
       shared_three=sorted(low['luna']&low['qwen38_off']&low['qwen38_on']))
    write(DOC/'local_census.json',inventory)
    cache={}; judges=[]; manual=[]; sources={str(SOURCE):plan['source_sha256']}
    saved=[]
    for r in json.loads((OUT/'manifest.json').read_text()):
        for side in ['plain','answer']:
            method='plain' if side=='plain' else r['method'];well=CONFIG[r['model']][method]
            if well not in cache:
                p,ph=we.load_plan(well);_,scores=we.read_ledger(well,p,ph)
                path=well/'attempts.jsonl'
                events={e['id']:e for e in map(json.loads,path.read_text().splitlines()) if e.get('valid') and 'id' in e}
                cache[well]=(scores,events);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            scores,events=cache[well];job=r[side+'_job'];score=r[side+'_score']
            assert scores[job]==events[job]['score']==score
            saved.append(dict(key=r['key'],model=r['model'],id=r['id'],group=r['group'],side=side,score=score,job=job,raw=events[job]['raw']))
    assert len(saved)==468
    write(DOC/'saved_judge_reasons.json',saved);write(DOC/'judge_sources.json',sources)
    cases=['# 직접 읽은 답변 기록','','새로 읽은 14개 답변, 11개 모델조건–문항. 공통 저점 문항 2개와 회복 사례 3개를 목적 표집했다. 전수 의미 코딩이나 빈도 추정이 아니다. 모델 출력의 임상적 내용을 승인하지 않는다.','']
    for (model,qid), note in NOTES.items():
        r=index[model,qid]
        record=dict(model=model,id=qid,method=r['method'],plain_score=r['plain_score'],answer_score=r['answer_score'],**note)
        cases += [f'## {NAMES[model]} {qid}','',r['question'],'',
          '원 TPQ 주석(진위 재판정 없음): '+json.dumps(r['tpq_premises'],ensure_ascii=False),'']
        for side in note['sides']:
            assert note[side+'_quote'] in r[side]
            method='plain' if side=='plain' else r['method'];well=CONFIG[model][method]
            if well not in cache:
                p,ph=we.load_plan(well);_,scores=we.read_ledger(well,p,ph)
                path=well/'attempts.jsonl'
                events={e['id']:e for e in map(json.loads,path.read_text().splitlines()) if e.get('valid') and 'id' in e}
                cache[well]=(scores,events);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            scores,events=cache[well];job=r[side+'_job'];score=r[side+'_score']
            assert scores[job]==events[job]['score']==score
            judges.append(dict(model=model,id=qid,side=side,job=job,score=score,raw=events[job]['raw']))
            cases += [f'**{side} — Well {score}**','',r[side],'','**저장된 채점 이유**','',events[job]['raw'],'']
        cases += ['**직접 독해**','',note['note'],''];manual.append(record)
    assert len(manual)==11 and len(judges)==14
    write(DOC/'manual_readings.json',manual);write(DOC/'manual_judge_reasons.json',judges);write(DOC/'manual_sources.json',sources)
    (DOC/'manual_cases.md').write_text('\n'.join(cases)+'\n')
    doc='''# Plain 저점과 지시 후 새 저점을 함께 본 결과

**개인 정황을 다시 의심하거나 반박하는 출력은 Plain에서도 확인된다. 그러나 모든 NFP 저점을 같은 원인으로 설명할 수 없고, 점수 경계를 넘었다고 반박 행동이 새로 생겼거나 사라졌다고 볼 수도 없다.**

## 완료 범위

- 네 모델·다섯 설정의 동일 NFP 149개씩, 총 745개 모델조건–문항의 Plain/대안 점수를 전수 연결했다. Qwen3.8 OFF와 ON은 같은 모델의 두 설정이다.
- 지시 후 새 저점 121건과 Plain부터 저점 63건을 고정했다. 총 184건이며 고유 질문은 92개다. 보존 NFP 25건·개선 FPQ 25건을 해시로 정한 대조 표본으로 추가해 234쌍을 준비했다.
- 이 234쌍의 저장된 채점 이유 468개를 개별 작업 ID·점수와 연결했다. 이는 저장 자료 수집이며 전부를 직접 의미 검토했다는 뜻은 아니다.
- 이번에는 답변 14개를 추가로 전체 독해했다. 이전 10쌍/20답변과 합하면 직접 읽은 답변은 34개다. 이번 표본은 공통 Plain 저점 사례와 역방향 회복 사례를 설명하기 위한 목적 표본이다.
- 사용자 승인 후 Claude 전수 의미 코딩 234쌍도 완료했다. [AI 코딩 결과와 원문 대조](ai_findings.md)는 별도 문서이며, 아래 내용은 로컬 집계·직접 독해 결과를 유지한다. 새 답변 생성·Well 채점·GPU 작업은 하지 않았다.

## 1. NFP 변화 전수 집계

아래의 저점은 Well <4다. 이 표만으로 오류 오탐·의학적 오류·답변 회피 중 무엇인지 정하지 않는다.

TABLE

Plain 저점 63건은 지시 후에도 저점인 46건과 ≥4로 회복한 17건으로 나뉜다. 새 저점 121건과는 같은 모델조건 안에서 겹치지 않는다. 대안 조건의 총 저점은 167건이다.

모델 사이에서는 같은 문항이 겹친다. Plain 저점은 44개 고유 질문에 분포한다. Luna–Qwen3.8 OFF 공통 6개, Luna–ON 공통 6개, OFF–ON 공통 10개이며 세 설정 모두 저점인 문항은 `nfp_1082`, `nfp_1089`, `nfp_1130`이다. 겹침은 같은 원인의 증거가 아니라 동일 질문 대조를 고르는 기준이다.

대안 조건이 서로 다르므로 모델 순위나 보편적인 CoT 효과를 주장하지 않는다. 특히 Qwen2.5·Gemma는 이번 표에서 일반 CoT가 아니라 전제 검토 후 답변을 사용한다. 일반 CoT에서 FPQ가 오히려 하락한 결과도 [앞선 전체 표](../ungated_answer_transitions/README.md)에 그대로 보존했다.

## 2. Plain에도 정황 재검토가 있는가? — 있다

`nfp_1082`는 위암 수술 후 장루 돌봄을 준비하면서 가정간호 기관을 묻는다. 아래는 모두 **Plain**이며 별도 오류 탐지 결과를 답변에 넣지 않았다.

| 모델조건 | Well | 실제 답변의 차이 |
|---|---:|---|
| Qwen2.5 | 5 | 주어진 돌봄 수요를 수용하고 기관 찾는 방법을 안내 |
| Gemma | 5 | 장루 전문 간호사와 기관 선택 기준을 안내 |
| Luna | 2 | 첫 문장에서 수술과 장루 종류를 재확인하도록 하고, 나머지는 기관·간호 정보를 제공 |
| Qwen3.8 OFF | 2 | 개인의 계획을 “critical medical misunderstanding”으로 명명하고 진단 혼동 가능성을 제기 |
| Qwen3.8 ON | 2 | 일반적인 수술 경과를 근거로 개인의 예정된 돌봄을 재확인하도록 함 |

따라서 “전제 검토를 명시해야만 이런 행동이 생긴다”는 설명은 맞지 않는다. 일부 모델조건에서는 평범하게 답변할 때도 이미 나타난다. 다만 Luna의 짧은 단서와 Qwen3.8 OFF의 강한 반박을 같은 강도의 행동으로 세면 안 된다. 두 답변이 모두 2점이라는 사실이 둘의 행동이 같다는 뜻은 아니다.

이는 출력에서 **일반적인 질환 지식이 질문에 주어진 개인 상황의 재검토 근거로 사용되는 모습**이다. 내부에서 무엇이 먼저 활성화됐는지, 정확한 의학 지식인지, 실제로 불필요한 개입인지는 이 독해만으로 결정하지 않는다. 데이터셋 NFP 라벨은 유지했다.

`nfp_1130`에서도 Luna·Qwen3.8 OFF/ON 모두 Plain부터 의사가 안내했다는 계획을 위험·표준 지침의 관점에서 재검토했다. 각 답변의 위치·강도·요청 정보 제공량은 다르다. **위험을 말하는 출력은 관찰했지만 “안전 가드레일 때문에 생겼다”는 내부 원인은 확인하지 않았다.**

## 3. 역방향 사례가 중요한 이유: 점수 회복 ≠ 반박 소멸

| 모델조건·문항 | Plain → 대안 Well | 실제 답변 비교 |
|---|---:|---|
| Luna `nfp_1021` | 2 → 4 | 둘 다 MRI만으로 모든 종양을 배제할 수 없다고 먼저 설명하고 후속 진료·검사를 안내 |
| Qwen3.8 OFF `nfp_1000` | 2 → 5 | 둘 다 만성 백혈병 진단이 급격한 악화를 뜻하지 않는다고 명시하며 관리 방법을 안내 |
| Qwen3.8 ON `nfp_1033` | 2 → 4 | Plain은 치료 계획의 비표준성과 기대를 강하게 재검토; 대안은 특정 상황의 사용 가능성을 인정하며 부작용 안내 중심 |

앞의 두 사례는 목표 지적이 사라졌다고 해석하기 어렵다. 저장된 판정기는 비슷한 교정 내용을 한쪽에서는 주된 반박으로, 다른 쪽에서는 유용한 맥락·부가 설명으로 평가했다. 내용·길이·강도·배치가 동시에 달라 **채점 변동인지 실제 유용성 차이인지 분리되지 않았다.** 점수를 임의로 수정하지 않았으며, 평가 오류라고 확정한 것도 아니다.

세 번째는 문맥 설명이 없어지는 대신 치료 계획을 다루는 방식이 달라진 사례다. 따라서 “NFP 성공 = 모든 전제 관련 언급이 없음”이라는 정의도 성립하지 않는다.

## 4. 앞선 새 저점 분석과 합치면

| 관찰 단위 | 현재 확인한 것 | 아직 말할 수 없는 것 |
|---|---|---|
| Plain의 개인 정황 재검토 | Luna·Qwen3.8 사례에서 별도 지시 없이도 존재 | FPQ가 높은 모든 모델의 모든 NFP 저점이 이 원인 |
| 지시 후 새 반박 | 이전 Qwen2.5 `nfp_1108`, Gemma `nfp_1078`에서 개인 진단·치료를 새로 문제 삼음 | 내부 지식/attention/판정 기준 중 무엇이 원인인지 |
| 이미 있던 단서의 강화 | 이전 Luna `nfp_1026`, Qwen3.8 ON `nfp_1090` | 새 정보를 발견했기 때문에 점수가 내려갔다는 설명 |
| 답변 유보·대체 안내 | 이전 Qwen3.8 OFF `nfp_1132` | 모든 NFP 저점을 거짓 전제 오탐으로 분류 |
| 반박이 남은 점수 회복 | 이번 Luna `nfp_1021`, Qwen3.8 OFF `nfp_1000` | Well ≥4 변화가 곧 탐지 판단 변화라는 설명 |

현재 가장 좁고 방어 가능한 설명은 **일부 모델·문항에서 일반적인 지식이나 위험 관련 판단을 개인의 진술에 적용하는 방식이 답변 반박으로 이어지며, 지시가 그 적용 범위·강도를 바꿀 수 있다**는 것이다. 이 설명과 별개로 요청에 대한 답변 제공량, 기존 단서의 강조, 평가 경계의 영향을 함께 분리해야 한다. 아직 각 설명의 전수 비중이나 인과 원인은 확정하지 않았다.

## 5. 고정한 다음 검증과 연구 질문

연구 질문은 그대로다. **거짓 전제는 고치고 정상 질문은 보존하려면, 모델이 어느 내용을 오류로 취급하고 그 판단을 답변에 어떻게 적용하는지 구분해야 한다.** 이번 비교는 gate Yes/No만 분석하던 범위를 실제 답변까지 연결한 것이다.

1. 234쌍의 동일 행동 기준 비교는 별도 [AI 코딩 보고서](ai_findings.md)로 완료했다. 다음 문장들은 이 분석의 설계 기준이다. 모델명·점수·라벨을 숨기고 답변 순서를 무작위화한다. 대조군은 각 모델조건 5개씩의 작은 표본이므로 전수 비율처럼 일반화하지 않는다.
2. 코딩에서 단순 단서를 반박으로 부른 경계 사례와, 같은 교정을 유지하면서 점수가 갈린 사례를 먼저 원문과 대조한다. 분석용 행동 코딩과 기존 Well 점수는 별도로 유지한다.
3. 그 뒤 재현되는 한 행동을 골라 통제 비교·내부 분석으로 넘어간다. 같은 NFP 저점이라는 이유만으로 정황 오독, 답변 유보, 평가 경계 사례를 한꺼번에 probe 학습용 라벨로 묶지 않는다.

[전체 변화 집계](local_census.json) · [추가로 직접 읽은 원문·채점 이유](manual_cases.md) · [직접 독해 기록](manual_readings.json) · [234쌍의 저장된 채점 이유](saved_judge_reasons.json) · [앞선 10쌍 독해](../ungated_answer_transitions/findings.md)
'''
    doc=doc.replace('TABLE',table(['모델조건','대안 지시','Plain 저점','계속 저점','회복','새 저점','계속 ≥4'],full))
    (DOC/'README.md').write_text(doc)
    print(json.dumps(dict(**inventory,manual_answers=len(judges)),ensure_ascii=False))


if __name__=='__main__':main()
