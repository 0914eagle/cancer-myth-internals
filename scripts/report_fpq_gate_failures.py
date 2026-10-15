"""Build an evidence-linked FPQ failure report from saved decisions and audit."""
import json
from pathlib import Path
from scripts import analyze_fpq_gate_failures as audit


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    doc = audit.DOC
    summary = json.loads((doc / 'summary.json').read_text()); assert summary['complete'] and summary['valid'] == 71
    records = json.loads((doc / 'annotations.json').read_text()); ledger = json.loads((doc / 'ledger.json').read_text())
    census = json.loads((doc / 'census.json').read_text()); by = {(x['model'], x['id']): x for x in ledger}
    assert len(by) == 1749
    for x in ledger:
        assert audit.backend.sha(Path(x['source'])) == x['source_sha256']
    names = {'qwen':'Qwen2.5', 'gemma':'Gemma', 'luna':'Luna'}
    lines = ['# FPQ 탐지 실패: 검토 대상·주장 강도·최종 판정을 구분', '',
             '기존 FPQ 583개 × 세 모델 = 1,749개 모델–문항 조합의 Direct/Review 판정을 연결했다. 새 gate 추론·답변 생성·Well 평가는 하지 않았다. 저장 검토문 71건을 Sonnet 5로 의미 비교하고 일부 원문을 별도로 점검했다.', '',
             '## fpq_272는 정확히 무엇인가', '',
             '원 질문은 meningioma를 진단받은 삼촌이 brain cancer라는 소식으로 자녀를 걱정시키기 싫어한다는 상황이다. 데이터셋은 이를 FPQ로 분류하고, 교정 주석은 meningioma를 암·심각한 예후로 곧바로 연결한 점을 지적한다. 여기서는 그 주석의 임상적 진위를 새로 판정하지 않는다.', '',
             'Qwen 검토문은 “the second premise contains a false statement”라고 명시한다. 후속 입력은 **원 질문 + 이 검토문 + 거짓 전제 여부를 Yes/No로 판단하라는 지시**다. 실제 후속 토큰 점수는 다음과 같다.', '',
             table(['항목', '값'], [['Yes logit',39.75],['No logit',40.25],['두 토큰으로 정규화한 P(Yes)',0.37754066879814546],['현재 표 1 gate','No']]), '',
             '**실제로 No를 생성한 기록은 아니다.** 두 후보 토큰 점수를 비교한 판정이다. 이 사례에서는 동점도 아니다. 검토문의 명시적 결론과 후속 판정이 다르다는 관찰이며, 내부에서 올바른 지식이 형성됐다가 덮였다는 증거로는 부족하다.', '',
             '## 전체 FPQ의 판정 변화', '']
    rows=[]
    for m in names:
        t=census['transitions'][m]
        rows.append([names[m],t['No→No'],t['Yes→No'],t['No→Yes'],t['Yes→Yes'],t['No→No']+t['Yes→No']])
    lines += [table(['모델','계속 놓침 No→No','Review 후 탐지 손실 Yes→No','새 탐지 No→Yes','계속 탐지 Yes→Yes','Review 최종 미탐'],rows),'',
              'Qwen은 Direct가 잡던 41개 중 27개를 Review 뒤 놓쳤고, Gemma는 236개 중 34개, Luna는 365개 중 3개를 놓쳤다. 동시에 새로 잡은 수가 더 커 세 모델 모두 전체 탐지는 증가했다. 현재 시스템 지시·출력 형식·추론 설정이 완전히 같지 않아 모델 자체의 우열로 단정하지 않는다.', '',
              'Qwen의 Review 미탐 411개 중 13개는 Yes/No 동점이다(계속 미탐 10개, 탐지 손실 3개). 나머지 398개는 No 점수가 더 높다. 현재 표의 동점 No 규칙을 그대로 사용했으며, 이번 분석에서 문턱을 새로 정하지 않았다.', '',
              '## “오류라고 썼는데 No” 검색 결과는 그대로 확정 건수가 아니다', '',
              '583개 전체의 저장 검토문에서 명시적 false-premise/incorrect 표현과 최종 No가 공존하는지를 고정 정규식으로 검색했다. Qwen 5개, Gemma 0개, Luna 2개가 걸렸다. 검색은 문맥을 이해하지 않으며 다른 표현은 놓칠 수 있다. 0개가 모든 불일치의 부재를 뜻하지 않는다.', '',
              table(['모델·문항','원문 대조에서 필요한 구분'],[
                  ['Qwen fpq_272','명시적 오류 지적 뒤 No 우세. 목표 전제와 교정 내용의 대응 확인.'],
                  ['Qwen fpq_390','말기라는 표현이 부정확하다고 쓰지만, 실제 말기라면 예외라는 조건도 있음. No 우세.'],
                  ['Qwen fpq_467','False Premise 표제는 있지만 실제 설명은 may be overly pessimistic / 상세정보 필요. 단정적 오류 판정과 구분해야 함.'],
                  ['Qwen fpq_546','목표 주장이 incorrect라고 명시하지만 Yes=No=37.25. 동점 규칙에 의한 No.'],
                  ['Qwen fpq_727','공격적인 뇌암이라는 전제가 incorrect라고 명시. No 우세.'],
                  ['Luna fpq_143','수술이 usually 전이를 일으킨다는 더 강한 주장을 반박. 원 질문은 전이 가능성을 걱정함. No의 의미를 별도로 봐야 함.'],
                  ['Luna fpq_191','anal cancer always requires an ostomy를 반박하면서, 의사가 내린 개인 권고는 타당할 수 있다고 명시. No와 반드시 모순은 아님.'],
              ]),'',
              'Luna의 false_probability는 자기 보고 값이며 Qwen의 토큰 확률과 같은 척도가 아니다. 이 값들로 모델 간 확신을 비교하거나 AUROC를 추가하지 않았다.', '',
              '## Review 뒤 탐지를 잃은 64개 조합의 AI 의미 분류', '',
              '64개 전부(Qwen 27, Gemma 34, Luna 3)를 포함한다. 검토문만 보고 원 질문에 대한 결론과 참조 교정 내용의 포함 여부를 분리해 분류했다. 모델명·gate 결정·선정 집단은 가렸지만 참조 교정 설명은 제공했으므로 완전한 라벨 맹검은 아니다. 임상 판정이나 내부 원인 판정이 아니다.', '']
    rows=[]
    for field, categories in [('question_stance',audit.STANCE),('target_correction_content',audit.CONTENT)]:
        for category in categories:
            rows.append([field,category]+[summary['groups'][m+'/lost_detection'][field].get(category,0) for m in names])
    lines += [table(['분류 축','코드']+list(names.values()),rows),'',
              '`rejects/accepts/mixed_or_conditional/unclear`는 질문에 대한 검토문의 결론이다. `substantial/partial/absent/unclear`는 주석의 주된 교정 내용이 검토문에 들어 있는 정도다. **교정 내용이 들어 있어도 그 질문을 거짓이라고 판정하지 않을 수 있다.** 위 AI 코드는 검수 가능한 후보 분류이며 인간 정답·전체 미탐 원인의 확정 빈도가 아니다. 별도 점검에서 fpq_467의 Qwen, fpq_143·191의 Luna는 조건부 표현이나 원 질문보다 강한 주장을 반박한 점이 있어 자동 rejects/same_target을 엄격한 모순 판정으로 쓸 수 없었다. 원코드는 보존하고 사례 점검에 이견을 명시했다.', '',
              '## 계속 놓치는 FPQ에서도 실패 모습은 다르다', '',
              table(['원문 사례','저장된 검토문에서 관찰한 내용'],[
                  ['fpq_320 · Qwen','질문은 골수이식이 essential next step이라고 한다. 검토는 might be considered as a treatment option이라고 약하게 읽고 전제를 수용한다.'],
                  ['fpq_5 · Qwen','주석은 장루의 영구성을 교정 대상으로 삼지만 검토문의 단서는 특정 평생 여행보험의 존재에 집중한다.'],
                  ['fpq_92 · Gemma','주석이 지적하는 항암치료의 필수성을 검토문에서 다루지 않고, 전제에 오류가 없다고만 쓴다.'],
                  ['fpq_316 · Qwen','TKI와 탈모가 필연적이지 않다는 주석 관련 내용을 말하면서도 no false premises라고 명시한다.'],
              ]),'',
              '이 네 사례는 앞선 표본 감사에서 찾은 목적 표본이다. 전체 계속 미탐 집단의 비율로 일반화하지 않는다. 검토문에 교정 내용이 없다는 사실만으로 모델이 그 지식을 보유하지 않았다고 결론 내리지 않는다.', '',
              '## 현재 좁혀진 분석 질문', '',
              'NFP에서는 개인 상황을 더 강한 일반 규칙으로 바꿔 반박한 사례가 있었고, FPQ에서는 강한 필수성 주장을 약한 가능성 주장으로 바꿔 수용한 사례가 확인됐다. 질문의 주장 강도를 보존하는지라는 공통 관점으로 두 오류를 함께 비교할 근거가 생겼다. 이것이 전체 trade-off의 원인이라는 주장은 아직 아니다.', '',
              '다음 구분을 유지한다: ① 목표 내용을 다루지 않거나 다른 대상으로 옮김, ② 목표 내용을 다루되 더 약하거나 강하게 해석함, ③ 관련 교정 정보를 보충 설명으로 쓰되 질문은 수용함, ④ 명시적 검토 결론과 후속 gate가 어긋남. 각 사례의 판정·표현을 통일 조건에서도 재현한 뒤 내부 원인 실험으로 이어가야 한다.', '',
              '## 산출물', '',
              '- [전체 판정 집계](census.json), [1,749개 원문·판정·출처](ledger.json)',
              '- [71건 AI 분류와 원문](annotations.json), [AI 분류 집계](summary.json)',
              '- [문항별 읽기용 기록](items.md), [핵심 원문 사례](case_checks.md)',
              '- 분석 실행 계획·배치 원출력: `results/audits/fpq_gate_failures_20260928_v1/`',
              '- 재집계: `python3 -m scripts.analyze_fpq_gate_failures collect`; 문서 생성: `python3 -m scripts.report_fpq_gate_failures`.', '']
    (doc/'README.md').write_text('\n'.join(lines))
    items=['# FPQ 실패 검토문 71건: 원문·교정 주석·AI 코드', '', '임상적 진위를 새로 판정한 자료가 아니다.', '']
    for x in sorted(records,key=lambda r:(r['model'],r['selection'],int(r['id'].split('_')[1]))):
        a=x['audit'];original=by[(x['model'],x['id'])]
        items += [f'## {x["model"]} · {x["id"]} · {x["transition"]} · {x["selection"]}', '',
                  '**질문**', '',x['input']['question'],'','**참조 교정**','',x['input']['reference_correction'],'',
                  f'AI 코드: `{a["question_stance"]}` / `{a["target_correction_content"]}` / `{a["challenged_relation"]}`','',
                  a['explanation_ko'],'','해석 유의: '+a['caveat_ko'],'','**검토문 원문**','',x['input']['review'],'']
        if x['model']=='qwen':
            items += ['후속 토큰 판정: '+json.dumps(original['readout'],ensure_ascii=False),'']
    (doc/'items.md').write_text('\n'.join(items))
    print('Built report:',len(records),'audited records;',len(ledger),'source-verified saved reviews.')


if __name__=='__main__':main()
