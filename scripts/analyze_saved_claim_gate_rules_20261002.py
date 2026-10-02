"""Descriptive ablation on saved Luna fixed-candidate records; no model calls."""
from collections import Counter
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/frontier_cli/luna_fixed_claim_factorial_20261001_v1'
OUT=ROOT/'docs/reviews/saved_claim_gate_rules_2026-10-02'
CONDS=['control','content','criterion','both']
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 questions={r['id']:r for r in json.loads((SOURCE/'evaluation_questions.json').read_text())}
 records={c:{p.parent.name:json.loads(p.read_text()) for p in (SOURCE/c).glob('*/record.json') if json.loads(p.read_text())['status']=='complete'} for c in CONDS}
 common=set.intersection(*(set(r) for r in records.values()))
 assert len(common)==721
 rows=[];items=[]
 for condition in CONDS:
  counts={}
  for i in sorted(common):
   r=records[condition][i]['result'];cc=r['claim_assessments'];lab=questions[i]['set']
   decisions=dict(model_boolean=r['has_false_premise'],any_false=any(c['assessment']=='false' for c in cc),
    false_and_relied_on=any(c['assessment']=='false' and c['question_relation']=='relied_on' for c in cc),
    false_or_unknown_relied_on=any(c['assessment'] in ['false','undetermined'] and c['question_relation']=='relied_on' for c in cc))
   for rule,d in decisions.items():
    counts.setdefault((rule,lab),Counter())[bool(d)]+=1
    items.append(dict(condition=condition,id=i,label=lab,rule=rule,yes=int(d)))
  for rule in decisions:
   f=counts[rule,'fpq'];n=counts[rule,'nfp'];nf=sum(f.values());nn=sum(n.values())
   rows.append(dict(condition=condition,rule=rule,fpq_yes=f[True],fpq_n=nf,nfp_yes=n[True],nfp_n=nn,tpr=f[True]/nf,fpr=n[True]/nn))
 for name,data in [('summary',rows),('item_decisions',items)]:
  with (OUT/(name+'.csv')).open('w') as out:
   w=csv.DictWriter(out,fieldnames=data[0].keys(),lineterminator='\n');w.writeheader();w.writerows(data)
 lines=['# 기존 Luna 출력의 판정 규칙 재집계','', '새 호출 없이 고정 후보 4조건 공통 721문항(FPQ575/NFP146)을 재집계했다. 저장된 진위·관계 표시를 그대로 사용한다. 새 범위 수정 실험과 별개다.','',
 '이는 사후 기술 통계이며 원문 충실성이나 목표 오류 일치를 새로 검증하지 않았다. 더 적극적인 규칙의 TPR 상승만으로 개선으로 인정하지 않는다. 자기보고 확률 AUROC는 사용하지 않는다.','',
 '| 조건 | 판정 규칙 | FPQ Yes /575 | NFP Yes /146 |','|---|---|---:|---:|']
 names=dict(model_boolean='모델의 원래 최종 판정',any_false='false 후보가 하나라도 있음',false_and_relied_on='false이면서 질문의 전제로 표시된 후보가 있음',false_or_unknown_relied_on='false 또는 판단 불가이며 질문의 전제로 표시된 후보가 있음')
 for r in rows:lines.append(f"| {r['condition']} | {names[r['rule']]} | {r['fpq_yes']} ({r['tpr']:.1%}) | {r['nfp_yes']} ({r['fpr']:.1%}) |")
 lines+=['','`any_false`는 질문에 없는 거짓 주장까지 양성으로 셀 수 있다. `false_and_relied_on`은 모델이 실제 목표 전제를 잘못 제외하면 놓친다. 판단 불가를 포함하는 규칙은 검증할 수 없는 개인 상황까지 양성으로 셀 수 있다. 따라서 어떤 규칙도 정답 관계·진위 oracle이 아니다.','', '재현: `python3 scripts/analyze_saved_claim_gate_rules_20261002.py`']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':main()
