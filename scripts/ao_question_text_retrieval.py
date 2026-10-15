"""Question-only TF-IDF diagnostic: lexical retrieval, never semantic scoring."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();out=root/'ao_analysis'
    config=json.loads((root/'runtime_gemma27.json').read_text())
    bundle=json.loads(Path(config['bundle']).read_text())
    lookup={r['id']:r for r in bundle['rows']}
    cohort={dataset:sorted([q for q in config['cohort'] if q['dataset']==dataset],key=lambda q:q['id']) for dataset in ['cancer_myth','crepe']}
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    matrices={};models={};indices={}
    for dataset,name in [('cancer_myth','fold0'),('crepe','official')]:
        split=next(s for s in bundle['splits'] if (s['dataset'],s['name'])==(dataset,name))
        vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=50000)
        vectorizer.fit([lookup[i]['question'] for i in split['train']])
        models[dataset]=vectorizer
        matrices[dataset]=vectorizer.transform([q['question'] for q in cohort[dataset]])
        indices[dataset]={q['id']:i for i,q in enumerate(cohort[dataset])}
    details=[]
    for reader,file in [('AO','ao_results.jsonl'),('NLA','nla_results.jsonl')]:
        rows=[json.loads(s) for s in (out/'snapshots'/file).open()]
        for dataset in cohort:
            selected=[r for r in rows if r['source']==dataset]
            texts=[r['ao_output'] if reader=='AO' else r['nla_output'] for r in selected]
            scores=(models[dataset].transform(texts) @ matrices[dataset].T).toarray()
            for r,values in zip(selected,scores):
                idx=indices[dataset][r['base_id']];own=float(values[idx])
                # Pessimistic rank among equal-score candidates; a zero score
                # ranks last instead of pretending the tie is a retrieval hit.
                rank=int(np.count_nonzero(values>=own-1e-12))
                details.append(dict(reader=reader,id=r['id'],dataset=dataset,question_id=r['base_id'],
                    condition=r['condition'],variant=str(r['requested_window_size']) if reader=='AO' else r['position_family'],
                    candidates=len(cohort[dataset]),own_question_cosine=own,conservative_rank=rank,
                    top1=own>0 and rank<=1,top5=own>0 and rank<=5))
    groups=defaultdict(list)
    for r in details:groups[r['reader'],r['dataset'],r['condition'],r['variant']].append(r)
    summary=[]
    for key,rows in sorted(groups.items()):
        summary.append(dict(zip(['reader','dataset','condition','variant'],key),n=len(rows),candidates=rows[0]['candidates'],
            mean_own_question_cosine=sum(r['own_question_cosine'] for r in rows)/len(rows),
            top1=sum(r['top1'] for r in rows),top5=sum(r['top5'] for r in rows)))
    (out/'lexical_retrieval_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details))
    (out/'lexical_retrieval_summary.json').write_text(json.dumps(dict(groups=summary,
        method='TF-IDF fitted on each official/fold train question pool only; explanation-to-question cosine; retrieve among385Cancer or400CREPE fixed training questions; pessimistic tie ranks.',
        cautions=['Lexical overlap diagnostic, not semantic accuracy, FPQ identification, answer performance, or held-out retrieval performance',
                  'Synonyms, boilerplate, explanation length, and near-duplicate Cancer variants affect ranks',
                  'NLA is a partial snapshot; cross-reader comparisons must restrict to shared completed questions and account for layers/query/batching']),indent=2)+'\n')
    print(json.dumps([r for r in summary if r['reader']=='AO'],ensure_ascii=False))


if __name__=='__main__':main()
