"""Freeze shared grouped Cancer 3-fold / official CREPE inputs; zero model calls."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paper_baselines import digest, norm, read_rows, sha, write_json


def split_groups(rows, n, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    if len({r['group_id'] for r in rows}) < n:
        raise ValueError('Too few groups')
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(
        rows, [r['label'] for r in rows], [r['group_id'] for r in rows]))


def prepare(cancer, crepe, registry_path, out, qa=None, group_overrides=None):
    reg = json.loads(registry_path.read_text())
    original = read_rows(cancer)
    rows = [dict(id=r['id'], dataset='cancer_myth', group_id=r['group_id'],
                 question=r['question'], label=int(r['set']=='fpq'),
                 presuppositions=[r['premise_text']], reference_answer=r['correction'],
                 exposure_history='previously_available_exploratory_benchmark') for r in original]
    sources = {'cancer':sha(cancer), 'registry':sha(registry_path)}
    for split in ('train','dev','test'):
        path = crepe / f'{split}.jsonl'
        sources['crepe_'+split] = sha(path)
        for r in read_rows(path):
            rows.append({**r, 'official_split':split, 'group_id':r.get('group_id',r['id']),
                         'exposure_history':'existing_official_split'})
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs')
    # Union known groups, identical questions, identical annotated claims.
    # This is a deterministic audit, NOT a claim of complete semantic deduplication.
    parent = {i:i for i in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b):
        a,b = sorted((find(a),find(b))); parent[b] = a
    seen = {}
    for r in rows:
        keys = [('group',r['dataset'],r['group_id']), ('question',norm(r['question']))]
        # Cancer premise_text sometimes contains attribution placeholders (e.g. From physicians).
        # Preserve curated Cancer groups; do not merge those on raw premise alone.
        if r['dataset'] == 'crepe' and r['label']:
            keys += [('crepe_premise',norm(p)) for p in r.get('presuppositions',[]) if norm(p)]
        for key in keys:
            if key in seen: union(r['id'],seen[key])
            else: seen[key] = r['id']
    if group_overrides:
        sources['group_overrides'] = sha(group_overrides)
        for group in json.loads(group_overrides.read_text()):
            for i in group:
                if i not in parent: raise ValueError('Unknown ID in override: '+i)
                union(group[0],i)
    shots = {norm(r['question']) for pack in reg['fixed_examples'].values() for r in pack['examples']}
    excluded_groups = {find(r['id']) for r in rows if norm(r['question']) in shots}
    labels = {}
    for r in rows: labels.setdefault(norm(r['question']),set()).add(r['label'])
    conflicts = {q for q,ys in labels.items() if len(ys)>1}
    excluded_groups |= {find(r['id']) for r in rows if norm(r['question']) in conflicts}
    dropped, kept = [], []
    for r in rows:
        r['group_id'] = find(r['id'])
        if r['group_id'] in excluded_groups:
            dropped.append({'id':r['id'],'reason':'fixed_example_or_conflict_group'}); continue
        kept.append(r)
    cancer_rows = sorted([r for r in kept if r['dataset']=='cancer_myth'],key=lambda r:r['id'])
    crepe_rows = sorted([r for r in kept if r['dataset']=='crepe'],key=lambda r:r['id'])
    # Quarantine cross-dataset linked groups rather than claim cross-dataset novelty.
    cross = {r['group_id'] for r in cancer_rows} & {r['group_id'] for r in crepe_rows}
    dropped += [{'id':r['id'],'reason':'cross_dataset_link'} for r in kept if r['group_id'] in cross]
    cancer_rows = [r for r in cancer_rows if r['group_id'] not in cross]
    crepe_rows = [r for r in crepe_rows if r['group_id'] not in cross]
    priority = {'train':0,'dev':1,'test':2}
    best = {}
    for r in crepe_rows: best[r['group_id']] = max(best.get(r['group_id'],-1),priority[r['official_split']])
    dropped += [{'id':r['id'],'reason':'official_test_dev_group_priority'} for r in crepe_rows
                if priority[r['official_split']] < best[r['group_id']]]
    crepe_rows = [r for r in crepe_rows if priority[r['official_split']] == best[r['group_id']]]
    splits = []
    for fold,(train_idx,test_idx) in enumerate(split_groups(cancer_rows,3,20261007)):
        outer = [cancer_rows[int(i)] for i in train_idx]
        ti,di = split_groups(outer,5,20261008+fold)[0]
        train = [outer[int(i)] for i in ti]; dev = [outer[int(i)] for i in di]
        splits.append(dict(dataset='cancer_myth',name=f'fold{fold}',
            train=[r['id'] for r in train],dev=[r['id'] for r in dev],
            test=[cancer_rows[int(i)]['id'] for i in test_idx]))
    splits.append(dict(dataset='crepe',name='official',**{s:[r['id'] for r in crepe_rows if r['official_split']==s]
                                                       for s in ('train','dev','test')}))
    allrows = cancer_rows+crepe_rows; lookup={r['id']:r for r in allrows}
    for s in splits:
        groups=[]
        for part in ('train','dev','test'):
            group={lookup[i]['group_id'] for i in s[part]}
            if any(group & prev for prev in groups): raise ValueError('Group leakage')
            groups.append(group)
            if {lookup[i]['label'] for i in s[part]} != {0,1}: raise ValueError('Missing label in split')
        # Common train-only smoke IDs for all models/methods.
        s['smoke']=[next(i for i in s['train'] if lookup[i]['label']==y) for y in (1,0)]
    qa_rows = read_rows(qa) if qa else []
    if qa:
        sources['qa'] = sha(qa)
        seenqa=set()
        adaptation_questions={norm(x['question']) for x in allrows} | shots
        for r in qa_rows:
            for k in ('id','dataset','question','metric','gold','source_revision','source_split'):
                if k not in r: raise ValueError('QA missing field: '+k)
            key=(r['dataset'],r['id'])
            if key in seenqa: raise ValueError('Duplicate QA ID')
            seenqa.add(key)
            if r['metric']=='choice' and str(r['gold']) not in r.get('choices',{}): raise ValueError('Bad QA choice')
            if r['metric']=='label' and r['gold'] not in r.get('allowed_labels',[]): raise ValueError('Bad QA label')
            if r['metric'] not in ('choice','label','numeric'): raise ValueError('Bad QA metric')
            if norm(r['question']) in adaptation_questions:
                raise ValueError('QA overlaps adaptation/example questions')
    value=dict(schema=1,protocol_id='paper_baselines_v3_nla_backbones_20261008',
        sources=sources,rows=allrows,splits=splits,qa=qa_rows,exclusions=dropped,
        audit={'group_basis':'existing Cancer groups, exact question links, CREPE exact premise links, optional reviewed overrides',
               'semantic_audit':'not_complete; report exploratory evaluation',
               'qa_status':'ready' if qa_rows else 'blocked_missing_frozen_QA_inputs'},
        counts={d:dict(Counter(str(r['label']) for r in allrows if r['dataset']==d)) for d in ('cancer_myth','crepe')})
    value['content_sha256']=digest(value)
    write_json(out,value,immutable=True)
    return value


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cancer',type=Path,default=ROOT/'configs/premise_sft/cancer_questions.jsonl')
    p.add_argument('--crepe',type=Path,default=ROOT/'results/fpqa_prompting/data/crepe')
    p.add_argument('--registry',type=Path,default=ROOT/'docs/reviews/paper_tables_2026-10-06/table2_prompt_registry_2026-10-07.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--qa-jsonl',type=Path)
    p.add_argument('--group-overrides',type=Path)
    a=p.parse_args(); result=prepare(a.cancer,a.crepe,a.registry,a.out,a.qa_jsonl,a.group_overrides)
    print(json.dumps({k:result[k] for k in ('content_sha256','counts','audit')},indent=2))
