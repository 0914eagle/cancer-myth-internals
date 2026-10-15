"""Snapshot complete AO and available NLA; export matched training evidence."""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
from pathlib import Path
import re


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def snapshot(path, destination):
    data = path.read_bytes()
    # A live writer may be partway through its next JSON line. Preserve only
    # complete newline-terminated rows, and record that a suffix was omitted.
    end = data.rfind(b'\n') + 1
    complete = data[:end]
    destination.write_bytes(complete)
    rows = [json.loads(line) for line in complete.splitlines()]
    assert len({r['id'] for r in rows}) == len(rows)
    return rows, dict(path=str(path), sha256=hashlib.sha256(complete).hexdigest(), rows=len(rows),
                     ignored_unterminated_bytes=len(data)-end)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve(); out = root/'ao_analysis'
    out.mkdir(exist_ok=True); (out/'snapshots').mkdir(exist_ok=True)
    config = json.loads((root/'runtime_gemma27.json').read_text())
    sample = json.loads((root/'analysis/review_sample.json').read_text())['questions']
    selected = {(r['dataset'],r['id']) for r in sample}
    ao, ao_meta = snapshot(root/'gemma27/ao_results.jsonl',out/'snapshots/ao_results.jsonl')
    nla, nla_meta = snapshot(root/'gemma27/av_results.jsonl',out/'snapshots/nla_results.jsonl')
    assert len(ao) == 4710, 'Main AO cohort must be complete before this export'
    assert all(r['layer']==31 and r['hidden_state_index']==32 and r['partition']=='train' for r in ao)
    assert all(r['layer']==41 and r['hidden_state_index']==42 and r['partition']=='train' for r in nla)
    ao_by = {(r['source'],r['base_id'],r['condition'],r['requested_window_size']):r for r in ao}
    nla_by = {(r['source'],r['base_id'],r['condition'],r['position_family']):r for r in nla}
    assert len(ao_by)==len(ao) and len(nla_by)==len(nla)
    widths = [1,5,20]; positions = ['question_last_token','prompt_keyword_1','assistant_boundary_last_token']
    native_path = out/'native_answers/answers.jsonl'
    native = {}
    if native_path.exists():
        data=native_path.read_bytes(); end=data.rfind(b'\n')+1
        for line in data[:end].splitlines():
            r=json.loads(line); native[r['dataset'],r['id'],r['condition']]=r
    paired=[]; review=[]
    gold={(r['dataset'],r['id']):r for r in json.loads(Path(config['bundle']).read_text())['rows']}
    for q in config['cohort']:
        wide=dict(dataset=q['dataset'],question_id=q['id'],category='FPQ' if q['label'] else 'NFP',question=q['question'])
        for condition in ['plain','balanced']:
            for width in widths:
                row=ao_by[q['dataset'],q['id'],condition,width]
                for field in ['ao_output','window_size','short_question_window']:
                    wide[f'{condition}__ao_{width}__{field}']=row[field]
                if (q['dataset'],q['id']) in selected:
                    review.append(dict(reader='AO',dataset=q['dataset'],question_id=q['id'],category=wide['category'],
                        question=q['question'],condition=condition,requested_window_size=width,actual_window_size=row['window_size'],
                        explanation=row['ao_output'],layer=31,activation_id=row['id'],
                        reference_answer_for_reviewer_only=gold[q['dataset'],q['id']].get('reference_answer')))
            for position in positions:
                row=nla_by.get((q['dataset'],q['id'],condition,position))
                wide[f'{condition}__nla_{position}__output']=row['nla_output'] if row else ''
                wide[f'{condition}__nla_{position}__available']=row is not None
                if row and (q['dataset'],q['id']) in selected:
                    review.append(dict(reader='NLA',dataset=q['dataset'],question_id=q['id'],category=wide['category'],
                        question=q['question'],condition=condition,position=position,token=row['selected_token_text'],
                        explanation=row['nla_output'],layer=41,activation_id=row['id'],
                        reference_answer_for_reviewer_only=gold[q['dataset'],q['id']].get('reference_answer')))
            answer=native.get((q['dataset'],q['id'],condition))
            wide[f'{condition}__native_answer']=answer['answer'] if answer else ''
            wide[f'{condition}__native_answer_available']=answer is not None
        paired.append(wide)
    with (out/'paired_comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    (out/'review_explanations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review))
    groups=defaultdict(list)
    for row in ao:groups[row['source'],row['category'],row['condition'],row['requested_window_size']].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        summary.append(dict(zip(['dataset','category','condition','requested_window_size'],key),
            n=len(rows),actual_sizes=dict(Counter(r['window_size'] for r in rows)),
            mean_words=sum(len(r['ao_output'].split()) for r in rows)/len(rows),
            correction_language=sum(bool(re.search(r'\b(?:misconception|incorrect|mistaken|false premise|correct your|correct the|correcting|correction|rebuttal|clarif\w*)\b',r['ao_output'],re.I)) for r in rows)))
    pairs=[]
    for width in widths:
        same=sum(ao_by[q['dataset'],q['id'],'plain',width]['ao_output']==ao_by[q['dataset'],q['id'],'balanced',width]['ao_output'] for q in config['cohort'])
        pairs.append(dict(requested_window_size=width,n_questions=785,exact_plain_balanced_matches=same))
    write(out/'aggregate_summary.json',dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ao=ao_meta,nla=nla_meta,expected_ao=4710,expected_nla=4710,n_questions=785,
        review_ao=sum(r['reader']=='AO' for r in review),review_nla=sum(r['reader']=='NLA' for r in review),
        native_review_answers=len(native),groups=summary,exact_condition_matches=pairs,
        layer_confounds='Common source Gemma3-27B, public AO decoder31 versus public NLA decoder41; distinct receiver/injection/query/batching protocols.',
        scope='Fixed training cohort only; outputs cannot be read as final-test scores or actual correction labels',
        token_windows='Consecutive question tokens ending at question last token; every source vector sees its causal prefix. Requested20 is cropped to actual question length.'))
    print('AO complete:',len(ao),'NLA snapshot:',len(nla),'Review AO:',48,'Review NLA:',sum(r['reader']=='NLA' for r in review),'Native:',len(native),flush=True)


if __name__=='__main__':main()
