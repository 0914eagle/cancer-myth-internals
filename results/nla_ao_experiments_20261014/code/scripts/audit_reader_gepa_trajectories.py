"""Audit saved reader trajectories over approved, public research payloads.

Runs on CPU and never loads a source model or verbalizer. The reader identity and
before/after order are withheld from the external auditor; answers are shown so
this is a retrospective content audit, not blind behavioral prediction.
"""
import hashlib
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reader_gepa_io import save
from src.fpqa_cli_backend import CLIBackend

P = ROOT / 'results/nla_ao_parallel_20261012/gepa_reader_pilot'
OUT = P / 'cpu_analysis_20261013'
SYSTEM = '''Audit saved activation interpretations and model answers for a research experiment.
Each item contains two anonymous views of the same question. A view contains a
reader description of pre-answer activations and the actual answer from that
generation. Reader descriptions are unverified, may hallucinate and may echo
generic instructions. You are not measuring beliefs or causality. The two views
have randomized order; do not infer chronology or reader identity.

Use the question and supplied annotation/reference to identify the SPECIFIC
target error. If annotation and question conflict, flag it; do not force the
annotation to be correct. A topic summary, a plausible continuation, generic
"check/correct the premise" language, or a wrong-topic correction is not specific
target recognition. Only mark recognition if the supplied description itself
states the relevant mistaken claim or correct alternative. Do not supply missing
facts on the reader's behalf. Calling an illness serious is not by itself a
specific claim about its misleading prognosis or classification.

For NFP, normal qualifications or uncertainty are not over-rebuttal. Mark
spurious correction only if the text attacks a sound or reported premise without
adequate basis. Absence of correction is not evidence of an internal normality
decision. Evidence fields must quote a short exact substring; use an empty string
when there is no evidence. Generic correction intent can be true even when
specific recognition is false. Additional information means relevant accurate
diagnostic information absent from the actual answer, not just different wording.

Return ONLY JSON {"items":[...]}, one item per input id. Each item has:
id; annotation_alignment (aligned/ambiguous/conflict/uncertain);
annotation_reason; views (object keyed A,B). Each view has:
reader_topic_match (yes/partial/no), reader_specific_target_error_identified(bool),
reader_explicit_correction_intent(bool), reader_spurious_correction_on_nfp(bool),
reader_generic_correction_language(bool), reader_evidence(string),
answer_specific_target_error_identified(bool), answer_explicit_correction(bool),
answer_spurious_correction_on_nfp(bool), answer_evidence(string),
new_information_beyond_answer(bool), mismatch(one of
reader_correct_text_not,reader_normal_text_overrebuttal,concordant,topic_only,uncertain),
reason(string). Always set specific_target_error_identified false for NFP.
Do not interpret a topic-only NFP reader as a normality decision; use topic_only
or uncertain rather than reader_normal_text_overrebuttal without positive evidence.
'''


def validate(item, source, order):
    assert item['id'] == source['id']
    assert item['annotation_alignment'] in {'aligned','ambiguous','conflict','uncertain'}
    assert isinstance(item['annotation_reason'], str)
    assert set(item['views']) == {'A','B'}
    for tag, is_current in zip(['A','B'], order):
        v = item['views'][tag]
        assert v['reader_topic_match'] in {'yes','partial','no'}
        for field in ['reader_specific_target_error_identified','reader_explicit_correction_intent',
                      'reader_spurious_correction_on_nfp','reader_generic_correction_language',
                      'answer_specific_target_error_identified','answer_explicit_correction',
                      'answer_spurious_correction_on_nfp','new_information_beyond_answer']:
            assert isinstance(v[field], bool)
        assert v['mismatch'] in {'reader_correct_text_not','reader_normal_text_overrebuttal','concordant','topic_only','uncertain'}
        for field in ['reader_evidence','answer_evidence','reason']: assert isinstance(v[field],str)
        prefix = 'current' if is_current else 'seed'
        assert not v['reader_evidence'] or v['reader_evidence'] in source[prefix + '_reader_explanation']
        assert not v['answer_evidence'] or v['answer_evidence'] in source[prefix + '_answer']
        if source['label'] == 0:
            assert not v['reader_specific_target_error_identified']
            assert not v['answer_specific_target_error_identified']


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path)
    args = parser.parse_args()
    if args.input_root is not None: OUT = args.input_root.resolve()
    system = SYSTEM
    if args.input_root is not None:
        system += ('\nFor this AO-query comparison, both views have the exact same activation and actual answer; only the reader query differs. '
            'Additionally return in EACH view: source_stance (accept/question/reject/unclear), stance_evidence (short exact substring), '
            'stance_matches_answer (boolean or null). A topic summary or your own factual judgment cannot establish source stance: use unclear. '
            'If the reader explicitly attributes a stance, compare that attribution to the actual answer; this is retrospective agreement, not a prediction.')
    assert json.loads((P / 'export_consent.json').read_text())['state'] == 'approved'
    config = json.loads((P / 'gpt_runtime.json').read_text())['judge_model']
    runtime = json.loads((P / 'runtime.json').read_text())
    lookup = {r['id']: r for r in json.loads(Path(runtime['bundle']).read_text())['rows']}
    all_rows = json.loads((OUT / 'reader_trajectories.json').read_text())
    rows = sorted([r for r in all_rows if not r['starting_prompt']],
                  key=lambda r: (r['dataset'],r['reader'],r['candidate_key'],r['row_id']))
    if args.input_root is None: assert len(rows) == 59
    assert rows
    allowed = {r['id'] for r in json.loads((P / 'export_source_rows.json').read_text())} if isinstance(json.loads((P / 'export_source_rows.json').read_text()),list) else None
    if allowed is not None: assert {r['row_id'] for r in rows} <= allowed
    save(OUT / 'trajectory_audit_protocol.json', dict(model=config,
        questions='Previously approved train100 only', interpretations=len(rows),
        identity_blinding='Reader name, model name, candidate ID and chronological order hidden; order randomized per record.',
        answer_visibility='Actual answers and gold reference shown for retrospective content comparison; not blind behavior prediction.',
        selection_bias=('Fixed same100 train questions, same activation and same answer; only AO query changes.'
                        if args.input_root is not None else 'Candidate-dependent training mini-batches; not representative or a fixed longitudinal panel.'),
        prompt=system, source='Saved files only; no GPU calls'))
    pending = [r for r in rows if not (OUT / 'trajectory_audits' / (hashlib.sha256(r['id'].encode()).hexdigest()+'.json')).exists()]
    for offset in range(0,len(pending),5):
        batch = pending[offset:offset+5]
        payload, orders = [], {}
        for index,r in enumerate(batch):
            anon = 'case_'+str(index)
            order = [False,True]
            random.Random(r['source_native_id']).shuffle(order)
            orders[anon] = (r,order)
            ref = lookup[r['row_id']]
            views = {}
            for tag,is_current in zip(['A','B'],order):
                prefix = 'current' if is_current else 'seed'
                views[tag] = {'reader_description':r[prefix+'_reader_explanation'],
                              'actual_answer':r[prefix+'_answer']}
            payload.append(dict(id=anon,question=r['question'],question_type='FPQ' if r['label']==1 else 'NFP',
                presuppositions=ref['presuppositions'],reference_answer=ref['reference_answer'],views=views))
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps({'items':payload},ensure_ascii=False)}]
        cache_id=hashlib.sha256(json.dumps([config,messages],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        cache=OUT/'audit_calls'/(cache_id+'.json')
        started=time.time()
        try:
            if cache.exists(): raw=json.loads(cache.read_text())['answer']
            else:
                raw,metadata=CLIBackend(config)(messages)
                save(cache,dict(answer=raw,messages=messages,config=config,metadata=metadata,seconds=time.time()-started))
            text=raw.strip()
            if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
            result=json.loads(text)
            assert len(result['items']) == len(batch)
            assert {x['id'] for x in result['items']} == set(orders)
            for item in result['items']:
                r,order=orders[item['id']]
                anonymous={**r,'id':item['id']}
                validate(item,anonymous,order)
                if args.input_root is not None:
                    for tag,is_current in zip(['A','B'],order):
                        view=item['views'][tag]
                        assert view['source_stance'] in {'accept','question','reject','unclear'}
                        assert isinstance(view['stance_evidence'],str)
                        assert not view['stance_evidence'] or view['stance_evidence'] in r[('current' if is_current else 'seed')+'_reader_explanation']
                        assert view['stance_matches_answer'] is None or isinstance(view['stance_matches_answer'],bool)
                seed=item['views']['A' if not order[0] else 'B']
                current=item['views']['A' if order[0] else 'B']
                path=OUT/'trajectory_audits'/(hashlib.sha256(r['id'].encode()).hexdigest()+'.json')
                save(path,dict(id=r['id'],dataset=r['dataset'],reader=r['reader'],row_id=r['row_id'],
                    label=r['label'],candidate_key=r['candidate_key'],valid=True,model=config,
                    anonymous_view_order=['current' if x else 'seed' for x in order],
                    audit=dict(annotation_alignment=item['annotation_alignment'],annotation_reason=item['annotation_reason'],
                        seed=seed,current=current,new_information_beyond_current_answer=current['new_information_beyond_answer'],mismatch=current['mismatch']),
                    limits='Exploratory GPT content audit, not independent human verification or source-belief measurement.'))
            done=len(list((OUT/'trajectory_audits').glob('*.json')))
            save(OUT/'trajectory_audit_status.json',dict(state='running',complete=done,expected=len(rows)))
            print('Audited',done,'/',len(rows),'elapsed',round(time.time()-started,1),flush=True)
        except Exception as error:
            save(OUT/'trajectory_audit_status.json',dict(state='failed',error=repr(error),completed_files=len(list((OUT/'trajectory_audits').glob('*.json')))))
            raise
    save(OUT/'trajectory_audit_status.json',dict(state='complete',complete=len(rows),expected=len(rows)))


if __name__=='__main__':main()
