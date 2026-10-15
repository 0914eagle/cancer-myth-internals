"""Matched wrong-question-description controls for the frozen reader-GEPA pilot.

All source prompts, data partitions, metric limits and reader queries are retained.
Only training reflection receives a different question's description, generated
under the *same current candidate*. Donor activations need one generation prefill,
not a completed donor answer. Donor labels are used only for train-only matching.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reader_gepa_io import PilotIO, save, load_prompt_tokenizer
from src.paper_baselines import digest
from src.fpqa_prompting import parse_rating, well_response_judge
from src.fpqa_cli_backend import CLIBackend

P = ROOT / 'results/nla_ao_parallel_20261012/gepa_reader_pilot'
OUT = P.parent / 'reader_feedback_controls_20261014'
STANCE_QUERY = (
    "Describe whether these activations provide evidence that the model accepts, questions, "
    "or rejects an assumption in the user's question. Identify the particular assumption "
    "when it is represented. Distinguish a topic summary from evidence of a stance. "
    "Do not decide whether the assumption is objectively true using your own knowledge. "
    "Do not assume that an error exists. If a stance is not supported by the activations, "
    "say that it is unclear."
)


def prepare():
    original = json.loads((P / 'plan.json').read_text())
    config = json.loads((P / 'runtime.json').read_text())
    bundle = json.loads(Path(config['bundle']).read_text())
    lookup = {r['id']: r for r in bundle['rows']}
    donors = {}
    for dataset, split in original['splits'].items():
        for label in [1, 0]:
            ids = sorted(i for i in split['train_ids'] if lookup[i]['label'] == label)
            assert len(ids) == 25 and len({lookup[i]['group_id'] for i in ids}) == 25
            random.Random(f'42:{dataset}:{label}').shuffle(ids)
            for i, recipient in enumerate(ids): donors[recipient] = ids[(i + 1) % len(ids)]
    for recipient, donor in donors.items():
        assert recipient != donor and lookup[recipient]['group_id'] != lookup[donor]['group_id']
        assert (lookup[recipient]['dataset'], lookup[recipient]['label']) == (lookup[donor]['dataset'], lookup[donor]['label'])
    native = {r['row_id']:r for r in (json.loads(f.read_text()) for f in (P/'native/results').glob('*.json'))
              if r.get('candidate_prompt') == original['starting_prompt'] and r.get('row_id') in donors}
    assert len(native) == 100
    assert all(len(r['nla_source_token_ids']) == 1 and 0 < len(r['ao_source_token_ids']) <= 20 for r in native.values())
    protocol = dict(protocol='reader_feedback_wrong_question_control_v1',
        source_pilot=str(P), original_plan_sha256=hashlib.sha256((P/'plan.json').read_bytes()).hexdigest(),
        reused_arms=['gepa','text','nla','ao'], new_arms=['nla_shuffled','ao_shuffled'],
        primary_contrasts={'nla':['gepa','nla','nla_shuffled'], 'ao':['gepa','ao','ao_shuffled']},
        text_definition='gepa already receives question, actual answer, label, Well score and judge rationale. text is a secondary GPT-generated-diagnostic arm.',
        splits=original['splits'], starting_prompt=original['starting_prompt'],
        metric_question_cap=100, pair_metric_cap=50, max_candidate_proposals=4, seed=42,
        prompt_token_limit=512, diagnostic_token_limit=512, source_batch_size=1,
        donor_map=donors, donor_matching='Fixed train-only derangement within dataset and FPQ/NFP; no same-group donors.',
        donor_candidate='Recipient and donor use byte-identical current candidate prompts.',
        donor_capture='Use cached same-candidate native activation if present; otherwise capture the first generation prefill with max_new_tokens=1. Donor answer is not evaluated or supplied.',
        donor_cost='Additional donor prefills/readers are recorded separately and are not objective metric calls. Total GPU cost need not equal the real-description arm.',
        reader_settings={'nla':{'decoder_block_zero_based':41,'hf_hidden_states_index':42,'positions':'question last token','vectors':1,'query':'Checkpoint canonical query, fixed'},
                         'ao':{'decoder_block_zero_based':31,'hf_hidden_states_index':32,'positions':'last up to20 question tokens','query':config['ao_prompt']}},
        extraction_timing='First generation prefill; selected position is the last question token, not the final assistant-header token and not an annotated premise-span endpoint.',
        reader_confound='Layers and input widths differ; conclusions compare the configured pipelines, not reader identity alone. Donor windows retain each donor question length.',
        ao_query_probe={'seed_questions':100,'queries':[config['ao_prompt'],STANCE_QUERY],
            'source_activations':'Exactly the same saved seed activation per question for both queries; no new source answers.',
            'selection_use':'Exploratory content audit only; never choose GEPA prompts using this probe or heldout answers.'},
        heldout='Reuse the identical pre-fixed8 heldout IDs per dataset, disjoint from candidate-dev. Select new prompts with GPT dev only, freeze, then use pinned Haiku. No selection using primary Haiku outcomes.',
        final_arms=6, final_answer_records=96,
        prior_paths={'/data2/heejae/cancer_myth_internals/results/gepa_nla_pilot_20261013_v2/protocol.json':'Not present on this host; not read or merged',
                     '/data2/heejae/tools/pilot_gepa_ao.py':'Not present on this host; user-reported AO L46 is a separate unverified setup'},
        completion_order='Existing AO and original64 heldout evaluation first; new controls then query probe. All new design choices registered before original heldout scoring.')
    path = OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        for k,v in protocol.items(): assert existing[k]==v, 'Protocol changed: '+k
        return existing,config,lookup
    assert not list((P/'heldout_haiku').glob('*.json')), 'Register before inspecting primary heldout scores'
    protocol.update(preregistered_utc=time.time(),primary_heldout_scores_at_registration=0)
    save(path,protocol)
    return protocol,config,lookup


class ControlIO(PilotIO):
    def capture_donor(self, row, prompt):
        messages=[{'role':'system','content':self.registry['transport_common']+'\n\n'+prompt},
                  {'role':'user','content':row['question']}]
        full_id=digest([self.baseline['models']['gemma27'],messages,'native_reader_gepa_v1'])
        full=self.out/'native/results'/(full_id+'.json')
        if full.exists(): return json.loads(full.read_text())
        ident=digest([self.baseline['models']['gemma27'],messages,'native_reader_gepa_prefill_only_v1'])
        request=self.out/'native/requests'/(ident+'.json')
        if not request.exists():save(request,dict(id=ident,row=row,messages=messages,candidate_prompt=prompt,capture_only=True))
        response=self.out/'native/results'/(ident+'.json');start=time.monotonic()
        while not response.exists():
            error=self.out/'native/results'/(ident+'.error.json')
            if error.exists():raise RuntimeError(error.read_text())
            if time.monotonic()-start>900:raise TimeoutError('Donor prefill request exceeded900seconds')
            time.sleep(1)
        result=json.loads(response.read_text())
        assert result['messages']==messages and result['capture_only']
        return result


def run():
    protocol,config,lookup=prepare()
    assert json.loads((P/'pipeline_status.json').read_text())['state']=='complete', 'Finish original pilot first'
    assert json.loads((P/'export_consent.json').read_text())['state']=='approved'
    io=ControlIO(P/'runtime.json');status=OUT/'status.json'
    from transformers import AutoTokenizer
    from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
    tok=load_prompt_tokenizer(io.baseline)
    def state(stage,**kw):save(status,dict(state='running',stage=stage,updated_utc=time.time(),**kw))
    def pairs(ids,partition):
        groups=[[lookup[i] for i in ids if lookup[i]['label']==y] for y in [1,0]]
        for g in groups:random.Random(42).shuffle(g)
        return [dict(pair=[a,b],partition=partition) for a,b in zip(*groups)]
    selections={}
    try:
        for reader in ['nla','ao']:
            arm=reader+'_shuffled'
            for dataset,split in protocol['splits'].items():
                d=OUT/'optimization'/dataset/arm;d.mkdir(parents=True,exist_ok=True)
                if (d/'selected.json').exists():
                    selections[dataset,arm]=json.loads((d/'selected.json').read_text());continue
                state('optimization',dataset=dataset,arm=arm)
                def evaluate(candidate,example):
                    if set(candidate)!={'answer'}:raise ValueError('Candidate components changed')
                    if len(tok.encode(candidate['answer'],add_special_tokens=False))>512:
                        return 0.,{'error':'Candidate exceeds fixed512token limit','valid':False}
                    feedback=[]
                    for row in example['pair']:
                        native=io.native(row,candidate['answer']);judge=io.judge(row,native)
                        info=dict(id=row['id'],label=row['label'],question=row['question'],answer=native['answer'],rating=judge['score'],reason=judge['reason'])
                        if example['partition']=='train':
                            donor=lookup[protocol['donor_map'][row['id']]]
                            donor_native=io.capture_donor(donor,candidate['answer'])
                            assert donor_native['candidate_prompt']==native['candidate_prompt']
                            extra=io.reader(reader,donor_native)
                            tokens=tok.encode(extra['explanation'],add_special_tokens=False)
                            info.update(additional_diagnostic=tok.decode(tokens[:512]),diagnostic_source=reader,
                                        diagnostic_clipped=len(tokens)>512)
                            # Reflection sees identical schema and source name, never the donor ID or condition label.
                            save(d/'extras'/(native['id']+'.json'),dict(extra,recipient_id=row['id'],donor_id=donor['id'],
                                recipient_native_id=native['id'],donor_native_id=donor_native['id'],
                                donor_capture_only=bool(donor_native.get('capture_only')),
                                supplied_diagnostic=info['additional_diagnostic'],clipped=len(tokens)>512))
                        feedback.append(info)
                    score=sum(r['rating']/5 for r in feedback)/2
                    with (d/'evaluations.jsonl').open('a') as f:
                        f.write(json.dumps(dict(candidate=candidate,partition=example['partition'],score=score,examples=feedback,timestamp=time.time()),ensure_ascii=False)+'\n')
                    return score,{'examples':feedback}
                def reflect(messages):
                    if isinstance(messages,str):messages=[{'role':'user','content':messages}]
                    if not any(m['role']=='system' for m in messages):
                        messages=[{'role':'system','content':'Improve the supplied prompt using training feedback. Return the requested candidate.'}]+messages
                    if [m['role'] for m in messages]!=['system','user']:
                        messages=[{'role':'system','content':'Improve the prompt following the supplied optimization conversation.'},
                                  {'role':'user','content':json.dumps(messages,ensure_ascii=False)}]
                    return io.external('reflection',messages)
                result=optimize_anything(seed_candidate={'answer':protocol['starting_prompt']},evaluator=evaluate,
                    dataset=pairs(split['train_ids'],'train'),valset=pairs(split['candidate_dev_ids'],'dev'),
                    objective='Maximize balanced FPQ correction and normal-question preservation. Only questions are available at inference.',
                    background='Optimize only the answer prompt. No tools or retrieval. The prompt must fit512tokenizer tokens. '
                        'Additional diagnostic text, when present, may be unreliable and is not gold evidence of source beliefs. '
                        'Use reference-grounded grading feedback to assess it. Do not copy example-specific reference facts into a general instruction.',
                    config=GEPAConfig(engine=EngineConfig(run_dir=str(d/'engine'),seed=42,max_metric_calls=50,
                        max_candidate_proposals=4,parallel=False,max_workers=1,cache_evaluation=False),
                        reflection=ReflectionConfig(reflection_lm=reflect,reflection_minibatch_size=2)))
                assert len(tok.encode(result.best_candidate['answer'],add_special_tokens=False))<=512
                record=dict(candidate=result.best_candidate,best_idx=result.best_idx,dataset=dataset,arm=arm,
                    selection='Frozen GPT candidate-dev best; no heldout access',seed=42)
                save(d/'selected.json',record);selections[dataset,arm]=record
                print('Selected',dataset,arm,flush=True)
            io.stop_reader()
        save(OUT/'frozen_candidates.json',{dataset:{arm:selections[dataset,arm] for arm in protocol['new_arms']} for dataset in protocol['splits']})
        haiku=json.loads((P/'haiku_runtime.json').read_text())['judge_model']
        scored=[]
        # Original heldout scores are read only after every new candidate is frozen.
        scored.extend(json.loads((P/'heldout_validation_scored.json').read_text()))
        for dataset,split in protocol['splits'].items():
            for arm in protocol['new_arms']:
                for ident in split['heldout_validation_ids']:
                    state('heldout_generation_scoring',dataset=dataset,arm=arm,row_id=ident)
                    native=io.native(lookup[ident],selections[dataset,arm]['candidate']['answer'])
                    cache_id=hashlib.sha256(json.dumps([haiku,native['id']],sort_keys=True).encode()).hexdigest()
                    path=P/'heldout_haiku'/(cache_id+'.json')
                    if path.exists():judge=json.loads(path.read_text())
                    elif native['completed']:
                        raw,meta=CLIBackend(haiku)(well_response_judge(ROOT/io.baseline['well_root'],lookup[ident],native['answer']))
                        rating=parse_rating(raw)
                        if rating is None:raise ValueError('Unparseable pinned Haiku score')
                        judge=dict(score=rating,reason=raw,metadata=meta);save(path,judge)
                    else:judge=dict(score=0,reason='Incomplete answer',incomplete=True);save(path,judge)
                    scored.append(dict(dataset=dataset,arm=arm,row_id=ident,label=lookup[ident]['label'],native=native,judgment=judge))
        assert len(scored)==96
        save(OUT/'heldout_scored.json',scored)
        summary=[]
        for dataset in protocol['splits']:
            for arm in protocol['reused_arms']+protocol['new_arms']:
                group=[r for r in scored if (r['dataset'],r['arm'])==(dataset,arm)]
                fpq=[r['judgment']['score'] for r in group if r['label']==1]
                nfp=[r['judgment']['score'] for r in group if r['label']==0]
                assert len(fpq)==len(nfp)==4
                summary.append(dict(dataset=dataset,arm=arm,fpq_mean=sum(fpq)/4,nfp_mean=sum(nfp)/4,
                    balanced_normalized=(sum(fpq)+sum(nfp))/40,fpq_n=4,nfp_n=4))
        save(OUT/'comparison_summary.json',dict(results=summary,limits='Tiny fixed heldout pilot subset, not official test. New controls registered before primary heldout scoring; no Haiku reselection.'))
        with (OUT/'comparison_summary.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
        save(status,dict(state='complete',stage='heldout_comparison',records=96,updated_utc=time.time()))
    except Exception as e:
        save(status,dict(state='failed',error=repr(e),updated_utc=time.time(),cached_results_preserved=True));raise
    finally:io.stop_reader()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:
        plan,_,_=prepare();print('Registered',len(plan['donor_map']),'train-only donors;4 new dataset/arm optimizations.')
    else:run()
