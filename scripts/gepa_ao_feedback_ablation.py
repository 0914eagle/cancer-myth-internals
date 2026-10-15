"""CREPE: matched Text, Text+AO, shuffled AO, and strict AO-only feedback."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reader_feedback_controls import P,ControlIO
from scripts.reader_gepa_io import save,load_prompt_tokenizer
from scripts.run_ao_behavior_query_probe import BEHAVIOR_QUERY,DIRECTORY as PROBE
from src.paper_baselines import digest
from src.fpqa_cli_backend import CLIBackend
from src.fpqa_prompting import parse_rating,well_response_judge

DATASET=os.environ.get('FPQA_GEPA_DATASET','crepe')
QUERY_KIND=os.environ.get('FPQA_GEPA_QUERY','behavior')
assert DATASET in {'crepe','cancer_myth'} and QUERY_KIND in {'behavior','semantic'}
NATIVE_GPUS=os.environ.get('FPQA_GEPA_NATIVE_GPUS','4,5')
READER_GPUS=os.environ.get('FPQA_GEPA_READER_GPUS','6,7')
OUT=P.parent/f'gepa_ao_{QUERY_KIND}_feedback_{DATASET}_luna_20261014'
ARMS=(['ao_only','text','text_ao','text_ao_shuffled'] if QUERY_KIND=='behavior'
      else ['text','text_ao','text_ao_shuffled'])
AO_QUERY=(BEHAVIOR_QUERY if QUERY_KIND=='behavior'
          else 'Describe the semantic content represented by these activations.')


def feedback_record(arm,row,native,judge,description=None):
    if arm=='ao_only':
        assert isinstance(description,str)
        return {'ao_description':description}
    info=dict(question=row['question'],answer=native['answer'],label=row['label'],
        presuppositions=row['presuppositions'],reference_answer=row['reference_answer'],
        rating=judge['score'],grading_reason=judge['reason'])
    if arm!='text':
        assert isinstance(description,str)
        info['ao_description']=description
    return info


def prepare():
    original=json.loads((P/'plan.json').read_text());config=json.loads((P/'runtime.json').read_text())
    bundle=json.loads(Path(config['bundle']).read_text());lookup={r['id']:r for r in bundle['rows']}
    split=original['splits'][DATASET];donors={}
    for label in [1,0]:
        ids=sorted(i for i in split['train_ids'] if lookup[i]['label']==label)
        assert len(ids)==25 and len({lookup[i]['group_id'] for i in ids})==25
        random.Random(f'42:{DATASET}:{label}').shuffle(ids)
        donors.update({recipient:ids[(j+1)%len(ids)] for j,recipient in enumerate(ids)})
    for a,b in donors.items():
        assert a!=b and lookup[a]['group_id']!=lookup[b]['group_id'] and lookup[a]['label']==lookup[b]['label']
    gpt=json.loads((P/'gpt_runtime.json').read_text())
    for role,value in gpt.items():
        if role.endswith('_model'):
            assert value['backend']=='codex'
            value['model']='gpt-6-luna'
    gpt['note']='User-requested Luna for all GEPA GPT roles; fresh judge and optimizer caches; final Haiku unchanged.'
    protocol=dict(protocol='ao_feedback_ablation_luna_v1',dataset=DATASET,source_model='google/gemma-3-27b-it',
        arms=ARMS,splits=split,starting_prompt=original['starting_prompt'],ao_query=AO_QUERY,
        ao_layer_zero_based=31,ao_positions='Question-end last up to20 tokens, first prefill before answer tokens',
        reflection_fields={arm:list(feedback_record(arm,dict(question='',label=0,presuppositions=[],reference_answer=''),
            {'answer':''},{'score':0,'reason':''},'')) for arm in ARMS},
        ao_only='Current candidate and AO descriptions only; no question text, answers, labels, references, per-example scores or judge rationale in reflection.',
        scalar_objective='Actual answer Well scores retained for GEPA acceptance and GPT dev selection in every arm, including AO-only.',
        question_metric_cap=100,pair_metric_cap=50,max_candidate_proposals=4,reflection_minibatch_pairs=2,
        prompt_token_limit=512,diagnostic_token_limit=512,seed=42,source_batch_size=1,source_max_new_tokens=8192,
        donor_map=donors,donor_matching='Train-only different group, same label, same current candidate; no donor identity in reflection',
        native_gpus=NATIVE_GPUS,reader_gpus=READER_GPUS,
        final_evaluation='Freeze all Luna-dev-selected candidates, then pinned Haiku on existing8 heldout dev IDs; exploratory, not official test.',
        prior_exposure='Reader query proposed after viewing earlier train outputs and prior pilot heldout results; no new independent-confirmation claim.',
        equality='Same GEPA library, seed, train50/dev8, starting prompt, objective budget and common reflection template across arms. Cached identical generations and judgments reused.',
        gpt=gpt,judge=gpt['judge_model'],migration='Preserve prior Sol runs; reuse source generations and reader outputs only. No Sol judgments, reflection calls, optimizer state or selected prompts reused.')
    path=OUT/'protocol.json'
    if path.exists():
        previous=json.loads(path.read_text())
        # Placement may change by user instruction; scientific settings remain frozen.
        assert {k:v for k,v in previous.items() if k not in {'native_gpus','reader_gpus'}}=={
            k:v for k,v in protocol.items() if k not in {'native_gpus','reader_gpus'}}
    else:save(path,protocol)
    save(OUT/'execution_hardware.json',dict(native_gpus=NATIVE_GPUS,reader_gpus=READER_GPUS,
        initial_native_gpus=json.loads(path.read_text())['native_gpus'],
        initial_reader_gpus=json.loads(path.read_text())['reader_gpus'],updated_utc=time.time()))
    config={**config,'artifact_root':str(OUT),'native_gpus':NATIVE_GPUS,'reader_gpus':READER_GPUS,'ao_prompt':AO_QUERY}
    save(OUT/'runtime.json',config)
    if (OUT/'gpt_runtime.json').exists():assert json.loads((OUT/'gpt_runtime.json').read_text())==gpt
    else:save(OUT/'gpt_runtime.json',gpt)
    for name in ['haiku_runtime.json','export_consent.json']:
        if not (OUT/name).exists():save(OUT/name,json.loads((P/name).read_text()))
    # Read-only individual cache links: never link a writable queue directory.
    for name in ['native/results']:
        target=OUT/name;target.mkdir(parents=True,exist_ok=True)
        for origin in [P,P.parent/'gepa_ao_feedback_ablation_crepe_20261014']:
            for source in (origin/name).glob('*.json'):
                if source.name.endswith('.error.json'):continue
                destination=target/source.name
                if not destination.exists():destination.symlink_to(source.resolve())
    for r in json.loads((PROBE/'query_outputs.json').read_text()):
        if QUERY_KIND!='behavior':continue
        ident=digest(['ao_behavior_gepa_v1',r['source_native_id'],AO_QUERY])
        path=OUT/'readers/ao/results'/(ident+'.json')
        if not path.exists():save(path,dict(r['behavior'],id=ident,reused_probe_output_id=r['behavior']['id']))
    return protocol,config,lookup


class BehaviorIO(ControlIO):
    def external(self,role,messages):
        assert self.gpt[role+'_model']['model']=='gpt-6-luna'
        config=self.gpt[role+'_model']
        ident=digest([role,config,messages,'reader_gepa_gpt_v1'])
        if (self.out/'external_calls'/(ident+'.json')).exists():return super().external(role,messages)
        # Both active suites share one provider request at a time and a 429 cooldown.
        gate=P.parent/'luna_gepa_provider_cooldown.json'
        with (P.parent/'luna_gepa_provider.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            deadline=json.loads(gate.read_text()).get('retry_after_utc',0) if gate.exists() else 0
            while time.time()<deadline:
                save(self.out/'provider_gate_status.json',dict(state='waiting_provider',retry_after_utc=deadline,model='gpt-6-luna'))
                time.sleep(min(5,deadline-time.time()))
            save(self.out/'provider_gate_status.json',dict(state='calling',role=role,model='gpt-6-luna',updated_utc=time.time()))
            try:
                result=super().external(role,messages)
            except Exception as error:
                if any(word in str(error).lower() for word in ['429','rate limit','usage limit','quota','hit your limit']):
                    save(gate,dict(retry_after_utc=time.time()+600,reason=str(error),model='gpt-6-luna'))
                raise
            save(self.out/'provider_gate_status.json',dict(state='idle',model='gpt-6-luna',updated_utc=time.time()))
            return result

    def start_reader(self,kind='ao'):
        assert kind=='ao'
        if self.reader_proc is not None and self.reader_proc.poll() is None:return
        directory=self.out/'readers/ao';directory.mkdir(parents=True,exist_ok=True)
        (directory/'STOP').unlink(missing_ok=True)
        self.reader_log=(directory/'worker.log').open('a')
        self.reader_proc=subprocess.Popen(['/data3/heejae/uv/fpqa_ao_20261012/bin/python','-u',
            'scripts/reader_gepa_verbalizer_worker.py','--config',str(self.config_path),'--kind','ao','--gpus',READER_GPUS],
            cwd=ROOT,env={**os.environ,'HF_HOME':'/data3/heejae/hf_cache','HF_HUB_OFFLINE':'1'},
            stdout=self.reader_log,stderr=self.reader_log)
        self.reader_kind='ao'

    def behavior(self,native):
        ident=digest(['ao_behavior_gepa_v1',native['id'],AO_QUERY])
        path=self.out/'readers/ao/results'/(ident+'.json')
        reuse=native['reader_reuse'].get('ao',{})
        if QUERY_KIND=='semantic' and not path.exists() and reuse.get('bitwise_equal'):
            old=self.ao_by_path[reuse['original_activation_path']]
            save(path,dict(id=ident,explanation=old['ao_output'],query=AO_QUERY,
                source_native_id=native['id'],source_activation_path=native['ao_activation_path'],
                source_text_provided=False,reused_original_output_id=old['id'],bytewise_verified=True))
        if not path.exists():
            save(self.out/'readers/ao/requests'/(ident+'.json'),dict(id=ident,native_id=native['id'],
                activation_path=native['ao_activation_path'],query=AO_QUERY))
            self.start_reader();began=time.monotonic()
            while not path.exists():
                if self.reader_proc.poll() is not None:raise RuntimeError('Owned AO behavior reader exited')
                if time.monotonic()-began>900:raise TimeoutError('AO behavior request exceeded900seconds')
                time.sleep(1)
        result=json.loads(path.read_text())
        assert result['query']==AO_QUERY and result['source_native_id']==native['id']
        assert result['source_activation_path']==native['ao_activation_path'] and result['source_text_provided'] is False
        return result


def run():
    protocol,config,lookup=prepare();io=BehaviorIO(OUT/'runtime.json')
    from gepa.optimize_anything import GEPAConfig,EngineConfig,ReflectionConfig,optimize_anything
    tok=load_prompt_tokenizer(io.baseline);split=protocol['splits'];selected={}
    def state(stage,**kw):save(OUT/'status.json',dict(state='running',stage=stage,updated_utc=time.time(),**kw))
    def pairs(ids,partition):
        groups=[[lookup[i] for i in ids if lookup[i]['label']==y] for y in [1,0]]
        for g in groups:random.Random(42).shuffle(g)
        return [dict(pair=[a,b],partition=partition) for a,b in zip(*groups)]
    try:
        for arm in ARMS:
            directory=OUT/'optimization'/arm;directory.mkdir(parents=True,exist_ok=True)
            path=directory/'selected.json'
            if path.exists():selected[arm]=json.loads(path.read_text());continue
            state('optimization',arm=arm)
            def evaluate(candidate,example):
                if set(candidate)!={'answer'}:raise ValueError('Candidate components changed')
                if len(tok.encode(candidate['answer'],add_special_tokens=False))>512:
                    return 0.,{'error':'Candidate exceeds fixed512token limit'}
                feedback=[];objective=[]
                for row in example['pair']:
                    native=io.native(row,candidate['answer']);judge=io.judge(row,native)
                    description=None;extra=None
                    if example['partition']=='train' and arm!='text':
                        donor=lookup[protocol['donor_map'][row['id']]] if arm=='text_ao_shuffled' else row
                        read_native=io.capture_donor(donor,candidate['answer']) if arm=='text_ao_shuffled' else native
                        extra=io.behavior(read_native)
                        tokens=tok.encode(extra['explanation'],add_special_tokens=False)
                        description=tok.decode(tokens[:512])
                        save(directory/'extras'/(native['id']+'.json'),dict(extra,recipient_id=row['id'],donor_id=donor['id'],
                            supplied_diagnostic=description,clipped=len(tokens)>512))
                    # Dev scores never become reflection feedback; no extra reader calls on dev.
                    info=feedback_record(arm,row,native,judge,description) if example['partition']=='train' else {}
                    if arm=='ao_only' and example['partition']=='train':assert set(info)=={'ao_description'}
                    feedback.append(info)
                    objective.append(dict(row_id=row['id'],native_id=native['id'],score=judge['score']))
                score=sum(r['score']/5 for r in objective)/2
                with (directory/'evaluations.jsonl').open('a') as f:
                    f.write(json.dumps(dict(candidate=candidate,partition=example['partition'],score=score,
                        objective_records=objective,reflection_feedback=feedback,timestamp=time.time()),ensure_ascii=False)+'\n')
                return score,{'examples':feedback}
            def reflect(messages):
                if isinstance(messages,str):messages=[{'role':'user','content':messages}]
                if not any(m['role']=='system' for m in messages):
                    messages=[{'role':'system','content':'Improve the supplied prompt using training feedback. Return the requested candidate.'}]+messages
                assert [m['role'] for m in messages]==['system','user']
                with (directory/'reflection_payloads.jsonl').open('a') as f:
                    f.write(json.dumps(dict(messages=messages,arm=arm,timestamp=time.time()),ensure_ascii=False)+'\n')
                return io.external('reflection',messages)
            result=optimize_anything(seed_candidate={'answer':protocol['starting_prompt']},evaluator=evaluate,
                dataset=pairs(split['train_ids'],'train'),valset=pairs(split['candidate_dev_ids'],'dev'),
                objective='Maximize balanced FPQ correction and normal-question preservation. Only questions are available at inference.',
                background='Optimize only the answer prompt. No tools or retrieval. The prompt must fit512tokenizer tokens. '
                    'Use only the supplied feedback. Activation descriptions may be unreliable and are not evidence of source beliefs. '
                    'Do not assume that a described future response actually occurred. Do not copy example-specific facts into a general instruction.',
                config=GEPAConfig(engine=EngineConfig(run_dir=str(directory/'engine'),seed=42,max_metric_calls=50,
                    max_candidate_proposals=4,parallel=False,max_workers=1,cache_evaluation=False),
                    reflection=ReflectionConfig(reflection_lm=reflect,reflection_minibatch_size=2)))
            assert set(result.best_candidate)=={'answer'} and len(tok.encode(result.best_candidate['answer'],add_special_tokens=False))<=512
            record=dict(candidate=result.best_candidate,best_idx=result.best_idx,arm=arm,
                selection='Frozen GPT candidate-dev selection, no Haiku reselection',question_metric_cap=100)
            save(path,record);selected[arm]=record;print('Selected',arm,result.best_idx,flush=True)
        save(OUT/'frozen_candidates.json',selected);io.stop_reader()
        # Existing tiny heldout panel is an exploratory comparison, not a new official test.
        haiku=json.loads((OUT/'haiku_runtime.json').read_text())['judge_model'];scored=[]
        for arm in ARMS:
            for ident in split['heldout_validation_ids']:
                state('heldout',arm=arm,row_id=ident)
                native=io.native(lookup[ident],selected[arm]['candidate']['answer'])
                cache_id=hashlib.sha256(json.dumps([haiku,native['id']],sort_keys=True).encode()).hexdigest()
                path=OUT/'heldout_haiku'/(cache_id+'.json')
                old=P/'heldout_haiku'/path.name
                if path.exists():judge=json.loads(path.read_text())
                elif old.exists():judge=json.loads(old.read_text());save(path,judge)
                elif native['completed']:
                    raw,meta=CLIBackend(haiku)(well_response_judge(ROOT/io.baseline['well_root'],lookup[ident],native['answer']))
                    rating=parse_rating(raw)
                    if rating is None:raise ValueError('Unparseable pinned Haiku score')
                    judge=dict(score=rating,reason=raw,metadata=meta);save(path,judge)
                else:judge=dict(score=0,reason='Incomplete answer',incomplete=True);save(path,judge)
                scored.append(dict(arm=arm,row_id=ident,label=lookup[ident]['label'],native=native,judgment=judge))
                save(OUT/'heldout_scored_partial.json',scored)
        summary=[]
        for arm in ARMS:
            fpq=[r['judgment']['score'] for r in scored if r['arm']==arm and r['label']==1]
            nfp=[r['judgment']['score'] for r in scored if r['arm']==arm and r['label']==0]
            assert len(fpq)==len(nfp)==4
            summary.append(dict(arm=arm,fpq_mean=sum(fpq)/4,nfp_mean=sum(nfp)/4,balanced_normalized=(sum(fpq)+sum(nfp))/40))
        save(OUT/'heldout_scored.json',scored);save(OUT/'comparison_summary.json',dict(results=summary,
            limits='Exploratory existing8 heldout-dev questions, not official test; query chosen after prior pilot results.'))
        with (OUT/'comparison_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        save(OUT/'status.json',dict(state='complete',records=8*len(ARMS),updated_utc=time.time()))
    except Exception as error:
        save(OUT/'status.json',dict(state='failed',error=repr(error),updated_utc=time.time()));raise
    finally:io.stop_reader()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare();print('Prepared',DATASET,QUERY_KIND,ARMS,'with Luna, train50/dev8/heldout8,100question objective cap each')
    else:run()
