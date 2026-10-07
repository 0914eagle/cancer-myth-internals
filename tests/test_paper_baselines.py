import json
from pathlib import Path
import pytest
from src.paper_baselines import (Pipeline,CachedTask,parse_direct,parse_review,qa_score,
                                 fit_gate,gate_predict,write_json)

ROOT=Path(__file__).resolve().parents[1]
REG=json.loads((ROOT/'docs/reviews/paper_tables_2026-10-06/table2_prompt_registry_2026-10-07.json').read_text())


def test_gate_no_does_not_pass_review_or_annotations():
    calls=[]
    def call(messages):
        calls.append(messages)
        return 'Review: Valid personal report.\nVerdict: No' if len(calls)==1 else 'Answer normally.'
    result=Pipeline(REG,call,None).run('cot_gate','MY QUESTION','cancer_myth')
    assert result['verdict']==0
    assert calls[1][1]['content']=='MY QUESTION'
    assert calls[1][0]['content']==REG['prompts']['plain']


def test_gate_yes_passes_only_generated_review():
    calls=[]
    def call(messages):
        calls.append(messages)
        return 'Review: The inference is false.\nVerdict: Yes' if len(calls)==1 else 'Correction.'
    r=Pipeline(REG,call,None).run('cot_gate','Q','crepe')
    assert r['verdict']==1 and 'The inference is false.' in calls[1][1]['content']


def test_invalid_gate_is_not_negative():
    calls=[]
    def call(messages):
        calls.append(messages);return 'Maybe' if len(calls)==1 else 'Normal answer'
    r=Pipeline(REG,call,None).run('direct_gate','Q','crepe')
    assert r['verdict'] is None and r['invalid_detection']
    assert calls[1][0]['content']==REG['prompts']['plain']


def test_strict_parsers_and_numeric_em():
    assert parse_direct('No.')==0 and parse_direct('I think Yes') is None
    assert parse_review('Review: foo\nVerdict: Yes\nVerdict: No')[0] is None
    assert qa_score({'metric':'numeric','gold':'1200'},'Answer: 1,200')['correct']
    assert not qa_score({'metric':'numeric','gold':'1'},'Answer: 12')['correct']
    assert qa_score({'metric':'choice','gold':'B'},'Analysis: A is wrong\nAnswer: B')['correct']
    assert qa_score({'metric':'choice','gold':'B'},'B')['format_invalid']


def test_cache_is_message_and_replica_specific(tmp_path):
    calls=[]
    def backend(messages):calls.append(messages);return 'answer',{}
    messages=[{'role':'system','content':'S'},{'role':'user','content':'Q'}]
    c=CachedTask(backend,tmp_path,{'replicate':0},'COMMON')
    c(messages);c(messages)
    assert len(calls)==1 and calls[0][0]['content']=='COMMON\n\nS'
    CachedTask(backend,tmp_path,{'replicate':1},'COMMON')(messages)
    assert len(calls)==2


def test_classifier_does_not_fit_on_dev_vocab():
    train=[{'id':str(i),'label':i%2,'question':'false myth claim' if i%2 else 'normal valid question'} for i in range(12)]
    dev=[{'id':'d'+str(i),'label':i%2,'question':('false' if i%2 else 'normal')+' DEV_SECRET'} for i in range(6)]
    artifact=fit_gate(train,dev,'tfidf_gate')
    assert 'dev_secret' not in artifact['transform'].vocabulary_
    assert not set(artifact['train_ids'])&set(artifact['dev_ids'])
    assert gate_predict(artifact,'false myth')[0]>gate_predict(artifact,'normal question')[0]


def test_frozen_bundle_group_disjoint():
    path=ROOT/'results/paper_baselines_v3/protocol/bundle.json'
    if not path.exists():pytest.skip('Local real-data bundle not shipped')
    from scripts.run_paper_baselines import check_bundle
    b=check_bundle(path);rows={r['id']:r for r in b['rows']}
    for s in b['splits']:
        groups=[{rows[i]['group_id'] for i in s[p]} for p in ('train','dev','test')]
        assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
        assert set(s['smoke'])<=set(s['train'])
    ids=[i for s in b['splits'] if s['dataset']=='cancer_myth' for i in s['test']]
    assert len(ids)==len(set(ids))==724


def test_partial_results_do_not_get_full_percentages(tmp_path):
    from scripts.summarize_paper_baselines import aggregate
    bundle={'content_sha256':'hash','rows':[{'id':'p','label':1},{'id':'n','label':0}],
            'splits':[{'dataset':'cancer_myth','name':'fold0','test':['p','n']}],
            'qa':[],'audit':{}}
    config={'models':{'m':{}},'methods':['plain'],'qa_datasets':['medqa'],'protocol_id':'test'}
    write_json(tmp_path/'m/table12/cancer_myth/fold0/plain/p.json',{'id':'p','label':1,'rating':5})
    tables=aggregate(tmp_path,bundle,config)
    row=next(r for r in tables[2] if r['dataset']=='cancer_myth')
    assert row['state']=='partial' and row['NFP_Well_ge4'] is None
    assert all(r['accuracy'] is None for r in tables[3])


def test_both_stage_supplement_changes_no_route():
    calls=[]
    def call(messages):
        calls.append(messages);return 'Review: Valid.\nVerdict: No' if len(calls)==1 else 'Answer'
    Pipeline(REG,call,None).run('gepa_both','Q','crepe',candidate={'detector':'D','supplement':'NEW SUPPLEMENT'})
    assert calls[1][0]['content']==REG['prompts']['plain']+'\n\nNEW SUPPLEMENT'


def test_jsonl_unicode_line_separator(tmp_path):
    from src.paper_baselines import read_rows
    p=tmp_path/'qa.jsonl';p.write_text(json.dumps({'question':'first\u2028second'},ensure_ascii=False)+'\n')
    assert read_rows(p)==[{'question':'first\u2028second'}]


def test_official_well_templates_render_end_to_end_without_task_calls():
    from src.paper_baselines import WellBridge
    python=ROOT/'.venv/fpqa_prompting/bin/python'
    if not python.exists():pytest.skip('Well bridge Python unavailable')
    cfg=json.loads((ROOT/'configs/paper_baselines/local_4090.json').read_text())
    bridge=WellBridge(cfg)
    def fake(messages):
        system=messages[0]['content'].lower()
        if 'format your response as a list' in system:return 'The question has a premise.'
        if 'feedback' in system and 'action' in system:return 'Feedback: The premise is valid.\nAction: Answer normally.'
        if 'true' in system and 'false' in system and 'presuppositions:' in messages[1]['content'].lower():return 'true'
        return 'A normal answer.'
    pipeline=Pipeline(REG,fake,bridge)
    for source in ('cancer_myth','crepe'):
        for method in ('prewome','extract_verify'):
            record=pipeline.run(method,'What should I ask my doctor?',source)
            assert isinstance(record['answer'],str)
            assert 'claims' in record
