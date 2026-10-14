"""Export AO-generated text with question, candidate and answer provenance."""
from collections import Counter
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/nla_ao_parallel_20261012'
P=SOURCE/'gepa_reader_pilot'
OUT=Path('/home/eagle0914/cancer-myth/results/ao_text_outputs_20261014')
OUT.mkdir(parents=True,exist_ok=True)


def load(p):return json.loads(p.read_text())


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def jsonl(name,rows):
    with (OUT/name).open('w') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def csv_file(name,rows,fields):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def prompt_id(text):return hashlib.sha256(text.encode()).hexdigest()[:16]


runtime=load(P/'runtime.json');plan=load(P/'plan.json')
baseline=load(Path(runtime['baseline_config']))
registry=load(ROOT/baseline['registry'])
catalog={}
fixed=[]
for line in (SOURCE/'gemma27/ao_results.jsonl').read_text().splitlines():
    r=json.loads(line)
    assert r['layer']==31 and r['hidden_state_index']==32
    assert runtime['ao_prompt'] in r['ao_prompt_rendered']
    fixed.append(dict(id=r['id'],question_id=r['base_id'],dataset=r['source'],question_type=r['category'],
        condition=r['condition'],question=r['prompt'],source_layer_zero_based=31,hf_hidden_state_index=32,
        requested_window_size=r['requested_window_size'],actual_window_size=r['window_size'],
        source_token_positions=r['source_positions'],source_token_ids=r['source_token_ids'],
        ao_query=runtime['ao_prompt'],ao_text=r['ao_output'],ao_rendered_prompt=r['ao_prompt_rendered'],
        source_model_revision=r['source_model_revision'],ao_revision=r['ao_model_revision'],
        source_activation_path=r['activation_path'],source_answer_status='Activation-only original capture; no original answer recorded.'))
assert len(fixed)==4710 and len({r['question_id'] for r in fixed})==785
jsonl('fixed785_ao_texts.jsonl',fixed)
csv_file('fixed785_ao_texts.csv',fixed,['question_id','dataset','question_type','condition','requested_window_size',
    'actual_window_size','source_layer_zero_based','question','ao_query','ao_text'])

initial=[]
for f in (P/'analysis').glob('*.json'):
    r=load(f)
    if 'row_id' not in r:continue
    assert r['ao']['bytewise_verified']
    initial.append(dict(question_id=r['row_id'],dataset=r['dataset'],question_type='FPQ' if r['label'] else 'NFP',
        question=r['question'],candidate_id=prompt_id(r['native']['candidate_prompt']),
        source_prompt=r['native']['candidate_prompt'],ao_query=runtime['ao_prompt'],ao_text=r['ao']['explanation'],
        source_answer=r['native']['answer'],source_answer_finish_reason=r['native']['finish_reason'],
        well_score=r['judgment']['score'],well_reason=r['judgment']['reason'],
        source_native_id=r['native']['id'],activation_reuse_bytewise_verified=True,
        scope='Fixed100 train questions, Plain prompt; not heldout.'))
initial.sort(key=lambda r:(r['dataset'],r['question_id']))
assert len(initial)==100
jsonl('plain100_ao_question_answer_pairs.jsonl',initial)
csv_file('plain100_ao_question_answer_pairs.csv',initial,['question_id','dataset','question_type','question','ao_text','source_answer','well_score'])
initial_lookup={r['question_id']:r for r in initial}
natives={r['id']:r for r in (load(f) for f in (P/'native/results').glob('*.json')) if 'row_id' in r}
feedback=[];selected_prompts={};decoded_ids=set()
for dataset in ['cancer_myth','crepe']:
    d=P/'optimization'/dataset/'ao'
    selection=load(d/'selected.json');selected_prompts[dataset]=selection
    pool=load(d/'engine/candidates.json');indices={r['answer']:i for i,r in enumerate(pool)}
    emitted={(e['candidate']['answer'],x['id']) for e in (json.loads(s) for s in (d/'evaluations.jsonl').read_text().splitlines())
             for x in e['examples'] if 'additional_diagnostic' in x}
    for f in (d/'extras').glob('*.json'):
        extra=load(f);n=natives[f.stem];judge=load(P/'judgments'/(n['id']+'.json'))
        assert n['dataset']==dataset and n['row_id'] in plan['splits'][dataset]['train_ids']
        candidate=prompt_id(n['candidate_prompt'])
        catalog[candidate]=dict(prompt=n['candidate_prompt'])
        item=dict(dataset=dataset,question_id=n['row_id'],question_type=initial_lookup[n['row_id']]['question_type'],
            question=n['question'],candidate_id=candidate,engine_pool_index=indices.get(n['candidate_prompt']),
            is_plain_seed=n['candidate_prompt']==plan['starting_prompt'],
            is_selected_prompt=n['candidate_prompt']==selection['candidate']['answer'],
            supplied_in_completed_gepa_evaluation=(n['candidate_prompt'],n['row_id']) in emitted,
            ao_query=runtime['ao_prompt'],ao_text=extra['explanation'],
            supplied_to_gepa=extra['supplied_diagnostic'],diagnostic_clipped=extra['clipped'],
            source_answer=n['answer'],source_answer_finish_reason=n['finish_reason'],well_score=judge['score'],
            well_reason=judge['reason'],source_native_id=n['id'],
            source_layer_zero_based=31,hf_hidden_state_index=32,
            token_window_size=len(n['ao_source_token_ids']),
            initial_plain_ao_text=initial_lookup[n['row_id']]['ao_text'],
            initial_plain_answer=initial_lookup[n['row_id']]['source_answer'],
            initial_plain_well_score=initial_lookup[n['row_id']]['well_score'])
        feedback.append(item)
        if not extra.get('reused',False):decoded_ids.add(extra['id'])
feedback.sort(key=lambda r:(r['dataset'],r['engine_pool_index'] if r['engine_pool_index'] is not None else 999,r['candidate_id'],r['question_id']))
assert len(feedback)==67 and len(decoded_ids)==43
assert sum(r['supplied_in_completed_gepa_evaluation'] for r in feedback)==66
jsonl('gepa_ao_explanations.jsonl',feedback)
csv_file('gepa_ao_explanations.csv',feedback,['dataset','question_id','question_type','candidate_id','engine_pool_index',
    'is_plain_seed','is_selected_prompt','question','ao_query','ao_text','source_answer','well_score','initial_plain_ao_text','initial_plain_well_score'])
jsonl('fresh_ao_raw_outputs.jsonl',[load(P/'readers/ao/results'/(i+'.json')) for i in sorted(decoded_ids)])
save('candidate_prompt_catalog.json',catalog);save('selected_ao_prompts.json',selected_prompts)
heldout=[r for r in load(P/'heldout_validation_scored.json') if r['arm']=='ao']
assert len(heldout)==16
jsonl('ao_heldout_answers_haiku.jsonl',[dict(dataset=r['dataset'],question_id=r['row_id'],
    question_type='FPQ' if r['label'] else 'NFP',question=r['native']['question'],
    source_prompt=r['native']['candidate_prompt'],source_answer=r['native']['answer'],
    haiku_well_score=r['judgment']['score'],haiku_reason=r['judgment']['reason'],
    ao_description_at_inference=False,
    note='Source answers using the frozen AO-optimized prompt; no new AO description requested for heldout inference.') for r in heldout])
for name in ['comparison_summary.json','comparison_summary.csv','haiku_runtime.json']:
    shutil.copyfile(P/name,OUT/('heldout_'+name if name.startswith('comparison') else name))
controls=SOURCE/'reader_feedback_controls_20261014'
save('protocol.json',dict(source_model='google/gemma-3-27b-it',ao_query=runtime['ao_prompt'],
    decoder_block_zero_based=31,hf_hidden_states_index=32,
    fixed785_windows=[1,5,20],gepa_window='Last up to20 question tokens',
    position='Question-end activations captured in the first generation prefill, before answer tokens; not a premise-span endpoint.',
    reader_input='Activation vectors and fixed query; original question, answer and gold labels not supplied as text.',
    fixed785_source_prompts={k:v for k,v in registry['prompts'].items() if k in ['plain','balanced']},
    actual_token_layer_verification=str(controls/'actual_protocol_verification.json'),
    new_stance_query_comparison='Registered separately; not included as completed outputs in this snapshot.',
    limitations=['AO descriptions are unverified generated interpretations, not source-belief measurements.',
        'Candidate-dependent training samples repeat questions;67 feedback records are not67 independent questions.',
        'Heldout results use4FPQ+4NFP per dataset; not official test/main-table performance.',
        'These files preserve original English AO text without GPT rewriting.']))
shutil.copyfile(controls/'actual_protocol_verification.json',OUT/'token_layer_verification.json')

all_texts=['# GEPA 중 저장된 AO 설명 전체67개','',
    '영어 출력은 AO 원문이다. 완료된 평가 피드백66개와 중단 당시 저장만 된1개를 모두 보존하며 각 항목에 구분했다. 후보별 표본이 달라 이 목록 순서를 성능 향상 순서로 해석하면 안 된다.','']
for i,r in enumerate(feedback,1):
    delivered='완료된 평가 피드백에 포함' if r['supplied_in_completed_gepa_evaluation'] else '저장만 됨; 완료된 평가 피드백에는 없음'
    selected='최종 선택 프롬프트' if r['is_selected_prompt'] else ('초기 Plain' if r['is_plain_seed'] else '탐색 후보')
    all_texts += [f'## {i}. {r["dataset"]} · {r["question_id"]} · {r["question_type"]}','',r['question'],'',
        f'후보 `{r["candidate_id"]}` · {selected} · {delivered} · GPT Well {r["well_score"]}/5','',
        '````text',r['ao_text'],'````','']
(OUT/'GEPA_AO_TEXTS.md').write_text('\n'.join(all_texts))
plain_texts=['# 초기 Plain100의 AO 설명 전체','', '질문과 AO 원문만 나란히 표시했다. 실제 답변은 대응 JSONL/CSV에서 확인할 수 있다.','']
for r in initial:
    plain_texts += [f'## {r["dataset"]} · {r["question_id"]} · {r["question_type"]}','',r['question'],'',
                   '````text',r['ao_text'],'````','']
(OUT/'PLAIN100_AO_TEXTS.md').write_text('\n'.join(plain_texts))
examples=['# AO 실제 출력 예시','',
    '아래 영어 문장은 AO가 실제 생성한 원문이다. 전체 원문은 JSONL/CSV에 있다. 질문별 다양한 출력 형태를 확인하는 예시이며 전체 성공률을 대표하는 표본은 아니다. 최적화 후보는 최종 선택된 프롬프트와 다를 수 있다.','']
preview_ids=['fpq_44','fpq_706','nfp_1029','crepe:2018-04484']
for question_type in ['FPQ','NFP']:
    candidate=next((r['question_id'] for r in feedback if r['is_selected_prompt'] and not r['is_plain_seed'] and r['question_type']==question_type),None)
    if candidate and candidate not in preview_ids:preview_ids.append(candidate)
for ident in preview_ids:
    seed=initial_lookup[ident]
    examples += ['## '+ident+' · '+seed['question_type'],'',seed['question'],'',
        '**초기 Plain AO 원문**','', '> '+seed['ao_text'],'']
    cases=[r for r in feedback if r['question_id']==ident and not r['is_plain_seed']]
    for r in cases:
        tag='최종 선택 프롬프트' if r['is_selected_prompt'] else '탐색 후보'
        pool_tag=f'pool {r["engine_pool_index"]}' if r['engine_pool_index'] is not None else 'pool 미편입 시도'
        examples += [f'**{tag} · {pool_tag} · `{r["candidate_id"]}`**','',
            '> '+r['ao_text'],'',f'원 모델 답변의 GPT Well 점수: {r["well_score"]}/5. 답변 원문과 프롬프트는 `gepa_ao_explanations.jsonl`과 `candidate_prompt_catalog.json`에서 같은 ID로 확인할 수 있다.','']
(OUT/'EXAMPLES.md').write_text('\n'.join(examples))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
save('SNAPSHOT.json',dict(created_kst=now,fixed785_ao_records=4710,plain100_pairs=100,
    captured_gepa_ao_records=len(feedback),gepa_ao_feedback_records=66,captured_without_completed_evaluation=1,
    fresh_ao_decodes=len(decoded_ids),
    feedback_by_dataset=dict(Counter(r['dataset'] for r in feedback)),
    unique_feedback_questions=len({r['question_id'] for r in feedback}),
    heldout_ao_answers=16,selected_ao_conditions_complete=2,
    source_root=str(SOURCE),controls_status=load(controls/'controller_status.json')))
readme='''# AO가 실제 생성한 설명 텍스트

Gemma 3 27B의 내부 활성을 AO로 읽은 실제 영어 출력과 대응 질문·답변을 담았다.
생성 텍스트를 GPT로 다시 쓰거나 요약하지 않았다.

먼저 [실제 예시](EXAMPLES.md)를 보면 된다.
[GEPA 설명 전체67개](GEPA_AO_TEXTS.md)와 [초기 Plain100 설명 전체](PLAIN100_AO_TEXTS.md)도 GitHub에서 바로 읽을 수 있다.

| 파일 | 내용 |
|---|---|
| `fixed785_ao_texts.jsonl` / `.csv` | 고정785문항 × Plain/Balanced × 1/5/최대20토큰 = 4,710개 AO 설명. 각 행에 질문과 AO 원문 포함 |
| `plain100_ao_question_answer_pairs.jsonl` / `.csv` | 같은100문항의 Plain AO 설명과 실제 답변·GPT 채점. 기존 활성과 바이트 일치 확인 후 설명 연결 |
| `gepa_ao_explanations.jsonl` / `.csv` | AO-GEPA 중 저장한67개 설명. 질문·후보 ID·AO 원문·실제 답변·점수·초기 설명을 나란히 저장. 완료된 평가 피드백에 포함된 것은66개 |
| `fresh_ao_raw_outputs.jsonl` | 그중 새로 디코딩한43개 AO의 원본 결과와 실제 reader 프롬프트 |
| `candidate_prompt_catalog.json` | 후보 ID별 원 모델 입력 프롬프트 |
| `selected_ao_prompts.json` | Cancer-Myth·CREPE 최종 선택 프롬프트 |
| `ao_heldout_answers_haiku.jsonl` | 고정 프롬프트의 별도 보류16문항 답변과 Haiku 채점 |
| `heldout_comparison_summary.csv` | 기본 GEPA·텍스트·NLA·AO의 공통 보류 문항 점수 |
| `protocol.json` / `token_layer_verification.json` | AO 질문·레이어·토큰 위치 및 실제 저장값 검증 |

AO 질문은 `Describe the semantic content represented by these activations.`였다.
source decoder block31(0기반), HF hidden_states[32]를 사용했다.
GEPA에서는 질문 마지막 최대20토큰을 읽었다. 질문·정답·실제 답변 텍스트를 AO에 주지 않았다.

4,710개 초기 출력의 원 활성 추출에서는 원 모델 답변을 생성하지 않았다.
이후 Plain100 및 GEPA 답변은 별도 실제 생성에서 얻고 활성 대응을 확인했다.
67개 저장 설명에는 초기 설명 재사용24개와 새 디코딩43개가 포함된다.
이 중1개는 중단 당시 저장됐지만 완료된 평가 피드백 기록에는 없다.
`supplied_in_completed_gepa_evaluation` 필드로66개 완료 기록과 구분한다.
같은 질문이 여러 후보에서 반복되므로67문항으로 해석하면 안 된다.
보류 문항에서는 AO 설명을 추가 생성하지 않고, 고정된 프롬프트로 답변만 생성했다.

AO 출력은 모델의 믿음이나 사실의 정답이 아니다. 긍정 사례만 골라 성공률을 계산하지 않았다.
새 수용·의심·거부 질문 비교와 다른 문항 설명 대조는 별도 진행 중이며 이 파일들의 완료 결과와 구분한다.
'''
(OUT/'README.md').write_text(readme)
(OUT/'code').mkdir(exist_ok=True)
shutil.copyfile(Path(__file__),OUT/'code/export_ao_text_outputs.py')
manifest=[]
for f in sorted(OUT.rglob('*')):
    if f.is_file() and f.name!='MANIFEST.sha256.json':
        manifest.append(dict(path=str(f.relative_to(OUT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
save('MANIFEST.sha256.json',manifest)
archive=OUT.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as t:t.add(OUT,arcname=OUT.name)
print(json.dumps(dict(directory=str(OUT),files=len(manifest)+1,uncompressed_MB=round(sum(r['bytes'] for r in manifest)/1024**2,2),
    archive_MB=round(archive.stat().st_size/1024**2,2),captured_gepa_ao_records=len(feedback),completed_feedback_records=66,
    fresh_decodes=len(decoded_ids)),ensure_ascii=False,indent=2))
