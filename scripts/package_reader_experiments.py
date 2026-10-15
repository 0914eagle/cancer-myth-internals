"""Portable text-result snapshot; tensor/checkpoint payloads are indexed, not copied."""
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/nla_ao_parallel_20261012'
WORKSPACE=Path('/home/eagle0914/cancer-myth')
DEST=WORKSPACE/'results/nla_ao_experiments_20261014'
DEST.mkdir(parents=True,exist_ok=True)
queries={};questions={};counts={};omitted=[]


def save(rel,data):
    f=DEST/rel;f.parent.mkdir(parents=True,exist_ok=True)
    f.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def lines(path):
    if not path.exists():return []
    raw=path.read_text().splitlines();out=[]
    for i,line in enumerate(raw):
        if not line.strip():continue
        try:out.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(raw)-1:omitted.append({'path':str(path),'reason':'Incomplete trailing append at snapshot time'})
            else:raise
    return out


def jsonl(rel,rows):
    f=DEST/rel;f.parent.mkdir(parents=True,exist_ok=True)
    with f.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    counts[str(rel)]=len(rows)


def copy(source,rel):
    if not source.exists():return
    target=DEST/rel;target.parent.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.jsonl':jsonl(rel,lines(source))
    else:shutil.copyfile(source,target)


def question(dataset,ident,text,category):
    k=dataset+'|'+ident
    row=dict(id=ident,dataset=dataset,question=text,category=category)
    if k in questions:assert questions[k]==row
    else:questions[k]=row
    return k


for model in ['qwen25','gemma3','gemma27']:
    d=SOURCE/model
    descriptions=[]
    for row in lines(d/'av_results.jsonl'):
        assert row['hidden_state_index']==row['layer']+1
        q=question(row['source'],row['base_id'],row['prompt'],row['category'])
        query_id=hashlib.sha256(row['query'].encode()).hexdigest();queries[query_id]=row['query']
        descriptions.append(dict(activation_id=row['id'],question_key=q,condition=row['condition'],
            case_id=row['case_id'],layer=row['layer'],hidden_state_index=row['hidden_state_index'],
            layer_index_convention=row['layer_index_convention'],position=row['position'],position_family=row['position_family'],
            selected_token_id=row['selected_token_id'],selected_token_text=row['selected_token_text'],
            activation_mode=row['activation_mode'],source_model_revision=row['source_model_revision'],
            query_id=query_id,explanation=row['nla_output'],finish_reason=row['finish_reason'],
            output_tokens=row['output_tokens'],injection_scale=row['injection_scale'],
            original_activation_path=row['activation_path']))
    assert len(descriptions)==4710
    jsonl(f'models/{model}/nla_descriptions.jsonl',descriptions)
    source_answers=[]
    for row in lines(d/'answers.jsonl'):
        source_answers.append({k:v for k,v in row.items() if k not in ['answer_token_ids','messages']})
    jsonl(f'models/{model}/source_answers.jsonl',source_answers)
    if (d/'scored.jsonl').exists():
        fields=['id','base_id','case_id','condition','layer','hidden_state_index','position_family',
                'recon_mse','recon_cos','question_text_control_mse','question_text_control_cos','interpretation']
        jsonl(f'models/{model}/ar_scalar_scores.jsonl',[{k:r[k] for k in fields if k in r} for r in lines(d/'scored.jsonl')])
    for name in ['summary.json','capture_status.json','pipeline_status.json','nla_pipeline_status.json','ao_pipeline_status.json','ar_status.json','av_config.yaml','ao_execution_policy.json','av_results.timing.json']:
        copy(d/name,Path('models')/model/name)
    if model=='gemma27':
        ao=[]
        for row in lines(d/'ao_results.jsonl'):
            q=question(row['source'],row['base_id'],row['prompt'],row['category'])
            query_id=hashlib.sha256(row['ao_prompt_rendered'].encode()).hexdigest();queries[query_id]=row['ao_prompt_rendered']
            ao.append(dict(id=row['id'],question_key=q,condition=row['condition'],layer=row['layer'],
                hidden_state_index=row['hidden_state_index'],requested_window_size=row['requested_window_size'],
                window_size=row['window_size'],source_positions=row['source_positions'],
                source_token_ids=row['source_token_ids'],query_id=query_id,explanation=row['ao_output'],
                source_model_revision=row['source_model_revision'],ao_model_revision=row['ao_model_revision'],
                original_activation_path=row['activation_path']))
        assert len(ao)==4710;jsonl('models/gemma27/ao_descriptions.jsonl',ao)
assert len(questions)==785
jsonl('questions_785.jsonl',list(questions.values()));save('reader_queries.json',queries)

for name in ['execution_plan.md','verbalizer_protocol_audit.json','capture_verification.json','protocol_checksums.json',
             'runtime_source_with_gemma27.json','download_specs.json','adapter_config.json','ao_config.json']:
    copy(SOURCE/name,Path('protocols')/name)
for sub in ['analysis','ao_analysis','ao_prompt_probe','gold_span_probe','source_false_premise_probe']:
    d=SOURCE/sub
    for name in ['summary.json','aggregate_summary.json','qualitative_summary.json','matched_qualitative_summary.json',
                 'qualitative_annotations.json','native_behavior_audit.json','native_bytewise_verification.json',
                 'native_trace_summary.json','inspection_report.md','protocol.md','runtime.json','artifact_checksums.json']:
        copy(d/name,Path('earlier_studies')/sub/name)
    for name in ['paired_comparison.jsonl','annotated_comparison.jsonl','review_evidence.jsonl']:
        f=d/name
        if f.exists() and f.stat().st_size<4*1024**2:copy(f,Path('earlier_studies')/sub/name)

pilot=SOURCE/'gepa_reader_pilot'
for name in ['plan.json','protocol.md','protocol_checksums.json','gepa_contract_verification.json','gpt_runtime.json',
             'haiku_runtime.json','pipeline_status.json','supervisor_status.json','recovery_20261013.json',
             'recovery_20261014.json','hardware_recovery_20261013.json']:
    copy(pilot/name,Path('gepa_pilot')/name)
for f in (pilot/'cpu_analysis_20261013').glob('*'):
    if f.is_file():copy(f,Path('gepa_pilot/cpu_analysis_20261013')/f.name)
for f in (pilot/'cpu_analysis_20261013/trajectory_audits').glob('*.json'):
    copy(f,Path('gepa_pilot/cpu_analysis_20261013/trajectory_audits')/f.name)
for f in (pilot/'analysis').glob('*.json'):
    copy(f,Path('gepa_pilot/initial100_audit')/f.name)
native=[]
for f in (pilot/'native/results').glob('*.json'):
    r=json.loads(f.read_text())
    if 'row_id' in r:native.append(r)
jsonl('gepa_pilot/native_answers.jsonl',native)
jsonl('gepa_pilot/judgments.jsonl',[json.loads(f.read_text()) for f in (pilot/'judgments').glob('*.json')])
for dataset in ['cancer_myth','crepe']:
    for arm in ['gepa','text','nla','ao']:
        d=pilot/'optimization'/dataset/arm
        for name in ['selected.json','evaluations.jsonl','engine/candidates.json','engine/run_log.json']:
            copy(d/name,Path('gepa_pilot/optimization')/dataset/arm/name)
        for f in (d/'extras').glob('*.json'):copy(f,Path('gepa_pilot/optimization')/dataset/arm/'extras'/f.name)
for name in ['comparison_summary.json','comparison_summary.csv','frozen_candidates.json','heldout_validation_scored.json']:
    copy(pilot/name,Path('gepa_pilot')/name)
controls=SOURCE/'reader_feedback_controls_20261014'
for name in ['protocol.json','actual_protocol_verification.json','controller_status.json','status.json','comparison_summary.json','comparison_summary.csv']:
    copy(controls/name,Path('registered_controls_20261014')/name)
for dataset in ['cancer_myth','crepe']:
    for arm in ['nla_shuffled','ao_shuffled']:
        d=controls/'optimization'/dataset/arm
        for name in ['selected.json','evaluations.jsonl','engine/candidates.json','engine/run_log.json']:
            copy(d/name,Path('registered_controls_20261014/optimization')/dataset/arm/name)
        for f in (d/'extras').glob('*.json'):
            copy(f,Path('registered_controls_20261014/optimization')/dataset/arm/'extras'/f.name)
for name in ['frozen_candidates.json','heldout_scored.json']:
    copy(controls/name,Path('registered_controls_20261014')/name)
for f in (controls/'ao_query_probe').rglob('*.json'):
    if 'audit_calls' not in f.parts:
        copy(f,Path('registered_controls_20261014/ao_query_probe')/f.relative_to(controls/'ao_query_probe'))
for f in (SOURCE/'figures').glob('*'):
    if f.suffix in ['.png','.svg','.pdf']:copy(f,Path('figures')/f.name)

script_names=['analyze_reader_gepa_snapshot.py','audit_reader_gepa_trajectories.py','summarize_reader_gepa_audits.py',
    'plot_reader_gepa_snapshot.py','plot_reader_gepa_method.py','reader_gepa_io.py','reader_gepa_native_worker.py',
    'reader_gepa_verbalizer_worker.py','run_reader_gepa_pilot.py','supervise_reader_gepa_pilot.py',
    'reader_feedback_controls.py','supervise_reader_feedback_controls.py','run_ao_stance_query_probe.py',
    'verify_reader_protocol_snapshot.py','package_reader_experiments.py','summarize_ao_query_probe.py',
    'nla_generation_capture.py','nla_connection.py']
for name in script_names:copy(ROOT/'scripts'/name,Path('code/scripts')/name)
for name in ['paper_baselines.py','fpqa_cli_backend.py','fpqa_prompting.py']:
    copy(ROOT/'src'/name,Path('code/src')/name)
copy(ROOT/'docs/reviews/paper_tables_2026-10-06/table2_prompt_registry_2026-10-07.json',
     'protocols/table2_prompt_registry_2026-10-07.json')
tensor_index=[]
for parent in [SOURCE/'qwen25',SOURCE/'gemma3',SOURCE/'gemma27',pilot/'native/activations']:
    files=[f for f in parent.rglob('*.pt') if f.is_file()]
    tensor_index.append(dict(original_directory=str(parent),pt_files=len(files),bytes=sum(f.stat().st_size for f in files)))
save('external_tensor_index.json',tensor_index)
pipeline=json.loads((pilot/'pipeline_status.json').read_text())
snapshot=dict(created_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
    source_worktree=str(ROOT),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    snapshot_type='Text results, settings and code; live stages may be incomplete.',counts=counts,
    selected_conditions=len(list((pilot/'optimization').rglob('selected.json'))),current_pipeline=pipeline,
    controls_status=json.loads((controls/'controller_status.json').read_text()),
    incomplete_trailing_appends_omitted=omitted,
    not_copied=['Model checkpoints','Activation/reconstruction tensors','Provider raw events and auth files','GEPA pickle engine state'],
    portability='Saved text results are readable independently; executing the code needs the pinned base repository, official reader dependencies and configured model/cache paths.')
save('SNAPSHOT.json',snapshot)
readme='''NLA·AO 실험 결과 묶음 — 2026-10-14 스냅샷

먼저 볼 파일: gepa_pilot/cpu_analysis_20261013/report.html
점수 CSV: gepa_pilot/cpu_analysis_20261013/selected_dev_scores.csv
새 대조 계획: registered_controls_20261014/protocol.json
실제 토큰·레이어 검증: registered_controls_20261014/actual_protocol_verification.json

포함: 고정785문항, Qwen2.5-7B/Gemma3-12B/27B의 NLA 설명 각4710건,
Gemma27 AO 설명4710건, Qwen/Gemma12 AR 스칼라 점수, 당시 답변과 탐색 분석,
GEPA 답변·채점·후보 프롬프트·진행 상태, 재현용 코드 스냅샷.

주의: Gemma27의 초기785문항×2조건은 hidden-state-only 추출이어서
models/gemma27/source_answers.jsonl의 answer가 비어 있다. 실제 생성 답변은
gepa_pilot/native_answers.jsonl 및 별도 소규모 검증 결과에 있다.

dev 점수는 후보 선택 자료이며 최종 test 결과가 아니다. 보고서의2026-10-13
분석은 당시6조건 완료 시점의 기록이다. 현재 상태는 SNAPSHOT.json을 본다.
미완료 AO/Haiku/새 대조는 완료된 결과로 해석하지 않는다.
AR 값은 활성 방향 복원 지표이며 답변 교정이나 설명 사실성 점수가 아니다.
GPT 탐색 판정과 주석 불일치 표시는 사람 검증된 정답이 아니다.

hidden state·모델 가중치는 external_tensor_index.json의 원본 위치에 보존했다.
코드 실행에는 원본 저장소와 공식 NLA/AO 의존성 및 고정 캐시 경로가 필요하다.
이 묶음에는 로그인 정보·API 인증 파일을 포함하지 않았다.

서버에서 갱신:
cd /home/eagle0914/cancer-myth/external/paper_baselines_b81ace2
python3 scripts/package_reader_experiments.py
'''
(DEST/'README.txt').write_text(readme)
files=[]
for f in sorted(DEST.rglob('*')):
    if f.is_file() and f.name!='MANIFEST.sha256.json':
        files.append(dict(path=str(f.relative_to(DEST)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
save('MANIFEST.sha256.json',files)
archive=DEST.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(DEST,arcname=DEST.name)
print(json.dumps({'directory':str(DEST),'files':len(files)+1,'uncompressed_MB':round(sum(r['bytes'] for r in files)/1024**2,2),
                  'archive':str(archive),'archive_MB':round(archive.stat().st_size/1024**2,2),
                  'selected_conditions':snapshot['selected_conditions']},ensure_ascii=False,indent=2))
