"""Training audit and four matched GEPA arms; frozen held-out pilot validation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reader_gepa_io import PilotIO, save, load_prompt_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    a = parser.parse_args()
    c = json.loads(a.config.read_text()); out = Path(c['artifact_root'])
    baseline = json.loads(Path(c['baseline_config']).read_text())
    bundle = json.loads(Path(c['bundle']).read_text())
    lookup = {r['id']: r for r in bundle['rows']}
    splits = {}
    for dataset, name in [('cancer_myth', 'fold0'), ('crepe', 'official')]:
        split = next(s for s in bundle['splits'] if (s['dataset'], s['name']) == (dataset, name))
        train_ids = [r['id'] for r in c['cohort'] if r['dataset'] == dataset]
        dev_ids = []; validation_ids = []; seen = set()
        for label in [1, 0]:
            pool = sorted([lookup[i] for i in split['dev'] if lookup[i]['label'] == label], key=lambda r: r['id'])
            random.Random(42).shuffle(pool); chosen = []
            for row in pool:
                if row['group_id'] in seen: continue
                seen.add(row['group_id']); chosen.append(row['id'])
                if len(chosen) == 8: break
            assert len(chosen) == 8
            dev_ids += chosen[:4]; validation_ids += chosen[4:]
        assert len(train_ids) == 50 and len(dev_ids) == len(validation_ids) == 8
        assert set(train_ids) <= set(split['train'])
        assert set(dev_ids + validation_ids) <= set(split['dev'])
        assert not set(train_ids) & set(dev_ids + validation_ids)
        assert not set(dev_ids) & set(validation_ids)
        assert not set(train_ids + dev_ids + validation_ids) & set(split['test'])
        splits[dataset] = dict(train_ids=train_ids, candidate_dev_ids=dev_ids, heldout_validation_ids=validation_ids,
            source_split=name, main_protocol_dev_limit_unchanged=True)
    plan = dict(protocol='reader_gepa_training_pilot_v1', stage='pilot_not_main_table', models=['gemma27'],
        arms=['gepa', 'text', 'nla', 'ao'], splits=splits, seed=42,
        starting_prompt=json.loads((ROOT / baseline['registry']).read_text())['prompts']['plain'],
        question_metric_budget_per_dataset_arm=100, pair_metric_budget=50,
        max_candidate_proposals=4, reflection_minibatch_pairs=2, prompt_token_limit=512,
        diagnostic_token_limit=512, source_batch_size=1, source_max_new_tokens=8192,
        additional_readers='Only opt_train evaluator feedback; candidate dev/heldout validation use answer scores only',
        final_validation='Fixed8 held-out official/fold dev questions per dataset, Haiku only after all GPT-selected candidates frozen; not official test',
        gpt=json.loads((out / 'gpt_runtime.json').read_text()),
        library='gepa0.1.1 optimize_anything, same algorithm/limits/seed across arms',
        engine_cache_evaluation=False,
        cache_note='Installed0.1.1 cache key(candidate_hash,integer example_id) collides across train/dev list loaders; disabled uniformly. Native/judge caches key actual messages/questions.',
        source_hardware='GPU0,1 persistent Gemma27; GPU2,3 persistent NLA/AO when needed',
        main_protocol_note='Pilot uses4+4 candidate dev; it does not replace planned CREPE100+100/main baseline dev selection.')
    plan_path = out / 'plan.json'
    if plan_path.exists(): assert json.loads(plan_path.read_text()) == plan
    else: save(plan_path, plan)
    if a.prepare_only:
        print('Prepared train50/dev8/heldout8 per dataset;4arms;100question metric cap per arm/dataset', flush=True)
        return
    io = PilotIO(a.config)
    status = out / 'pipeline_status.json'
    def state(stage, **kwargs): save(status, dict(state='running', stage=stage, updated_utc=time.time(), **kwargs))
    def parse_json(text):
        text = text.strip()
        if text.startswith('```'): text = '\n'.join(text.splitlines()[1:-1])
        return json.loads(text)
    try:
        # Obtain actual answers before attributing the previously saved explanations.
        analysis_dir = out / 'analysis'; analysis_dir.mkdir(exist_ok=True)
        audit_records = []
        for index, sample in enumerate(c['cohort']):
            state('plain100_audit', complete=index, expected=100)
            row = lookup[sample['id']]
            native = io.native(row, plan['starting_prompt'])
            judgment = io.judge(row, native)
            nla = io.reader('nla', native); ao = io.reader('ao', native)
            path = analysis_dir / (row['id'].replace(':', '_') + '.json')
            if path.exists(): audit = json.loads(path.read_text())
            else:
                payload = dict(question=row['question'], label=row['label'], presuppositions=row['presuppositions'],
                    reference_answer=row['reference_answer'], actual_answer=native['answer'],
                    rating=judgment['score'], grading_reason=judgment['reason'],
                    nla_explanation=nla['explanation'], ao_explanation=ao['explanation'])
                system = ('Audit explanations of pre-answer activations against the supplied question, reference, actual answer and grading feedback. '
                    'Reader outputs are unverified hypotheses. Do not assume named nodes or source beliefs were measured. '
                    'Return only JSON with annotation_issue(boolean), annotation_issue_reason(string), '
                    'actual_behavior(one of target_corrected,target_not_corrected,normal_answer,premise_challenged,uncertain), '
                    'nla and ao(each an object with topic_match(boolean), specific_target_error_identified(boolean), '
                    'behavior_agrees_with_actual(boolean or null), adds_diagnostic_information_beyond_text(boolean), evidence(string)). '
                    'For NFP, a topic summary is not proof the source deemed the premise sound. '
                    'Generic correction language is not specific target error identification. Extra information must add something beyond the visible answer and grading reason. '
                    'Flag question/annotation mismatches; your audit is not ground truth or causal evidence.')
                raw = io.external('judge', [{'role': 'system', 'content': system},
                                            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
                try:
                    analysis = parse_json(raw)
                    assert isinstance(analysis, dict) and isinstance(analysis['annotation_issue'], bool)
                    assert isinstance(analysis['annotation_issue_reason'], str)
                    assert analysis['actual_behavior'] in {'target_corrected', 'target_not_corrected', 'normal_answer', 'premise_challenged', 'uncertain'}
                    for kind in ['nla', 'ao']:
                        fields = analysis[kind]
                        assert isinstance(fields, dict) and isinstance(fields['evidence'], str)
                        for field in ['topic_match', 'specific_target_error_identified', 'adds_diagnostic_information_beyond_text']:
                            assert isinstance(fields[field], bool)
                        assert fields['behavior_agrees_with_actual'] is None or isinstance(fields['behavior_agrees_with_actual'], bool)
                except Exception:
                    analysis = dict(format_invalid=True, raw=raw)
                audit = dict(row_id=row['id'], dataset=row['dataset'], label=row['label'], question=row['question'],
                    native=native, judgment=judgment, nla=nla, ao=ao, audit=analysis,
                    audit_author='GPT6.1sol exploratory audit, not an independent human judgment')
                save(path, audit)
            audit_records.append(audit)
        save(analysis_dir / 'summary.json', dict(state='complete', questions=100,
            all_completed_answers=sum(x['native']['completed'] for x in audit_records),
            bytewise_verified_nla_reuses=sum(x['nla'].get('reused', False) for x in audit_records),
            bytewise_verified_ao_reuses=sum(x['ao'].get('reused', False) for x in audit_records),
            gpt_annotation_issue_flags=sum(bool(x['audit'].get('annotation_issue')) for x in audit_records),
            limits='GPT-only diagnostic opinions; same100 train questions; no causal/knowledge claims; do not select only positive cases'))
        io.stop_reader()
        from transformers import AutoTokenizer
        from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything
        tokenizer = load_prompt_tokenizer(baseline)
        def pairs(ids, partition):
            groups = [[lookup[i] for i in ids if lookup[i]['label'] == y] for y in [1, 0]]
            assert len(groups[0]) == len(groups[1])
            for group in groups: random.Random(42).shuffle(group)
            return [dict(pair=[x, y], partition=partition) for x, y in zip(*groups)]
        selections = {}
        for arm in plan['arms']:
            for dataset in ['cancer_myth', 'crepe']:
                state('gepa_optimization', arm=arm, dataset=dataset)
                directory = out / 'optimization' / dataset / arm; directory.mkdir(parents=True, exist_ok=True)
                selected = directory / 'selected.json'
                if selected.exists():
                    selections[(dataset, arm)] = json.loads(selected.read_text()); continue
                train = pairs(splits[dataset]['train_ids'], 'train')
                dev = pairs(splits[dataset]['candidate_dev_ids'], 'dev')
                def evaluate(candidate, example):
                    if set(candidate) != {'answer'}: raise ValueError('Candidate components changed')
                    if len(tokenizer.encode(candidate['answer'], add_special_tokens=False)) > 512:
                        return 0., {'error': 'Candidate exceeds fixed512token limit', 'valid': False}
                    feedback = []
                    for row in example['pair']:
                        native = io.native(row, candidate['answer'])
                        judgment = io.judge(row, native)
                        info = dict(id=row['id'], label=row['label'], question=row['question'],
                            answer=native['answer'], rating=judgment['score'], reason=judgment['reason'])
                        if example['partition'] == 'train':
                            extra = io.extra(arm, row, native, judgment)
                            if extra is not None:
                                tokens = tokenizer.encode(extra['explanation'], add_special_tokens=False)
                                info['additional_diagnostic'] = tokenizer.decode(tokens[:512])
                                info['diagnostic_source'] = arm
                                info['diagnostic_clipped'] = len(tokens) > 512
                                save(directory / 'extras' / (native['id'] + '.json'), dict(extra,
                                    supplied_diagnostic=info['additional_diagnostic'], clipped=len(tokens) > 512))
                        feedback.append(info)
                    score = sum(x['rating'] / 5 for x in feedback) / 2
                    event = dict(candidate=candidate, partition=example['partition'], score=score, examples=feedback, timestamp=time.time())
                    with (directory / 'evaluations.jsonl').open('a') as stream: stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                    return score, {'examples': feedback}
                def reflect(messages):
                    if isinstance(messages, str): messages = [{'role': 'user', 'content': messages}]
                    if not any(m['role'] == 'system' for m in messages):
                        messages = [{'role': 'system', 'content': 'Improve the supplied prompt using training feedback. Return the requested candidate.'}] + messages
                    if [m['role'] for m in messages] != ['system', 'user']:
                        messages = [{'role': 'system', 'content': 'Improve the prompt following the supplied optimization conversation.'},
                                    {'role': 'user', 'content': json.dumps(messages, ensure_ascii=False)}]
                    return io.external('reflection', messages)
                result = optimize_anything(seed_candidate={'answer': plan['starting_prompt']}, evaluator=evaluate,
                    dataset=train, valset=dev,
                    objective='Maximize balanced FPQ correction and normal-question preservation. Only questions are available at inference.',
                    background='Optimize only the answer prompt. No tools or retrieval. The prompt must fit512tokenizer tokens. '
                        'Additional diagnostic text, when present, may be unreliable and is not gold evidence of source beliefs. '
                        'Use reference-grounded grading feedback to assess it. Do not copy example-specific reference facts into a general instruction.',
                    config=GEPAConfig(engine=EngineConfig(run_dir=str(directory / 'engine'), seed=42,
                        max_metric_calls=50, max_candidate_proposals=4, parallel=False, max_workers=1,
                        cache_evaluation=False), reflection=ReflectionConfig(reflection_lm=reflect, reflection_minibatch_size=2)))
                candidate = result.best_candidate
                assert set(candidate) == {'answer'} and len(tokenizer.encode(candidate['answer'], add_special_tokens=False)) <= 512
                record = dict(candidate=candidate, best_idx=result.best_idx, selection='Frozen GPT candidate-dev best; heldout not accessed',
                    dataset=dataset, arm=arm, seed=42, budget_question_cap=100, max_candidate_proposals=4)
                save(selected, record); selections[(dataset, arm)] = record
                print('Selected', dataset, arm, 'candidate', result.best_idx, flush=True)
            io.stop_reader()
        # All candidates are frozen before any independent heldout answer/Haiku evaluation.
        save(out / 'frozen_candidates.json', {dataset: {arm: selections[(dataset, arm)] for arm in plan['arms']}
                                            for dataset in splits})
        io.stop_reader()
        haiku_path = out / 'haiku_runtime.json'
        validation_records = []
        for dataset in splits:
            for arm in plan['arms']:
                for ident in splits[dataset]['heldout_validation_ids']:
                    state('heldout_validation_generation', dataset=dataset, arm=arm, row_id=ident)
                    native = io.native(lookup[ident], selections[(dataset, arm)]['candidate']['answer'])
                    validation_records.append(dict(dataset=dataset, arm=arm, row_id=ident, native=native))
        save(out / 'heldout_validation_answers.json', validation_records)
        if not haiku_path.exists():
            save(status, dict(state='awaiting_haiku_configuration', stage='heldout_scoring',
                generated_answers=len(validation_records), detail='Candidates frozen and answers saved; no fallback judge'))
            return
        from src.fpqa_cli_backend import CLIBackend
        from src.fpqa_prompting import parse_rating, well_response_judge
        hc = json.loads(haiku_path.read_text())['judge_model']
        scored = []
        for item in validation_records:
            state('heldout_haiku_scoring', complete=len(scored), expected=len(validation_records))
            key = hashlib.sha256(json.dumps([hc, item['native']['id']], sort_keys=True).encode()).hexdigest()
            path = out / 'heldout_haiku' / (key + '.json')
            if path.exists(): judgment = json.loads(path.read_text())
            else:
                if item['native']['completed']:
                    answer, metadata = CLIBackend(hc)(well_response_judge(ROOT / baseline['well_root'], lookup[item['row_id']], item['native']['answer']))
                    score = parse_rating(answer)
                    if score is None: raise ValueError('Unparseable heldout Haiku result')
                    judgment = dict(score=score, reason=answer, metadata=metadata)
                else: judgment = dict(score=0, reason='Incomplete answer', incomplete=True)
                save(path, judgment)
            scored.append(dict(item, judgment=judgment, label=lookup[item['row_id']]['label']))
        save(out / 'heldout_validation_scored.json', scored)
        summary = []
        for dataset in splits:
            for arm in plan['arms']:
                group = [x for x in scored if (x['dataset'], x['arm']) == (dataset, arm)]
                fpq = [x['judgment']['score'] for x in group if x['label']]
                nfp = [x['judgment']['score'] for x in group if not x['label']]
                summary.append(dict(dataset=dataset, arm=arm, fpq_mean=sum(fpq) / len(fpq), nfp_mean=sum(nfp) / len(nfp),
                    balanced_normalized=(sum(fpq) / len(fpq) + sum(nfp) / len(nfp)) / 10,
                    fpq_ge4=sum(x >= 4 for x in fpq), nfp_ge4=sum(x >= 4 for x in nfp), fpq_n=len(fpq), nfp_n=len(nfp)))
        save(out / 'comparison_summary.json', dict(state='complete', results=summary,
            limits='Tiny heldout pilot dev subset; not official dataset test or main-table results; no candidate re-selection using Haiku'))
        with (out / 'comparison_summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        save(status, dict(state='complete', stage='heldout_pilot_comparison', records=len(scored), updated_utc=time.time()))
        save(out / 'native/STOP', {'reason': 'All authorized pilot stages complete'})
    except Exception as error:
        save(status, dict(state='failed', error=repr(error), updated_utc=time.time(), resumes_from_cached_artifacts=True))
        raise
    finally:
        io.stop_reader()


if __name__ == '__main__':
    main()
