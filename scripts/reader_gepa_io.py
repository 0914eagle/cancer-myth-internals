"""Auditable cached provider calls and file-queue model inference for the pilot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paper_baselines import digest
from src.fpqa_cli_backend import CLIBackend
from src.fpqa_prompting import parse_rating, well_response_judge


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_prompt_tokenizer(baseline):
    # Resolve the pinned snapshot first. Passing a hub ID can trigger a remote
    # model-info lookup in Transformers' regex check despite local_files_only.
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(repo_id=baseline['gepa']['prompt_tokenizer'],
        revision=baseline['gepa']['tokenizer_revision'], cache_dir=baseline['cache_root'], local_files_only=True)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


class PilotIO:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.c = json.loads(self.config_path.read_text())
        self.out = Path(self.c['artifact_root'])
        self.baseline = json.loads(Path(self.c['baseline_config']).read_text())
        self.registry = json.loads((ROOT / self.baseline['registry']).read_text())
        self.gpt = json.loads((self.out / 'gpt_runtime.json').read_text())
        self.reader_proc = None; self.reader_kind = None; self.reader_log = None
        old = Path(self.c['original_artifact_root'])
        self.original = {
            'nla': {r['id']: r for r in (json.loads(s) for s in (old / 'gemma27/av_results.jsonl').read_text().splitlines())},
            'ao': {r['id']: r for r in (json.loads(s) for s in (old / 'gemma27/ao_results.jsonl').read_text().splitlines())
                   if r['requested_window_size'] == 20},
        }
        # AO output IDs differ from the manifest ID. Index by immutable source path.
        self.ao_by_path = {r['activation_path']: r for r in self.original['ao'].values()}

    def external(self, role, messages):
        config = self.gpt[role + '_model']
        ident = digest([role, config, messages, 'reader_gepa_gpt_v1'])
        path = self.out / 'external_calls' / (ident + '.json')
        if path.exists(): return json.loads(path.read_text())['answer']
        began = time.perf_counter()
        try:
            answer, metadata = CLIBackend(config)(messages)
        except Exception as error:
            save(self.out / 'external_errors' / (ident + '.json'), dict(role=role, config=config,
                error=repr(error), timestamp=time.time()))
            raise
        save(path, dict(role=role, config=config, messages=messages, answer=answer,
                        metadata=metadata, seconds=time.perf_counter() - began))
        return answer

    def native(self, row, prompt):
        messages = [{'role': 'system', 'content': self.registry['transport_common'] + '\n\n' + prompt},
                    {'role': 'user', 'content': row['question']}]
        ident = digest([self.baseline['models']['gemma27'], messages, 'native_reader_gepa_v1'])
        request = self.out / 'native/requests' / (ident + '.json')
        if not request.exists(): save(request, dict(id=ident, row=row, messages=messages, candidate_prompt=prompt))
        response = self.out / 'native/results' / (ident + '.json')
        began = time.monotonic()
        while not response.exists():
            error = response.with_name(ident + '.error.json')
            if error.exists(): raise RuntimeError(error.read_text())
            if time.monotonic() - began > 1800: raise TimeoutError('Native worker request exceeded1800seconds: ' + ident)
            time.sleep(1)
        result = json.loads(response.read_text())
        assert result['messages'] == messages and result['source_model_revision'] == self.baseline['models']['gemma27']['revision']
        return result

    def judge(self, row, native):
        path = self.out / 'judgments' / (native['id'] + '.json')
        if path.exists(): return json.loads(path.read_text())
        if not native['completed']:
            result = dict(score=0, reason='Incomplete native answer; no judge call.', native_id=native['id'], incomplete=True)
        else:
            text = self.external('judge', well_response_judge(ROOT / self.baseline['well_root'], row, native['answer']))
            score = parse_rating(text)
            if score is None: raise ValueError('Unparseable pinned Well judge result')
            result = dict(score=score, reason=text, native_id=native['id'], incomplete=False)
        save(path, result)
        return result

    def stop_reader(self):
        if self.reader_proc is not None:
            save(self.out / 'readers' / self.reader_kind / 'STOP', {'reason': 'Switch/completion of this owned pilot reader'})
            try: self.reader_proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.reader_proc.terminate(); self.reader_proc.wait(timeout=60)
            if self.reader_log: self.reader_log.close()
            self.reader_proc = None; self.reader_kind = None; self.reader_log = None

    def start_reader(self, kind):
        if self.reader_kind == kind and self.reader_proc is not None and self.reader_proc.poll() is None: return
        self.stop_reader()
        directory = self.out / 'readers' / kind; directory.mkdir(parents=True, exist_ok=True)
        (directory / 'STOP').unlink(missing_ok=True)
        status = directory / 'status.json'; status.unlink(missing_ok=True)
        python = '/data3/heejae/uv/medical_nla/bin/python' if kind == 'nla' else '/data3/heejae/uv/fpqa_ao_20261012/bin/python'
        self.reader_log = (directory / 'worker.log').open('a')
        self.reader_proc = subprocess.Popen([python, '-u', 'scripts/reader_gepa_verbalizer_worker.py',
            '--config', str(self.config_path), '--kind', kind, '--gpus', '2,3'], cwd=ROOT,
            env={**os.environ, 'HF_HOME': '/data3/heejae/hf_cache', 'HF_HUB_OFFLINE': '1'},
            stdout=self.reader_log, stderr=self.reader_log)
        self.reader_kind = kind

    def reader(self, kind, native):
        reuse = native['reader_reuse'].get(kind, {})
        if reuse.get('bitwise_equal'):
            original = self.original['nla'][reuse['original_activation_id']] if kind == 'nla' else self.ao_by_path[reuse['original_activation_path']]
            result = dict(explanation=original['nla_output' if kind == 'nla' else 'ao_output'], reused=True,
                original_output_id=original['id'], source_native_id=native['id'], bytewise_verified=True,
                query=original['query' if kind == 'nla' else 'ao_prompt_rendered'])
            assert kind == 'nla' or original['ao_prompt_rendered'].endswith('<start_of_turn>model\n')
            return result
        ident = digest([kind, native['id'], 'reader_gepa_fixed_query_v1'])
        response = self.out / 'readers' / kind / 'results' / (ident + '.json')
        if response.exists(): return json.loads(response.read_text())
        request = self.out / 'readers' / kind / 'requests' / (ident + '.json')
        save(request, dict(id=ident, native_id=native['id'], activation_path=native[kind + '_activation_path']))
        self.start_reader(kind)
        began = time.monotonic()
        while not response.exists():
            if self.reader_proc.poll() is not None: raise RuntimeError(kind + ' reader worker exited; inspect worker.log')
            if time.monotonic() - began > 600: raise TimeoutError(kind + ' reader request timed out')
            time.sleep(1)
        return json.loads(response.read_text())

    def extra(self, arm, row, native, judgment):
        if arm == 'gepa': return None
        if arm in ('nla', 'ao'): return self.reader(arm, native)
        if arm != 'text': raise ValueError('Unknown arm')
        inputs = dict(question=row['question'], answer=native['answer'], label=row['label'],
                      rating=judgment['score'], grading_reason=judgment['reason'])
        explanation = self.external('text_explanation', [
            {'role': 'system', 'content': 'Describe diagnostic patterns in the supplied question, answer and grading feedback for prompt optimization. Use100-150words. Distinguish observed behavior from speculation. Identify the relevant premise, answer focus and correction or over-rebuttal behavior. Do not claim access to hidden states. Do not write a new prompt.'},
            {'role': 'user', 'content': json.dumps(inputs, ensure_ascii=False)}])
        return dict(explanation=explanation, source='Question/answer/grade only; no activation', reused=False)
