"""Execute pinned Well code unchanged; replace only model transport.

Run in a dedicated process: upstream uses top-level module names. RAG and API
SDK imports are replaced by fail-closed transport shims, not protocol rewrites.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from src.fpqa_cli_backend import CLIBackend
from src.fpqa_prompting import freeze, shared_call_slot

WELL_REVISION = "6c9770f65da6e9c50250cf94e35b07f258288e38"


class FatalTransportError(BaseException):
    """Escape GEPA's catch-and-continue proposal handler on infrastructure failure."""


def verify_source(root):
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != WELL_REVISION:
        raise ValueError(f"Unexpected Well revision: {revision}")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "*.py"], text=True):
        raise ValueError("Well Python sources were modified")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("GEPA", "response", "data_gen/template", "utils", "constant")
            for p in sorted((root / directory).glob("*.py"))}


class UncachedCaller:
    """Audit every invocation. No answer memoization: upstream cache default is off."""
    def __init__(self, config, out):
        self.config, self.out = config, Path(out)
        self.clients = {r: CLIBackend({**config[f"{r}_model"], "exact_well_messages": True})
                        for r in ("task", "judge", "reflection")}
        self.n = 0

    def __call__(self, role, messages):
        if (self.out / "STOP").exists():
            raise FatalTransportError("STOP requested")
        self.n += 1
        path = self.out / "calls" / f"{self.n:06d}_{role}.json"
        request = {"role": role, "config": self.config[f"{role}_model"], "messages": messages}
        freeze(path.with_suffix(".request.json"), request)
        try:
            with shared_call_slot(self.config[f"{role}_model"]["backend"]):
                answer, meta = self.clients[role](messages)
        except Exception as error:
            freeze(path.with_suffix(".error.json"), {"role": role, "error_type": type(error).__name__, "message": str(error)})
            freeze(self.out / "FAILED.json", {"role": role, "call": self.n, "error_type": type(error).__name__, "message": str(error)})
            raise FatalTransportError(f"{role} call {self.n} failed; aborting optimization") from error
        freeze(path, {**request, "text": answer, "metadata": meta})
        print(f"call {self.n}: {role} complete", flush=True)
        return answer


def load_upstream(root, call, record=None):
    """Import actual upstream evaluator, parser, templates and run_gepa function."""
    root = Path(root).resolve()
    sys.path.insert(0, str(root))
    role = ContextVar("well_model_role", default="judge")

    def unavailable(*a, **kw):
        raise RuntimeError("Only no_passages and CLI model transport are supported")

    def transport(messages, model, **kwargs):
        if kwargs.get("web_search"):
            unavailable()
        return SimpleNamespace(text=call(role.get(), messages))

    # Avoid installing CUDA, retrieval or API clients for the no-RAG experiment.
    shims = {
        "utils.RAG_utils": {"run_RAG": unavailable},
        "utils.gemini_utils": {
            "call_gemini_one_by_one_api": transport,
            "extract_gemini_response_text": lambda response: response.text,
            "save_gemini_thinking_trace": lambda *a, **kw: None,
        },
        "utils.openai_utils": {
            "DEFAULT_OPENAI_API_KEY": None,
            "DEFAULT_OPENAI_BASE_URL": "https://proxy.vectorinstitute.ai/v1",
            "call_openai_one_by_one_api": unavailable,
            "extract_openai_response_text": unavailable,
        },
    }
    for name, attrs in shims.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        sys.modules[name] = module
    balanced = importlib.import_module("GEPA.optimize_direct_qa_balanced")
    direct = importlib.import_module("GEPA.optimize_direct_qa")
    original_evaluator = balanced.BalancedDirectQAGEPAEvaluator
    original_reflector = balanced.create_reflection_lm

    class TransportEvaluator(original_evaluator):
        def run_task_model(self, messages):
            token = role.set("task")
            try:
                return super().run_task_model(messages)
            finally:
                role.reset(token)

        def __call__(self, candidate, example):
            result = super().__call__(candidate, example)
            if record is not None:
                record(candidate, example, result)
            return result

    def reflector(**kwargs):
        upstream_reflector = original_reflector(**kwargs)
        def run(prompt):
            token = role.set("reflection")
            try:
                return upstream_reflector(prompt)
            finally:
                role.reset(token)
        return run

    balanced.BalancedDirectQAGEPAEvaluator = TransportEvaluator
    balanced.create_reflection_lm = reflector
    return balanced, direct, original_evaluator


def make_args(balanced, family, data_dir, out, config):
    parser = argparse.ArgumentParser()
    # These are the original Gemini subparser's arguments; the model is replaced.
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--thinking_level", default=None)
    parser.add_argument("--web_search", action="store_true")
    balanced.add_common_args(parser)
    argv = ["--model_name", config["task_model"]["model"],
            "--evaluator_model_name", config["judge_model"]["model"],
            "--reflection_model_name", config["reflection_model"]["model"],
            "--dataset_family", family, "--out_dir", str(out),
            "--fpq_val_limit", "50", "--tpq_val_limit", "50",
            "--run_dir", str(out / "gepa")]
    for split in ("train", "val"):
        for kind in ("fpq", "tpq"):
            argv += [f"--{kind}_{split}_path", str(data_dir / f"{kind}_{split}.jsonl")]
    # Budget, seed, objective, background, caching and proposal settings all come
    # from the upstream parser. No parallel/feedback/prompt overrides.
    return parser.parse_args([*argv, "no_passages"])


def write_inputs(base, out):
    """Only schema conversion; preserve selected questions/references and order."""
    counts = {}
    for split, source in (("train", "opt_train"), ("val", "opt_dev")):
        rows = [json.loads(line) for line in (base / f"{source}.jsonl").read_text().splitlines() if line.strip()]
        for kind, label in (("fpq", 1), ("tpq", 0)):
            selected = [r for r in rows if r["label"] == label]
            if not selected:
                raise ValueError("Balanced optimization needs both labels")
            data = [{"id": r["id"], "question": r["question"],
                     "presuppositions": r["presuppositions"], "answer": r["reference_answer"],
                     "few_shot_data": [], "passages": []} for r in selected]
            path = out / f"{kind}_{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
            if path.exists() and path.read_text() != content:
                raise ValueError("Prepared source data changed")
            path.write_text(content)
            counts[f"{kind}_{split}"] = len(data)
    return counts
