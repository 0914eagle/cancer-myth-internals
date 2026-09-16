"""One text-in, text-out call through the OpenAI API, `codex exec`, or `claude -p`.

Follows medical_nla/scripts/run_judge.py. codex is an agent, not a completion
endpoint, so its stdout is a transcript; `--output-last-message` writes only
the final reply to a file, and the banner is scanned for the model actually
used (codex picks a default unless --model says otherwise, so "whatever codex
chose" is not a provenance record).

`claude` is the Claude Code CLI in print mode (`claude -p`): the prompt goes in
on stdin, every tool is disabled, nothing is persisted, and the JSON result
carries the reply plus the model that served it. It is a writer/aligner
backend (twins, paraphrases, span alignment); the reported judge stays the
one the table names, and the same identity check applies.

Neither the judge nor the aligner may be the backbone under study; the
family check refuses gemma/llama/qwen judges unless explicitly overridden.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

BACKBONE_MARKERS = ("gemma", "llama", "qwen", "nla-")


def check_judge_identity(model: str, allow_same_family: bool = False) -> None:
    lowered = (model or "").lower()
    if any(marker in lowered for marker in BACKBONE_MARKERS):
        if not allow_same_family:
            raise SystemExit(
                f"refusing: '{model}' looks like a backbone under study. Judging a model's "
                "responses with that family is the objection this project cannot survive. "
                "Pass --allow-same-family only for a sanity check, never for a reported number."
            )
        print(f"[warn] judge '{model}' shares a backbone family", file=sys.stderr)


def parse_codex_banner_model(*streams: str) -> str:
    for stream in streams:
        for line in (stream or "").splitlines()[:20]:
            stripped = line.strip()
            if stripped.startswith("model:"):
                return stripped.split(":", 1)[1].strip()
    return ""


def run_codex(prompt: str, model: str, timeout: int, codex_cmd: str = "codex") -> tuple[str, str]:
    """One read-only `codex exec`. Returns (answer, model_used)."""
    handle, out_file = tempfile.mkstemp(suffix=".codex.txt")
    os.close(handle)
    try:
        cmd = codex_cmd.split() + [
            "exec", "--sandbox", "read-only", "--ephemeral",
            "--skip-git-repo-check", "--output-last-message", out_file,
        ]
        if model:
            cmd += ["--model", model]
        cmd += ["-"]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            err_lines = [ln for ln in (proc.stderr + proc.stdout).splitlines() if "ERROR" in ln or "error" in ln.lower()]
            detail = " | ".join(err_lines[-3:]) if err_lines else proc.stderr[-300:]
            raise RuntimeError(f"codex exec failed ({proc.returncode}): {detail[:600]}")
        answer = Path(out_file).read_text(encoding="utf-8").strip()
        if not answer:
            raise RuntimeError(f"codex wrote no final message; stdout tail: {proc.stdout[-400:]!r}")
        return answer, parse_codex_banner_model(proc.stdout, proc.stderr) or (model or "codex-default")
    finally:
        Path(out_file).unlink(missing_ok=True)


def parse_claude_result(stdout: str, fallback_model: str) -> tuple[str, str]:
    """The last JSON object on stdout is the `--output-format json` result."""
    import json

    text = (stdout or "").strip()
    start = text.rfind("\n{")
    payload = text[start + 1:] if start >= 0 else text
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"claude -p returned no JSON result; stdout tail: {text[-400:]!r}") from exc
    if not isinstance(data, dict) or data.get("type") != "result":
        raise RuntimeError("claude -p JSON is not a result record")
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        raise RuntimeError(f"claude -p error ({data.get('subtype')}): {str(data.get('result', ''))[:300]}")
    answer = str(data.get("result") or "").strip()
    if not answer:
        raise RuntimeError("claude -p wrote an empty result")
    usage = data.get("modelUsage") or {}
    served = ",".join(sorted(k for k in usage if isinstance(k, str))) if isinstance(usage, dict) else ""
    return answer, served or fallback_model or "claude-default"


def run_claude(prompt: str, model: str, timeout: int, claude_cmd: str = "claude") -> tuple[str, str]:
    """One tool-less `claude -p`. Returns (answer, model_used)."""
    cmd = claude_cmd.split() + [
        "-p", "--bare", "--tools", "", "--no-session-persistence",
        "--output-format", "json", "--permission-mode", "default",
    ]
    if model:
        cmd += ["--model", model]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout)[-300:]
        raise RuntimeError(f"claude -p failed ({proc.returncode}): {detail}")
    return parse_claude_result(proc.stdout, model)


def run_openai(prompt: str, model: str, timeout: int, *, temperature: float = 0.0, max_tokens: int = 400) -> tuple[str, str]:
    from openai import OpenAI

    client = OpenAI(timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip(), model


def make_caller(
    backend: str,
    model: str,
    *,
    timeout: int = 180,
    codex_cmd: str = "codex",
    claude_cmd: str = "claude",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> Callable[[str], tuple[str, str]]:
    """Returns f(prompt) -> (text, model_used) for the chosen backend."""
    if backend == "codex":
        return lambda prompt: run_codex(prompt, model, timeout, codex_cmd)
    if backend == "claude":
        return lambda prompt: run_claude(prompt, model, timeout, claude_cmd)
    if backend == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("backend=openai but OPENAI_API_KEY is not set")
        return lambda prompt: run_openai(prompt, model, timeout, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"unknown backend {backend!r}")


def backend_available(backend: str, codex_cmd: str = "codex", claude_cmd: str = "claude") -> bool:
    import shutil

    if backend == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if backend == "codex":
        return shutil.which(codex_cmd.split()[0]) is not None
    if backend == "claude":
        return shutil.which(claude_cmd.split()[0]) is not None
    return False
