"""Score responses with the Cancer-Myth judge prompts (validate.py, validate_nfp.py).

FPQ rows get Sharpness in {-1, 0, +1} (PCR = share of +1, PCS = mean); NFP
rows get {-1, +1} from validate_nfp.py (NFP = share of +1); TPQ rows are
scored with the NFP rubric using the annotated true presupposition as the
"possible hallucination".

Transport is `--backend codex` (default; `codex exec`, no API key) or
`--backend openai`. The paper's judge was GPT-4o; whatever judge is used
here must first pass scripts/calibrate_judge.py against the GPT-4o scores
stored in Cancer-Myth's all_data.json, and one judge must score every
condition of one table.

Resumable by id, provenance in every row, a lock so two judges never
append to the same file, --dry-run prices the job.
"""

from __future__ import annotations

import argparse
import atexit
import datetime as dt
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import construct_prompt_fpq, construct_prompt_nfp, parse_score
from src.llm_backend import check_judge_identity, make_caller

CHARS_PER_TOKEN = 4.0


def build_prompt(row: dict, q: dict, examples_fpq: list, examples_nfp: list) -> tuple[str, str]:
    if row["set"] == "fpq":
        return "fpq", construct_prompt_fpq(q["question"], q["correction"], row["response"], examples_fpq)
    if row["set"] == "nfp":
        return "nfp", construct_prompt_nfp(q["question"], q["hallucination_text"], row["response"], examples_nfp)
    if row["set"] == "tpq":
        return "nfp", construct_prompt_nfp(q["question"], q["premise_text"], row["response"], examples_nfp)
    raise ValueError(f"unknown set {row['set']}")


def acquire_lock(out_path: Path) -> None:
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"another judge is writing {out_path} ({lock_path} exists). Wait, or if none is "
            f"running: rm {lock_path}"
        )
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    atexit.register(lambda: lock_path.unlink(missing_ok=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=["codex", "openai"], default=None, help="default: config judge.backend")
    parser.add_argument("--model", default=None, help="default: config judge.model (openai) / codex default")
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=None, help="openai only")
    parser.add_argument("--paper-protocol", action="store_true", help="openai only: temperature 0.7 as in validate.py")
    parser.add_argument("--allow-same-family", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    judge_cfg = cfg["judge"]
    backend = args.backend or judge_cfg.get("backend", "codex")
    if args.model is not None:
        model = args.model
    else:
        model = judge_cfg.get("model", "gpt-4o") if backend == "openai" else judge_cfg.get("codex_model", "")
    temperature = 0.7 if args.paper_protocol else (
        args.temperature if args.temperature is not None else float(judge_cfg.get("temperature", 0.0))
    )
    examples_fpq = load_json(judge_cfg["examples_fpq"])
    examples_nfp = load_json(judge_cfg["examples_nfp"])

    questions = {q["id"]: q for q in read_jsonl(args.questions)}
    by_text: dict[str, list[dict]] = {}
    for q in questions.values():
        by_text.setdefault(q["question"], []).append(q)

    rows = list(read_jsonl(args.responses))
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.output)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    jobs = []
    for row in rows:
        targets = by_text.get(row["question"]) or [questions[row["id"]]]
        for q in targets:
            job_id = f"{row['id']}::{q['id']}"
            if job_id in done:
                continue
            kind, prompt = build_prompt({**row, "set": q["set"]}, q, examples_fpq, examples_nfp)
            jobs.append((job_id, row, q, kind, prompt))
    print(f"[judge] {len(jobs)} prompts to score ({len(done)} done) via {backend} model={model or 'backend default'} -> {out_path}", flush=True)

    if args.dry_run:
        tokens = sum(int(len(p) / CHARS_PER_TOKEN) for *_, p in jobs)
        print(f"[dry-run] ~{tokens:,} input tokens over {len(jobs)} calls")
        return
    if not jobs:
        return

    check_judge_identity(model, args.allow_same_family)
    call = make_caller(backend, model, timeout=args.timeout, codex_cmd=args.codex_cmd,
                       temperature=temperature, max_tokens=int(judge_cfg.get("max_tokens", 400)))
    acquire_lock(out_path)
    failures, consecutive = 0, 0
    for n, (job_id, row, q, kind, prompt) in enumerate(jobs, start=1):
        text, model_used = "", model
        for attempt in range(4):
            try:
                text, model_used = call(prompt)
                break
            except Exception as exc:  # noqa: BLE001 - rate limits, transient errors
                wait = 2 ** attempt
                print(f"[judge] {job_id}: {exc!r}; retry in {wait}s", flush=True)
                time.sleep(wait)
        if not text:
            failures += 1
            consecutive += 1
            print(f"[judge] {job_id}: FAILED, skipped (rerun to retry)", file=sys.stderr)
            if consecutive >= 3:
                raise SystemExit("three consecutive failures -- backend/model not usable; fix and rerun")
            continue
        consecutive = 0
        score, parsed = parse_score(text)
        append_jsonl(
            out_path,
            {
                "id": job_id,
                "response_id": row["id"],
                "question_id": q["id"],
                "set": q["set"],
                "rubric": kind,
                "sharpness": int(score.get("Sharpness", 1)),
                "reason": score.get("Reason"),
                "judge_parsed": parsed,
                "judge_raw": text,
                "judge_backend": backend,
                "judge_model": model_used,
                "judge_temperature": temperature if backend == "openai" else None,
                "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "response_model_id": row.get("model_id"),
            },
        )
        if args.sleep:
            time.sleep(args.sleep)
        if n % 50 == 0 or n == len(jobs):
            print(f"[judge] {n}/{len(jobs)} (failures {failures})", flush=True)
    print(f"[done] {out_path}")
    if failures:
        raise SystemExit(f"{failures} rows failed; rerun the same command to retry them")


if __name__ == "__main__":
    main()
