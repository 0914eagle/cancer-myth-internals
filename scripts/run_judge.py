"""Score responses with the Cancer-Myth judge (GPT-4o, validate.py prompts).

FPQ rows get the Sharpness score in {-1, 0, +1} (PCR = share of +1, PCS =
mean); NFP rows get {-1, +1} from validate_nfp.py's rubric (NFP = share of
+1). TPQ rows are scored with the NFP rubric using the annotated true
presupposition as the "possible hallucination".

Following medical_nla/scripts/run_judge.py: resumable by id, provenance in
every row (judge model, temperature, timestamp), and --dry-run prices the job
before contacting anything.

Input rows need {id, set, question, response}; the question table supplies
the correction / hallucination text by id.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import construct_prompt_fpq, construct_prompt_nfp, parse_score

CHARS_PER_TOKEN = 4.0


def build_prompt(row: dict, q: dict, examples_fpq: list, examples_nfp: list) -> tuple[str, str]:
    if row["set"] == "fpq":
        return "fpq", construct_prompt_fpq(q["question"], q["correction"], row["response"], examples_fpq)
    if row["set"] == "nfp":
        return "nfp", construct_prompt_nfp(q["question"], q["hallucination_text"], row["response"], examples_nfp)
    if row["set"] == "tpq":
        return "nfp", construct_prompt_nfp(q["question"], q["premise_text"], row["response"], examples_nfp)
    raise ValueError(f"unknown set {row['set']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--paper-protocol", action="store_true", help="temperature 0.7 as in validate.py")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    judge_cfg = cfg["judge"]
    model = args.model or judge_cfg["model"]
    temperature = 0.7 if args.paper_protocol else (
        args.temperature if args.temperature is not None else float(judge_cfg["temperature"])
    )
    examples_fpq = load_json(judge_cfg["examples_fpq"])
    examples_nfp = load_json(judge_cfg["examples_nfp"])

    questions = {q["id"]: q for q in read_jsonl(args.questions)}
    # A TPQ row shares its text with an NFP row; responses are keyed by whichever
    # id generated them, so map by question text as well.
    by_text = {}
    for q in questions.values():
        by_text.setdefault(q["question"], []).append(q)

    rows = list(read_jsonl(args.responses))
    if args.limit:
        rows = rows[: args.limit]
    out_path = Path(args.output)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    jobs = []
    for row in rows:
        # One response can be judged under several question rows (nfp + tpq).
        targets = by_text.get(row["question"]) or [questions[row["id"]]]
        for q in targets:
            job_id = f"{row['id']}::{q['id']}"
            if job_id in done:
                continue
            kind, prompt = build_prompt({**row, "set": q["set"]}, q, examples_fpq, examples_nfp)
            jobs.append((job_id, row, q, kind, prompt))
    print(f"[judge] {len(jobs)} prompts to score ({len(done)} already done) -> {out_path}", flush=True)

    if args.dry_run:
        tokens = sum(int(len(p) / CHARS_PER_TOKEN) for *_, p in jobs)
        print(f"[dry-run] ~{tokens:,} input tokens over {len(jobs)} calls with {model}")
        return
    if not jobs:
        return

    from openai import OpenAI

    client = OpenAI()
    for n, (job_id, row, q, kind, prompt) in enumerate(jobs, start=1):
        for attempt in range(5):
            try:
                reply = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=int(judge_cfg.get("max_tokens", 400)),
                )
                text = reply.choices[0].message.content or ""
                break
            except Exception as exc:  # noqa: BLE001 - rate limits, transient errors
                wait = 2**attempt
                print(f"[judge] {job_id}: {exc!r}; retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            text = ""
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
                "judge_model": model,
                "judge_temperature": temperature,
                "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "response_model_id": row.get("model_id"),
            },
        )
        if args.sleep:
            time.sleep(args.sleep)
        if n % 50 == 0 or n == len(jobs):
            print(f"[judge] {n}/{len(jobs)}", flush=True)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
