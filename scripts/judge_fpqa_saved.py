"""Score already generated answers without invoking the generation model."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_fpqa_prompt_data import sha
from src.fpqa_prompting import (RecordedCaller, digest, freeze, load_rows, parse_rating,
                                response_metrics, seed_prompt, task_messages, well_crepe_judge,
                                judge_template_name)


def score_one(caller, well_root, row, answer_record):
    if answer_record["question"] != row["question"] or answer_record["id"] != row["id"]:
        raise ValueError("Answer/question identity mismatch")
    expected = digest(task_messages(seed_prompt("response", answer_record["method"]), row))
    if answer_record["messages_sha256"] != expected:
        raise ValueError("Answer prompt mismatch")
    result = caller("judge", well_crepe_judge(well_root, row, answer_record["answer"]))
    return {**answer_record, "label": row["label"], "judgment": result,
            "score": parse_rating(result), "status": "scored"}


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    config = {"judge_model": config["judge_model"]}
    roots = {"cancer_myth": args.cancer_data, "crepe": args.crepe_data}
    jobs = []
    for dataset, root in roots.items():
        manifest = json.loads((root / "manifest.json").read_text())
        path = root / dataset / "test.jsonl"
        if sha(path) != manifest["files"][str(path.relative_to(root))]:
            raise ValueError("Data changed")
        for row in load_rows(path):
            for method in ("plain", "balanced"):
                jobs.append((dataset, method, row))
    plan = dict(config=config, generation_run_sha256=sha(args.answers / "run.json"),
                data={d: sha(p / "manifest.json") for d, p in roots.items()},
                templates={judge_template_name(d, y): sha(args.well_root / f"data_gen/template/{judge_template_name(d, y)}_template.py") for d in roots for y in (0, 1)},
                expected=len(jobs), generation_calls=0, workers=5,
                code_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT / "src/fpqa_prompting.py", ROOT / "src/fpqa_cli_backend.py")})
    freeze(args.out / "run.json", plan)
    caller = RecordedCaller(config, args.out)
    errors = []
    records = {}
    def key(job):
        return job[0] + "/" + job[1] + "/" + digest(job[2]["id"])
    def answer_path(job):
        return args.answers / job[0] / job[1] / "answers" / f"{digest(job[2]['id'])}.json"
    def output_path(job):
        return args.out / job[0] / job[1] / "records" / f"{digest(job[2]['id'])}.json"
    for job in jobs:
        path = output_path(job)
        if path.exists():
            r = json.loads(path.read_text())
            if r["source_sha256"] != sha(answer_path(job)):
                raise ValueError("Previously scored answer changed")
            records[key(job)] = r
    def work(job):
        source = answer_path(job)
        record = score_one(caller, args.well_root, job[2], json.loads(source.read_text()))
        record["source_sha256"] = sha(source)
        freeze(output_path(job), record)
        return record
    new = 0
    def report(state):
        counts = {f"{d}/{m}": sum(r["dataset"] == d and r["method"] == m for r in records.values())
                  for d in roots for m in ("plain", "balanced")}
        status = dict(state=state, at=time.time(), valid=len(records), expected=len(jobs), counts=counts,
                      errors=errors, generation_calls=0)
        tmp = args.out / "status.tmp"
        tmp.write_text(json.dumps(status, indent=2)); tmp.replace(args.out / "status.json")
        print(json.dumps(status), flush=True)
    stop = args.out / "STOP"
    with ThreadPoolExecutor(max_workers=5) as pool:
        active = {}
        report("running")
        while len(records) < len(jobs):
            used = set(active.values())
            if not errors and not stop.exists():
                for job in jobs:
                    if len(active) == 5 or (args.max_new and new >= args.max_new):
                        break
                    if key(job) not in records and key(job) not in used and answer_path(job).exists():
                        active[pool.submit(work, job)] = key(job)
                        new += 1
            if not active:
                if errors or stop.exists() or (args.max_new and new >= args.max_new):
                    break
                source_status = json.loads((args.answers / "status.json").read_text())
                if source_status["state"] in ("failed", "stopped_by_user", "complete"):
                    errors.append(dict(error="Generation ended with missing answers", source_state=source_status["state"]))
                    break
                report("waiting_generation")
                time.sleep(30)
                continue
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                k = active.pop(future)
                try:
                    records[k] = future.result()
                except Exception as exc:
                    errors.append(dict(key=k, error=str(exc)[-2000:]))
            report("draining_error" if errors else "running")
    for d in roots:
        for m in ("plain", "balanced"):
            subset = [r for r in records.values() if r["dataset"] == d and r["method"] == m]
            expected = sum(j[0] == d and j[1] == m for j in jobs)
            if len(subset) == expected:
                freeze(args.out / d / m / "metrics.json", dict(n=expected, partial=False, metrics=response_metrics(subset)))
    report("failed" if errors else "complete" if len(records) == len(jobs) else "stopped_by_user" if stop.exists() else "partial")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cancer-data", type=Path, default=ROOT / "results/fpqa_prompting/cancer_data_v1")
    p.add_argument("--crepe-data", type=Path, default=ROOT / "results/fpqa_prompting/data")
    p.add_argument("--well-root", type=Path, default=ROOT / "external/fpqa_20261002/Well")
    p.add_argument("--config", type=Path, default=ROOT / "configs/fpqa_prompting/luna6_sonnet55.json")
    p.add_argument("--max-new", type=int, default=0)
    args = p.parse_args()
    main(args)
