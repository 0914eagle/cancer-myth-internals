"""Generate only Luna Plain/balanced test answers; no judge or GEPA calls.

One shared pool limits both datasets together. STOP or the first failure stops new
submissions, drains active calls, and preserves completed outputs. No auto retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_fpqa_prompt_data import sha
from src.fpqa_prompting import RecordedCaller, digest, freeze, load_rows, seed_prompt, task_messages, validate_splits


def collect_jobs(data_roots):
    lanes, sources = [], {}
    for dataset, root in data_roots.items():
        manifest = json.loads((root / "manifest.json").read_text())
        for rel, expected in manifest["files"].items():
            if rel.startswith(dataset + "/") and sha(root / rel) != expected:
                raise ValueError("Prepared data changed: " + rel)
        splits = [load_rows(root / dataset / f"{s}.jsonl") for s in ("train", "dev", "test")]
        validate_splits(*splits)
        sources[dataset] = dict(root=str(root.resolve()), manifest_sha256=sha(root / "manifest.json"),
                                test_n=len(splits[2]))
        for method in ("plain", "balanced"):
            lanes.append([(dataset, method, row) for row in splits[2]])
    jobs = [lane[i] for i in range(max(map(len, lanes))) for lane in lanes if i < len(lane)]
    return jobs, sources


def record_path(out, job):
    dataset, method, row = job
    return out / dataset / method / "answers" / f"{digest(row['id'])}.json"


def generate_one(caller, out, job):
    dataset, method, row = job
    prompt = seed_prompt("response", method)
    messages = task_messages(prompt, row)
    answer = caller("task", messages)
    record = dict(id=row["id"], dataset=dataset, method=method, split="test",
                  question=row["question"], answer=answer, prompt=prompt,
                  messages_sha256=digest(messages), status="generated_unscored")
    freeze(record_path(out, job), record)
    return record


def reuse_answers(source, out, jobs, generation_config):
    source_plan = json.loads((source / "run.json").read_text())
    if source_plan["config"]["task_model"] != generation_config["task_model"]:
        raise ValueError("Cannot reuse answers from a different task model configuration")
    copied = 0
    for job in jobs:
        old, dest = record_path(source, job), record_path(out, job)
        if not old.exists() or dest.exists():
            continue
        r = json.loads(old.read_text())
        expected = digest(task_messages(seed_prompt("response", job[1]), job[2]))
        if (r["id"] != job[2]["id"] or r["dataset"] != job[0] or r["method"] != job[1]
                or r["question"] != job[2]["question"] or r["messages_sha256"] != expected or not r["answer"]):
            raise ValueError("Reused answer identity mismatch")
        freeze(dest, {**r, "reused_source": str(old.resolve()), "reused_source_sha256": sha(old)})
        copied += 1
    return copied


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads(args.config.read_text())
    # Explicit whitelist: this process cannot instantiate a judge or reflector.
    generation_config = {"task_model": config["task_model"]}
    jobs, sources = collect_jobs({"cancer_myth": args.cancer_data, "crepe": args.crepe_data})
    plan = dict(task="response_generation_only", sources=sources, config=generation_config,
                methods={m: seed_prompt("response", m) for m in ("plain", "balanced")},
                expected_answers=len(jobs), global_workers=args.workers,
                judge_calls=0, reflection_calls=0, automatic_retries=False,
                code_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                             (Path(__file__), ROOT / "src/fpqa_prompting.py", ROOT / "src/fpqa_cli_backend.py")},
                cli_version=subprocess.run([config["task_model"]["backend"], "--version"],
                                           check=True, capture_output=True, text=True).stdout.strip())
    if args.reuse_answers:
        plan["reuse_source_manifest_sha256"] = sha(args.reuse_answers / "run.json")
    freeze(args.out / "run.json", plan)
    if args.preflight:
        print(json.dumps(plan, indent=2)); return
    if args.reuse_answers:
        print(f"Reused {reuse_answers(args.reuse_answers, args.out, jobs, generation_config)} answers", flush=True)
    caller = RecordedCaller(generation_config, args.out)
    counts = {f"{d}/{m}": 0 for d in sources for m in ("plain", "balanced")}
    pending = []
    for job in jobs:
        path = record_path(args.out, job)
        if path.exists():
            r = json.loads(path.read_text())
            if r["messages_sha256"] != digest(task_messages(seed_prompt("response", job[1]), job[2])) or not r["answer"]:
                raise ValueError("Cached answer mismatch")
            counts[f"{job[0]}/{job[1]}"] += 1
        else:
            pending.append(job)
    if args.max_new:
        pending = pending[:args.max_new]
    errors = []
    def status(state):
        value = dict(state=state, at=time.time(), valid=sum(counts.values()), expected=len(jobs),
                     counts=counts, errors=errors, judge_calls=0, reflection_calls=0)
        temp = args.out / "status.tmp"
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(args.out / "status.json")
        print(json.dumps(value), flush=True)
    status("running")
    stop = args.out / "STOP"
    queue = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        def submit():
            if stop.exists() or errors:
                return
            job = next(queue, None)
            if job:
                active[pool.submit(generate_one, caller, args.out, job)] = job
        for _ in range(args.workers):
            submit()
        while active:
            done, _ = wait(active, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    future.result()
                    counts[f"{job[0]}/{job[1]}"] += 1
                except Exception as exc:
                    errors.append(dict(dataset=job[0], method=job[1], id=job[2]["id"], error=str(exc)[-2000:]))
            for _ in done:
                submit()
            status("draining_error" if errors else "draining_stop" if stop.exists() else "running")
    state = "failed" if errors else "complete" if sum(counts.values()) == len(jobs) else "stopped_by_user" if stop.exists() else "partial"
    status(state)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crepe-data", type=Path, default=ROOT / "results/fpqa_prompting/data")
    parser.add_argument("--cancer-data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/fpqa_prompting/luna6_sonnet55.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--reuse-answers", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 5 or args.max_new < 0:
        parser.error("workers must be 1..5; max-new must be nonnegative")
    run(args)
