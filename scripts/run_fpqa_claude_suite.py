"""Parallel saved-answer scoring and two GEPA runs; fail-stop, no quota retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    limits = out / "call_limits"
    limits.mkdir(exist_ok=True)
    if (limits / "STOP").exists():
        raise RuntimeError("Suite is stopped; explicit review is required before resuming")
    env = {**os.environ, "FPQA_CALL_LIMIT_DIR": str(limits.resolve())}
    python = str(ROOT / ".venv/fpqa_prompting/bin/python")
    runner = str(ROOT / "scripts/run_fpqa_prompt_experiment.py")
    config = str(ROOT / "configs/fpqa_prompting/luna6_sonnet55.json")
    lanes = {"baseline_judgments": [[python, "-u", str(ROOT / "scripts/judge_fpqa_saved.py"),
              "--answers", str(ROOT / "results/fpqa_prompting/luna6_baselines_20261002_v1"),
              "--out", str(out / "baseline_judgments")]]}
    for dataset, data in (("cancer_myth", "cancer_data_v1"), ("crepe", "data")):
        common = ["--dataset", dataset, "--data", str(ROOT / "results/fpqa_prompting" / data),
                  "--task", "response", "--method", "gepa", "--config", config]
        optimize = out / dataset / "optimization"
        evaluate = out / dataset / "gepa_test"
        lanes[dataset] = [
            [python, "-u", runner, "optimize", *common, "--out", str(optimize)],
            [python, "-u", runner, "evaluate", *common, "--split", "test", "--prompt",
             str(optimize / "optimized_prompt.json"), "--out", str(evaluate)]]
    plan = dict(lanes=lanes, claude_total_concurrency=5, new_codex_total_concurrency=5,
                note="Existing baseline generation is independent; no judge/reflector GPT calls. No automatic retries.")
    path = out / "suite_plan.json"
    text = json.dumps(plan, indent=2) + "\n"
    if path.exists() and path.read_text() != text:
        raise ValueError("Suite plan changed")
    path.write_text(text)
    active, states = {}, {}
    def start(name, index):
        with (out / f"{name}.log").open("a") as log:
            proc = subprocess.Popen(lanes[name][index], cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        active[name] = (proc, index)
        states[name] = dict(state="running", stage=index, pid=proc.pid)
    def report(state):
        temp = out / "suite_status.tmp"
        temp.write_text(json.dumps(dict(state=state, at=time.time(), lanes=states), indent=2) + "\n")
        temp.replace(out / "suite_status.json")
    for name in lanes:
        start(name, 0)
    failed = False
    report("running")
    while active:
        if (out / "STOP").exists():
            (limits / "STOP").write_text("User requested suite stop\n")
        for name, (proc, index) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[name]
            if code:
                failed = True
                states[name] = dict(state="failed", stage=index, returncode=code)
                (limits / "STOP").write_text(f"Lane {name} exited {code}; inspect logs before resuming.\n")
            elif index + 1 < len(lanes[name]) and not (limits / "STOP").exists():
                start(name, index + 1)
            else:
                states[name] = dict(state="complete" if index + 1 == len(lanes[name]) else "stopped", stage=index)
        report("draining_failure" if failed else "stopping" if (limits / "STOP").exists() else "running")
        if active:
            time.sleep(5)
    report("failed" if failed else "stopped" if (limits / "STOP").exists() else "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
