"""Luna-only test100 protocol: prioritize GEPA before bulk final judging."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def commands(out):
    python = str(ROOT / ".venv/fpqa_prompting/bin/python")
    scripts = ROOT / "scripts"
    config = str(ROOT / "configs/fpqa_prompting/luna_only/luna6.json")
    cancer = str(ROOT / "results/fpqa_prompting/cancer_data_test100_v2")
    first, second = {}, {}
    for dataset, data in (("cancer_myth", cancer), ("crepe", str(ROOT / "results/fpqa_prompting/data"))):
        common = ["--dataset", dataset, "--data", data, "--task", "response", "--method", "gepa", "--config", config]
        first[dataset + "_optimize"] = [python, "-u", str(scripts / "run_fpqa_prompt_experiment.py"),
             "optimize", *common, "--out", str(out / dataset / "optimization")]
        second[dataset + "_test"] = [python, "-u", str(scripts / "run_fpqa_prompt_experiment.py"),
             "evaluate", *common, "--split", "test", "--prompt", str(out / dataset / "optimization/optimized_prompt.json"),
             "--out", str(out / dataset / "gepa_test")]
    first["baseline_generation"] = [python, "-u", str(scripts / "generate_fpqa_baselines.py"),
         "--cancer-data", cancer, "--config", config, "--out", str(out / "baselines"),
         "--reuse-answers", str(ROOT / "results/fpqa_prompting/luna6_baselines_20261002_v1")]
    second["baseline_judgments"] = [python, "-u", str(scripts / "judge_fpqa_saved.py"),
         "--answers", str(out / "baselines"), "--cancer-data", cancer,
         "--config", config, "--out", str(out / "baseline_judgments")]
    return [first, second]


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "OPTIMIZATION_ONLY.json").exists():
        raise RuntimeError("User restricted this run to existing optimization processes; do not launch scoring phases")
    lock = (out / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    limits = out / "call_limits"
    limits.mkdir(exist_ok=True)
    if (limits / "STOP").exists() or (out / "STOP").exists():
        raise RuntimeError("Suite stopped; inspect before resuming")
    plan = dict(phases=commands(out), global_codex_concurrency=5, all_roles="gpt-6-luna",
                rationale="Finish GEPA before bulk test scoring. Do not reuse old optimizer states or Sonnet judgments.")
    path = out / "suite_plan.json"
    serialized = json.dumps(plan, indent=2) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Plan changed")
    path.write_text(serialized)
    env = {**os.environ, "FPQA_CALL_LIMIT_DIR": str(limits.resolve())}
    states = {}
    def report(state, phase):
        tmp = out / "suite_status.tmp"
        tmp.write_text(json.dumps(dict(state=state, phase=phase, at=time.time(), lanes=states), indent=2) + "\n")
        tmp.replace(out / "suite_status.json")
    for phase, jobs in enumerate(plan["phases"]):
        active = {}
        for name, command in jobs.items():
            with (out / f"{name}.log").open("a") as log:
                p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            active[name] = p
            states[name] = dict(state="running", pid=p.pid)
        failed = False
        report("running", phase)
        while active:
            if (out / "STOP").exists():
                (limits / "STOP").write_text("User requested stop\n")
            for name, p in list(active.items()):
                code = p.poll()
                if code is None:
                    continue
                del active[name]
                states[name] = dict(state="failed" if code else "complete", returncode=code)
                if code:
                    failed = True
                    (limits / "STOP").write_text(f"{name} exited {code}; no automatic retries\n")
            report("draining_failure" if failed else "running", phase)
            if active:
                time.sleep(5)
        if failed or (limits / "STOP").exists():
            report("failed" if failed else "stopped", phase)
            return
    report("complete", len(plan["phases"]) - 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    main(args.out.resolve())
