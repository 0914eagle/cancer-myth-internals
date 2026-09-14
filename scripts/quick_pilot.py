"""Fixed 30 FPQ + 15 NFP dev comparison, 2 new generation conditions, <=225 judge calls.

Uses unchanged Cancer-Myth prompts with one Terra judge. No legacy scores or test.
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_terra_judge import events_by_id
from scripts.reevaluate_pilot_nfp import load_generation
from scripts.run_judge import build_prompt
from scripts.run_pilot import comparison_identity
from src.config import load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import parse_score
from src.llm_backend import make_caller
from src.pilot import digest, file_digest, frozen_json, load_manifest, output_lock

MODEL = "gpt-5.6-terra"
BASE = ("plain", "fp_identification", "premise_cot")
NEW = {"premise_cot_1024": ("premise_cot", 1024), "steering_L21_a0.1": ("steering", 128)}
METHODS = (*BASE, *NEW)


def code_hashes():
    return {str(ROOT / name): file_digest(ROOT / name) for name in (
        "scripts/quick_pilot.py", "scripts/run_pilot.py", "src/pilot.py",
        "src/pilot_model.py", "src/steering.py", "src/judge_prompts.py", "src/llm_backend.py",
    )}


def prepare(pilot, out, config):
    pilot, out, config = Path(pilot).resolve(), Path(out).resolve(), Path(config).resolve()
    cfg = load_config(config)
    mp = pilot / "split/manifest.json"
    manifest = load_manifest(mp)
    questions = {q["id"]: q for q in manifest["questions"] if q["partition"] == "dev"}
    rng = random.Random(17)
    selected = set()
    for kind, n in (("fpq", 30), ("nfp", 15)):
        pool = sorted(i for i, q in questions.items() if q["set"] == kind)
        if len(pool) < n:
            raise ValueError(f"Need {n} dev {kind} questions")
        selected.update(rng.sample(pool, n))
    chosen = [q for q in questions.values() if q["id"] in selected]
    sources = {str(mp): file_digest(mp), **code_hashes()}
    responses, runs = {}, {}
    for method in BASE:
        rows, hashes = load_generation(pilot, method, manifest, questions)
        sources.update(hashes)
        runs[method] = load_json(pilot / "dev" / f"{method}.jsonl.run.json")
        responses[method] = {i: rows[i] for i in selected}
    identity = runs["plain"]["identity"]
    for run in runs.values():
        if (run["identity"] != identity or run["max_new_tokens"] != 512
                or run["review_tokens"] != 128 or run["batch_size"] != 1):
            raise ValueError("Need matching baseline identity, final=512, review=128, batch=1")
    impl = digest({n: file_digest(ROOT / "src" / n) for n in ("pilot.py", "pilot_model.py", "steering.py")})
    # model_identity is JSON-round-tripped before saving: YAML's integer GPU
    # keys (max_memory: {0: ...}) become strings. Compare their JSON form.
    if digest(identity["source_model"]) != digest(cfg["source_model"]):
        raise ValueError(
            "Baseline source model changed; use its original config. "
            f"Stored: {identity['source_model']!r}; configured: {cfg['source_model']!r}"
        )
    refresh = identity["implementation_hash"] != impl
    current_identity = {**identity, "implementation_hash": impl}
    fit = pilot / "fit/fit.json"
    fit_spec = load_json(fit)
    if (comparison_identity(fit_spec["identity"]) != comparison_identity(identity)
            or fit_spec["manifest_hash"] != manifest["manifest_hash"]):
        raise ValueError("Fit direction differs from baseline model/split")
    direction_path = pilot / "fit/directions.npz"
    import numpy as np
    with np.load(direction_path) as arr:
        key = f"L21_c_pair{fit_spec['prefix_tokens']}"
        if key not in arr or "L21_norm_scale" not in arr:
            raise ValueError("Need already fitted L21 C direction and scale")
        v, scale = arr[key], float(arr["L21_norm_scale"])
        if (v.ndim != 1 or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1, atol=1e-4)
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError("Invalid fitted direction or scale")
    for path in (fit, direction_path):
        sources[str(path)] = file_digest(path)
    examples = {}
    for kind in ("fpq", "nfp"):
        path = Path(cfg["judge"][f"examples_{kind}"])
        examples[kind] = load_json(path)
        sources[str(path.resolve())] = file_digest(path)
    plan = {"version": "quick-dev-45-v1", "pilot": str(pilot), "config_path": str(config),
            "config_hash": digest(cfg), "sources": sources, "manifest_hash": manifest["manifest_hash"],
            "identity": current_identity, "baseline_identity": identity,
            "refresh_baselines": refresh, "generation_conditions": list(METHODS if refresh else NEW),
            "fit_identity": fit_spec["identity"], "questions": chosen, "baseline_responses": responses,
            "examples": examples, "methods": list(METHODS), "backend": "codex", "model": MODEL,
            "seed": 17, "generation_budget": 225 if refresh else 90, "judge_budget_max": 225,
            "selection": "seed-17 within-set random sample of previously viewed dev; no score filtering"}
    with output_lock(out / "prepare"):
        frozen_json(out / "plan.json", plan)
        frozen_json(out / "question_ids.json", [q["id"] for q in chosen])
    print(f"Prepared 30 FPQ + 15 NFP. Generate {len(plan['generation_conditions'])} x 45 final answers.")
    if refresh:
        print("Code hash changed: regenerate ALL three baselines on this subset; keep original files untouched.")
        print(f"Baseline implementation: {identity['implementation_hash']}; current: {impl}")
    print("Original Cancer-Myth judge prompts + Terra; at most 225 calls, no retries/preflight. GPT calls now: 0.")
    return plan


def get_plan(out):
    plan = load_json(out / "plan.json")
    if plan["version"] != "quick-dev-45-v1" or plan["model"] != MODEL or plan["methods"] != list(METHODS):
        raise ValueError("Unsupported quick comparison plan")
    for name, expected in plan["sources"].items():
        if file_digest(name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    if digest(load_config(plan["config_path"])) != plan["config_hash"]:
        raise ValueError("Configuration changed")
    expected = [q["id"] for q in plan["questions"]]
    if (len(set(expected)) != 45 or any(q["partition"] != "dev" for q in plan["questions"])
            or [q["set"] for q in plan["questions"]].count("fpq") != 30
            or [q["set"] for q in plan["questions"]].count("nfp") != 15
            or load_json(out / "question_ids.json") != expected):
        raise ValueError("Invalid frozen dev subset")
    return plan


def new_rows(out, plan, tag, complete=True):
    path = out / f"{tag}.jsonl"
    if not path.exists():
        return None
    run = load_json(str(path) + ".run.json")
    method, review = (tag, 128) if tag in BASE else NEW[tag]
    expected = {q["id"]: q for q in plan["questions"]}
    if (run["manifest_hash"] != plan["manifest_hash"] or run["partition"] != "dev"
            or run["method"] != method or run["review_tokens"] != review
            or run["max_new_tokens"] != 512 or run["batch_size"] != 1
            or run["identity"] != plan["identity"] or set(run["question_ids"]) != set(expected)):
        raise ValueError("New generation differs from frozen comparison")
    if method == "steering" and (run["layer"] != 21 or run["alpha"] != 0.1
            or run["direction_hash"] != plan["sources"][str(Path(plan["pilot"]) / "fit/directions.npz")]
            or run.get("fit_spec_hash") != plan["sources"][str(Path(plan["pilot"]) / "fit/fit.json")]
            or run.get("fit_source_identity") != plan["fit_identity"]):
        raise ValueError("Steering configuration changed")
    rows = list(read_jsonl(path)); indexed = {r["id"]: r for r in rows}
    if len(rows) != len(indexed) or not set(indexed) <= set(expected):
        raise ValueError("Duplicate/unexpected generated IDs")
    for i, r in indexed.items():
        if (r["run_hash"] != digest(run) or r["question"] != expected[i]["question"]
                or r["set"] != expected[i]["set"] or not r["response"].strip()):
            raise ValueError("Invalid generated content/provenance")
    if complete and set(indexed) != set(expected):
        return None
    return indexed


def generate(out):
    plan = get_plan(out)
    for tag in plan["generation_conditions"]:
        method, review = (tag, 128) if tag in BASE else NEW[tag]
        if new_rows(out, plan, tag) is not None:
            print(f"Reuse complete {tag}; no GPU load.")
            continue
        cmd = [sys.executable, str(ROOT / "scripts/run_pilot.py"), "generate",
               "--config", plan["config_path"], "--manifest", str(Path(plan["pilot"]) / "split/manifest.json"),
               "--partition", "dev", "--method", method, "--batch-size", "1",
               "--max-new-tokens", "512", "--review-tokens", str(review),
               "--question-ids", str(out / "question_ids.json"),
               "--identity-reference", str(Path(plan["pilot"]) / "dev/plain.jsonl.run.json"),
               "--output", str(out / f"{tag}.jsonl")]
        if method == "steering":
            cmd += ["--fit-dir", str(Path(plan["pilot"]) / "fit"), "--layer", "21", "--alpha", "0.1"]
            cmd += ["--reuse-fit-sha256", plan["sources"][str(Path(plan["pilot"]) / "fit/fit.json")]]
        if tag in BASE:
            cmd += ["--baseline-refresh"]
        subprocess.run(cmd, cwd=ROOT, check=True)
        if new_rows(out, plan, tag) is None:
            raise ValueError("Incomplete generation")
    if plan["refresh_baselines"]:
        audit = {}
        for tag in BASE:
            rows = new_rows(out, plan, tag)
            old = plan["baseline_responses"][tag]
            audit[tag] = {
                "answer_changed_ids": [i for i in rows if rows[i]["response"] != old[i]["response"]],
                "review_changed_ids": [i for i in rows if rows[i].get("review") != old[i].get("review")],
            }
        frozen_json(out / "baseline_refresh_audit.json", audit)
        print("Baseline refresh audit saved. All five scored rows use current-code answers.")
    print("Generation complete. GPT calls: 0. Next: plan-score, then score.")


def plan_score(out):
    plan = get_plan(out)
    responses = dict(plan["baseline_responses"])
    sources = {}
    for tag in plan["generation_conditions"]:
        responses[tag] = new_rows(out, plan, tag)
        if responses[tag] is None:
            raise ValueError("Run generate first")
        for path in (out / f"{tag}.jsonl", out / f"{tag}.jsonl.run.json"):
            sources[str(path)] = file_digest(path)
    jobs, mapping = {}, {}
    for method in METHODS:
        mapping[method] = {}
        for q in plan["questions"]:
            row = responses[method][q["id"]]
            kind, prompt = build_prompt(row, q, plan["examples"]["fpq"], plan["examples"]["nfp"])
            key = digest([q["id"], kind, prompt])
            jobs[key] = {"id": key, "question_id": q["id"], "set": kind,
                         "answer_hash": digest(row["response"]), "prompt": prompt}
            mapping[method][q["id"]] = key
    order = [{"id": k} for k in sorted(jobs)]
    random.Random(17).shuffle(order)
    spec = {"plan_hash": digest(plan), "sources": sources, "cases": jobs, "mapping": mapping,
            "order": order, "budget": len(jobs), "model": MODEL}
    assert len(jobs) <= 225
    with output_lock(out / "judge-prepare"):
        frozen_json(out / "judge_plan.json", spec)
    print(f"Frozen judge plan: {len(jobs)} unique inputs, {225-len(jobs)} duplicate answers shared. Calls now: 0.")
    return spec


def judge_plan(out):
    plan = get_plan(out)
    spec = load_json(out / "judge_plan.json")
    if spec["plan_hash"] != digest(plan) or spec["model"] != MODEL or spec["budget"] > 225:
        raise ValueError("Judge plan mismatch")
    for path, h in spec["sources"].items():
        if file_digest(path) != h:
            raise ValueError("Generated answers changed after judge planning")
    return plan, spec


def score(out, timeout=180):
    with output_lock(out / "score"):
        _, spec = judge_plan(out)
        run_hash = digest(spec)
        previous = events_by_id(out, spec, run_hash)
        print(f"Remaining calls: {spec['budget']-len(previous)}; original prompts; {MODEL}; no retries/preflight.")
        if len(previous) == spec["budget"]:
            return
        call = make_caller("codex", MODEL, timeout=timeout)
        failures = 0
        for job in spec["order"]:
            key = job["id"]
            if key in previous:
                continue
            common = {"id": key, "run_hash": run_hash}
            append_jsonl(out / "attempts.jsonl", {**common, "status": "started", "at": dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                raw, used = call(spec["cases"][key]["prompt"])
                parsed, ok = parse_score(raw)
                allowed = {-1, 0, 1} if spec["cases"][key]["set"] == "fpq" else {-1, 1}
                valid = ok and parsed["Sharpness"] in allowed and used == MODEL
                event = {**common, "status": "finished", "raw": raw, "model": used, "valid": valid,
                         "score": parsed["Sharpness"] if valid else None, "reason": parsed.get("Reason")}
            except Exception as exc:
                event = {**common, "status": "finished", "valid": False, "score": None, "error": str(exc)}
            append_jsonl(out / "attempts.jsonl", event)
            failures = 0 if event["valid"] else failures+1
            print(f"{spec['cases'][key]['question_id']}: {event['score'] if event['valid'] else 'invalid'}", flush=True)
            if failures >= 3 or event.get("model", MODEL) != MODEL or "error" in event:
                print("Stopped on backend/model failure or 3 invalid replies. Attempts consumed; no automatic retry.")
                break


def report(out):
    plan, spec = judge_plan(out)
    events = events_by_id(out, spec, digest(spec))
    values = {}
    for key, rows in events.items():
        e = rows[-1]
        if e["status"] == "finished" and e.get("valid"):
            parsed, ok = parse_score(e["raw"])
            allowed = {-1, 0, 1} if spec["cases"][key]["set"] == "fpq" else {-1, 1}
            if not ok or parsed["Sharpness"] not in allowed or e["model"] != MODEL or parsed["Sharpness"] != e["score"]:
                raise ValueError("Invalid stored score")
            values[key] = e["score"]
    mapped = {m: {i: values.get(k) for i,k in spec["mapping"][m].items()} for m in METHODS}
    lines = ["# Quick Gemma dev comparison", "", "Original Cancer-Myth prompts/few-shot; judge: codex gpt-5.6-terra.",
             "Not exact paper reproduction: judge and split differ. NFP is reference-targeted, not overall QA accuracy.",
             "Exploratory, previously inspected dev: 30 FPQ + 15 NFP. No preservation guarantee or test evaluation.",
             f"Baselines regenerated on current code: {plan['refresh_baselines']}. Fit values reused unchanged with source identity recorded.",
             f"Attempted {len(events)}/{spec['budget']}; valid {len(values)}. Missing/invalid are never passes.", "",
             "| Method | FPQ valid | PCR % | PCS | NFP valid | NFP pass % | FPQ rescue/harm (pairs) | NFP rescue/harm (pairs) |",
             "|---|---:|---:|---:|---:|---:|---|---|"]
    for method in METHODS:
        cells=[];pairs=[]
        for kind in ("fpq", "nfp"):
            ids=[q["id"] for q in plan["questions"] if q["set"]==kind]
            vs=[mapped[method][i] for i in ids]; valid=[v for v in vs if v is not None]
            cells.append(f"{len(valid)}/{len(ids)}")
            cells.append(f"{100*sum(v==1 for v in valid)/len(ids):.1f}" if len(valid)==len(ids) else "incomplete")
            if kind=="fpq":cells.append(f"{sum(valid)/len(ids):.3f}" if len(valid)==len(ids) else "incomplete")
            pairs_ids=[i for i in ids if mapped[method][i] is not None and mapped['plain'][i] is not None]
            rescue=sum(mapped[method][i]==1 and mapped['plain'][i]!=1 for i in pairs_ids)
            harm=sum(mapped[method][i]!=1 and mapped['plain'][i]==1 for i in pairs_ids)
            pairs.append(f"{rescue}/{harm} ({len(pairs_ids)})")
        lines.append('| '+ ' | '.join([method,*cells,*pairs])+' |')
    lines+=['','CoT rows differ only in review-token cap (128 vs 1024); final cap 512.',
            'Steering: fixed L21 paired-C, alpha=0.1 x fit residual scale, unconditional. No internal gate in this run.',
            'Use these counts to choose the next experiment; do not search thresholds or claim significance from this small subset.']
    text='\n'.join(lines)+'\n';(out/'report.md').write_text(text);print(text)
    return mapped


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','generate','plan-score','score','report','all'])
    p.add_argument('--pilot-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--config',default='configs/gemma2_9b.yaml');p.add_argument('--timeout',type=int,default=180)
    args=p.parse_args();out=Path(args.out_dir).resolve()
    if args.stage in {'prepare','all'}:prepare(args.pilot_dir,out,args.config)
    if args.stage in {'generate','all'}:generate(out)
    if args.stage in {'plan-score','all'}:plan_score(out)
    if args.stage in {'score','all'}:score(out,args.timeout)
    if args.stage in {'report','all'}:report(out)


if __name__=='__main__':main()
