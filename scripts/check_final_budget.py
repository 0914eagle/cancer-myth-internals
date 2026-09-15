"""Fit-only 512/1024 final-answer budget diagnostic. No judge backend.

Uses the existing generation runtime unchanged in a separate cache. One frozen
CoT reasoning trace is shared across both final budgets. Original runs are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline_generation as bg
from src.baseline_suite import load_suite
from src.config import load_config
from src.pilot import digest, file_digest, output_lock

METHODS = ("plain", "zero_shot_cot")
BUDGETS = (512, 1024)


def select_questions(rows, seed=17):
    """Choose unique fit source groups without reading answers or scores."""
    selected, groups = [], set()
    for kind, count in (("fpq", 8), ("nfp", 4)):
        pool = sorted((q for q in rows if q["partition"] == "fit" and q["set"] == kind),
                      key=lambda q: digest([seed, q["id"]]))
        chosen = []
        for q in pool:
            if q["group_id"] not in groups:
                chosen.append(q)
                groups.add(q["group_id"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Need {count} unique fit groups for {kind}")
        selected.extend(chosen)
    return selected


def implementation():
    return {str(p): file_digest(p) for p in (
        Path(__file__).resolve(), ROOT / "src/baseline_generation.py",
        ROOT / "src/pilot.py", ROOT / "src/pilot_model.py", ROOT / "src/steering.py")}


def prepare(suite_dir, out, config):
    if out == suite_dir or out == suite_dir / "model" or suite_dir.is_relative_to(out):
        raise ValueError("Use a separate diagnostic directory")
    suite = load_suite(suite_dir)
    cases = select_questions(suite["questions"])
    sources = [suite_dir / "suite.json", suite_dir / "questions.jsonl",
               suite_dir / "model/identity.json"]
    identity = json.loads(sources[-1].read_text())
    if identity["implementation_sha256"] != file_digest(ROOT / "src/baseline_generation.py"):
        raise ValueError("Source generation code differs; do not silently change protocol")
    for method in METHODS:
        p = suite_dir / "model/generation" / method / "spec.json"
        spec = json.loads(p.read_text())
        if (spec["final_tokens"] != 512 or spec["review_tokens"] != 1024
                or spec["identity"] != identity["identity"] or spec["prompts"] != bg.PROMPTS):
            raise ValueError(f"Unexpected source protocol for {method}")
        sources.append(p)
    snapshots = {}
    for q in cases:
        p = suite_dir / "model/generation/plain/cache" / (digest(q["id"]) + ".json")
        row = json.loads(p.read_text())
        if row.get("status") != "complete" or row.get("question") != q["question"]:
            raise ValueError(f"Missing valid source Plain: {q['id']}")
        snapshots[q["id"]] = row
        sources.append(p)
    cfg = load_config(config)
    # JSON-normalize numeric YAML keys, as done by model_identity.
    cfg = json.loads(json.dumps(cfg))
    if cfg["source_model"] != identity["identity"]["source_model"]:
        raise ValueError("Config differs from source model")
    plan = {"version": "final-budget-fit-v1", "cases": cases, "config": cfg,
            "source_identity": identity, "source_plain": snapshots,
            "sources": {str(p): file_digest(p) for p in sources},
            "implementation": implementation(), "methods": list(METHODS),
            "final_budgets": list(BUDGETS), "review_tokens": 1024,
            "selection": "seed 17 hashed IDs; 8 FPQ + 4 NFP unique fit groups; no outcome selection",
            "max_final_generations": 48, "max_reasoning_generations": 12, "judge_calls": 0}
    with output_lock(out / "prepare"):
        bg._atomic_json(out / "plan.json", plan, frozen=True)
    print(f"Prepared 12 fit questions; at most 48 final + 12 reasoning generations. GPT calls: 0. {out}")
    return plan


def verify(plan):
    for path, expected in {**plan["sources"], **plan["implementation"]}.items():
        if file_digest(path) != expected:
            raise ValueError(f"Frozen source/code changed: {path}")


def record_path(out, method, budget, qid):
    return out / "answers" / method / str(budget) / (digest(qid) + ".json")


def generate(out, plan):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
        raise ValueError("Set CUDA_VISIBLE_DEVICES to one physical GPU: 0 or 1")
    verify(plan)
    signature = digest(plan)
    tasks = [(q, m, b) for q in plan["cases"] for m in METHODS for b in BUDGETS]
    with output_lock(out / "generate"):
        missing = []
        for q, m, b in tasks:
            if bg._cache_record(record_path(out, m, b, q["id"]), digest([signature, q["id"], m, b])) is None:
                missing.append((q, m, b))
        if not missing:
            print("All 48 final answers already stored. No model loaded.")
            return
        runtime = bg.make_runtime(plan["config"])
        if runtime.identity != plan["source_identity"]["identity"]:
            raise ValueError("Runtime model/tokenizer/software differs from source")
        for i, (q, method, budget) in enumerate(missing, 1):
            # Only question text and ID enter inference. References/labels stay offline.
            row = bg._answer_one(runtime, out / "model", {"id": q["id"], "question": q["question"]},
                                 method, budget, plan["review_tokens"], 512)
            row["signature"] = digest([signature, q["id"], method, budget])
            bg._atomic_json(record_path(out, method, budget, q["id"]), row, frozen=True)
            print(f"[budget] {i}/{len(missing)} {method} {q['id']} final={budget} "
                  f"cap_hit={row['details']['answer']['cap_hit']}", flush=True)


def report(out, plan):
    rows = {}
    for q in plan["cases"]:
        for m in METHODS:
            for b in BUDGETS:
                p = record_path(out, m, b, q["id"])
                if p.exists():
                    rows[q["id"], m, b] = bg._cache_record(p, digest([digest(plan), q["id"], m, b]))
    lines = ["# Final-answer budget diagnostic", "",
             "Fit only: 8 FPQ + 4 NFP, unique source groups, selected without scores or answer inspection.",
             "Exploratory length diagnostic, not correction accuracy or an unbiased suite estimate.",
             "Same prompts and greedy decoding; same saved CoT reasoning per paired answer. Review cap=1024.",
             "Report uses saved files only. GPT calls: 0. Missing answers are not successes.", "",
             "| Method | Final cap | Stored/12 | Length stops | EOS stops | Review length stops |",
             "|---|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        for b in BUDGETS:
            rs = [rows[q["id"], m, b] for q in plan["cases"] if (q["id"], m, b) in rows]
            cap = sum(r["details"]["answer"]["cap_hit"] for r in rs)
            eos = sum(r["details"]["answer"]["ended_with_eos"] for r in rs)
            review_cap = sum(r["details"].get("reasoning", {}).get("cap_hit", False) for r in rs)
            lines.append(f"| {m} | {b} | {len(rs)}/12 | {cap} | {eos} | {review_cap} |")
    lines += ["", "## Paired checks", "",
              "Text-prefix checks use decoded text, not exact token IDs. Divergence requires inspection;",
              "it is not automatically a correction change. Longer output is not automatically better.", ""]
    for m in METHODS:
        pairs = [(rows[q["id"], m, 512], rows[q["id"], m, 1024]) for q in plan["cases"]
                 if (q["id"], m, 512) in rows and (q["id"], m, 1024) in rows]
        changed = [a["id"] for a, b in pairs if a["response"] != b["response"]]
        diverged = [a["id"] for a, b in pairs if not b["response"].startswith(a["response"])]
        resolved = [a["id"] for a, b in pairs if a["details"]["answer"]["cap_hit"]
                    and b["details"]["answer"]["ended_with_eos"]]
        review_mismatch = [a["id"] for a, b in pairs
                           if a["details"].get("reasoning") != b["details"].get("reasoning")]
        lines += [f"- {m}: paired={len(pairs)}, changed={len(changed)}, 512 length → 1024 EOS={len(resolved)}.",
                  f"  Prefix divergence IDs: {diverged}. Reasoning mismatch IDs: {review_mismatch}."]
    controls = [rows[q["id"], "plain", 512] for q in plan["cases"] if (q["id"], "plain", 512) in rows]
    different = [r["id"] for r in controls if r["response"] != plan["source_plain"][r["id"]]["response"]]
    lines += [f"- Source Plain replay: {len(controls)} compared; different IDs: {different}.",
              "", "Full paired answers: `examples.md`. Inspect the added text for actual corrections,",
              "new factual claims, unnecessary rebuttals and repetition; no automatic medical verdict is produced."]
    examples = ["# Paired fit answers: final cap 512 vs 1024", "",
                "Selected fit diagnostic only. Reference text is benchmark annotation, not independently verified truth."]
    for q in plan["cases"]:
        examples += ["", f"## {q['id']} ({q['set']})", "", q["question"], "",
                     "Benchmark reference (not used in generation):", "", str(q.get("correction", ""))]
        for m in METHODS:
            first = rows.get((q["id"], m, 512))
            if first and "reasoning" in first["details"]:
                examples += ["", f"### {m}: shared reasoning", "", first["details"]["reasoning"]["text"]]
            for b in BUDGETS:
                r = rows.get((q["id"], m, b))
                examples += ["", f"### {m}: final cap {b}", ""]
                if r:
                    a = r["details"]["answer"]
                    examples += [f"Tokens={a['output_tokens']}; stop={a['stop_reason']}", "", r["response"]]
                else:
                    examples.append("MISSING")
    with output_lock(out / "report"):
        (out / "report.md").write_text("\n".join(lines) + "\n")
        (out / "examples.md").write_text("\n".join(examples) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "generate", "report", "all"))
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/qwen25_7b.yaml")
    args = p.parse_args()
    suite_dir, out = args.suite_dir.resolve(), args.out_dir.resolve()
    if args.stage in {"prepare", "all"}:
        plan = prepare(suite_dir, out, args.config)
    else:
        plan = json.loads((out / "plan.json").read_text())
    if args.stage in {"generate", "all"}:
        generate(out, plan)
    if args.stage in {"report", "all"}:
        report(out, plan)


if __name__ == "__main__":
    main()
