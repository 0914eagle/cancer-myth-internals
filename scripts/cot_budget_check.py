"""Compare 128 -> 1024 review-token budget on saved dev CoT FPQ score-0 cases.

Explicit Gemma generation only; no judge calls. Old scores select cases, not new labels.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_pilot_judge import audit
from scripts.reevaluate_pilot_nfp import load_generation
from scripts.run_pilot import comparison_identity
from src.jsonl import load_json, read_jsonl
from src.pilot import check_resume, digest, file_digest, frozen_json, load_manifest, output_lock

METHOD = "premise_cot"
NEW_BUDGET = 1024


def implementation_hash():
    return digest(
        {
            name: file_digest(ROOT / "src" / name)
            for name in ("pilot.py", "pilot_model.py", "steering.py")
        }
    )


def prepare(pilot, out):
    manifest_path = pilot / "split/manifest.json"
    manifest = load_manifest(manifest_path)
    questions = {q["id"]: q for q in manifest["questions"] if q["partition"] == "dev"}
    audit(pilot, "gpt-5.6-sol", METHOD)
    responses, sources = load_generation(pilot, METHOD, manifest, questions)
    run_path = pilot / "dev/premise_cot.jsonl.run.json"
    old_run = load_json(run_path)
    if old_run["review_tokens"] != 128 or old_run["batch_size"] != 1:
        raise ValueError(
            "This paired subset check requires the old run to use review_tokens=128, batch_size=1"
        )
    score_path = pilot / "dev/premise_cot_judge_codex_gpt-5.6-sol.jsonl"
    scores = {r["question_id"]: r for r in read_jsonl(score_path)}
    ids = [
        qid for qid, q in questions.items() if q["set"] == "fpq" and scores[qid]["sharpness"] == 0
    ]
    if not ids or any(not responses[i].get("review") for i in ids):
        raise ValueError("Need saved score-0 FPQ reviews")
    sources.update(
        {
            str(p.resolve()): file_digest(p)
            for p in (manifest_path, score_path, Path(str(score_path) + ".run.json"))
        }
    )
    cases = [
        {
            "id": i,
            "question": questions[i]["question"],
            "reference": questions[i]["correction"],
            "old_review": responses[i]["review"],
            "old_answer": responses[i]["response"],
            "old_review_output_tokens": responses[i]["review_output_tokens"],
        }
        for i in ids
    ]
    plan = {
        "version": "cot-review-budget-1024-v1",
        "manifest_hash": manifest["manifest_hash"],
        "sources": sources,
        "old_run": old_run,
        "new_review_tokens": NEW_BUDGET,
        "cases": cases,
        "question_ids": ids,
        "selected_by": "old Sol FPQ score 0 on viewed dev",
        "current_implementation_hash": implementation_hash(),
    }
    out.mkdir(parents=True, exist_ok=True)
    with output_lock(out / "prepare"):
        frozen_json(out / "plan.json", plan)
        frozen_json(out / "question_ids.json", ids)
    return plan


def verify_sources(plan):
    for path, expected in plan["sources"].items():
        if file_digest(Path(path)) != expected:
            raise ValueError(f"Baseline source changed: {path}")
    if plan.get("current_implementation_hash", implementation_hash()) != implementation_hash():
        raise ValueError("Generation implementation changed after plan preparation")


def verify_replay(out, plan):
    path = out / "premise_cot_r128_control.jsonl"
    run = load_json(Path(str(path) + ".run.json"))
    old = plan["old_run"]
    if comparison_identity(run["identity"]) != comparison_identity(old["identity"]):
        raise ValueError("Replay control model identity differs")
    for key in (
        "review_tokens",
        "max_new_tokens",
        "batch_size",
        "seed",
        "decoding",
        "method",
        "partition",
        "manifest_hash",
    ):
        if run[key] != old[key]:
            raise ValueError(f"Replay control setting differs: {key}")
    ids = set(plan["question_ids"])
    if check_resume(path, run, ids) != ids:
        raise ValueError("Replay control incomplete")
    control = {r["id"]: r for r in read_jsonl(path)}
    changed = [
        c["id"]
        for c in plan["cases"]
        if control[c["id"]]["question"] != c["question"]
        or control[c["id"]].get("review") != c["old_review"]
        or control[c["id"]].get("response") != c["old_answer"]
    ]
    audit_result = {
        "changed_ids": changed,
        "output_hash": file_digest(path),
        "run_hash": file_digest(Path(str(path) + ".run.json")),
    }
    frozen_json(out / "control_audit.json", audit_result)
    if changed:
        raise ValueError(
            f"128-token replay differs on {changed}; inspect control before attributing changes to length"
        )
    return audit_result


def report(out):
    plan = load_json(out / "plan.json")
    verify_sources(plan)
    path = out / "premise_cot_r1024.jsonl"
    run_path = Path(str(path) + ".run.json")
    run = load_json(run_path)
    old = plan["old_run"]
    for key in (
        "manifest_hash",
        "partition",
        "method",
        "max_new_tokens",
        "batch_size",
        "seed",
        "decoding",
    ):
        if run[key] != old[key]:
            raise ValueError(f"Comparison setting changed: {key}")
    if comparison_identity(run["identity"]) != comparison_identity(old["identity"]):
        raise ValueError("Comparison model/runtime identity changed")
    control_audit = None
    if run["identity"].get("implementation_hash") != old["identity"].get("implementation_hash"):
        control_audit = verify_replay(out, plan)
        control_run = load_json(out / "premise_cot_r128_control.jsonl.run.json")
        if control_run["identity"] != run["identity"]:
            raise ValueError("New generation and replay implementation differ")
    if (
        run["review_tokens"] != NEW_BUDGET
        or run["question_ids"] != plan["question_ids"]
        or run["diagnostic_subset_hash"] != file_digest(out / "question_ids.json")
    ):
        raise ValueError("New run has wrong review budget/subset")
    ids = set(plan["question_ids"])
    if check_resume(path, run, ids) != ids:
        raise ValueError("Generation incomplete; resume all/generate before report")
    new = {r["id"]: r for r in read_jsonl(path)}
    for c in plan["cases"]:
        r = new[c["id"]]
        if r["question"] != c["question"] or not r.get("review") or not r.get("response"):
            raise ValueError("Missing/mismatched generated content")
    at_cap = sum(new[i]["review_output_tokens"] == NEW_BUDGET for i in ids)
    changed_review = sum(new[c["id"]]["review"] != c["old_review"] for c in plan["cases"])
    changed_answer = sum(new[c["id"]]["response"] != c["old_answer"] for c in plan["cases"])
    summary = {
        "plan_hash": digest(plan),
        "new_output_hash": file_digest(path),
        "new_run_hash": file_digest(run_path),
        "n": len(ids),
        "reviews_changed": changed_review,
        "answers_changed": changed_answer,
        "reviews_at_1024_tokens": at_cap,
        "judge_calls": 0,
        "scores_assigned": 0,
        "control_audit": control_audit,
    }
    lines = [
        "# CoT review budget: 128 → 1024 — selected dev cases",
        "",
        f"{len(ids)} old score-0 FPQs. GPU generation performed; GPT judge calls: 0. Test not used.",
        f"Final-answer budget unchanged: {old['max_new_tokens']}; batch size 1; prompts/model identity unchanged.",
        "1024 is a maximum, not a required length. Hitting the cap alone does not prove truncation.",
        "This is a selected diagnostic sample; no PCR, population improvement, or new scores are inferred.",
        f"Reviews changed: {changed_review}/{len(ids)}; answers changed: {changed_answer}/{len(ids)}; new reviews at cap: {at_cap}.",
        "A longer review also changes the input to the answer stage; this is the effect of the review-budget change on the full pipeline.",
        "",
    ]
    for c in plan["cases"]:
        r = new[c["id"]]
        lines += [
            f"## {c['id']}",
            "",
            "**Question**",
            "",
            c["question"],
            "",
            "**Supplied reference — not independently validated**",
            "",
            c["reference"],
            "",
            f"**Old review (128 budget; {c['old_review_output_tokens']} output tokens)**",
            "",
            c["old_review"],
            "",
            f"**New review (1024 budget; {r['review_output_tokens']} output tokens)**",
            "",
            r["review"],
            "",
            "**Old final answer**",
            "",
            c["old_answer"],
            "",
            "**New final answer**",
            "",
            r["response"],
            "",
            "**Target-premise recognition before/after + quotes:**",
            "",
            "**Correction accuracy before/after + quotes:**",
            "",
            "**REVIEW / unresolved reference or medical issue:**",
            "",
        ]
    with output_lock(out / "report"):
        frozen_json(out / "summary.json", summary)
        # Preserve a user's filled review on rerun.
        review_path = out / "review.md"
        if not review_path.exists():
            review_path.write_text("\n".join(lines) + "\n")
    print(summary)
    print(f"Review: {out / 'review.md'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "generate", "report", "all"))
    p.add_argument("--pilot-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--config", default="configs/gemma2_9b.yaml")
    a = p.parse_args()
    if a.stage in ("prepare", "all"):
        plan = prepare(a.pilot_dir, a.out_dir)
        print(
            f"Prepared {len(plan['question_ids'])} cases; old scores unchanged; no judge calls.",
            flush=True,
        )
    if a.stage in ("generate", "all"):
        plan = load_json(a.out_dir / "plan.json")
        verify_sources(plan)
        old_run = a.pilot_dir / "dev/premise_cot.jsonl.run.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_pilot.py"),
            "generate",
            "--config",
            a.config,
            "--manifest",
            str(a.pilot_dir / "split/manifest.json"),
            "--partition",
            "dev",
            "--method",
            METHOD,
            "--question-ids",
            str(a.out_dir / "question_ids.json"),
            "--identity-reference",
            str(old_run),
            "--batch-size",
            "1",
            "--max-new-tokens",
            str(plan["old_run"]["max_new_tokens"]),
        ]
        budgets = [NEW_BUDGET]
        if plan["old_run"]["identity"].get("implementation_hash") != implementation_hash():
            budgets = [128, NEW_BUDGET]
            print(
                "Legacy code hash differs: replay 128 on the same cases first; proceed only if all old reviews/answers match.",
                flush=True,
            )
        for budget in budgets:
            filename = (
                "premise_cot_r128_control.jsonl" if budget == 128 else "premise_cot_r1024.jsonl"
            )
            subprocess.run(
                command + ["--review-tokens", str(budget), "--output", str(a.out_dir / filename)],
                check=True,
                cwd=ROOT,
            )
            if budget == 128:
                verify_replay(a.out_dir, plan)
    if a.stage in ("report", "all"):
        report(a.out_dir)


if __name__ == "__main__":
    main()
