"""SFT-only follow-up: append 'Answer the question.' to each final prompt.

Original two-call CoT reasoning is unchanged. Historical Base scores are read
from the pinned advisor snapshot, never regenerated or presented as a matched
answer-instruction control. Original no-instruction generation is preserved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import premise_table2 as original
from scripts import premise_sft as sft
from src import baseline_generation as bg
from src import well_eval as well
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import digest, file_digest, frozen_json, output_lock

INSTRUCTION = "Answer the question."
CONDITIONS = ("cancer", "crepe")
METHODS = original.METHODS
BASE_CSV = ROOT / "docs/reviews/advisor_presentation_2026-09-23/answer_scores.csv"
BASE_PROVENANCE = BASE_CSV.parent / "provenance.json"


def base_scores(questions):
    inventory = {q["id"]: q for q in questions}
    rows, seen = [], set()
    with BASE_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != "qwen" or row["method"] not in METHODS or row["id"] not in inventory:
                continue
            key = (row["method"], row["id"])
            rating = int(row["score"])
            if key in seen or not 0 <= rating <= 5 or row["set"] != inventory[row["id"]]["set"]:
                raise ValueError("Invalid historical Base score mapping")
            seen.add(key)
            rows.append({**row, "score": rating})
    expected = {(m, qid) for m in METHODS for qid in inventory}
    if seen != expected:
        raise ValueError("Historical Base scores do not cover the same test questions")
    provenance = json.loads(BASE_PROVENANCE.read_text())
    if provenance["judge"] != "claude-sonnet-5":
        raise ValueError("Historical Base judge differs")
    return rows


def prepare(art, out, cancer_run, crepe_run):
    out = Path(out)
    plan = original.prepare(art, out, cancer_run, crepe_run)
    protocol = {"version": "premise-table2-answer-v1", "active_conditions": list(CONDITIONS),
                "answer_instruction": INSTRUCTION, "append_to": "final user prompt for all three methods",
                "cot_reasoning_prompt_changed": False, "implementation_sha256": file_digest(__file__),
                "base_csv_sha256": file_digest(BASE_CSV), "base_provenance_sha256": file_digest(BASE_PROVENANCE),
                "base_generation": "none; historical 1024-token Qwen/Sonnet scores on the same 147 IDs",
                "base_prompt_instruction_added": False,
                "comparison_note": "Historical Base lacks the appended answer instruction. Treat it as a reference, not a matched prompt estimate of the SFT effect."}
    frozen_json(out / "answer_instruction_protocol.json", protocol)
    rows = base_scores(plan["questions"])
    frozen_json(out / "reused_base_scores.json", {"sources": protocol, "rows": rows})
    return {"counts": plan["counts"], "sft_answer_rows": len(CONDITIONS)*len(METHODS)*len(plan["questions"]),
            "reused_base_scores": len(rows)}


def load_plan(out):
    out = Path(out)
    plan = original.load_plan(out)
    protocol = json.loads((out / "answer_instruction_protocol.json").read_text())
    if protocol["implementation_sha256"] != file_digest(__file__) or protocol["answer_instruction"] != INSTRUCTION:
        raise ValueError("Answer-instruction protocol changed; use a new run")
    if protocol["base_csv_sha256"] != file_digest(BASE_CSV):
        raise ValueError("Historical Base source changed")
    return plan, protocol


def answer_one(runtime, out, question, method, final_tokens, review_tokens):
    text, details = question["question"], {}
    prompt = text
    if method == "zero_shot_cot":
        reasoning = bg._shared_generate(runtime, out, bg.COT_REASON.format(question=text), review_tokens)
        details["reasoning"] = reasoning
        prompt = bg.COT_FINAL.format(question=text, review=reasoning["text"])
    elif method == "fp_unconditional":
        prompt = bg.FP_ANSWER.format(question=text)
    elif method != "plain":
        raise ValueError("Unsupported prompt method")
    prompt += "\n\n" + INSTRUCTION
    answer = bg._shared_generate(runtime, out, prompt, final_tokens)
    form = original.output_form(answer["text"])
    details["answer"] = {k: v for k, v in answer.items() if k != "text"}
    details["final_prompt_sha256"] = digest(prompt)
    details["answer_instruction"] = INSTRUCTION
    return {"id": question["id"], "question": text, "prompt_method": method,
            "status": "complete", "response": answer["text"], "details": details,
            "output_form": form, "final_answer_present": form not in {"empty", "diagnostic_only", "diagnostic_incomplete"}}


def generate(out, condition):
    if condition not in CONDITIONS:
        raise ValueError("Only SFT models are generated in this follow-up")
    out = Path(out)
    plan, _ = load_plan(out)
    run = Path(plan["conditions"][condition])
    complete = json.loads((run / "complete.json").read_text())
    manifest = json.loads((run / "train_manifest.json").read_text())
    if complete["smoke_only"] or complete["last_epoch"] != 3 or manifest["model_revision"] != sft.MODEL_REVISION:
        raise ValueError("Need completed 3-epoch SFT")
    if set(manifest["ids"]) & {q["id"] for q in plan["questions"]}:
        raise ValueError("Training IDs overlap test")
    bg.gpu_guard({"2", "3"})
    cfg = {"condition": condition, "model_lock": plan["model_lock"], "seed": plan["seed"],
           "adapter": str(run / "adapter")}
    model_out = out / "models" / condition
    rows = bg._question_rows(plan["questions"])
    with output_lock(model_out / "generation"):
        runtime = original.PinnedRuntime(cfg)
        runtime.identity.update(answer_instruction=INSTRUCTION,
                                answer_protocol_sha256=file_digest(__file__))
        bg._freeze_identity(model_out, runtime, cfg)
        for method in METHODS:
            folder = model_out / "generation" / method
            spec = bg._spec(runtime, rows, "generation", method=method,
                            final_tokens=plan["final_tokens"], review_tokens=plan["review_tokens"],
                            answer_instruction=INSTRUCTION, cot_reasoning_prompt_changed=False)
            bg._atomic_json(folder / "spec.json", spec, frozen=True)
            exported = []
            for index, q in enumerate(rows, 1):
                signature = digest([spec, q])
                path = folder / "cache" / (digest(q["id"]) + ".json")
                saved = bg._cache_record(path, signature)
                if saved is None:
                    record = answer_one(runtime, model_out, q, method, plan["final_tokens"], plan["review_tokens"])
                    record.update(condition=condition, method=f"{condition}_{method}", signature=signature)
                    bg._atomic_json(path, record, frozen=True)
                    print(f"[{condition}/{method}] {index}/{len(rows)}: {record['output_form']}", flush=True)
                else:
                    record = saved
                exported.append(record)
            write_jsonl(out / "answers" / f"{condition}_{method}.jsonl", exported)


def judge_plan(out):
    out = Path(out)
    plan, _ = load_plan(out)
    expected = {q["id"] for q in plan["questions"]}
    paths = [out / "answers" / f"{c}_{m}.jsonl" for c in CONDITIONS for m in METHODS]
    for p in paths:
        rows = list(read_jsonl(p))
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError("Incomplete SFT generation")
    return well.prepare(out / "questions.jsonl", paths, out / "well", backend="claude", model="claude-sonnet-5")


def report(out):
    out = Path(out)
    plan, protocol = load_plan(out)
    summary = well.report(out / "well")
    judge, plan_hash = well.load_plan(out / "well")
    _, scores = well.read_ledger(out / "well", judge, plan_hash)
    historical = {(r["method"], r["id"]): r["score"] for r in base_scores(plan["questions"])}
    results, cases = [], []
    for condition in ("historical_base", *CONDITIONS):
        for method in METHODS:
            rows = {} if condition == "historical_base" else {r["id"]: r for r in read_jsonl(out / "answers" / f"{condition}_{method}.jsonl")}
            for kind in ("fpq", "nfp"):
                qs = [q for q in plan["questions"] if q["set"] == kind]
                counts, forms = Counter(), Counter()
                success4 = success5 = capped = reasoning_capped = 0
                for q in qs:
                    if condition == "historical_base":
                        rating = historical[method, q["id"]]
                        answered = True  # Historical CSV contains ratings, not raw responses.
                    else:
                        r = rows[q["id"]]
                        rating = scores.get(judge["mapping"][f"{condition}_{method}"][q["id"]])
                        answered = r["final_answer_present"]
                        forms[r["output_form"]] += 1
                        capped += bool(r["details"]["answer"]["cap_hit"])
                        reasoning_capped += bool(r["details"].get("reasoning", {}).get("cap_hit"))
                        cases.append({"condition": condition, "prompt_method": method, "id": q["id"],
                            "set": kind, "question": q["question"], "response": r["response"],
                            "well_rating": rating, "output_form": r["output_form"],
                            "historical_base_rating": historical[method, q["id"]],
                            "matched_prompt_comparison": False, "final_truncated": r["details"]["answer"]["cap_hit"]})
                    if rating is not None:
                        counts[rating] += 1
                        success4 += rating >= 4 and answered
                        success5 += rating == 5 and answered
                valid = sum(counts.values())
                readable = sum(counts[v] for v in range(1, 6))
                results.append({"condition": condition, "prompt_method": method, "set": kind,
                    "expected": len(qs), "valid": valid, "missing": len(qs)-valid,
                    "answer_instruction_added": condition != "historical_base",
                    "s4_count": success4, "s5_count": success5,
                    "s4_over_valid": success4/valid if valid else None,
                    "s5_over_valid": success5/valid if valid else None,
                    "raw_well_counts": dict(counts),
                    "raw_well_mean_1_to_5": sum(v*counts[v] for v in range(1,6))/readable if readable else None,
                    "output_forms": dict(forms) if rows else None,
                    "final_truncated": capped if rows else None,
                    "reasoning_truncated": reasoning_capped if rows else None})
    result = {"complete": summary["unstarted"] == 0 and summary["invalid_or_interrupted"] == 0,
              "judge": "claude-sonnet-5", "protocol": protocol, "rows": results}
    sft.write_json(out / "table2.json", result)
    write_jsonl(out / "table2_cases.jsonl", cases)
    lines = ["# SFT + explicit final answer instruction", "", protocol["comparison_note"], "",
             "Diagnostic-only/incomplete outputs cannot count as answer success; raw Well scores remain separate. Missing ratings are not imputed. Mean excludes zero.", "",
             f"Complete: {result['complete']}", "",
             "| Model | Prompt | Set | Answer instruction | Valid/expected | ≥4 | S5 | Raw Well mean 1–5 | Diagnostic-only/incomplete | Final cap |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        pct = lambda v: "NA" if v is None else f"{100*v:.1f}%"
        mean = r["raw_well_mean_1_to_5"]
        forms = r["output_forms"]
        diag = sum(forms.get(k,0) for k in ("diagnostic_only","diagnostic_incomplete")) if forms is not None else "NA"
        lines.append(f"| {r['condition']} | {r['prompt_method']} | {r['set']} | {r['answer_instruction_added']} | {r['valid']}/{r['expected']} | {pct(r['s4_over_valid'])} | {pct(r['s5_over_valid'])} | {f'{mean:.3f}' if mean is not None else 'NA'} | {diag} | {r['final_truncated'] if r['final_truncated'] is not None else 'NA'} |")
    (out / "table2.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "generate", "judge-plan", "score", "report"):
        p = subs.add_parser(stage)
        p.add_argument("--out", required=True)
        if stage == "prepare":
            p.add_argument("--art", required=True)
            p.add_argument("--cancer-run", required=True)
            p.add_argument("--crepe-run", required=True)
        elif stage == "generate":
            p.add_argument("--condition", choices=CONDITIONS, required=True)
        elif stage == "score":
            p.add_argument("--max-calls", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.art, args.out, args.cancer_run, args.crepe_run)
    elif args.stage == "generate":
        generate(args.out, args.condition)
        return
    elif args.stage == "judge-plan":
        result = judge_plan(args.out)
    elif args.stage == "score":
        result = well.score(Path(args.out) / "well", max_calls=args.max_calls, timeout=240)
    else:
        result = {"complete": report(args.out)["complete"]}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
