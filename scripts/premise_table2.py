"""Matched Plain/CoT/unconditional-FP answers before and after premise SFT.

Uses the existing baseline prompts, two-call CoT and greedy generation exactly.
prepare/report are offline; generate uses one GPU; score explicitly caps calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import premise_sft as sft
from src import baseline_generation as bg
from src.jsonl import read_jsonl, write_jsonl
from src.pilot import digest, file_digest, frozen_json, normalized
from src import well_eval as well

METHODS = ("plain", "zero_shot_cot", "fp_unconditional")
CONDITIONS = ("base", "cancer", "crepe")
VERSION = "premise-table2-v1"


def prepare(art, out, cancer_run, crepe_run):
    art, out = Path(art).resolve(), Path(out).resolve()
    # Verify the bundled source and its frozen grouped split before joining references.
    sft.audit_splits(sft.cancer_rows())
    canonical = list(read_jsonl(sft.BUNDLE / "cancer_questions.jsonl"))
    questions = [dict(q) for q in canonical if q["partition"] == "test"]
    prepared = list(read_jsonl(art / "data/cancer/test.jsonl"))
    if [(q["id"], q["question"]) for q in questions] != [(q["id"], q["question"]) for q in prepared]:
        raise ValueError("Prepared test differs from the canonical inventory")
    demos = {normalized(q) for q in well.example_questions()}
    if any(normalized(q["question"]) in demos for q in questions):
        raise ValueError("Test overlaps Well demonstrations; do not silently change denominator")
    eval_texts = {normalized(q["question"]) for q in questions}
    for name in ("cancer", "crepe"):
        fit = list(read_jsonl(art / f"data/{name}/fit.jsonl"))
        if eval_texts & {normalized(q["question"]) for q in fit}:
            raise ValueError(f"Test question appears in {name} training")
    references = {}
    raw = art / "raw/cancer/validation.jsonl"
    for r in read_jsonl(raw):
        key = normalized(r["question"])
        pair = (r["source_myth"], r["presupposition_correction"])
        if key in references and references[key] != pair:
            raise ValueError("Conflicting original FPQ references")
        references[key] = pair
    reference_changes = []
    for q in questions:
        if q["set"] == "fpq":
            premise, correction = references[normalized(q["question"])]
            if (q["premise_text"], q["correction"]) != (premise, correction):
                reference_changes.append({"id": q["id"], "bundle_premise": q["premise_text"],
                                          "bundle_correction": q["correction"]})
            q["premise_text"], q["correction"] = premise, correction
        well.judge_messages(q, "")
    lock = art / "downloads.json"
    sft.model_location(lock)
    plan = {"version": VERSION, "art": str(art), "model_lock": str(lock),
            "model_lock_sha256": file_digest(lock), "questions": questions,
            "counts": dict(Counter(q["set"] for q in questions)),
            "conditions": {"base": None, "cancer": str(art / "runs" / cancer_run),
                           "crepe": str(art / "runs" / crepe_run)},
            "methods": list(METHODS), "final_tokens": 1024, "review_tokens": 1024,
            "seed": 17, "decoding": "greedy; one native user turn per call; no added system",
            "prompts": {k: bg.PROMPTS[k] for k in ("cot_reason", "cot_final", "fp_answer")},
            "sources": {"canonical_sha256": file_digest(sft.BUNDLE / "cancer_questions.jsonl"),
                        "original_fpq_sha256": file_digest(raw)},
            "reference_changes": reference_changes,
            "implementation_sha256": file_digest(__file__),
            "baseline_generation_sha256": file_digest(bg.__file__)}
    frozen_json(out / "experiment.json", plan)
    path = out / "questions.jsonl"
    if path.exists():
        if list(read_jsonl(path)) != questions:
            raise ValueError("Evaluation inventory changed")
    else:
        write_jsonl(path, questions)
    return plan


def load_plan(out):
    out = Path(out)
    plan = json.loads((out / "experiment.json").read_text())
    if plan["version"] != VERSION or plan["implementation_sha256"] != file_digest(__file__):
        raise ValueError("Experiment implementation changed; use a new run")
    if plan["baseline_generation_sha256"] != file_digest(bg.__file__):
        raise ValueError("Baseline generation implementation changed")
    if plan["model_lock_sha256"] != file_digest(plan["model_lock"]):
        raise ValueError("Model lock changed")
    if list(read_jsonl(out / "questions.jsonl")) != plan["questions"]:
        raise ValueError("Evaluation inventory changed")
    return plan


class PinnedRuntime(bg.ModelRuntime):
    """Reuse baseline rendering/decoding with the pinned base or a completed adapter."""
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        bg.gpu_guard({"2", "3"})
        if torch.__version__ != "2.5.1+cu121" or torch.cuda.device_count() != 1:
            raise ValueError("Expected pinned torch and exactly one visible GPU")
        torch.manual_seed(cfg["seed"])
        self.torch = torch
        path = sft.model_location(cfg["model_lock"])
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, use_safetensors=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map=None).to("cuda:0")
        adapter = cfg.get("adapter")
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True
        self.identity = {"model_id": sft.MODEL, "model_revision": sft.MODEL_REVISION,
                         "condition": cfg["condition"], "adapter": adapter,
                         "adapter_sha256": file_digest(Path(adapter) / "adapter_model.safetensors") if adapter else None,
                         "adapter_config_sha256": file_digest(Path(adapter) / "adapter_config.json") if adapter else None,
                         "chat_template_sha256": digest(self.tokenizer.chat_template),
                         "versions": sft.runtime(), "seed": cfg["seed"],
                         "table2_implementation_sha256": file_digest(__file__)}


def output_form(text):
    """Keep diagnostic-only output distinct without rewriting the model answer."""
    value = text.strip()
    if not value:
        return "empty"
    if value.startswith("```"):
        match = re.match(r"```(?:json)?\s*\n?(.*?)\n?```(.*)$", value, re.S | re.I)
        if match:
            value = match[1].strip() + "\n" + match[2].strip()
    try:
        obj, end = json.JSONDecoder().raw_decode(value)
    except ValueError:
        if re.match(r'^\{\s*"(?:has_false_premise|false_premises)"', value):
            return "diagnostic_incomplete"
        return "text"
    if isinstance(obj, dict) and ("has_false_premise" in obj or "false_premises" in obj):
        tail = value[end:].strip()
        if tail in ("", "Answer:"):
            return "diagnostic_only"
        return "diagnostic_with_answer"
    return "text"


def method_name(condition, method):
    return "plain" if condition == "base" and method == "plain" else f"{condition}_{method}"


def generate(out, condition, methods):
    out = Path(out)
    plan = load_plan(out)
    run = plan["conditions"][condition]
    if run:
        complete = json.loads((Path(run) / "complete.json").read_text())
        manifest = json.loads((Path(run) / "train_manifest.json").read_text())
        if complete["smoke_only"] or complete["last_epoch"] != 3 or manifest["model_revision"] != sft.MODEL_REVISION:
            raise ValueError("Need completed 3-epoch main training from the pinned base")
        test_ids = {q["id"] for q in plan["questions"]}
        if test_ids & set(manifest["ids"]):
            raise ValueError("Training IDs overlap test")
    if not methods or any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        raise ValueError("Select the three authorized prompt conditions")
    cfg = {"condition": condition, "model_lock": plan["model_lock"], "seed": plan["seed"],
           "adapter": str(Path(run) / "adapter") if run else None}
    model_out = out / "models" / condition
    bg.run_generation(plan["questions"], cfg, model_out, methods,
                      final_tokens=plan["final_tokens"], review_tokens=plan["review_tokens"],
                      runtime_factory=PinnedRuntime, allowed_gpus={"2", "3"})
    for method in methods:
        rows = bg.load_generation_records(model_out, method)
        if len(rows) != len(plan["questions"]) or any(r["status"] != "complete" for r in rows):
            raise ValueError(f"Incomplete generation for {condition}/{method}; inspect item caches")
        for r in rows:
            r["prompt_method"], r["condition"] = method, condition
            r["method"] = method_name(condition, method)
            r["output_form"] = output_form(r["response"])
            r["final_answer_present"] = r["output_form"] not in {"empty", "diagnostic_only", "diagnostic_incomplete"}
        write_jsonl(out / "answers" / f"{condition}_{method}.jsonl", rows)
    print(json.dumps({"condition": condition, "completed_methods": methods}), flush=True)


def judge_plan(out, backend, model):
    out = Path(out)
    plan = load_plan(out)
    paths = [out / "answers" / f"{c}_{m}.jsonl" for c in CONDITIONS for m in METHODS]
    expected = {q["id"] for q in plan["questions"]}
    for path in paths:
        rows = list(read_jsonl(path))
        if len(rows) != len(expected) or {r["id"] for r in rows} != expected:
            raise ValueError(f"Incomplete answer file: {path}")
    return well.prepare(out / "questions.jsonl", paths, out / "well", backend=backend, model=model)


def report(out):
    out = Path(out)
    plan = load_plan(out)
    summary = well.report(out / "well")
    judge, plan_hash = well.load_plan(out / "well")
    _, scores = well.read_ledger(out / "well", judge, plan_hash)
    results, cases = [], []
    for condition in CONDITIONS:
        for method in METHODS:
            name = method_name(condition, method)
            rows = {r["id"]: r for r in read_jsonl(out / "answers" / f"{condition}_{method}.jsonl")}
            base_rows = {r["id"]: r for r in read_jsonl(out / "answers" / f"base_{method}.jsonl")}
            base_name = method_name("base", method)
            for kind in ("fpq", "nfp"):
                qs = [q for q in plan["questions"] if q["set"] == kind]
                counts, forms = Counter(), Counter()
                successes4 = successes5 = valid = paired = rescue = harm = 0
                capped = reasoning_capped = 0
                for q in qs:
                    r = rows[q["id"]]
                    rating = scores.get(judge["mapping"][name][q["id"]])
                    base_rating = scores.get(judge["mapping"][base_name][q["id"]])
                    forms[r["output_form"]] += 1
                    capped += bool(r["details"]["answer"]["cap_hit"])
                    reasoning_capped += bool(r["details"].get("reasoning", {}).get("cap_hit"))
                    success4 = rating is not None and rating >= 4 and r["final_answer_present"]
                    success5 = rating == 5 and r["final_answer_present"]
                    if rating is not None:
                        valid += 1
                        counts[rating] += 1
                        successes4 += success4
                        successes5 += success5
                    # The base and adapter are paired under the SAME prompt method.
                    base_row = base_rows[q["id"]]
                    if rating is not None and base_rating is not None:
                        paired += 1
                        base_success4 = base_rating >= 4 and base_row["final_answer_present"]
                        rescue += success4 and not base_success4
                        harm += base_success4 and not success4
                    cases.append({"condition": condition, "prompt_method": method, "id": q["id"],
                                  "set": kind, "question": q["question"], "response": r["response"],
                                  "base_response": base_row["response"], "well_rating": rating,
                                  "base_well_rating": base_rating, "output_form": r["output_form"],
                                  "final_answer_present": r["final_answer_present"],
                                  "final_truncated": r["details"]["answer"]["cap_hit"]})
                readable = sum(counts[v] for v in range(1, 6))
                results.append({"condition": condition, "prompt_method": method, "set": kind,
                    "expected": len(qs), "valid": valid, "missing": len(qs)-valid,
                    "answer_s4_count": successes4, "answer_s5_count": successes5,
                    "answer_s4_over_valid": successes4/valid if valid else None,
                    "answer_s5_over_valid": successes5/valid if valid else None,
                    "raw_judge_counts_0_to_5": dict(counts),
                    "raw_judge_mean_1_to_5": sum(v*counts[v] for v in range(1, 6))/readable if readable else None,
                    "mean_denominator": readable, "output_forms": dict(forms),
                    "final_truncated": capped, "reasoning_truncated": reasoning_capped,
                    "paired_base": {"n": paired, "s4_rescue": rescue, "s4_harm": harm}})
    result = {"judge": {"backend": judge["backend"], "model": judge["model"]},
              "complete": summary["unstarted"] == 0 and summary["invalid_or_interrupted"] == 0,
              "note": "Same held-out 147 questions across nine rows. Rates use valid judgments; missing shown. Diagnostic-only/incomplete outputs never count as answer successes. Raw judge scores retained separately. Mean excludes zero. Fixed 1024 budgets; length stops reported, not silently regenerated.",
              "rows": results}
    sft.write_json(out / "table2.json", result)
    write_jsonl(out / "table2_cases.jsonl", cases)
    lines = ["# SFT before/after: Table 2", "", result["note"], "",
             f"Judge: {judge['backend']} / {judge['model']}. Complete: {result['complete']}", "",
             "| Model | Prompt | Set | Valid/expected | ≥4 | S5 | Mean 1–5 | Diagnostic-only/incomplete | Final cap |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        def percent(v):
            return "NA" if v is None else f"{100*v:.1f}%"
        mean = r["raw_judge_mean_1_to_5"]
        diag = sum(r["output_forms"].get(k, 0) for k in ("diagnostic_only", "diagnostic_incomplete"))
        lines.append(f"| {r['condition']} | {r['prompt_method']} | {r['set']} | {r['valid']}/{r['expected']} | {percent(r['answer_s4_over_valid'])} | {percent(r['answer_s5_over_valid'])} | {mean if mean is not None else 'NA'} | {diag} | {r['final_truncated']} |")
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
            p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
        elif stage == "judge-plan":
            p.add_argument("--backend", choices=well.BACKENDS, default="claude")
            p.add_argument("--model", default="claude-sonnet-5")
        elif stage == "score":
            p.add_argument("--max-calls", type=int, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        value = prepare(args.art, args.out, args.cancer_run, args.crepe_run)
        print(json.dumps({"counts": value["counts"], "methods": value["methods"], "max_answer_rows": 9*len(value["questions"])}))
    elif args.stage == "generate":
        generate(args.out, args.condition, args.methods)
    elif args.stage == "judge-plan":
        print(json.dumps(judge_plan(args.out, args.backend, args.model)))
    elif args.stage == "score":
        print(json.dumps(well.score(Path(args.out)/"well", max_calls=args.max_calls, timeout=240)))
    else:
        value = report(args.out)
        print(json.dumps({"complete": value["complete"], "table": str(Path(args.out)/"table2.md")}))


if __name__ == "__main__":
    main()
