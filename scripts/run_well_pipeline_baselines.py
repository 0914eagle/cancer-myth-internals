#!/usr/bin/env python3
"""Well PreWoMe / extract-check-answer on frozen Cancer-Myth Luna test questions.

No RAG. Official four-shot examples; primary matched comparison excludes example
overlap. Stage-level resume does not resample completed outputs. Transport errors
stop the run; their outputs are not assigned clinical scores.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpqa_prompting import digest, freeze, load_rows, response_metrics
from src.fpqa_well_upstream import (UncachedCaller, FatalTransportError, load_upstream,
                                     make_args, verify_source, WELL_REVISION)
from src.fpqa_well_pipelines import load_templates, extract, answer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(question):
    return " ".join(question.casefold().split())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--well-root", type=Path, default=ROOT/"external/fpqa_20261002/Well")
    p.add_argument("--reference-root", type=Path,
                   default=ROOT/"results/fpqa_prompting/well_upstream_test_v1_20261003_run")
    p.add_argument("--test", type=Path,
                   default=ROOT/"results/fpqa_prompting/cancer_data_test100_v2/cancer_myth/test.jsonl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke", action="store_true", help="First FPQ and NFP only; same frozen run")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if not 1 <= a.workers <= 5:
        p.error("workers must be 1..5")
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out/".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/"STOP").exists():
        raise RuntimeError("STOP present")
    if (out/"run.json").exists() and not (a.resume or a.preflight):
        raise RuntimeError("Use --resume for an existing run")
    source = verify_source(a.well_root)
    reference = json.loads((a.reference_root/"run.json").read_text())
    if reference["source_sha256"] != source:
        raise ValueError("Reference Well source changed")
    config = reference["configs"]["cancer_myth"]
    rows = load_rows(a.test)
    old = reference["provenance"]["cancer_myth"]
    if old["test_sha256"] != sha(a.test) or old["test_ids"] != [r["id"] for r in rows]:
        raise ValueError("Reference test set differs")
    if Counter(r["label"] for r in rows) != {1: 100, 0: 100} or len({r['id'] for r in rows}) != 200:
        raise ValueError("Expected 200 unique questions, 100/100")
    fs_path = a.well_root/"data_gen/CancerMyth/few_shot_data.json"
    few_shot = json.loads(fs_path.read_text())
    if few_shot != json.loads((a.well_root/"data_gen/CancerMythNFP/few_shot_data.json").read_text()):
        raise ValueError("Label-specific few-shot examples differ")
    overlap = [r["id"] for r in rows if normalized(r["question"]) in
               {normalized(x["question"]) for x in few_shot}]
    references = {}
    reference_hashes = {}
    for method in ("plain", "gepa"):
        references[method] = []
        for row in rows:
            path = a.reference_root/"cancer_myth"/method/digest(row["id"])/"record.json"
            rr = json.loads(path.read_text())
            if (rr["id"], rr["label"], rr["feedback"]["question"]) != (row["id"], row["label"], row["question"]):
                raise ValueError("Reference record mismatch")
            references[method].append(rr)
            reference_hashes[f"{method}/{row['id']}"] = sha(path)
    local = threading.local(); stopped = threading.Event()
    def dispatch(role, messages):
        return local.call("judge", role, messages)
    balanced, direct, _ = load_upstream(a.well_root, dispatch)
    templates = load_templates(a.well_root)
    original_args = make_args(balanced, "CancerMyth", out/"unused_inputs", out, config)
    from constant.constant import MODEL_FINAL_ANSWER_KEY
    upstream_files = ["prompting/run_prewome.py", "prompting/run_presupposition_pipeline.py",
                      "response/response.py", "data_gen/CancerMyth/few_shot_data.json",
                      "data_gen/CancerMythNFP/few_shot_data.json"]
    plan = {"protocol": "well_pipeline_official_fourshot_no_rag_luna_v1",
            "well_revision": WELL_REVISION, "upstream_sha256": {f: sha(a.well_root/f) for f in upstream_files},
            "judge_source_sha256": source, "test_sha256": sha(a.test), "config": config,
            "test_ids": [r["id"] for r in rows], "fewshot_overlap_ids": overlap,
            "n_answers": 400, "n_primary_fpq": 100-sum(r["id"] in overlap and r["label"] == 1 for r in rows),
            "n_primary_nfp": 100-sum(r["id"] in overlap and r["label"] == 0 for r in rows),
            "reference_root": str(a.reference_root.resolve()), "reference_sha256": reference_hashes,
            "adapter_sha256": {f: sha(ROOT/f) for f in ["src/fpqa_well_pipelines.py",
                "scripts/run_well_pipeline_baselines.py", "src/fpqa_well_upstream.py", "src/fpqa_cli_backend.py"]},
            "transport_differences": reference["remaining_transport_differences"],
            "scope": "Exploratory comparison on existing test. Official four-shot pipelines vs zero-shot Plain/optimized GEPA; not an isolated decomposition ablation. Primary analysis excludes demonstration overlap in all methods. Model and transport differ from original Gemini study."}
    freeze(out/"preflight.json", plan)
    freeze(out/"few_shot_data.json", few_shot)
    if a.preflight:
        print(json.dumps({k: plan[k] for k in ("protocol", "n_answers", "fewshot_overlap_ids", "n_primary_fpq", "n_primary_nfp")}, indent=2))
        return
    freeze(out/"run.json", plan)
    selected = [next(r for r in rows if r['label'] == y) for y in (1, 0)] if a.smoke else rows

    def process(row):
        folder = out/"items"/row["id"]
        def call(stage, role, messages):
            if stopped.is_set() or (out/"STOP").exists():
                raise FatalTransportError("STOP requested")
            location = folder/"stages"/stage
            request = {"role": role, "messages": messages, "config": config[f"{role}_model"]}
            freeze(location/"request.json", request)
            saved = location/"result.json"
            if saved.exists():
                return json.loads(saved.read_text())["text"]
            attempts = location/"attempts"
            attempt = attempts/f"{len(list(attempts.glob('*')))+1:03d}"
            text = UncachedCaller(config, attempt)(role, messages)
            if not text.strip():
                raise ValueError("Empty model output; not a valid completed stage")
            freeze(saved, {"text": text, "attempt": str(attempt.relative_to(out))})
            return text
        try:
            claims = extract(row["question"], few_shot, templates, call)
            freeze(folder/"claims.json", {"id": row["id"], "claims": claims})
            for method in ("prewome", "atomic"):
                path = folder/method/"record.json"
                if path.exists():
                    continue
                generated = answer(row["question"], few_shot, claims, method, templates, call)
                # Score the generated answer with the same unchanged official judge
                # classes and zero judge demonstrations as the reference GEPA run.
                def judge_call(stage, role, messages):
                    return call(f"{method}/judge", role, messages)
                local.call = judge_call
                evaluator = balanced.BalancedDirectQAGEPAEvaluator(original_args, "gemini")
                evaluator.ScoreTemplateClass = evaluator.template_classes['fpq' if row['label'] else 'tpq']
                dp = {"id": row["id"], "question": row["question"],
                      "answer": row["reference_answer"], "presuppositions": row["presuppositions"],
                      "few_shot_data": [], "passages": [], MODEL_FINAL_ANSWER_KEY: generated["answer"]}
                score, explanation = evaluator.score_response(dp)
                freeze(path, {"id": row["id"], "label": row["label"], "method": method,
                    "question": row["question"], "claims": claims, **generated,
                    "score": score, "score_explanation": explanation,
                    "fewshot_overlap": row["id"] in overlap})
                print(f"COMPLETE {row['id']} {method}: {score}", flush=True)
        except BaseException as error:
            stopped.set()
            freeze(folder/"errors"/f"{len(list((folder/'errors').glob('*')))+1:03d}.json",
                   {"error_type": type(error).__name__, "message": str(error)})
            raise

    def status(state):
        data = {"state": state, "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "total": 400, "completed": 0, "methods": {}, "fewshot_overlap_ids": overlap}
        for method in ("prewome", "atomic"):
            records = [json.loads(path.read_text()) for path in (out/"items").glob(f"*/{method}/record.json")]
            data["completed"] += len(records)
            done_ids = {r['id'] for r in records if not r['fewshot_overlap']}
            data['methods'][method] = {"n_completed": len(records),
                "all_200_including_overlap": response_metrics(records),
                "primary_excluding_overlap": response_metrics([r for r in records if not r['fewshot_overlap']]),
                "matched_reference_on_completed_primary_ids": {
                    m: response_metrics([r for r in rr if r['id'] in done_ids]) for m, rr in references.items()}}
        temp = out/"status.tmp"; temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
        temp.replace(out/"status.json")
        return data
    status("running")
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = [pool.submit(process, row) for row in selected]
            for future in as_completed(futures):
                future.result(); status("running")
        data = status("smoke_complete" if a.smoke else "complete")
        if not a.smoke:
            if data["completed"] != 400:
                raise RuntimeError("Incomplete run cannot be marked complete")
            freeze(out/"metrics.json", data)
    except BaseException:
        status("failed")
        raise


if __name__ == "__main__":
    main()
