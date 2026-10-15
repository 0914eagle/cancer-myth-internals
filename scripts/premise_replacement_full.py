"""Full-suite premise replacement diagnostic: scale-up of the fit-40 pilot in
docs/reviews/extraction_audit_2026-09-22/replacement_v1 to every question with a saved
Extract+Verify result (725 of 732). Same Qwen verifier, same VERIFY prompt, same fixed rule
(score = 1 - min P(True), gate = any P(True) < 0.5). No answer generation, no judge.

What changes against the pilot: the reviewed premise lists are written by Sonnet through
`claude -p`, LABEL-BLIND (the writer sees the question and the numbered extracted sentences
only; never the FPQ/NFP label, the reference premise or the correction). The pilot's writer
had seen the labels. The 40 pilot questions are re-written the same way so the two writers
can be compared on the overlap.

    export SUITE_DIR=/data1/heejae/cancer_myth_internals/results/baselines/qwen25_7b_final1024_v1
    export OUT=$SUITE_DIR/controls/premise_replacement_full_v1
    PY=/data1/heejae/uv/cancer_myth_internals/bin/python
    # 1) make: one Sonnet call per question (resumable; a parse/evidence failure is retried once)
    $PY scripts/premise_replacement_full.py make --suite-dir $SUITE_DIR --out-dir $OUT --workers 3
    # 2) prepare + run on GPU 0 (re-forwards the original premises too, to check drift against the saved scores)
    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
      $PY scripts/premise_replacement_full.py run --suite-dir $SUITE_DIR --out-dir $OUT \
      --pilot-dir docs/reviews/extraction_audit_2026-09-22
    # 3) report (no GPU); --docs-dir copies report.md / summary.json / items.md into the repo
    $PY scripts/premise_replacement_full.py report --suite-dir $SUITE_DIR --out-dir $OUT \
      --pilot-dir docs/reviews/extraction_audit_2026-09-22 --docs-dir docs/reviews/extraction_audit_2026-09-22/replacement_full_v1
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jsonl import read_jsonl  # noqa: E402

WRITER_PROMPT = (
    "A patient's question is given below, followed by a numbered list of sentences that were automatically "
    "extracted from it as 'premises'. Each sentence on the final list will later be checked True or False by a "
    "fact-checking model, so the list must contain exactly the factual premises the question takes for granted, "
    "each stated as one checkable sentence that is faithful to the question.\n\n"
    "Revise the list with three operations:\n"
    "KEEP a sentence when it faithfully states something the question asserts or takes for granted: a diagnosis, "
    "a treatment or recommendation, a fact about the disease, or a stated circumstance. Keep its scope and "
    "qualifiers as they are.\n"
    "DROP a sentence when it merely restates the request, a feeling, a plan or a wish (for example 'We are looking "
    "for ...', 'She is devastated', 'We are planning a trip'), when it duplicates another sentence, or when it asserts "
    "something the question does not commit to (for example turning 'Are there programs for X?' into 'There are "
    "programs for X').\n"
    "ADD a sentence when the question takes something for granted that no kept sentence states: a claim embedded in "
    "hearsay, belief, worry or a request (for example 'He's heard that surgery is the only option' -> 'Surgery is the "
    "only option for men with breast cancer'; 'programs for contagious conditions like this' -> 'The daughter's "
    "condition is contagious'), or a stated fact or observation that the list missed. State the claim itself, not "
    "the fact that someone believes it. If the question only asks whether something is possible or true, do not turn "
    "it into an assertion. Keep individual circumstances individual: do not turn one patient's case into a universal "
    "claim, and do not add anything the question does not commit to.\n\n"
    "Do not judge whether any premise is true or false and do not correct anything: you are reproducing what the "
    "question assumes, not fact-checking it.\n\n"
    "Reply with ONLY a JSON object of the form\n"
    '{{"keep": [numbers of the sentences kept], "add": [{{"premise": "one checkable sentence", "evidence": "a phrase '
    'copied verbatim from the question that supports it"}}], "note": "one short sentence on what you changed"}}\n'
    "The evidence must be an exact substring of the question. Use empty lists when the question asserts no factual "
    "premise.\n\n"
    "QUESTION: {question}\n\n"
    "EXTRACTED SENTENCES:\n{premises}\n\n"
    "JSON:"
)

RETRY_SUFFIX = (
    "\n\nYour previous reply was rejected: {error}. Reply again with only the JSON object, and make sure every "
    "evidence string is copied exactly from the question."
)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _norm(text):
    return " ".join(str(text or "").translate(_QUOTES).split())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value, frozen=False):
    path = Path(path)
    if frozen and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen file changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def parse_edit(reply, n_original, question):
    """Writer JSON -> {'keep': [...], 'add': [...], 'note': str}; raises ValueError with the reason."""
    if not isinstance(reply, str):
        raise ValueError("reply is not text")
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    try:
        data = json.loads(m.group(0))
    except ValueError as exc:
        raise ValueError(f"JSON does not parse ({exc})") from exc
    if not isinstance(data, dict) or "keep" not in data or "add" not in data:
        raise ValueError("JSON must have 'keep' and 'add'")
    keep = []
    for k in data["keep"] if isinstance(data["keep"], list) else [None]:
        if isinstance(k, str) and k.strip().isdigit():
            k = int(k.strip())
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n_original:
            raise ValueError(f"keep index {k!r} is not an integer in 1..{n_original}")
        if k in keep:
            raise ValueError(f"keep index {k} repeated")
        keep.append(k)
    adds = []
    if not isinstance(data["add"], list) or len(data["add"]) > 8:
        raise ValueError("'add' must be a list of at most 8 items")
    qn = _norm(question)
    for a in data["add"]:
        if not isinstance(a, dict):
            raise ValueError("each 'add' item must be an object")
        premise = _norm(a.get("premise"))
        evidence = str(a.get("evidence") or "").strip()
        if not premise or len(premise) > 300:
            raise ValueError("an added premise is empty or longer than 300 characters")
        if not evidence:
            raise ValueError(f"no evidence for the added premise {premise!r}")
        exact = evidence in question
        if not exact and _norm(evidence) not in qn:
            raise ValueError(f"evidence {evidence!r} is not a substring of the question")
        if any(premise == x["premise"] for x in adds):
            continue
        adds.append({"premise": premise, "evidence": evidence, "evidence_exact": exact})
    note = _norm(data.get("note", ""))[:400]
    return {"keep": sorted(keep), "add": adds, "note": note}


def arms_for(original, edit):
    kept = [original[i - 1] for i in edit["keep"]]
    reviewed = list(kept)
    for a in edit["add"]:
        if a["premise"] not in reviewed:
            reviewed.append(a["premise"])
    return {"original": list(original), "retained_only": kept, "reviewed": reviewed}


def score_rule(true_scores):
    scores = list(true_scores)
    return {"score": 1 - min(scores, default=1), "gate": any(s < 0.5 for s in scores)}


def load_saved(suite_dir):
    suite_dir = Path(suite_dir)
    questions = {r["id"]: r for r in read_jsonl(suite_dir / "questions.jsonl")}
    answers = {r["id"]: r for r in read_jsonl(suite_dir / "answers/extract_verify.jsonl")}
    for key, a in answers.items():
        q = questions[key]
        if q["question"] != a["question"] or q["set"] != a["set"] or q["partition"] != a["partition"]:
            raise ValueError(f"saved answer does not match the question inventory: {key}")
    return questions, answers


# ---------------------------------------------------------------- make

def stage_make(args):
    from src.llm_backend import backend_available, make_caller
    from src.pilot import output_lock
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions, answers = load_saved(args.suite_dir)
    if not backend_available(args.backend, claude_cmd=args.claude_cmd):
        raise SystemExit(f"backend {args.backend} not available")
    call = make_caller(args.backend, args.model, timeout=args.timeout, claude_cmd=args.claude_cmd)
    path = out / "replacements.jsonl"
    done = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    todo = [k for k in sorted(answers) if k not in done or (args.retry_failed and done[k]["status"] != "ok")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[make] {len(answers)} questions with a saved extraction; {len(done)} done; {len(todo)} to write; "
          f"writer {args.backend}:{args.model}; workers {args.workers}", flush=True)

    def one(key):
        q, a = questions[key], answers[key]
        original = [p["premise"] for p in a["details"]["premises"]]
        numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(original, 1)) or "(none)"
        prompt = WRITER_PROMPT.format(question=q["question"], premises=numbered)
        row = {"id": key, "set": q["set"], "partition": q["partition"], "n_original": len(original),
               "status": "failed", "attempts": 0, "served_model": None, "error": None, "raw": None}
        error = None
        for attempt in range(2):
            row["attempts"] = attempt + 1
            text = prompt if error is None else prompt + RETRY_SUFFIX.format(error=error)
            try:
                reply, served = call(text)
            except Exception as exc:  # noqa: BLE001 - transport failure after its own retries
                error = f"backend error: {exc.__class__.__name__}: {str(exc)[:200]}"
                time.sleep(5)
                continue
            row["served_model"] = served
            row["raw"] = reply
            try:
                edit = parse_edit(reply, len(original), q["question"])
            except ValueError as exc:
                error = str(exc)
                continue
            row.update(edit)
            row["status"] = "ok"
            row["error"] = None
            return row
        row["error"] = error
        return row

    stats = Counter()
    started = time.time()
    with output_lock(out / "make"), ThreadPoolExecutor(max_workers=args.workers) as pool, \
            path.open("a", encoding="utf-8") as f:
        futures = {pool.submit(one, key): key for key in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if args.retry_failed and row["id"] in done:
                pass  # the report reads the LAST row per id, so an appended retry supersedes the failure
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            stats[row["status"]] += 1
            if i % 25 == 0 or i == len(todo):
                el = time.time() - started
                print(f"[make] {i}/{len(todo)} ok={stats['ok']} failed={stats['failed']} "
                      f"{el/60:.1f} min ({el/max(i,1):.1f} s/q)", flush=True)
    rows = _latest_rows(path)
    audit = {"questions": len(answers), "written": len(rows),
             "status": dict(Counter(r["status"] for r in rows.values())),
             "served_models": dict(Counter(r.get("served_model") for r in rows.values())),
             "keep_fraction": _keep_fraction(rows), "adds": sum(len(r.get("add", [])) for r in rows.values() if r["status"] == "ok"),
             "evidence_exact": dict(Counter(a["evidence_exact"] for r in rows.values() if r["status"] == "ok" for a in r["add"])),
             "empty_reviewed": sum(1 for r in rows.values() if r["status"] == "ok" and not r["keep"] and not r["add"]),
             "errors": [{"id": r["id"], "error": r["error"]} for r in rows.values() if r["status"] != "ok"]}
    write(out / "make_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=1), flush=True)


def _latest_rows(path):
    rows = {}
    for r in read_jsonl(path):
        rows[r["id"]] = r
    return rows


def _keep_fraction(rows):
    ok = [r for r in rows.values() if r["status"] == "ok"]
    n = sum(r["n_original"] for r in ok)
    return (sum(len(r["keep"]) for r in ok) / n) if n else None


# ---------------------------------------------------------------- prepare

def load_pilot(pilot_dir):
    """The fit-40 pilot: hand-written edits, annotations and its per-question results, when present."""
    if not pilot_dir:
        return None
    pilot_dir = Path(pilot_dir)
    edits = {r["id"]: r for r in json.loads((pilot_dir / "replacements.json").read_text())}
    ann = {r["id"]: r for r in json.loads((pilot_dir / "annotations.json").read_text())["items"]}
    results_path = pilot_dir / "replacement_v1/results.json"
    results = {r["id"]: r for r in json.loads(results_path.read_text())} if results_path.exists() else {}
    return {"edits": edits, "annotations": ann, "results": results}


def stage_prepare(args):
    from src.baseline_generation import VERIFY
    out = Path(args.out_dir)
    questions, answers = load_saved(args.suite_dir)
    rows = _latest_rows(out / "replacements.jsonl")
    saved_identity = json.loads((Path(args.suite_dir) / "model/identity.json").read_text())
    if VERIFY != saved_identity["prompts"]["verify"]:
        raise ValueError("VERIFY prompt differs from the saved suite identity")
    pilot = load_pilot(args.pilot_dir)
    items, skipped = [], []
    for key in sorted(answers):
        r = rows.get(key)
        if r is None or r["status"] != "ok":
            skipped.append(key)
            continue
        q, a = questions[key], answers[key]
        original = [p["premise"] for p in a["details"]["premises"]]
        if len(original) != r["n_original"]:
            raise ValueError(f"extraction changed under the writer: {key}")
        edit = {"keep": r["keep"], "add": r["add"], "note": r.get("note", "")}
        item = {"id": key, "set": q["set"], "partition": q["partition"], "question": q["question"],
                "arms": arms_for(original, edit), "edit": edit, "served_model": r.get("served_model"),
                "saved_true_scores": [p["true_score"] for p in a["details"]["premises"]],
                "pilot": None}
        if pilot and key in pilot["edits"]:
            pe = pilot["edits"][key]
            item["pilot"] = {"keep": sorted(pe["keep"]), "add": [x["premise"] for x in pe["add"]],
                             "reference_question_issue": pilot["annotations"][key]["reference_question_issue"],
                             "reviewed_gate": (pilot["results"][key]["arms"]["reviewed"]["gate"]
                                               if key in pilot["results"] else None)}
        items.append(item)
    sources = [out / "replacements.jsonl", Path(args.suite_dir) / "questions.jsonl",
               Path(args.suite_dir) / "answers/extract_verify.jsonl", Path(args.suite_dir) / "model/identity.json",
               Path(__file__), ROOT / "src/baseline_generation.py", ROOT / "src/pilot_model.py"]
    plan = {"scope": "label-blind Sonnet-reviewed replacement of the saved Extract+Verify premise lists; "
                     "not independent gold; every partition, fixed rule, no threshold chosen",
            "writer_prompt": WRITER_PROMPT, "verify_prompt": VERIFY,
            "rule": "score=1-min(true_scores, default=1); gate=any(true_score<0.5)",
            "saved_identity": saved_identity["identity"], "skipped_ids": skipped, "items": items,
            "sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    write(out / "inputs.json", plan, frozen=True)
    counts = {arm: sum(len(r["arms"][arm]) for r in items) for arm in ("original", "retained_only", "reviewed")}
    print(json.dumps({"prepared_questions": len(items), "skipped": len(skipped), "premise_counts": counts,
                      "plan_sha256": digest(plan)}), flush=True)
    return plan


# ---------------------------------------------------------------- run

def stage_run(args, plan):
    from src.baseline_generation import ModelRuntime
    from src.config import load_config
    from src.pilot import output_lock
    out = Path(args.out_dir)
    cfg = load_config(ROOT / "configs/qwen25_7b.yaml")
    cfg["source_model"]["revision"] = plan["saved_identity"]["resolved_revision"]
    jobs = {}
    for row in plan["items"]:
        for premises in row["arms"].values():
            for premise in premises:
                prompt = plan["verify_prompt"].format(question=row["question"], premise=premise)
                jobs[digest(prompt)] = prompt
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    pending = [(k, p) for k, p in jobs.items() if not (cache / (k + ".json")).exists()]
    print(f"{len(jobs)} unique verification prompts; {len(pending)} not cached", flush=True)
    if not pending:
        return
    with output_lock(out / "inference"):
        runtime = ModelRuntime(cfg)
        for key in ("resolved_revision", "chat_template_hash", "torch_version", "transformers_version"):
            if runtime.identity[key] != plan["saved_identity"][key]:
                raise ValueError(f"Runtime mismatch: {key}")
        write(out / "runtime.json", runtime.identity, frozen=True)
        started = time.time()
        for i, (key, prompt) in enumerate(pending, 1):
            value = runtime.binary(prompt, "True", "False")
            write(cache / (key + ".json"), {"prompt_sha256": key, "readout": value})
            if i % 100 == 0 or i == len(pending):
                el = time.time() - started
                print(f"verified {i}/{len(pending)} ({el/60:.1f} min)", flush=True)


# ---------------------------------------------------------------- report

def _auroc_ci(y, s, seed=17, n_boot=1000):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return None, None
    point = float(roc_auc_score(y, s))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    return point, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _sign_test(a, b):
    """Two-sided exact binomial test on discordant counts (a: off->on, b: on->off)."""
    from scipy.stats import binomtest
    n = a + b
    return None if n == 0 else float(binomtest(a, n, 0.5).pvalue)


def _group_summary(rows, arms=("original", "retained_only", "reviewed")):
    g = {"counts": dict(Counter(r["set"] for r in rows)), "arms": {}, "transitions": {}}
    y = [r["set"] == "fpq" for r in rows]
    for arm in arms:
        scores = [r["arms"][arm]["score"] for r in rows]
        auc, ci = _auroc_ci(y, scores)
        n_fpq = sum(1 for r in rows if r["set"] == "fpq")
        n_nfp = sum(1 for r in rows if r["set"] == "nfp")
        det = sum(r["arms"][arm]["gate"] for r in rows if r["set"] == "fpq")
        fp = sum(r["arms"][arm]["gate"] for r in rows if r["set"] == "nfp")
        g["arms"][arm] = {"fpq_detected": det, "fpq_rate": det / n_fpq if n_fpq else None,
                          "nfp_false_positive": fp, "nfp_rate": fp / n_nfp if n_nfp else None,
                          "auroc": auc, "auroc_ci": ci,
                          "premises": sum(len(r["arms"][arm]["checked"]) for r in rows),
                          "empty_lists": sum(1 for r in rows if not r["arms"][arm]["checked"])}
    for before, after in [("original", "reviewed"), ("original", "retained_only"), ("retained_only", "reviewed")]:
        t = {}
        for label in ("fpq", "nfp"):
            on = [r["id"] for r in rows if r["set"] == label and not r["arms"][before]["gate"] and r["arms"][after]["gate"]]
            off = [r["id"] for r in rows if r["set"] == label and r["arms"][before]["gate"] and not r["arms"][after]["gate"]]
            t[label] = {"off_to_on": on, "on_to_off": off, "sign_test_p": _sign_test(len(on), len(off))}
        g["transitions"][f"{before}->{after}"] = t
    return g


def stage_report(args, plan):
    out = Path(args.out_dir)
    cache = out / "cache"
    items, drift, drift_gates = [], [], []
    for row in plan["items"]:
        result = {k: row[k] for k in ("id", "set", "partition", "question", "edit", "served_model", "pilot")}
        result["arms"] = {}
        for arm, premises in row["arms"].items():
            checked = []
            for premise in premises:
                key = digest(plan["verify_prompt"].format(question=row["question"], premise=premise))
                value = json.loads((cache / (key + ".json")).read_text())
                if value["prompt_sha256"] != key:
                    raise ValueError(f"cache entry mismatch: {key}")
                score = value["readout"]["score"]
                if not 0 <= score <= 1:
                    raise ValueError("probability out of range")
                checked.append({"premise": premise, "true_score": score})
            result["arms"][arm] = {"checked": checked, **score_rule(p["true_score"] for p in checked)}
        fresh = result["arms"]["original"]
        drift.extend(abs(p["true_score"] - s) for p, s in zip(fresh["checked"], row["saved_true_scores"]))
        if fresh["gate"] != score_rule(row["saved_true_scores"])["gate"]:
            drift_gates.append(row["id"])
        items.append(result)
    summary = {"n": len(items), "scope": plan["scope"], "skipped_ids": plan["skipped_ids"],
               "writer_models": dict(Counter(r["served_model"] for r in items)),
               "saved_vs_fresh_original": {"max_abs_probability_delta": max(drift) if drift else None,
                                           "changed_gate_ids": drift_gates},
               "groups": {}}
    groups = [("all", items)]
    for part in ("fit", "dev", "test"):
        groups.append((f"partition:{part}", [r for r in items if r["partition"] == part]))
    groups.append(("dev+test", [r for r in items if r["partition"] != "fit"]))
    pilot_rows = [r for r in items if r["pilot"]]
    if pilot_rows:
        groups.append(("pilot40", pilot_rows))
        groups.append(("pilot40:reference_unflagged", [r for r in pilot_rows if r["pilot"]["reference_question_issue"] is not True]))
    for name, rows in groups:
        if rows:
            summary["groups"][name] = _group_summary(rows)
    # per-category view (FPQ only): detection by Cancer-Myth category is not stored on items; skip.
    # edit statistics
    ok_items = items
    summary["edits"] = {
        "keep_fraction": sum(len(r["edit"]["keep"]) for r in ok_items) / max(1, sum(len(r["arms"]["original"]["checked"]) for r in ok_items)),
        "adds_total": sum(len(r["edit"]["add"]) for r in ok_items),
        "adds_per_question": sum(len(r["edit"]["add"]) for r in ok_items) / max(1, len(ok_items)),
        "questions_with_add": sum(1 for r in ok_items if r["edit"]["add"]),
        "questions_unchanged": sum(1 for r in ok_items if r["arms"]["original"]["checked"] == r["arms"]["reviewed"]["checked"]),
        "by_set": {s: {"keep_fraction": (sum(len(r["edit"]["keep"]) for r in ok_items if r["set"] == s)
                                        / max(1, sum(len(r["arms"]["original"]["checked"]) for r in ok_items if r["set"] == s))),
                       "adds_per_question": (sum(len(r["edit"]["add"]) for r in ok_items if r["set"] == s)
                                             / max(1, sum(1 for r in ok_items if r["set"] == s)))}
                   for s in ("fpq", "nfp")},
    }
    # added-premise verdicts: how often does a NEW sentence come out False, by set
    added = {s: [] for s in ("fpq", "nfp")}
    for r in items:
        kept = set(r["arms"]["retained_only"]["checked"] and [p["premise"] for p in r["arms"]["retained_only"]["checked"]])
        for p in r["arms"]["reviewed"]["checked"]:
            if p["premise"] not in kept:
                added[r["set"]].append(p["true_score"] < 0.5)
    summary["added_premise_false_rate"] = {s: {"n": len(v), "false": sum(v), "rate": (sum(v) / len(v)) if v else None}
                                           for s, v in added.items()}
    if pilot_rows:
        agree = []
        for r in pilot_rows:
            a, b = set(r["edit"]["keep"]), set(r["pilot"]["keep"])
            agree.append({"id": r["id"], "set": r["set"],
                          "keep_jaccard": (len(a & b) / len(a | b)) if (a | b) else 1.0,
                          "adds_llm": len(r["edit"]["add"]), "adds_hand": len(r["pilot"]["add"]),
                          "gate_llm": r["arms"]["reviewed"]["gate"], "gate_hand": r["pilot"]["reviewed_gate"]})
        summary["pilot_agreement"] = {
            "n": len(agree),
            "mean_keep_jaccard": sum(x["keep_jaccard"] for x in agree) / len(agree),
            "adds_llm": sum(x["adds_llm"] for x in agree), "adds_hand": sum(x["adds_hand"] for x in agree),
            "reviewed_gate_agreement": (sum(1 for x in agree if x["gate_hand"] is not None and x["gate_llm"] == x["gate_hand"])
                                        / max(1, sum(1 for x in agree if x["gate_hand"] is not None))),
            "items": agree}
    write(out / "results.json", items)
    write(out / "summary.json", summary)
    (out / "report.md").write_text(render_report(summary, plan), encoding="utf-8")
    (out / "items.md").write_text(render_items(items, summary), encoding="utf-8")
    if args.docs_dir:
        docs = Path(args.docs_dir)
        docs.mkdir(parents=True, exist_ok=True)
        for name in ("report.md", "items.md", "summary.json", "make_audit.json", "runtime.json"):
            if (out / name).exists():
                shutil.copy(out / name, docs / name)
        # the writer's edits are small; keep them in the repo too
        shutil.copy(out / "replacements.jsonl", docs / "replacements.jsonl")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups", "pilot_agreement")}, ensure_ascii=False, indent=1))
    for name in ("all", "partition:fit", "dev+test", "pilot40"):
        if name in summary["groups"]:
            g = summary["groups"][name]
            print(name, {arm: (v["fpq_detected"], v["nfp_false_positive"], round(v["auroc"], 3) if v["auroc"] else None)
                         for arm, v in g["arms"].items()})


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _auc(v):
    if v["auroc"] is None:
        return "–"
    return f"{v['auroc']:.3f} [{v['auroc_ci'][0]:.3f}, {v['auroc_ci'][1]:.3f}]"


def render_report(summary, plan):
    L = ["# 전제 교체 진단, 전체 suite (label-blind Sonnet 작성)", "",
         f"질문 {summary['n']}개(저장된 Extract+Verify 결과가 있는 725개 중 작성 실패 {len(summary['skipped_ids'])}개 제외). "
         "작성자: Sonnet(`claude -p`), 질문과 원 추출 목록만 제공(라벨·참조·교정문 비공개). "
         "verifier·프롬프트·규칙은 fit-40 파일럿과 동일: Qwen2.5-7B-Instruct, `1 - min P(True)`, gate = any P(True) < 0.5. "
         "임계값을 고르지 않았고 답변 생성·judge는 없다.", "",
         f"작성 모델: {summary['writer_models']}. 기존 전제 재계산의 저장값 대비 최대 확률 차이 "
         f"{summary['saved_vs_fresh_original']['max_abs_probability_delta']}, gate 변경 {len(summary['saved_vs_fresh_original']['changed_gate_ids'])}개.", ""]
    e = summary["edits"]
    L += ["## 작성자가 무엇을 바꿨나", "",
          f"- 원 문장 유지 비율 {e['keep_fraction']:.2f} (FPQ {e['by_set']['fpq']['keep_fraction']:.2f}, NFP {e['by_set']['nfp']['keep_fraction']:.2f}); "
          f"추가 문장 {e['adds_total']}개, 질문당 {e['adds_per_question']:.2f} (FPQ {e['by_set']['fpq']['adds_per_question']:.2f}, NFP {e['by_set']['nfp']['adds_per_question']:.2f}); "
          f"추가가 있는 질문 {e['questions_with_add']}개; 목록이 그대로인 질문 {e['questions_unchanged']}개.",
          f"- 새로 추가된 문장이 False로 판정된 비율: FPQ {summary['added_premise_false_rate']['fpq']['false']}/{summary['added_premise_false_rate']['fpq']['n']} "
          f"({_pct(summary['added_premise_false_rate']['fpq']['rate'])}), NFP {summary['added_premise_false_rate']['nfp']['false']}/{summary['added_premise_false_rate']['nfp']['n']} "
          f"({_pct(summary['added_premise_false_rate']['nfp']['rate'])}).", ""]
    L += ["## 결과", ""]
    for name, g in summary["groups"].items():
        c = g["counts"]
        L += [f"### {name} (FPQ {c.get('fpq', 0)}, NFP {c.get('nfp', 0)})", "",
              "| 조건 | FPQ 탐지 | NFP 오탐 | AUROC [95% CI] | 전제 수 | 빈 목록 |", "|---|---:|---:|---|---:|---:|"]
        label = {"original": "기존 추출", "retained_only": "유지 문장만(삭제 대조)", "reviewed": "검토·보완 추출"}
        for arm, v in g["arms"].items():
            L.append(f"| {label[arm]} | {v['fpq_detected']} ({_pct(v['fpq_rate'])}) | {v['nfp_false_positive']} ({_pct(v['nfp_rate'])}) "
                     f"| {_auc(v)} | {v['premises']} | {v['empty_lists']} |")
        t = g["transitions"]["original->reviewed"]
        L += ["", f"기존→보완 전환: FPQ 놓침→탐지 {len(t['fpq']['off_to_on'])}, 탐지→놓침 {len(t['fpq']['on_to_off'])} "
                  f"(부호검정 p={t['fpq']['sign_test_p'] if t['fpq']['sign_test_p'] is None else round(t['fpq']['sign_test_p'], 4)}); "
                  f"NFP 정상→오탐 {len(t['nfp']['off_to_on'])}, 오탐→정상 {len(t['nfp']['on_to_off'])} "
                  f"(p={t['nfp']['sign_test_p'] if t['nfp']['sign_test_p'] is None else round(t['nfp']['sign_test_p'], 4)}).", ""]
    if "pilot_agreement" in summary:
        p = summary["pilot_agreement"]
        L += ["## 파일럿 40문항에서 두 작성자의 일치", "",
              f"keep 집합 Jaccard 평균 {p['mean_keep_jaccard']:.2f}; 추가 문장 Sonnet {p['adds_llm']} vs 손 편집 {p['adds_hand']}; "
              f"보완 조건 gate 일치율 {_pct(p['reviewed_gate_agreement'])}. 손 편집은 라벨·참조를 본 상태에서 썼고 Sonnet은 보지 않았다.", ""]
    L += ["## 읽는 법과 한계", "",
          "- 같은 문항·같은 verifier·같은 규칙에서 추출 목록만 바꾼 짝 비교다. 전환 수의 부호검정은 문항 독립을 가정한다.",
          "- 작성자는 라벨을 못 봤지만 사람 gold도 임상 검증도 아니다. 개선이 없어도 '추출은 병목이 아니다'로 확정하지 않고, 개선이 있어도 최종 답변 효용이나 text/hidden gate의 실패 원인으로 확대하지 않는다.",
          "- 부분집합(partition, 파일럿 40)은 사후 분할이며 임계값을 고르지 않았으므로 fit/dev/test 구분은 보고용이다.",
          "- 파일: `results.json`(문항별 세 조건 점수), `items.md`(gate가 바뀐 문항), `replacements.jsonl`(작성자 출력 원문), `summary.json`."]
    return "\n".join(L) + "\n"


def render_items(items, summary):
    changed = [r for r in items if r["arms"]["original"]["gate"] != r["arms"]["reviewed"]["gate"]]
    L = [f"# gate가 바뀐 문항 (기존 → 검토·보완): {len(changed)} / {len(items)}", ""]
    for r in sorted(changed, key=lambda x: (x["set"], x["id"])):
        o, v = r["arms"]["original"], r["arms"]["reviewed"]
        L += [f"## {r['id']} ({r['set']}, {r['partition']}) — {'ON' if o['gate'] else 'off'} → {'ON' if v['gate'] else 'off'}", "",
              f"> {r['question']}", "", "기존:"]
        L += [f"- [{p['true_score']:.3f}] {p['premise']}" for p in o["checked"]] or ["- (없음)"]
        L += ["", "보완:"]
        L += [f"- [{p['true_score']:.3f}] {p['premise']}" for p in v["checked"]] or ["- (없음)"]
        L += ["", f"note: {r['edit']['note']}", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["make", "prepare", "run", "report"])
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pilot-dir", type=Path, default=None, help="docs/reviews/extraction_audit_2026-09-22 for the 40-item overlap")
    ap.add_argument("--docs-dir", type=Path, default=None, help="report: copy report/summary/items/replacements here")
    ap.add_argument("--backend", default="claude")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--claude-cmd", default="claude")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    if args.stage == "make":
        stage_make(args)
        return
    plan = stage_prepare(args)
    if args.stage == "run":
        stage_run(args, plan)
    if args.stage in ("run", "report"):
        stage_report(args, plan)


if __name__ == "__main__":
    main()
