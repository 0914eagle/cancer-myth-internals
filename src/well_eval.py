"""Frozen, capped Well-Actually answer evaluation; importing performs no calls.

The upstream 0--5 rubric is preserved. Invalid/missing judgments never become
scores. The published split and judge are NOT reproduced by our default run.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import json
import os
from pathlib import Path
import re

from src.jsonl import read_jsonl
from src.llm_backend import check_judge_identity, make_caller
from src.pilot import digest, file_digest, frozen_json, normalized, output_lock

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "prompts" / "well"
VERSION = "well-answer-eval-v1"
DEFAULT_MODEL = "gpt-5.6-terra"


def template_manifest():
    """Verify vendored artifacts on every prepare; no upstream code is executed."""
    info = json.loads((TEMPLATES / "sources.json").read_text())
    for record in info["files"]:
        if file_digest(TEMPLATES / record["local_file"]) != record["sha256"]:
            raise ValueError("Vendored Well source hash mismatch")
    for name, sha in info["derived_templates"].items():
        if file_digest(TEMPLATES / name) != sha:
            raise ValueError("Derived Well template hash mismatch")
    return info


def example_questions() -> list[str]:
    """Questions embedded in the active judge templates, not unused JSON examples."""
    return list(template_manifest()["example_questions"])


def parse_rating(raw):
    """One terminal 'Rating: X' or one JSON Rating int; reject ambiguity/fallbacks."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.S | re.I)
    if fenced:
        text = fenced[1].strip()
    if text.startswith("{"):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate key")
                result[key] = value
            return result
        try:
            data = json.loads(text, object_pairs_hook=unique)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or set(data) - {"Rating", "Reason", "Explanation"}:
            return None
        value = data.get("Rating")
        if any(not isinstance(data[k], str) for k in ("Reason", "Explanation") if k in data):
            return None
        return value if type(value) is int and 0 <= value <= 5 else None
    # Count all rating markers, including malformed ones, to reject contradictions.
    if len(re.findall(r"\bRating\s*:", text, re.I)) != 1:
        return None
    match = re.search(r"\bRating\s*:\s*([0-5])(?:\*\*|__)?\s*[.。]?\s*$", text, re.I)
    return int(match[1]) if match else None


def judge_messages(question, answer):
    """Faithful upstream role/content, including its TPQ false-claim reminder."""
    kind = question["set"]
    if kind not in {"fpq", "nfp"}:
        raise ValueError("Expected canonical fpq/nfp set")
    fields = {"question": question["question"], "model_final_answer": answer}
    if kind == "fpq":
        fields["presupposition"] = question.get("premise_text") or question.get("source_myth")
        fields["correction"] = question.get("correction")
        if any(not isinstance(fields[k], str) or not fields[k].strip()
               for k in ("presupposition", "correction")):
            raise ValueError("FPQ needs premise_text/source_myth and correction separately; join original HF references")
    return [{"role": "system", "content": (TEMPLATES / f"{kind}_system.txt").read_text()},
            {"role": "user", "content": (TEMPLATES / f"{kind}_user.txt").read_text().format(**fields)}]


def transport_prompt(messages):
    # make_caller has a text-only interface. Preserve both strings, record adapter.
    return "\n\n".join(f"[{m['role'].upper()}]\n{m['content']}" for m in messages)


def _source(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": file_digest(p)}


def prepare(questions_path, answer_paths, out_dir, *, backend="codex", model=DEFAULT_MODEL):
    """Freeze complete prompts, answer mapping and provenance; never call a model."""
    if backend not in {"codex", "openai"} or not model:
        raise ValueError("Explicit supported backend/model required")
    check_judge_identity(model)
    sources = template_manifest()
    demos = {normalized(q) for q in example_questions()}
    questions, texts = {}, set()
    for row in read_jsonl(questions_path):
        qid, text = row.get("id"), row.get("question")
        if not isinstance(qid, str) or not qid or qid in questions:
            raise ValueError("Empty/duplicate/non-string question ID")
        if not isinstance(text, str) or not normalized(text) or normalized(text) in texts:
            raise ValueError("Empty/duplicate question text")
        if normalized(text) in demos:
            raise ValueError(f"Judge demonstration overlaps evaluation: {qid}; exclude it")
        if row.get("set") not in {"fpq", "nfp"}:
            raise ValueError("Question set must be fpq or nfp")
        judge_messages(row, "")  # Validate original reference mapping before any call.
        questions[qid] = dict(row)
        texts.add(normalized(text))
    if not questions:
        raise ValueError("No questions")
    jobs, mapping, answer_hashes = {}, {}, {}
    for path in answer_paths:
        file_rows = 0
        for row in read_jsonl(path):
            file_rows += 1
            qid, method, response = row.get("id"), row.get("method"), row.get("response")
            if qid not in questions or not isinstance(method, str) or not method.strip():
                raise ValueError("Unknown answer ID or missing method")
            if qid in mapping.setdefault(method, {}):
                raise ValueError("Duplicate method/question answer")
            if not isinstance(response, str):
                raise ValueError("Answer response must be a string; empty text is valid to judge")
            question = questions[qid]
            if "question" in row and row["question"] != question["question"]:
                raise ValueError("Answer question text differs from inventory")
            for key in ("set", "partition"):
                if key in row and key in question and row[key] != question[key]:
                    raise ValueError(f"Answer {key} differs from inventory")
            if row.get("status", "complete") not in {"complete", "completed", "finished"}:
                raise ValueError("Do not judge unfinished generation records")
            messages = judge_messages(question, response)
            jobid = digest({"backend": backend, "model": model, "messages": messages})
            jobs.setdefault(jobid, {"messages": messages, "prompt": transport_prompt(messages)})
            mapping[method][qid] = jobid
            answer_hashes.setdefault(method, {})[qid] = digest(row)
        if not file_rows:
            raise ValueError(f"Empty answer file cannot establish its method: {path}; finish generation first")
    if not jobs:
        raise ValueError("No answer rows to evaluate")
    plan = {"version": VERSION, "backend": backend, "model": model,
            "transport": "system/user contents serialized as [SYSTEM]/[USER] via text-only make_caller",
            "not_exact_paper_reproduction": "Own split and default Terra judge; upstream uses another judge/protocol",
            "questions": questions, "jobs": jobs, "mapping": mapping, "answer_hashes": answer_hashes,
            "sources": {"questions": _source(questions_path), "answers": [_source(p) for p in answer_paths],
                        "templates": sources, "implementation_sha256": file_digest(__file__)},
            "call_settings": {"temperature": 0.0, "max_tokens": 1024},
            "order": sorted(jobs), "expected_per_method": len(questions)}
    envelope = {"plan_hash": digest(plan), "plan": plan}
    out = Path(out_dir)
    with output_lock(out / "judge"):
        frozen_json(out / "judge_plan.json", envelope)
    return {"unique_calls": len(jobs), "answer_rows": sum(map(len, mapping.values())),
            "shared_calls": sum(map(len, mapping.values())) - len(jobs), "calls_now": 0}


def load_plan(out_dir):
    saved = json.loads((Path(out_dir) / "judge_plan.json").read_text())
    plan = saved["plan"]
    if plan.get("version") != VERSION or digest(plan) != saved.get("plan_hash"):
        raise ValueError("Modified/unsupported frozen judge plan")
    return plan, saved["plan_hash"]


def _append(path, event):
    """Persist started before a call, including flush/fsync for interruption safety."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_ledger(out_dir, plan, plan_hash):
    path = Path(out_dir) / "attempts.jsonl"
    events = {}
    if not path.exists():
        return events, {}
    for row in read_jsonl(path):
        jobid = row.get("id")
        if row.get("plan_hash") != plan_hash or jobid not in plan["jobs"]:
            raise ValueError("Ledger provenance/ID mismatch")
        previous = events.get(jobid)
        if row.get("status") == "started":
            if previous is not None:
                raise ValueError("Repeated started job: retries forbidden")
        elif row.get("status") == "finished":
            if previous is None or previous["status"] != "started":
                raise ValueError("Finished event without one prior started event")
        else:
            raise ValueError("Unknown ledger event")
        events[jobid] = row
    scores = {}
    for jobid, row in events.items():
        if row["status"] != "finished":
            continue
        rating = parse_rating(row.get("raw"))
        valid = rating is not None and row.get("model") == plan["model"] and "error" not in row
        if row.get("valid") != valid or row.get("score") != (rating if valid else None):
            raise ValueError("Stored validity/score differs from raw reply")
        if valid:
            scores[jobid] = rating
    return events, scores


def score(out_dir, *, max_calls, timeout=180, codex_cmd="codex", caller=None):
    """At most max_calls NEW attempts. Started/invalid jobs are never retried."""
    if type(max_calls) is not int or max_calls < 1:
        raise ValueError("max_calls must be an explicit positive integer")
    if type(timeout) is not int or timeout < 1:
        raise ValueError("timeout must be a positive integer")
    out = Path(out_dir)
    with output_lock(out / "judge"):
        plan, plan_hash = load_plan(out)
        if plan["sources"]["implementation_sha256"] != file_digest(__file__):
            raise ValueError("Judge implementation changed; use the pinned code or a new run directory")
        previous, _ = read_ledger(out, plan, plan_hash)
        pending = [jobid for jobid in plan["order"] if jobid not in previous][:max_calls]
        if not pending:
            return {"new_calls": 0, "remaining_unstarted": 0}
        call = caller or make_caller(plan["backend"], plan["model"], timeout=timeout,
                                     codex_cmd=codex_cmd, **plan["call_settings"])
        calls, invalid_streak = 0, 0
        for jobid in pending:
            common = {"id": jobid, "plan_hash": plan_hash}
            _append(out / "attempts.jsonl", {**common, "status": "started",
                    "at": dt.datetime.now(dt.timezone.utc).isoformat()})
            calls += 1
            try:
                raw, used = call(plan["jobs"][jobid]["prompt"])
                rating = parse_rating(raw)
                valid = rating is not None and used == plan["model"]
                event = {**common, "status": "finished", "raw": raw, "model": used,
                         "valid": valid, "score": rating if valid else None}
            except Exception as exc:
                event = {**common, "status": "finished", "valid": False, "score": None,
                         "error": f"{type(exc).__name__}: {exc}"}
            _append(out / "attempts.jsonl", event)
            invalid_streak = 0 if event["valid"] else invalid_streak + 1
            print(f"{jobid[:12]}: {event['score'] if event['valid'] else 'invalid'}", flush=True)
            if "error" in event or event.get("model") != plan["model"] or invalid_streak >= 3:
                break
        return {"new_calls": calls, "remaining_unstarted": len(plan["jobs"]) - len(previous) - calls}


def _summary(values, expected):
    counts = Counter(v for v in values if v is not None)
    readable = sum(counts[v] for v in range(1, 6))
    return {"expected": expected, "valid": sum(counts.values()),
            "missing": expected - sum(counts.values()),
            "counts_0_to_5": {str(v): counts[v] for v in range(6)},
            "mean_1_to_5": sum(v * counts[v] for v in range(1, 6)) / readable if readable else None,
            "mean_denominator": readable, "s5_count": counts[5],
            "s5_over_all": counts[5] / expected if expected else None,
            "gibberish_count": counts[0], "gibberish_over_all": counts[0] / expected if expected else None}


def report(out_dir):
    """Offline report includes all denominators, zeros, missingness and pairs."""
    out = Path(out_dir)
    with output_lock(out / "judge"):
        plan, plan_hash = load_plan(out)
        events, scores = read_ledger(out, plan, plan_hash)
        mapped = {m: {qid: scores.get(jobs.get(qid)) for qid in plan["questions"]}
                  for m, jobs in plan["mapping"].items()}
        report_data = {"version": VERSION, "plan_hash": plan_hash,
                       "ledger_sha256": file_digest(out / "attempts.jsonl") if events else None,
                       "attempted": len(events), "unique_valid": len(scores),
                       "invalid_or_interrupted": len(events) - len(scores),
                       "unstarted": len(plan["jobs"]) - len(events), "calls_now": 0, "methods": {}}
        lines = ["# Well-Actually answer evaluation", "",
                 f"Judge: {plan['backend']} / {plan['model']}. Original pinned 0–5 templates and embedded examples.",
                 "Own split/judge/transport; not an exact paper reproduction. TPQ is not general medical QA accuracy.",
                 f"Unique attempted {len(events)}/{len(plan['jobs'])}; valid {len(scores)}; offline report calls: 0.",
                 "Mean excludes score 0 and missing; S5/all and gibberish/all use ALL expected questions.",
                 "If incomplete, S5/all is observed coverage, not an estimate with missing scores imputed as failures.", "",
                 "| Method | Set | Valid/expected | Missing | Counts 0/1/2/3/4/5 | Mean 1–5 (n) | S5/all | Gibberish/all | S5 rescue/harm (pairs) |",
                 "|---|---|---:|---:|---|---:|---:|---:|---|"]
        for method, values in sorted(mapped.items()):
            report_data["methods"][method] = {}
            for kind in ("fpq", "nfp"):
                ids = [qid for qid, q in plan["questions"].items() if q["set"] == kind]
                result = _summary([values[qid] for qid in ids], len(ids))
                paired = [qid for qid in ids if values[qid] is not None
                          and mapped.get("plain", {}).get(qid) is not None]
                rescue = sum(values[qid] == 5 and mapped["plain"][qid] != 5 for qid in paired)
                harm = sum(values[qid] != 5 and mapped["plain"][qid] == 5 for qid in paired)
                result["paired_plain"] = {"n": len(paired), "s5_rescue": rescue, "s5_harm": harm}
                report_data["methods"][method][kind] = result
                mean = f"{result['mean_1_to_5']:.3f}" if result["mean_1_to_5"] is not None else "NA"
                def percent(value):
                    return f"{100 * value:.1f}%" if value is not None else "NA"
                lines.append(f"| {method} | {kind} | {result['valid']}/{len(ids)} | {result['missing']} | "
                             + "/".join(str(result["counts_0_to_5"][str(v)]) for v in range(6))
                             + f" | {mean} ({result['mean_denominator']}) | {result['s5_count']}/{len(ids)} "
                             + f"({percent(result['s5_over_all'])}) | {result['gibberish_count']}/{len(ids)} "
                             + f"({percent(result['gibberish_over_all'])}) | {rescue}/{harm} ({len(paired)}) |")
        frozen_json(out / "report_snapshots" / f"{report_data['ledger_sha256'] or 'empty'}.json", report_data)
        (out / "report.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n")
        (out / "report.md").write_text("\n".join(lines) + "\n")
        return report_data
