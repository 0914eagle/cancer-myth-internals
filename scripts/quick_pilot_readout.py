"""Resume a frozen quick pilot with whitespace-independent JSON score parsing.

Does not change prompts, generation code, frozen plans, or existing ledger rows.
Report reparses stored replies without model calls; score calls unstarted jobs only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import quick_pilot as quick
from scripts.check_terra_judge import events_by_id
from src.jsonl import append_jsonl
from src.llm_backend import make_caller
from src.pilot import digest, file_digest, frozen_json, output_lock

READOUT = "quick-json-v2"


def parse_reply(raw, kind):
    """One complete JSON object (optionally fenced); never infer a missing score."""
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced[1]

    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key")
            obj[key] = value
        return obj

    try:
        parsed = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None
    allowed = {-1, 0, 1} if kind == "fpq" else {-1, 1}
    if (kind not in {"fpq", "nfp"} or not isinstance(parsed, dict)
            or type(parsed.get("Sharpness")) is not int or parsed["Sharpness"] not in allowed
            or ("Reason" in parsed and not isinstance(parsed["Reason"], str))):
        return None
    return parsed


def load_run(out):
    # All original hashes remain checked, including unchanged original parser and runner.
    plan, spec = quick.judge_plan(out)
    protocol = {"readout": READOUT, "judge_plan_hash": digest(spec),
                "source_hash": file_digest(__file__), "model": quick.MODEL,
                "rule": "single JSON object; integer Sharpness; original set-specific score range; no fallback"}
    frozen_json(out / "readout_json_v2_protocol.json", protocol)
    return plan, spec, protocol


def read_scores(out, spec):
    events = events_by_id(out, spec, digest(spec))
    values, recovered, missing = {}, [], []
    for key, rows in events.items():
        event = rows[-1]
        parsed = None
        if (event["status"] == "finished" and event.get("model") == quick.MODEL
                and "error" not in event):
            parsed = parse_reply(event.get("raw"), spec["cases"][key]["set"])
        if event.get("valid"):
            if parsed is None or type(event.get("score")) is not int or event["score"] != parsed["Sharpness"]:
                raise ValueError("Previously valid stored score disagrees with raw reply")
        if parsed is None:
            missing.append(key)
            continue
        values[key] = parsed["Sharpness"]
        if not event.get("valid"):
            recovered.append({"id": key, "question_id": spec["cases"][key]["question_id"],
                              "score": parsed["Sharpness"], "raw_hash": digest(event["raw"])})
    return events, values, recovered, missing


def score(out, timeout=180):
    # Share the old runner's lock and ledger: started calls always consume budget.
    with output_lock(out / "score"):
        _, spec, protocol = load_run(out)
        previous, _, recovered, _ = read_scores(out, spec)
        remaining = spec["budget"] - len(previous)
        print(f"Recovered {len(recovered)} stored replies offline; remaining calls: {remaining}; {quick.MODEL}.", flush=True)
        if remaining == 0:
            return
        call = make_caller("codex", quick.MODEL, timeout=timeout)
        failures = 0
        for job in spec["order"]:
            key = job["id"]
            if key in previous:
                continue
            common = {"id": key, "run_hash": digest(spec), "readout": READOUT,
                      "readout_protocol_hash": digest(protocol)}
            append_jsonl(out / "attempts.jsonl", {**common, "status": "started",
                         "at": dt.datetime.now(dt.timezone.utc).isoformat()})
            try:
                raw, used = call(spec["cases"][key]["prompt"])
                parsed = parse_reply(raw, spec["cases"][key]["set"])
                valid = parsed is not None and used == quick.MODEL
                event = {**common, "status": "finished", "raw": raw, "model": used,
                         "valid": valid, "score": parsed["Sharpness"] if valid else None,
                         "reason": parsed.get("Reason") if parsed else None}
            except Exception as exc:
                event = {**common, "status": "finished", "valid": False, "score": None, "error": str(exc)}
            append_jsonl(out / "attempts.jsonl", event)
            failures = 0 if event["valid"] else failures + 1
            print(f"{spec['cases'][key]['question_id']}: {event['score'] if event['valid'] else 'invalid'}", flush=True)
            if failures >= 3 or event.get("model", quick.MODEL) != quick.MODEL or "error" in event:
                print("Stopped on backend/model failure or 3 invalid replies. No automatic retry.")
                break


def report(out):
    with output_lock(out / "score"):
        plan, spec, protocol = load_run(out)
        events, values, recovered, missing = read_scores(out, spec)
        ledger = out / "attempts.jsonl"
        ledger_hash = file_digest(ledger) if ledger.exists() else None
        audit = {"protocol": protocol, "ledger_sha256": ledger_hash,
                 "attempted": len(events), "valid": len(values), "recovered": recovered,
                 "invalid_or_interrupted_ids": missing, "unstarted": spec["budget"] - len(events),
                 "changed_previously_valid_scores": 0}
        # Snapshot audit is immutable; prior ledger and report.md are untouched.
        frozen_json(out / "readout_audits" / f"{ledger_hash or 'empty'}.json", audit)
        mapped = {m: {i: values.get(k) for i, k in spec["mapping"][m].items()} for m in quick.METHODS}
        lines = ["# Quick Gemma dev comparison — JSON readout v2", "",
                 "Original Cancer-Myth prompts/few-shot unchanged; judge: codex gpt-5.6-terra.",
                 "Parser accepts compact/multiline JSON; no rubric changes or score imputation. Report calls: 0.",
                 "Exploratory dev: 30 FPQ + 15 NFP. Judge/split differ from paper; NFP is reference-targeted, not QA accuracy.",
                 f"Attempted {len(events)}/{spec['budget']}; valid {len(values)}; recovered {len(recovered)}; unstarted {audit['unstarted']}.",
                 f"Attempt ledger SHA-256: `{ledger_hash}`.", "",
                 "| Method | FPQ valid | PCR % | PCS | NFP valid | NFP pass % | FPQ rescue/harm (pairs) | NFP rescue/harm (pairs) |",
                 "|---|---:|---:|---:|---:|---:|---|---|"]
        for method in quick.METHODS:
            cells, pairs = [], []
            for kind in ("fpq", "nfp"):
                ids = [q["id"] for q in plan["questions"] if q["set"] == kind]
                vs = [mapped[method][i] for i in ids]
                valid = [v for v in vs if v is not None]
                complete = len(valid) == len(ids)
                cells += [f"{len(valid)}/{len(ids)}", f"{100*sum(v == 1 for v in valid)/len(ids):.1f}" if complete else "incomplete"]
                if kind == "fpq":
                    cells.append(f"{sum(valid)/len(ids):.3f}" if complete else "incomplete")
                paired = [i for i in ids if mapped[method][i] is not None and mapped['plain'][i] is not None]
                rescue = sum(mapped[method][i] == 1 and mapped['plain'][i] != 1 for i in paired)
                harm = sum(mapped[method][i] != 1 and mapped['plain'][i] == 1 for i in paired)
                pairs.append(f"{rescue}/{harm} ({len(paired)})")
            lines.append("| " + " | ".join([method, *cells, *pairs]) + " |")
        lines += ["", "CoT review caps: 128 vs 1024; final answer cap 512.",
                  "Steering: unconditional L21 paired-C, alpha=0.1 x fit residual scale; no gate.",
                  "No preservation guarantee, new generation, or test evaluation."]
        text = "\n".join(lines) + "\n"
        (out / "report_json_v2.md").write_text(text)
        print(text)
        return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["report", "score", "resume"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if args.stage in {"score", "resume"}:
        score(out, args.timeout)
    if args.stage in {"report", "resume"}:
        report(out)


if __name__ == "__main__":
    main()
