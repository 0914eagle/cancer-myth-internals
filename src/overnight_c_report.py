"""Offline, failure-aware summaries of frozen overnight C diagnostics.

This module deliberately has no model/backend imports. Rebuilding a report never
generates answers, changes scores, or launches a judge.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


STATUSES = ("complete", "failed", "timed_out", "skipped", "pending")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _average(values):
    values = [v for v in values if v is not None]
    return mean(values) if values else None


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path: Path, text: str):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _question_id(task):
    q = task.get("question", {})
    return str(task.get("question_id", q.get("id", task.get("id", "")) if isinstance(q, dict) else task.get("id", "")))


def _direction(task):
    d = task.get("direction") or {}
    if isinstance(d, str):
        d = {"name": d}
    return {
        "name": str(d.get("name", d.get("key", "unspecified"))),
        "layer": d.get("layer", task.get("layer")),
        "key": d.get("key"),
        "path": d.get("path"),
        "random_seed": d.get("random_seed"),
    }


def _pair_key(task):
    """Do not reuse an alpha=0 control for a different pair or direction."""
    return json.dumps([_direction(task), _question_id(task), task.get("positive"),
                       task.get("negative")], sort_keys=True, ensure_ascii=False)


def _logp(result, side, segment="full"):
    block = result.get(side, {})
    if not isinstance(block, dict):
        return None
    if segment != "full":
        block = block.get(segment, {})
    return _finite(block.get("mean_logp")) if isinstance(block, dict) else None


def descriptive_auc(scores, labels):
    """P(positive score > negative score), with half credit for ties."""
    positive = [s for s, y in zip(scores, labels) if y == 1 and _finite(s) is not None]
    negative = [s for s, y in zip(scores, labels) if y == -1 and _finite(s) is not None]
    if not positive or not negative:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def _records(out, tasks):
    records, artifact_warnings = [], []
    known = {str(t["id"]) for t in tasks}
    for task in tasks:
        task_id = str(task["id"])
        path = out / "tasks" / f"{task_id}.json"
        if not path.exists():
            records.append({"id": task_id, "status": "pending", "task": task, "result": {}})
            continue
        try:
            record = _json(path)
            if not isinstance(record, dict):
                raise ValueError("task artifact must be an object")
            if str(record.get("id")) != task_id:
                raise ValueError("artifact ID differs from planned ID")
            if record.get("status") not in STATUSES[:-1]:
                raise ValueError(f"unknown terminal status: {record.get('status')!r}")
            if record.get("task", task) != task:
                raise ValueError("artifact task differs from frozen plan")
            if not isinstance(record.get("result", {}), dict):
                raise ValueError("result must be an object")
            record["task"] = task
        except (ValueError, TypeError, OSError) as exc:
            record = {"id": task_id, "status": "failed", "task": task,
                      "result": {}, "error": f"Unreadable/inconsistent artifact: {exc}"}
        records.append(record)
    for path in sorted((out / "tasks").glob("*.json")):
        if path.stem not in known:
            artifact_warnings.append(f"Unplanned artifact excluded: {path.name}")
    return records, artifact_warnings


def _dose_summary(records):
    controls = defaultdict(list)
    groups = defaultdict(list)
    for r in records:
        t = r["task"]
        if t.get("kind") != "dose":
            continue
        d = _direction(t)
        key = json.dumps([d, t.get("group", "unspecified"), t.get("policy", "all"), t.get("alpha")], sort_keys=True)
        groups[key].append(r)
        # Ablation is active even at alpha=0; it must never be its own zero control.
        if r["status"] == "complete" and _finite(t.get("alpha")) == 0 and t.get("policy") != "ablate":
            controls[_pair_key(t)].append(r)
    summary, pairs = [], []
    for key, group in groups.items():
        direction, task_group, policy, alpha = json.loads(key)
        deltas = []
        for r in group:
            if r["status"] != "complete":
                continue
            t = r["task"]
            candidates = controls.get(_pair_key(t), [])
            exact = [c for c in candidates if c["task"].get("policy", "all") == policy]
            all_controls = [c for c in candidates if c["task"].get("policy", "all") == "all"]
            chosen = exact or all_controls
            # Multiple controls are ambiguous, never silently pick a preferred run.
            if len(chosen) != 1:
                continue
            control = chosen[0]
            point = {"task_id": r["id"], "question_id": _question_id(t),
                     "control_task_id": control["id"], "direction": direction,
                     "group": task_group, "policy": policy, "alpha": alpha}
            for segment in ("full", "prefix32", "rest"):
                p, n = (_logp(r.get("result", {}), s, segment) for s in ("positive", "negative"))
                p0, n0 = (_logp(control.get("result", {}), s, segment) for s in ("positive", "negative"))
                valid = all(x is not None for x in (p, n, p0, n0))
                point[segment] = {
                    "positive_mean_logp": p, "negative_mean_logp": n,
                    "margin": p - n if p is not None and n is not None else None,
                    "positive_delta": p - p0 if valid else None,
                    "negative_delta": n - n0 if valid else None,
                    "delta_margin": (p - n) - (p0 - n0) if valid else None,
                }
            if point["full"]["delta_margin"] is not None:
                deltas.append(point)
                pairs.append(point)
        segments = {}
        for segment in ("full", "prefix32", "rest"):
            rows = [p[segment] for p in deltas if p[segment]["delta_margin"] is not None]
            segments[segment] = {"n": len(rows), **{
                name: _average(p[name] for p in rows)
                for name in ("delta_margin", "positive_delta", "negative_delta", "margin")}}
        summary.append({"direction": direction, "group": task_group, "policy": policy, "alpha": alpha,
                        "planned": len(group), "complete": sum(r["status"] == "complete" for r in group),
                        "paired": len(deltas), **segments})
    summary.sort(key=lambda g: (g["direction"]["name"], str(g["direction"]["layer"]), g["group"], g["policy"], float(g["alpha"] or 0)))
    return summary, pairs


def _project_summary(records):
    groups = defaultdict(list)
    for r in records:
        if r["task"].get("kind") == "project":
            groups[json.dumps(_direction(r["task"]), sort_keys=True)].append(r)
    summary = []
    for key, group in groups.items():
        for pooling in ("prefix32", "full"):
            scores, labels = [], []
            for r in group:
                if r["status"] != "complete":
                    continue
                t, result = r["task"], r.get("result", {})
                label = t.get("label", t.get("score", result.get("label")))
                kind = t.get("set", t.get("question", {}).get("set") if isinstance(t.get("question"), dict) else None)
                block = result.get(pooling, {})
                score = _finite(block.get("dot")) if isinstance(block, dict) else None
                # Only original +/-1 FPQ labels; zeros and NFP are different targets.
                if kind == "nfp" or isinstance(label, bool) or label not in (-1, 1) or score is None:
                    continue
                scores.append(score)
                labels.append(label)
            summary.append({"direction": json.loads(key), "pooling": pooling,
                            "planned": len(group), "complete": sum(r["status"] == "complete" for r in group),
                            "positive": labels.count(1), "negative": labels.count(-1),
                            "auroc": descriptive_auc(scores, labels)})
    return summary


def _generation_summary(records):
    groups = defaultdict(list)
    for r in records:
        t = r["task"]
        if t.get("kind") == "generate":
            key = json.dumps([_direction(t), t.get("partition", "unspecified"),
                              t.get("method", "generate"), t.get("policy", "all"), t.get("alpha")], sort_keys=True)
            groups[key].append(r)
    summary = []
    for key, group in groups.items():
        direction, partition, method, policy, alpha = json.loads(key)
        complete = [r for r in group if r["status"] == "complete"]
        paired = [r for r in complete if isinstance(r["task"].get("baseline_response"), str)
                  and isinstance(r.get("result", {}).get("response"), str)]
        summary.append({"direction": direction, "partition": partition, "method": method,
                        "policy": policy, "alpha": alpha,
                        "planned": len(group), "complete": len(complete),
                        "mean_output_tokens": _average(_finite(r.get("result", {}).get("output_tokens")) for r in complete),
                        "cap_hits": sum(r.get("result", {}).get("hit_token_cap") is True for r in complete),
                        "compared_to_plain": len(paired),
                        "changed_from_plain": sum(r["result"]["response"] != r["task"]["baseline_response"] for r in paired)})
    return summary


def _steering_audit_summary(records):
    """Aggregate measured changes by predictor position, not by answer count."""
    groups = defaultdict(list)
    for record in records:
        task = record["task"]
        kind = task.get("kind")
        if kind not in ("dose", "generate"):
            continue
        for side in (("positive", "negative") if kind == "dose" else ("generation",)):
            key = json.dumps([kind, side, _direction(task), task.get("policy", "all"), task.get("alpha")], sort_keys=True)
            groups[key].append(record)
    rows = []
    for key, group in groups.items():
        kind, side, direction, policy, alpha = json.loads(key)
        complete = [r for r in group if r["status"] == "complete"]
        valid, invalid, not_applicable = [], [], 0
        for record in complete:
            audit = record.get("result", {}).get("steering_audit")
            if kind == "dose":
                audit = audit.get(side) if isinstance(audit, dict) else None
            if audit is None and not record["task"].get("direction"):
                not_applicable += 1
                continue
            reason = None
            if not isinstance(audit, dict):
                reason = "missing steering audit"
            else:
                n = _finite(audit.get("modified_predictor_positions"))
                if n is None or n < 0 or not n.is_integer():
                    reason = "invalid modified-position count"
                elif audit.get("policy", policy) != policy:
                    reason = "audit policy differs from task policy"
                else:
                    names = ("mean_actual_delta_norm", "mean_actual_relative_delta", "max_actual_relative_delta")
                    if n:
                        names += ("mean_original_hidden_norm",)
                    if any(_finite(audit.get(name)) is None or audit[name] < 0 for name in names):
                        reason = "missing/nonfinite/negative measured norm"
                    elif n == 0 and any(audit[name] != 0 for name in names):
                        reason = "nonzero delta with zero modified positions"
            if reason:
                invalid.append({"task_id": record["id"], "reason": reason})
            else:
                valid.append(audit)
        positions = sum(int(a["modified_predictor_positions"]) for a in valid)

        def weighted(name):
            if positions:
                return sum(a[name] * a["modified_predictor_positions"] for a in valid
                           if a["modified_predictor_positions"] > 0) / positions
            # Alpha=0/no-op audits record a zero delta, but no hidden norm was measured.
            return 0.0 if valid and name != "mean_original_hidden_norm" else None

        rows.append({"kind": kind, "side": side, "direction": direction, "policy": policy,
                     "alpha": alpha, "planned": len(group), "complete": len(complete),
                     "valid_audits": len(valid), "not_applicable": not_applicable,
                     "invalid_audits": invalid, "modified_predictor_positions": positions,
                     "alpha_abs_values": sorted({_finite(a.get("alpha_abs")) for a in valid
                                                 if _finite(a.get("alpha_abs")) is not None}),
                     **{name: weighted(name) for name in ("mean_actual_delta_norm", "mean_original_hidden_norm", "mean_actual_relative_delta")},
                     "max_actual_relative_delta": max((a["max_actual_relative_delta"] for a in valid), default=None)})
    rows.sort(key=lambda r: (r["kind"], r["direction"]["name"], str(r["direction"]["layer"]),
                             r["policy"], float(r["alpha"] or 0), r["side"]))
    return rows


def _direction_comparisons(out, comparisons):
    rows = []
    for item in comparisons:
        row = {"name": item.get("name", "unnamed"), "status": "unavailable", "cosine": None}
        try:
            import numpy as np
            vectors = []
            for side in ("left", "right"):
                descriptor = item[side]
                path = Path(descriptor["path"])
                path = path if path.is_absolute() else out / path
                with np.load(path, allow_pickle=False) as bundle:
                    vectors.append(np.asarray(bundle[descriptor["key"]], dtype=float))
            left, right = vectors
            if left.ndim != 1 or left.shape != right.shape or not all(np.isfinite(v).all() for v in vectors):
                raise ValueError("directions must be finite, matching one-dimensional vectors")
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom == 0 or not math.isfinite(denom):
                raise ValueError("zero direction or nonfinite norm cannot define cosine")
            row.update(status="complete", cosine=float(np.clip(np.dot(left, right) / denom, -1, 1)), dimensions=len(left))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _examples(out, records):
    """Keep unjudged dev texts together without presenting fit examples as tests."""
    lines = ["# Unjudged dev generation examples", "",
             "These are model outputs for inspection, not endorsed medical advice or correctness labels. Original task artifacts retain fit generations separately.", ""]
    completed = [r for r in records if r["task"].get("kind") == "generate" and r["status"] == "complete"]
    fit_n = sum(r["task"].get("partition") == "fit" for r in completed)
    lines += [f"Completed fit generation tasks excluded from this comparison file: {fit_n}.", ""]
    groups = defaultdict(list)
    for r in completed:
        if r["task"].get("partition") != "fit":
            groups[_question_id(r["task"])].append(r)

    def quoted(text):
        text = str(text)
        # Markdown from model output is data, preserved inside an adequate fence.
        import re
        longest = max((len(m.group()) for m in re.finditer(r"`+", text)), default=0)
        fence = "`" * max(3, longest + 1)
        return [fence + "text", text, fence, ""]

    for question_id, group in sorted(groups.items()):
        task = group[0]["task"]
        q = task.get("question", "")
        q = q.get("question", q) if isinstance(q, dict) else q
        lines += [f"## {_cell(question_id)}", "", "Question:", ""] + quoted(q)
        if isinstance(task.get("baseline_response"), str):
            lines += ["Saved Plain answer:", ""] + quoted(task["baseline_response"])
        for r in group:
            t, result = r["task"], r.get("result", {})
            d = _direction(t)
            lines += [f"### {_cell(d['name'])}, L{d['layer']}, alpha={t.get('alpha')}, policy={_cell(t.get('policy', 'all'))}", "",
                      f"Task `{r['id']}`; output tokens: {result.get('output_tokens', 'unknown')}; token cap reached: {result.get('hit_token_cap', 'unknown')}. No judge score.", ""]
            lines += quoted(result.get("response", "[missing response]"))
    if not groups:
        lines += ["No completed dev generation examples yet.", ""]
    path = out / "generated_examples.md"
    _write(path, "\n".join(lines))
    return path


def _format(value):
    return "—" if value is None else f"{value:.4f}"


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _plot(out, rows):
    candidates = [r for r in rows if r["paired"] and _finite(r.get("alpha")) is not None]
    if not candidates:
        return None, None
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
        groups = defaultdict(list)
        for r in candidates:
            d = r["direction"]
            groups[f"{d['name']} L{d['layer']} / {r['group']} / {r['policy']}"].append(r)
        for label, points in groups.items():
            points.sort(key=lambda r: r["alpha"])
            if len(points) < 2:
                continue
            for ax, metric, title in zip(axes, ("delta_margin", "positive_delta", "negative_delta"),
                                         ("Change in correction/failure margin", "Correction answer change", "Failure answer change")):
                ax.plot([p["alpha"] for p in points], [p["full"][metric] for p in points], marker="o", label=label)
                ax.set_title(title)
        for ax in axes:
            ax.axhline(0, color="grey", linewidth=0.8)
            ax.set_xlabel("Alpha (fit residual scale)")
            ax.set_ylabel("Mean log probability per token: change vs alpha=0")
            ax.grid(alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=min(3, len(labels)), fontsize=7)
        fig.suptitle("Exploratory teacher-forced diagnostics; varying completed pair counts, no correctness judgment", fontsize=10)
        path = out / "curves.svg"
        fig.savefig(path)
        plt.close(fig)
        return path, None
    except Exception as exc:
        return None, f"Optional curve rendering unavailable: {type(exc).__name__}: {exc}"


def write_report(out) -> dict[str, Path]:
    """Rebuild an offline report, including missing and failed planned tasks."""
    out = Path(out)
    plan_path = out / "plan.json"
    plan = _json(plan_path)
    tasks = plan.get("tasks", [])
    ids = [str(t["id"]) for t in tasks]
    if len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("Plan task IDs must be unique safe filenames")
    records, warnings = _records(out, tasks)
    warnings = list(plan.get("warnings", [])) + warnings
    run_status = None
    if (out / "run_status.json").exists():
        try:
            run_status = _json(out / "run_status.json")
        except (ValueError, OSError) as exc:
            warnings.append(f"run_status.json could not be read: {exc}")
    counts = Counter(r["status"] for r in records)
    by_kind = {}
    for kind in sorted({r["task"].get("kind", "unknown") for r in records}):
        by_kind[kind] = dict(Counter(r["status"] for r in records if r["task"].get("kind", "unknown") == kind))
    doses, pairs = _dose_summary(records)
    projections = _project_summary(records)
    generations = _generation_summary(records)
    steering_audits = _steering_audit_summary(records)
    comparisons = _direction_comparisons(out, plan.get("direction_comparisons", []))
    examples_path = _examples(out, records)
    extractions = [{"id": r["id"], "status": r["status"],
                    "result": r.get("result", {}), "error": r.get("error")}
                   for r in records if r["task"].get("kind") == "extract"]
    figure, plot_warning = _plot(out, doses)
    if plot_warning:
        warnings.append(plot_warning)
    summary = {
        "version": "overnight-c-offline-report-v1",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "judge_calls": plan.get("judge_calls", 0), "report_model_calls": 0,
        "planned": len(tasks), "statuses": {s: counts[s] for s in STATUSES},
        "by_kind": by_kind, "run_status": run_status, "warnings": warnings,
        "fit_input_audit": {key: plan.get(key) for key in (
            "fit_pair_count", "fit_pair_authors", "vector_audit", "dev_pair_ids", "own_fit_ids")},
        "dose_groups": doses, "dose_pairs": pairs, "projection_groups": projections,
        "generation_groups": generations,
        "steering_audit_groups": steering_audits,
        "direction_comparisons": comparisons,
        "extractions": extractions,
        "tasks": [{"id": r["id"], "kind": r["task"].get("kind"), "status": r["status"],
                   "error": r.get("error"), "elapsed_seconds": r.get("elapsed_seconds")} for r in records],
        "interpretation": "Teacher-forced likelihood is a diagnostic proxy, not PCR or medical correctness. Dev is exploratory. No new judge calls. Missing/failed tasks remain visible; pair counts can differ across doses.",
    }
    lines = ["# Overnight C diagnostics", "",
             f"Planned tasks: {len(tasks)}. " + "; ".join(f"{s}: {counts[s]}" for s in STATUSES) + ".",
             "Report model calls: 0. This report uses existing local task artifacts only.",
             f"Planned judge calls: {plan.get('judge_calls', 0)}. Completed generation tasks: {sum(r['status'] == 'complete' and r['task'].get('kind') == 'generate' for r in records)} (includes fit candidate construction).",
             "Teacher-forced likelihood is a diagnostic proxy, **not PCR, medical correctness, or proof that C is a correction/style direction**. Dev is exploratory; no test evaluation.",
             "Projection separation is descriptive: reading a direction does not establish causal control. Generated answers below are unjudged.", ""]
    if run_status is not None:
        lines += ["## Run status", "", "```json", json.dumps(run_status, ensure_ascii=False, indent=2), "```", ""]
    if plan.get("vector_audit"):
        lines += ["## Frozen direction inputs", "",
                  f"Original complete fit pairs: {plan.get('fit_pair_count')}. Dev reference pairs: {len(plan.get('dev_pair_ids', []))}. Fit questions for own-model contrast: {len(plan.get('own_fit_ids', []))}.",
                  "Author counts below cover complete selected fit pairs, not unused one-sided references.", "",
                  "| Layer | Dimension | Unit-vector norm | Fit residual scale |", "|---|---:|---:|---:|"]
        for layer, values in plan["vector_audit"].items():
            lines.append(f"| {layer} | {values['dimension']} | {_format(values['norm'])} | {_format(values['scale'])} |")
        lines += ["", "```json", json.dumps(plan.get("fit_pair_authors", {}), ensure_ascii=False, indent=2), "```", ""]
    lines += ["## Completion by experiment", "", "| Kind | Complete | Failed | Timed out | Skipped | Pending |", "|---|---:|---:|---:|---:|---:|"]
    for kind, c in by_kind.items():
        lines.append(f"| {_cell(kind)} | " + " | ".join(str(c.get(s, 0)) for s in STATUSES) + " |")
    lines += ["", "## Dose response", "",
              "Each row compares the same pair and direction against alpha=0 (same policy, or an explicitly shared all-token zero control). Positive and negative answer changes are reported separately. Both may become less likely even when their difference rises.",
              "Pair counts can differ after failures/timeouts; compare per-pair entries in summary.json before interpreting a curve. No significance claims or automatic winner selection.", "",
              "| Direction | Layer | Group | Policy | Alpha | Paired / planned | Δ margin | Δ correction logp | Δ failure logp |", "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in doses:
        d, full = row["direction"], row["full"]
        name = d["name"] + (f" [random seed {d['random_seed']}]" if d["random_seed"] is not None else "")
        lines.append(f"| {_cell(name)} | {d['layer']} | {_cell(row['group'])} | {_cell(row['policy'])} | {row['alpha']} | {row['paired']}/{row['planned']} | {_format(full['delta_margin'])} | {_format(full['positive_delta'])} | {_format(full['negative_delta'])} |")
    if not doses:
        lines.append("| No dose artifacts planned | — | — | — | — | 0 | — | — | — |")
    if figure:
        lines += ["", "![Teacher-forced likelihood curves](curves.svg)"]
    lines += ["", "Prefix-32 and remaining-answer likelihoods and per-pair control IDs are retained in summary.json.", "", "## Projection diagnostic", "",
              "Only FPQ labels +1/−1 enter AUROC; 0, missing labels, and NFP are excluded. These are prior rubric labels, not a new clinical review. High projection predicts +1; no sign flipping to improve AUROC.", "",
              "| Direction | Layer | Pooling | Complete / planned | +1 | −1 | AUROC |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in projections:
        d = row["direction"]
        lines.append(f"| {_cell(d['name'])} | {d['layer']} | {row['pooling']} | {row['complete']}/{row['planned']} | {row['positive']} | {row['negative']} | {_format(row['auroc'])} |")
    lines += ["", "## Actual residual intervention", "",
              "Norms are measured after the residual update, including dtype rounding. Means are weighted by modified predictor positions, not by answer count. Relative change is the mean of each position's ||Δh|| / ||h||, not a ratio of aggregate means. Maximum relative change is the largest recorded position-level ratio.",
              "Dose positive/negative sequences and free generation are separate. Alpha=0 addition has zero modified positions; its hidden norm is unmeasured. Ablation remains active at alpha=0 and has no additive alpha_abs. A missing audit is never imputed as zero intervention.", "",
              "| Kind / sequence | Direction | Layer | Policy | Alpha | Additive alpha_abs | Valid audit / complete / planned | Modified positions | Mean actual ‖Δh‖ | Mean original ‖h‖ | Mean relative Δ | Max relative Δ |", "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in steering_audits:
        d = row["direction"]
        absolute = ", ".join(_format(value) for value in row["alpha_abs_values"]) or "—"
        lines.append(f"| {row['kind']} / {row['side']} | {_cell(d['name'])} | {d['layer']} | {_cell(row['policy'])} | {row['alpha']} | {absolute} | {row['valid_audits']}/{row['complete']}/{row['planned']} | {row['modified_predictor_positions']} | {_format(row['mean_actual_delta_norm'])} | {_format(row['mean_original_hidden_norm'])} | {_format(row['mean_actual_relative_delta'])} | {_format(row['max_actual_relative_delta'])} |")
    invalid_audits = [(row, item) for row in steering_audits for item in row["invalid_audits"]]
    if invalid_audits:
        lines += ["", "Missing or invalid steering audits (task completion status is retained):", ""]
        lines += [f"- `{item['task_id']}` ({row['side']}): {_cell(item['reason'])}." for row, item in invalid_audits]
    if any(row["not_applicable"] for row in steering_audits):
        lines += ["", "Rows with no direction/hook have no applicable intervention audit; these are marked in summary.json, not treated as measured zeros."]
    lines += ["", "## Unjudged generation checks", "", "Exact text changes do not establish improvement or a style-only effect. Token-cap hits flag truncation for inspection.", "",
              "| Direction / method | Split | Layer | Policy | Alpha | Complete / planned | Changed / compared to Plain | Mean output tokens | Cap hits |", "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in generations:
        d = row["direction"]
        lines.append(f"| {_cell(d['name'])} / {_cell(row['method'])} | {row['partition']} | {d['layer']} | {_cell(row['policy'])} | {row['alpha']} | {row['complete']}/{row['planned']} | {row['changed_from_plain']}/{row['compared_to_plain']} | {_format(row['mean_output_tokens'])} | {row['cap_hits']} |")
    lines += ["", "[Read all completed dev answer comparisons](generated_examples.md). Fit candidate-construction answers remain in their individual task artifacts."]
    if extractions:
        lines += ["", "## Candidate direction extraction", "", "Fit construction is not held-out evidence of correction. Pair counts below report available candidates; their outputs have no new correctness labels.", "",
                  "| Task | Status | Pairs | Layers | Unnormalized difference norms |", "|---|---|---:|---|---|"]
        for row in extractions:
            result = row["result"]
            lines.append(f"| {_cell(row['id'])} | {row['status']} | {result.get('n_pairs', '—')} | {_cell(result.get('layers', '—'))} | {_cell(result.get('unnormalized_direction_norms', '—'))} |")
    if comparisons:
        lines += ["", "## Direction overlap", "", "Cosine measures vector overlap only; neither high nor low overlap establishes a pure style or correction direction.", "", "| Comparison | Status | Cosine | Reason |", "|---|---|---:|---|"]
        for row in comparisons:
            lines.append(f"| {_cell(row['name'])} | {row['status']} | {_format(row['cosine'])} | {_cell(row.get('error', ''))} |")
    lines += ["", "## Failures and unstarted work", "", "| Task | Kind | Status | Reason |", "|---|---|---|---|"]
    unfinished = [r for r in records if r["status"] != "complete"]
    for r in unfinished:
        lines.append(f"| {_cell(r['id'])} | {_cell(r['task'].get('kind'))} | {r['status']} | {_cell(r.get('error') or 'No completed artifact')} |")
    if not unfinished:
        lines.append("| — | — | All planned artifacts complete | Numerical/semantic quality still requires inspection |")
    if warnings:
        lines += ["", "## Recorded limitations", ""] + [f"- {_cell(w)}" for w in warnings]
    lines += ["", f"Plan SHA-256: `{summary['plan_sha256']}`.", "Per-task inputs/results are preserved under tasks/. This report never modifies those artifacts.", ""]
    _write(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write(out / "report.md", "\n".join(lines))
    paths = {"report": out / "report.md", "summary": out / "summary.json", "examples": examples_path}
    if figure:
        paths["curves"] = figure
    return paths
