"""Snapshot existing results for the October 6 presentation; makes no model calls."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "results/fpqa_prompting"


def read(path):
    return json.loads(path.read_text())


def main():
    files = {
        "direct_answers": "well_upstream_test_v1_20261003_run/metrics.json",
        "medical_instruction_ablation": "well_medical_bullet_ablation_20261003/metrics.json",
        "pipelines": "well_pipelines_luna6_20261004/metrics.json",
        "uncertainty": "well_uncertainty_luna6_20261004_v1/metrics.json",
        "scope_factorial": "well_scope_factorial_luna6_20261004_v1/metrics.json",
        "verifier_gepa_status": "well_verifier_gepa_luna6_20261004_v1/status.json",
        "reliability_status": "well_reliability_20261005_v1/status_all.json",
    }
    snapshot = {"captured_utc": datetime.now(timezone.utc).isoformat(), "sources": {}, "results": {}}
    for name, rel in files.items():
        path = BASE / rel
        snapshot["sources"][name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        snapshot["results"][name] = read(path)

    folder = BASE / "well_reliability_20261005_v1"
    plan = read(folder / "run.json")
    rows = []
    for item in plan["audit_ids"]:
        for origin in ["B1", "B2"]:
            paths = [folder / "historical" / item / f"{origin}.json"]
            paths += [folder / "jobs/rejudge" / item / origin / str(n) / "record.json" for n in [1, 2]]
            scores = [read(p)["score"] for p in paths if p.exists()]
            rows.append({
                "id": item, "origin": origin,
                "selection": "flipped" if item in plan["flipped_ids"] else "stable_control",
                "scores": scores, "complete": len(scores) == 3,
                "all_scores_in_1_to_5": all(1 <= s <= 5 for s in scores),
                "threshold_changed": len({s >= 4 for s in scores}) > 1,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
            })
    valid = [r for r in rows if r["complete"] and r["all_scores_in_1_to_5"]]
    snapshot["rescoring"] = {
        "records": rows,
        "completed_stored_answers": sum(r["complete"] for r in rows),
        "valid_stored_answers": len(valid),
        "valid_answers_crossing_ge4": sum(r["threshold_changed"] for r in valid),
        "valid_groups": dict(Counter(r["selection"] for r in valid)),
        "crossing_by_group": dict(Counter(r["selection"] for r in valid if r["threshold_changed"])),
        "note": "Selected audit sample, not population judge error rate. Score 0 remains unchanged in source and is excluded only from this valid-score analysis.",
    }
    for name, rel in {
        "answer_gepa_cancer": "well_upstream_cli_v3_20261003/cancer_myth/run.json",
        "answer_gepa_crepe": "well_upstream_cli_v3_20261003/crepe/run.json",
        "verifier_gepa": "well_verifier_gepa_luna6_20261004_v1/run.json",
    }.items():
        d = read(BASE / rel)
        keys = ["protocol", "config", "counts", "n_train", "n_val", "gepa_version", "objective", "feedback", "weighting", "max_metric_calls", "test_repeats", "remaining_transport_differences"]
        snapshot.setdefault("settings", {})[name] = {k: d[k] for k in keys if k in d}
    (OUT / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(OUT / "snapshot.json"), "rescore_valid": len(valid), "rescore_changed": sum(r["threshold_changed"] for r in valid)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
