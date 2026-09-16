"""Calibrate a judge against the GPT-4o scores shipped with Cancer-Myth.

all_data.json carries, for every one of the 585 questions, the answers of
eight models and GPT-4o's Sharpness score for each. Scoring those same
answers with our judge and comparing gives, at no generation cost:

    agreement on the 3-way score, agreement on PCR (+1 vs not), Cohen's
    kappa, and the PCR each judge would report per model.

The paper's human check was PCR agreement (Appendix Table 2: 100% for GPT-4o
vs physicians). A judge that reproduces GPT-4o's PCR on these answers can
score our tables; one that does not, cannot -- whatever it costs.

    python scripts/calibrate_judge.py --backend codex --models GPT-4o Claude-3.5-Sonnet --n 150
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config
from src.jsonl import append_jsonl, load_json, read_jsonl
from src.judge_prompts import construct_prompt_fpq, parse_score
from src.llm_backend import check_judge_identity, make_caller


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[lab] * cb[lab] for lab in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backend", choices=["codex", "openai", "claude"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--claude-cmd", default="claude")
    parser.add_argument("--models", nargs="+", default=["GPT-4o", "Claude-3.5-Sonnet", "DeepSeek-R1"],
                        help="answer authors in all_data.json to score")
    parser.add_argument("--n", type=int, default=150, help="questions per answer model")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--allow-same-family", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    judge_cfg = cfg["judge"]
    backend = args.backend or judge_cfg.get("backend", "codex")
    model = args.model if args.model is not None else (
        judge_cfg.get("model", "gpt-4o") if backend == "openai"
        else judge_cfg.get("claude_model", "") if backend == "claude"
        else judge_cfg.get("codex_model", "")
    )
    examples = load_json(judge_cfg["examples_fpq"])
    data = load_json(Path(cfg["data"]["cancer_myth_repo"]) / "data" / "all_data.json")
    out_dir = ensure_dir(Path(args.out_dir or Path(cfg["paths"]["report_dir"]) / "judge_calibration" / f"{backend}_{(model or 'default').replace('/', '_')}"))
    scores_path = out_dir / "scores.jsonl"
    done = {r["id"] for r in read_jsonl(scores_path)} if scores_path.exists() else set()

    rng = random.Random(args.seed)
    items = list(range(len(data)))
    rng.shuffle(items)
    jobs = []
    for author in args.models:
        for i in items[: args.n]:
            row = data[i]
            if author not in row.get("answers", {}) or author not in row.get("scores", {}):
                continue
            job_id = f"{row['QID']}::{author}"
            if job_id in done:
                continue
            jobs.append((job_id, row, author))
    print(f"[calibrate] {len(jobs)} answers to score via {backend} ({len(done)} done) -> {out_dir}", flush=True)

    if jobs:
        check_judge_identity(model, args.allow_same_family)
        call = make_caller(backend, model, timeout=180, codex_cmd=args.codex_cmd, claude_cmd=args.claude_cmd, temperature=0.0)
        consecutive_failures = 0
        for n, (job_id, row, author) in enumerate(jobs, start=1):
            prompt = construct_prompt_fpq(row["example_question"], row["example_assumption"], row["answers"][author], examples)
            try:
                text, used = call(prompt)
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                consecutive_failures += 1
                print(f"[calibrate] {job_id}: {exc!r}", file=sys.stderr)
                if consecutive_failures >= 3:
                    raise SystemExit(
                        "three consecutive failures -- the backend/model is not usable "
                        "(e.g. 'not supported when using Codex with a ChatGPT account'); fix and rerun"
                    )
                continue
            score, parsed = parse_score(text)
            ref = row["scores"][author]
            append_jsonl(scores_path, {
                "id": job_id, "qid": row["QID"], "author": author, "category": row.get("category"),
                "ours": int(score.get("Sharpness", 1)), "gpt4o": int(ref["Sharpness"] if isinstance(ref, dict) else ref),
                "parsed": parsed, "judge_model": used, "judge_backend": backend, "raw": text,
            })
            if n % 25 == 0 or n == len(jobs):
                print(f"[calibrate] {n}/{len(jobs)}", flush=True)

    rows = list(read_jsonl(scores_path))
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["author"]].append(r)
    lines = [f"# Judge calibration: {backend} / {rows[0]['judge_model'] if rows else model}", "",
             "Reference = GPT-4o scores stored in Cancer-Myth all_data.json (the paper's judge).", "",
             "| answers by | n | 3-way agree | PCR agree | kappa | PCR (GPT-4o) | PCR (ours) | PCS (GPT-4o) | PCS (ours) | unparsed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    all_a, all_b = [], []
    for author, rs in sorted(by_author.items()):
        a = [r["gpt4o"] for r in rs]
        b = [r["ours"] for r in rs]
        all_a += a
        all_b += b
        n = len(rs)
        lines.append(
            f"| {author} | {n} | {100 * sum(x == y for x, y in zip(a, b)) / n:.1f} | "
            f"{100 * sum((x == 1) == (y == 1) for x, y in zip(a, b)) / n:.1f} | {kappa(a, b):.2f} | "
            f"{100 * sum(x == 1 for x in a) / n:.1f} | {100 * sum(y == 1 for y in b) / n:.1f} | "
            f"{sum(a) / n:.2f} | {sum(b) / n:.2f} | {sum(not r['parsed'] for r in rs)} |"
        )
    if all_a:
        n = len(all_a)
        lines += ["", f"Overall: n={n}, 3-way agreement {100 * sum(x == y for x, y in zip(all_a, all_b)) / n:.1f}%, "
                  f"PCR agreement {100 * sum((x == 1) == (y == 1) for x, y in zip(all_a, all_b)) / n:.1f}%, kappa {kappa(all_a, all_b):.2f}"]
        conf = Counter((x, y) for x, y in zip(all_a, all_b))
        lines += ["", "Confusion (rows GPT-4o, cols ours):", "", "| | -1 | 0 | +1 |", "|---|---:|---:|---:|"]
        for x in (-1, 0, 1):
            lines.append(f"| {x:+d} | " + " | ".join(str(conf[(x, y)]) for y in (-1, 0, 1)) + " |")
    report = "\n".join(lines) + "\n"
    (out_dir / "calibration.md").write_text(report, encoding="utf-8")
    print(report)
    print(json.dumps({"out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
