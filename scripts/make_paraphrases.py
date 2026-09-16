"""Register-matched paraphrases of every suite question (style control 2).

One writer rewrites FPQ and NFP alike in one plain register, keeping every
belief; a premise check (FPQ) and a fidelity check (all rows) drop rewrites
that changed content. See src/paraphrase.py and docs/29.

    python scripts/make_paraphrases.py --suite-dir $SUITE_DIR --backend claude
    python scripts/make_paraphrases.py --suite-dir $SUITE_DIR --backend codex --limit 20

Writes under <suite>/variants/para/: para.jsonl (checkpoint, one line per
attempted question; rerun resumes), questions_para.jsonl (accepted rows,
id=<id>_para, paraphrase_of=<id>), para_audit.md, para_sample.md.
Judge calls: 0. The writer is recorded per row; it is never the judge.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_rows import make_llm
from src.baseline_suite import load_suite
from src.config import load_config
from src.jsonl import append_jsonl, read_jsonl, write_jsonl
from src.paraphrase import PARAPHRASE_VERSION, make_paraphrase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--out-dir", default=None, help="default: <suite>/variants/para")
    parser.add_argument("--backend", choices=["codex", "openai", "claude"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--claude-cmd", default="claude")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-jaccard", type=float, default=0.6,
                        help="word overlap above which a rewrite is a near copy (retried once, then dropped)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    judge_cfg = cfg["judge"]
    backend = args.backend or judge_cfg.get("backend", "codex")
    model = args.model if args.model is not None else (
        judge_cfg.get("model", "gpt-4o") if backend == "openai"
        else judge_cfg.get("claude_model", "") if backend == "claude"
        else judge_cfg.get("codex_model", "")
    )
    llm = make_llm(backend, model, args.codex_cmd, claude_cmd=args.claude_cmd)
    if llm is None:
        raise SystemExit(f"backend {backend} unavailable")
    writer = f"{backend}:{model or 'default'}:{PARAPHRASE_VERSION}"
    if llm("Reply with exactly the word OK.") is None:
        raise SystemExit(f"backend {backend} answered nothing; check `claude auth status` (subscription login), "
                         "ANTHROPIC_API_KEY, or codex login before spending the run")

    suite = load_suite(args.suite_dir)
    rows = list(suite["questions"])
    if args.limit:
        # Half FPQ, half NFP, so the eyeball sample shows both registers.
        half = max(args.limit // 2, 1)
        rows = [q for q in rows if q["set"] == "fpq"][:half] + [q for q in rows if q["set"] == "nfp"][:args.limit - half]
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.suite_dir) / "variants" / "para"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "para.jsonl"
    # Last record per ID wins; "empty" rows are re-attempted (a dead backend used to be recorded that way).
    done = {}
    if progress.exists():
        for r in read_jsonl(progress):
            done[r["id"]] = r
        done = {k: v for k, v in done.items() if v.get("status") != "empty"}
    if any(d.get("writer") not in (None, writer) for d in done.values()):
        raise SystemExit(f"Checkpoint {progress} was written by a different writer or prompt version "
                         f"(now {writer}); delete it or use a new --out-dir")
    print(f"[para] {len(rows)} questions; {len(done)} already attempted; writer={writer}", flush=True)

    n_new, consecutive_failures = 0, 0
    for q in rows:
        if q["id"] in done:
            continue
        before = llm.failures["n"]
        row, status, audit = make_paraphrase(q, llm, writer=writer, max_jaccard=args.max_jaccard)
        if llm.failures["n"] > before:
            # Backend failure (timeout, auth, rate limit): not a property of the question; never recorded.
            consecutive_failures += 1
            print(f"[para] backend failure on {q['id']}; not recorded ({consecutive_failures} in a row)", flush=True)
            if consecutive_failures >= 3:
                raise SystemExit("three consecutive backend failures -- backend/login/limit not usable; "
                                 "fix and rerun (the checkpoint resumes)")
            continue
        consecutive_failures = 0
        record = {"id": q["id"], "status": status, "row": row, "writer": writer, **audit}
        append_jsonl(progress, record)
        done[q["id"]] = record
        n_new += 1
        if n_new % 25 == 0:
            print(f"[para] {n_new} attempted this run", flush=True)

    accepted = [d["row"] for d in done.values() if d.get("row")]
    write_jsonl(out_dir / "questions_para.jsonl", accepted)
    status = Counter(d["status"] for d in done.values())
    by_set = Counter(d["row"]["set"] for d in done.values() if d.get("row"))
    jacc = [d["jaccard"] for d in done.values() if d.get("jaccard") is not None]
    lines = ["# Paraphrase audit", "", f"writer: {writer}", "", "| status | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(status.items())]
    lines += ["", f"accepted: {len(accepted)} of {len(done)} attempted (fpq {by_set['fpq']}, nfp {by_set['nfp']})"]
    lines += [f"premise check applied: {sum(bool(d.get('premise_checked')) for d in done.values())} rows",
              f"near-copy retries: {sum(bool(d.get('retried')) for d in done.values())} rows (max_jaccard {args.max_jaccard})"]
    if jacc:
        jacc = sorted(jacc)
        lines += [f"word Jaccard original vs rewrite: median {jacc[len(jacc) // 2]:.2f}, "
                  f"min {jacc[0]:.2f}, max {jacc[-1]:.2f} (high = near copy, weak control)"]
    (out_dir / "para_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_id = {q["id"]: q for q in suite["questions"]}
    rng = random.Random(args.seed)
    sample = rng.sample(accepted, min(30, len(accepted)))
    md = ["# Paraphrases (check by eye)", ""]
    for r in sample:
        md += [f"## {r['paraphrase_of']} ({r['set']})",
               f"- original: {by_id[r['paraphrase_of']]['question']}",
               f"- rewrite:  {r['question']}", ""]
    (out_dir / "para_sample.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[para] status: {dict(status)}")
    print(f"[done] {out_dir / 'questions_para.jsonl'} ({len(accepted)} rows). Judge calls: 0.")


if __name__ == "__main__":
    main()
