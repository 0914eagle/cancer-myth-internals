"""Minimal pairs for the A readout: each fpq question with its premise made true.

Text-only TF-IDF separates fpq from NFP at AUROC 0.77 (fpq were written by
GPT-4o from myths, NFP are questions LLMs false-alarmed on), so a probe that
separates the two may be reading style. The control is a twin of every fpq
question that differs only inside the aligned premise span, where the false
belief is replaced by the correct one (Cancer-Myth's correction). The twin is
spliced, not rewritten, so everything outside the span is byte-identical;
a second LLM call confirms the false belief is gone.

    python scripts/make_true_twins.py --questions $ROWS/questions.jsonl --out-dir $ROWS

Writes questions_twins.jsonl (set=tpair, pair_id=<fpq id>),
activation_rows_twins.jsonl, twins.jsonl (checkpoint, one line per fpq),
twins_audit.md and twins_sample.md. Only fpq rows with an LLM-verified span
get a twin.
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
from src.config import load_config
from src.jsonl import append_jsonl, read_jsonl, write_jsonl
from src.rows import activation_rows, make_true_twin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", choices=["codex", "openai", "claude"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--claude-cmd", default="claude")
    parser.add_argument("--limit", type=int, default=None)
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "twins.jsonl"
    done = {r["id"]: r for r in read_jsonl(progress)} if progress.exists() else {}

    fpq = [q for q in read_jsonl(args.questions) if q.get("set") == "fpq" and q.get("align_method") == "llm"]
    if args.limit:
        fpq = fpq[: args.limit]
    print(f"[twins] {len(fpq)} fpq rows with a verified span; {len(done)} already attempted", flush=True)

    n_new = 0
    for q in fpq:
        if q["id"] in done:
            continue
        twin, status = make_true_twin(q, llm)
        append_jsonl(progress, {"id": q["id"], "status": status, "twin": twin})
        done[q["id"]] = {"id": q["id"], "status": status, "twin": twin}
        n_new += 1
        if n_new % 25 == 0:
            print(f"[twins] {n_new} attempted this run", flush=True)

    twins = [d["twin"] for d in done.values() if d.get("twin")]
    write_jsonl(out_dir / "questions_twins.jsonl", twins)
    write_jsonl(out_dir / "activation_rows_twins.jsonl", activation_rows(twins))
    status = Counter(d["status"] for d in done.values())
    lines = ["# True-premise twins audit", "", "| status | n |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(status.items())]
    lines += ["", f"twins: {len(twins)} of {len(fpq)} eligible fpq rows"]
    (out_dir / "twins_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    by_id = {q["id"]: q for q in read_jsonl(args.questions)}
    rng = random.Random(args.seed)
    sample = rng.sample(twins, min(30, len(twins)))
    md = ["# True-premise twins (check by eye)", ""]
    for t in sample:
        md += [
            f"## {t['pair_id']}",
            f"- false belief: {by_id[t['pair_id']].get('premise_text')}",
            f"- original span: **{t['replaced']}**",
            f"- twin span:     **{t['replaced_with']}**",
            f"- twin question: {t['question']}",
            "",
        ]
    (out_dir / "twins_sample.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[twins] status: {dict(status)}")
    print(f"[done] {out_dir / 'questions_twins.jsonl'} ({len(twins)} twins)")


if __name__ == "__main__":
    main()
