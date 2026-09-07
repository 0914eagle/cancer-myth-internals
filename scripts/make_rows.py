"""E0: build the question set and the activation rows for positions A-D.

Outputs (under ${ART}/data/e1_rows_v1/):

    questions.jsonl        one row per question: fpq 585, nfp 150, tpq (Well)
    activation_rows.jsonl  up to three extraction rows per question (A, B, D)
    alignment_audit.md     how many premise spans were found, by method
    sample_alignments.md   40 random (question, premise, span) triples to eyeball

Premise spans are found by an LLM (verbatim-substring extraction, verified)
through `codex exec` or the OpenAI API, with a content-word heuristic as
fallback when neither is available. Run
once; the row files are inputs to every later stage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config
from src.jsonl import read_jsonl, write_jsonl
from src.rows import activation_rows, align_premise, load_fpq, load_nfp, load_tpq


def make_llm(backend: str, model: str, codex_cmd: str):
    from src.llm_backend import backend_available, make_caller

    if not backend_available(backend, codex_cmd):
        return None
    call = make_caller(backend, model, timeout=120, codex_cmd=codex_cmd, temperature=0.0, max_tokens=120)
    return lambda prompt: call(prompt)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-name", default="e1_rows_v1")
    parser.add_argument("--align", choices=["llm", "heuristic", "none"], default="llm")
    parser.add_argument("--align-backend", choices=["codex", "openai"], default=None, help="default: config judge.backend")
    parser.add_argument("--align-model", default=None, help="default: config judge model for the backend")
    parser.add_argument("--codex-cmd", default="codex")
    parser.add_argument("--limit", type=int, default=None, help="Per set, for a smoke run.")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg["paths"]["data_dir"]) / args.out_name)

    fpq, nfp, tpq = load_fpq(cfg), load_nfp(cfg), load_tpq(cfg)
    if args.limit:
        fpq, nfp, tpq = fpq[: args.limit], nfp[: args.limit], tpq[: args.limit]
    print(f"[rows] fpq={len(fpq)} nfp={len(nfp)} tpq={len(tpq)}", flush=True)
    missing_premise = sum(1 for r in fpq if not r.get("premise_text"))
    print(f"[rows] fpq rows without premise text: {missing_premise}", flush=True)

    existing = out_dir / "questions.jsonl"
    cache: dict[str, dict] = {}
    if existing.exists():
        cache = {r["id"]: r for r in read_jsonl(existing)}
        print(f"[rows] reusing {len(cache)} alignments from {existing}", flush=True)

    judge_cfg = cfg["judge"]
    backend = args.align_backend or judge_cfg.get("backend", "codex")
    model = args.align_model if args.align_model is not None else (
        judge_cfg.get("model", "gpt-4o") if backend == "openai" else judge_cfg.get("codex_model", "")
    )
    llm = make_llm(backend, model, args.codex_cmd) if args.align == "llm" else None
    if args.align == "llm" and llm is None:
        print(f"[align] backend {backend} unavailable (no key / no codex on PATH) -> heuristic alignment", flush=True)
    elif llm is not None:
        print(f"[align] via {backend} model={model or 'backend default'}", flush=True)

    questions = []
    for row in fpq + nfp + tpq:
        prev = cache.get(row["id"])
        if prev and prev.get("premise_span") is not None and prev.get("align_method") == "llm":
            row.update(
                premise_span=prev["premise_span"],
                align_score=prev.get("align_score"),
                align_method=prev.get("align_method"),
            )
        elif args.align == "none" or not row.get("premise_text"):
            row.update(premise_span=None, align_score=None, align_method="none")
        else:
            span, score, method = align_premise(row["question"], row["premise_text"], llm=llm)
            row.update(premise_span=span, align_score=score, align_method=method)
        questions.append(row)

    write_jsonl(out_dir / "questions.jsonl", questions)
    act_rows = activation_rows(questions)
    write_jsonl(out_dir / "activation_rows.jsonl", act_rows)

    by_set = Counter((r["set"], r["align_method"]) for r in questions)
    lines = ["# Premise alignment audit", "", "| set | method | n |", "|---|---|---:|"]
    for (s, m), n in sorted(by_set.items()):
        lines.append(f"| {s} | {m} | {n} |")
    lines += [
        "",
        f"activation rows: {len(act_rows)} "
        f"(families: {dict(Counter(r['position_family'] for r in act_rows))})",
    ]
    (out_dir / "alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    aligned = [q for q in questions if q.get("premise_span")]
    sample = rng.sample(aligned, min(40, len(aligned)))
    out = ["# Sample alignments (check by eye)", ""]
    for q in sample:
        s, e = q["premise_span"]
        out += [
            f"## {q['id']} ({q['align_method']}, {q.get('align_score')})",
            f"- premise: {q['premise_text']}",
            f"- span: **{q['question'][s:e]}**",
            f"- question: {q['question']}",
            "",
        ]
    (out_dir / "sample_alignments.md").write_text("\n".join(out), encoding="utf-8")
    print(json.dumps({f"{s}/{m}": n for (s, m), n in sorted(by_set.items())}, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
