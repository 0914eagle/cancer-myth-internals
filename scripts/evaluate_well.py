"""Prepare offline, explicitly score capped calls, or report offline Well ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.well_eval import DEFAULT_MODEL, prepare, report, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    p = subs.add_parser("prepare", help="Freeze prompts/provenance; no model calls")
    p.add_argument("--questions", required=True)
    p.add_argument("--answers", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend", choices=("codex", "openai"), default="codex")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p = subs.add_parser("score", help="New calls only; never retry a started job")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-calls", type=int, required=True)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--codex-cmd", default="codex")
    p = subs.add_parser("report", help="Read saved replies; no model calls")
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.questions, args.answers, args.out_dir, backend=args.backend, model=args.model)
    elif args.stage == "score":
        result = score(args.out_dir, max_calls=args.max_calls, timeout=args.timeout, codex_cmd=args.codex_cmd)
    else:
        result = report(args.out_dir)
        print((Path(args.out_dir) / "report.md").read_text())
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
