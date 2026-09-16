"""Prepare offline, explicitly score capped calls, or report offline Well ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.well_eval import DEFAULT_MODEL, prepare, report, score, BACKENDS, DEFAULT_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    p = subs.add_parser("prepare", help="Freeze prompts/provenance; no model calls")
    p.add_argument("--questions", required=True)
    p.add_argument("--answers", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--backend", choices=BACKENDS, default="codex")
    p.add_argument("--model", default=None, help="default: gpt-5.6-terra for codex, claude-sonnet-5 for claude; required for openai")
    p = subs.add_parser("score", help="New calls only; never retry a started job")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-calls", type=int, required=True)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--codex-cmd", default="codex")
    p.add_argument("--claude-cmd", default="claude")
    p = subs.add_parser("report", help="Read saved replies; no model calls")
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        model = args.model or DEFAULT_MODELS.get(args.backend)
        if not model:
            parser.error("--model is required for backend openai")
        result = prepare(args.questions, args.answers, args.out_dir, backend=args.backend, model=model)
    elif args.stage == "score":
        result = score(args.out_dir, max_calls=args.max_calls, timeout=args.timeout, codex_cmd=args.codex_cmd,
                       claude_cmd=args.claude_cmd)
    else:
        result = report(args.out_dir)
        print((Path(args.out_dir) / "report.md").read_text())
        return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
