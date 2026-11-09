"""Explicit phase entrypoints; only execute can contact the model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="Independent MMSI official-prompt Direct prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze public inputs; zero model requests")
    prep.add_argument("--run", required=True)
    prep.add_argument("--source-manifest", required=True)
    prep.add_argument("--config", required=True)
    prep.add_argument("--api-id", required=True)
    selection = prep.add_mutually_exclusive_group()
    selection.add_argument("--per-type", type=int, default=1)
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--all", action="store_true")
    prep.add_argument("--exclude-file", help="JSON list of reviewed sample IDs; no reference answers")
    prep.add_argument("--max-output-tokens", type=int, default=8192)
    run = sub.add_parser("execute", help="Execute each selected question once; no retry or resume")
    run.add_argument("--run", required=True)
    run.add_argument("--config", required=True)
    for name in ("freeze", "export"):
        sub.add_parser(name).add_argument("--run", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--run", required=True)
    scoring.add_argument("--references", required=True)
    scoring.add_argument("--references-sha256", required=True)
    sub.add_parser("selfcheck", help="Synthetic fixtures, fake Responses client, networking disabled")
    args = parser.parse_args()
    try:
        if args.command == "selfcheck":
            from .selfcheck import selfcheck
            result = selfcheck()
        elif args.command == "prepare":
            from .direct import prepare
            excluded = json.loads(Path(args.exclude_file).read_text()) if args.exclude_file else []
            if not isinstance(excluded, list) or any(not isinstance(s, str) for s in excluded):
                raise ValueError("exclude_file_requires_public_id_list")
            result = prepare(
                args.run, args.source_manifest, args.config, args.api_id,
                per_type=args.per_type, sample_ids=args.ids, excluded=excluded,
                all_samples=args.all, max_output_tokens=args.max_output_tokens)
        elif args.command == "execute":
            from .direct import execute
            result = execute(args.run, args.config)
        elif args.command == "freeze":
            from .evaluation import freeze
            frozen = freeze(args.run)
            result = {"freeze_identity": frozen["freeze_identity"],
                      "status_counts": frozen["status_counts"], "questions": frozen["questions"]}
        elif args.command == "score":
            from .evaluation import score
            result = score(args.run, args.references, args.references_sha256)["overall"]
        else:
            from .evaluation import export
            result = export(args.run)
    except Exception as exc:
        # Arbitrary exception text may include a URL, credential or provider body.
        known = isinstance(exc, ValueError) and len(str(exc)) < 100 and all(
            char.islower() or char.isdigit() or char == "_" for char in str(exc))
        print(json.dumps({"status": "error", "error_code": str(exc) if known else "operation_failed",
                          "error_type": type(exc).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
