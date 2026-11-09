"""CLI phases keep host reference access separate from online model execution."""
import argparse
import json
from pathlib import Path
import sys

from ...traces import read, write


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    selection = sub.add_parser("select")
    selection.add_argument("--source", required=True)
    selection.add_argument("--cache-index", required=True)
    selection.add_argument("--reviewed", nargs="+", required=True)
    selection.add_argument("--output", required=True)
    selection.add_argument("--slots", type=int, default=6, choices=range(1, 11))
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--selection", required=True)
    preparation.add_argument("--tokenizer", required=True)
    preparation.add_argument("--directory", required=True)
    continuation = sub.add_parser("prepare-from-frozen")
    continuation.add_argument("--parent-directory", required=True)
    continuation.add_argument("--directory", required=True)
    qualification = sub.add_parser("qualify")
    qualification.add_argument("--credentials", required=True)
    qualification.add_argument("--slots", type=int, default=6, choices=range(1, 11))
    qualification.add_argument("--output", required=True)
    execution = sub.add_parser("execute")
    execution.add_argument("--directory", required=True)
    credential_source = execution.add_mutually_exclusive_group(required=True)
    credential_source.add_argument("--credentials")
    credential_source.add_argument("--credentials-stdin", action="store_true")
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--directory", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--directory", required=True)
    scoring.add_argument("--references", required=True)
    scoring.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    if args.phase == "select":
        from .inputs import reviewed_ids, selection as choose
        result = choose(args.source, args.cache_index, reviewed_ids(args.reviewed),
                        [f"pool-{i + 1:02d}" for i in range(args.slots)])
        write(args.output, result, exclusive=True)
        print(json.dumps({"questions": result["questions"], "max_requests": result["max_requests"],
                          "selection_identity": result["selection_identity"], "model_requests": 0}))
    elif args.phase == "qualify":
        from .pool import qualify, read_keys
        keys = dict(list(read_keys(args.credentials).items())[:args.slots])
        result = {"results": qualify(keys), "inference_requests": 0}
        write(args.output, result, exclusive=True)
        print(json.dumps(result))
    elif args.phase == "prepare":
        from .runner import prepare
        result = prepare(args.directory, read(args.selection), args.tokenizer)
        print(json.dumps({"run_identity": result["run_identity"], "questions": result["questions"],
                          "max_requests": result["max_requests"], "model_requests": 0}))
    elif args.phase == "prepare-from-frozen":
        from .continuation import prepare_from_frozen
        result = prepare_from_frozen(args.directory, args.parent_directory)
        print(json.dumps({"run_identity": result["run_identity"], "questions": result["questions"],
                          "max_requests": result["max_requests"], "model_requests": 0,
                          "combined_max_requests": result["continuation"]["combined_max_requests"]}))
    elif args.phase == "execute":
        from .pool import read_keys
        from .runner import execute, validate
        manifest = validate(args.directory)
        keys = json.loads(sys.stdin.buffer.read()) if args.credentials_stdin else read_keys(args.credentials)
        keys = {api: keys[api] for api in manifest["api_ids"]}
        if any(not isinstance(value, str) or not value for value in keys.values()):
            raise ValueError("invalid_transient_credentials")
        print(json.dumps(execute(args.directory, keys)))
    elif args.phase == "freeze":
        from .runner import freeze
        result = freeze(args.directory)
        print(json.dumps({"freeze_identity": result["freeze_identity"],
                          "predictions": len(result["predictions"])}))
    elif args.phase == "score":
        from .evaluation import score
        result = score(args.directory, args.references, args.reference_sha256)
        print(json.dumps(result["overall"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Avoid tracebacks containing service context or credential-bearing arguments.
        print('{"status":"failed","reason_code":"phase_failed"}', file=sys.stderr)
        raise SystemExit(2)
