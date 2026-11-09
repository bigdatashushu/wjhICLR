from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import unittest

from xcodeapi import XcodeClient, XcodeAPIError

from .config import ROOT, RunnerError, load_config


def csv_ints(value: str):
    try:
        values = tuple(int(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated task IDs, e.g. 0,1.") from None
    return values


def check(config) -> dict:
    client = XcodeClient(config.api_profile(), max_requests=0)
    payload = client.prepare(
        [{"role": "system", "content": "Return one JSON action."},
         {"role": "user", "content": "Offline serialization check."}],
        max_output_tokens=config.max_output_tokens,
        extra_body={"reasoning": {"effort": config.reasoning_effort}},
    )
    dependencies = {m: find_spec(m) is not None
                    for m in ("numpy", "PIL", "yaml", "torch", "robosuite", "mujoco", "bddl")}
    return {
        "model": payload["model"], "protocol": "responses",
        "model_requests": client.attempts, "credentials_read": False,
        "dependencies": dependencies, "official_source_present": config.upstream_root.is_dir(),
        "ready_for_simulator_preflight": all(dependencies.values()) and config.upstream_root.is_dir(),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="LIBERO official-environment evaluator with a GPT single-step action policy."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "configs/gpt6.1terra.json")
        command.add_argument("--libero-root")
        command.add_argument("--model")
        command.add_argument("--suites", help="Comma-separated suite names.")
        command.add_argument("--task-ids", type=csv_ints)
        command.add_argument("--episodes", dest="episodes_per_task", type=int)
        command.add_argument("--max-steps", type=int)
        command.add_argument("--max-requests", type=int)
        command.add_argument("--render-gpu-device-id", type=int)
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--credentials-file", type=Path)
    execute.add_argument("--resume", action="store_true",
                         help="Skip completed episodes; never retry an incomplete episode.")
    report = sub.add_parser("report")
    report.add_argument("--run-dir", type=Path, required=True)
    sub.add_parser("selfcheck", help="Offline contract tests; zero model requests, no simulator.")
    args = parser.parse_args(argv)
    try:
        if args.command == "selfcheck":
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
            return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
        if args.command in ("check", "prepare"):
            names = ("libero_root", "model", "task_ids", "episodes_per_task", "max_steps",
                     "max_requests", "render_gpu_device_id")
            overrides = {name: getattr(args, name) for name in names}
            overrides["suites"] = tuple(args.suites.split(",")) if args.suites else None
            config = load_config(args.config, **overrides)
        if args.command == "check":
            result = check(config)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready_for_simulator_preflight"] else 2
        from .runner import prepare, run, summarize
        if args.command == "prepare":
            manifest = prepare(config, args.output.resolve())
            result = {"prepared": True, "model_requests": 0,
                      "planned_episodes": len(manifest["episodes"]),
                      "evaluation_settings": manifest["evaluation_settings"],
                      "run_dir": str(args.output.resolve())}
        elif args.command == "run":
            result = run(args.run_dir.resolve(), credentials_file=args.credentials_file,
                         resume=args.resume)
        else:
            result = summarize(args.run_dir.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if args.command == "prepare" or result["complete"] else 2
    except (RunnerError, XcodeAPIError) as exc:
        print(f"LIBERORunner: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, ImportError, KeyError) as exc:
        # Avoid printing arbitrary provider data from an unexpected exception.
        print(f"LIBERORunner: {type(exc).__name__}; check local dependencies and run files.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
