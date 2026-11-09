"""Offline-first command line. Only commands with --live can use the API."""

from __future__ import annotations

import argparse
import json
import sys

from .client import XcodeClient
from .config import DEFAULT_CREDENTIALS, load_profiles
from .content import image_file_part, text_part
from .errors import XcodeAPIError


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Isolated Xcode experiment API client")
    commands = result.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Check configuration offline; do not read keys")
    preflight.add_argument("--config")
    for command in ("prepare", "models", "smoke"):
        child = commands.add_parser(command)
        child.add_argument("--profile", choices=("gpt", "claude", "gemini"), required=True)
        child.add_argument("--config", help="Optional profiles.local.json path")
        child.add_argument("--model", help="Exact model ID from your token group")
        child.add_argument("--credentials", default=str(DEFAULT_CREDENTIALS))
        if command != "prepare":
            child.add_argument("--live", action="store_true", help="Actually send one API request")
        if command != "models":
            child.add_argument("--image", action="append", default=[], help="Local image; order is preserved")
            child.add_argument("--max-output-tokens", type=int, default=64)
        if command == "smoke":
            child.add_argument("--print-text", action="store_true", help="Explicitly print generated text")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            profiles = load_profiles(args.config)
            checks = {}
            for name, profile in profiles.items():
                # Check all protocol builders even when the account model ID
                # has not been selected. This value is never sent or presented
                # as an available model.
                client = XcodeClient(profile.with_model(profile.model or "offline-validation-only"))
                client.prepare([{"role": "system", "content": "Reply briefly."},
                                {"role": "user", "content": "Reply with OK."}])
                checks[name] = {
                    "protocol": profile.protocol,
                    "model_configured": bool(profile.model),
                    "key_variable": profile.api_key_env,
                    "payload_builder_valid": True,
                }
            print(json.dumps({
                "mode": "offline", "network_requests": 0, "credentials_read": False,
                "profiles": checks,
            }, indent=2))
            return 0
        client = XcodeClient.from_profile(
            args.profile, config_file=args.config, model=args.model,
            credentials_file=args.credentials, max_requests=1,
        )
        if args.command == "models":
            if not args.live:
                raise XcodeAPIError("models requires --live; model listing is an authenticated API request.")
            models = client.list_models()
            # Only IDs are intentionally requested for this command.
            print(json.dumps({"profile": args.profile, "model_ids": [item["id"] for item in models],
                              "request_attempts": client.attempts}, indent=2))
            return 0
        parts = [text_part("Reply with OK.")]
        parts.extend(image_file_part(path) for path in args.image)
        payload = client.prepare(
            [{"role": "user", "content": parts}], max_output_tokens=args.max_output_tokens
        )
        if args.command == "prepare":
            print(json.dumps({
                "mode": "offline", "profile": args.profile, "protocol": client.profile.protocol,
                "model": client.profile.model, "image_count": len(args.image),
                "max_output_tokens": args.max_output_tokens,
                "payload_valid": True, "network_requests": 0, "credentials_read": False,
            }, indent=2))
            return 0
        if not args.live:
            raise XcodeAPIError("smoke requires --live; use prepare for offline validation.")
        reply = client.send(payload)
        print(json.dumps({
            "profile": args.profile, "request_attempts": client.attempts,
            "returned_model_observed": reply.returned_model is not None,
            "returned_model_matches": (
                reply.returned_model == reply.requested_model if reply.returned_model is not None else None
            ),
            **reply.metadata(),
        }, indent=2))
        if args.print_text:
            print(reply.text)
        return 0
    except XcodeAPIError as error:
        # Library errors contain only locally defined, sanitized diagnostics.
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
