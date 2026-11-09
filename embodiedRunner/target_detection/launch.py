"""Load the previously authorized local Qwen key into coordinator memory only."""
import argparse
import os
import re
import sys
from . import __main__ as direct
from .sources import ROOT, canonical, file_digest, implementation_hashes
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "target_detection/experiment-20261108-r03.json"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--environment")
    args = parser.parse_args()
    from pathlib import Path
    frozen = json.loads((ROOT / "target_detection/verification/frozen.json").read_text())
    if (implementation_hashes() != frozen["implementation_hashes"]
            or file_digest(args.config) != frozen["config_sha256"]):
        raise ValueError("frozen_launch_identity_changed")
    key_file = ROOT.parent / "qwenkey.md"
    if key_file.stat().st_mode & 0o077:
        raise ValueError("authorized_key_permissions_not_private")
    keys = set(re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", key_file.read_text()))
    if len(keys) != 1:
        raise ValueError("authorized_key_not_unique")
    os.environ["DASHSCOPE_API_KEY"] = keys.pop()
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    try:
        sys.argv = ["target_detection", "run", "--config", args.config,
                    "--run-id", args.run_id, "--manipulation-catalog",
                    str(ROOT / "official_direct/assets/manipulation_assets.json")]
        if args.environment:
            sys.argv += ["--environment", args.environment]
        return direct.main()
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("DASHSCOPE_BASE_URL", None)


if __name__ == "__main__":
    raise SystemExit(main())
