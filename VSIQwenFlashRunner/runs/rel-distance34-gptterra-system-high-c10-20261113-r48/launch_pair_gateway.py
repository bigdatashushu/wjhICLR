#!/usr/bin/env python3
"""Start an isolated API gateway using the node's existing credential loader."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=27046)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    root = args.run_root.resolve()
    assert not (root / "gateway_start.json").exists()
    config = json.loads((root / "gateway_config.json").read_text())
    xcode = config.get("provider", "bailian") == "xcodeapi"
    loader = Path("/home/cvailab/wangjh/harness3d/precompute-20261022-r01/ops/launch_perception_service.py")
    tree = ast.parse(loader.read_text())
    allowed = {"DASHSCOPE_API_KEY", "HARNESS3D_PERCEPTION_TOKEN"}
    values = {}
    path = Path("/home/cvailab/wangjh/harness3d/precompute-20261022-r01/.transfer.env")
    for line in path.read_text().splitlines():
        words = shlex.split(line)
        if words and words[0] == "export":
            words = words[1:]
        if len(words) == 1 and "=" in words[0]:
            key, value = words[0].split("=", 1)
            values[key] = value
    for name in ("HARNESS3D_TRANSFER_TOKEN", "HARNESS3D_PERCEPTION_TOKEN"):
        if values.get(name):
            os.environ[name] = values[name]
    assignments = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "os" and target.value.attr == "environ"
                    and isinstance(target.slice, ast.Constant) and target.slice.value in allowed):
                assignments.append(node)
                break
    assert assignments
    # Execute only the existing environment assignments, never its service launch.
    if not xcode:
        exec(compile(ast.Module(body=assignments, type_ignores=[]), str(loader), "exec"),
             {"os": os, "Path": Path})
        assert os.environ.get("DASHSCOPE_API_KEY")
    else:
        private = Path(config["key_pool_path"])
        assert private.is_file() and private.stat().st_mode & 0o077 == 0
        os.environ["VSI_XCODE_KEY_POOL"] = str(private)
    if not os.environ.get("HARNESS3D_PERCEPTION_TOKEN"):
        os.environ["HARNESS3D_PERCEPTION_TOKEN"] = (
            values.get("HARNESS3D_PERCEPTION_TOKEN") or values["HARNESS3D_TRANSFER_TOKEN"])
    receipt = {
        "started_at": time.time(), "pid": os.getpid(), "port": args.port,
        "max_concurrency": args.concurrency, "model_requests_at_start": 0,
        "credential_loader_sha256": hashlib.sha256(loader.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((root / "gateway_config.json").read_bytes()).hexdigest(),
        "command_sha256": hashlib.sha256(json.dumps(sys.argv).encode()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "gateway_start.json").write_text(json.dumps(receipt, indent=2) + "\n")
    sys.argv = [
        "model_rpc_gateway.py", "--config", str(root / "gateway_config.json"),
        "--root", str(root / "gateway_cache"), "--host", "0.0.0.0",
        "--port", str(args.port), "--max-concurrency", str(args.concurrency)]
    runpy.run_path(str(root / "code/source/scripts/model_rpc_gateway.py"), run_name="__main__")


if __name__ == "__main__":
    main()
