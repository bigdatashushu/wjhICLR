"""Run a prepared rollout; transport the existing API key only through SSH stdin."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def main():
    from xcodeapi.config import resolve_api_key
    from LIBERORunner.config import Config
    from LIBERORunner.scripts.remote_preflight import render_environment

    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    task_root = PurePosixPath(settings["remote_root"])
    run_dir = PurePosixPath(args.run_dir)
    if ".." in run_dir.parts or not run_dir.is_relative_to(task_root / "runs"):
        raise ValueError("Use a prepared run under this isolated task root.")
    code = (
        "import os,sys; "
        "os.environ['XCODE_GPT_API_KEY']=sys.stdin.read().strip(); "
        "from LIBERORunner.scripts.network import configure; configure(); "
        "from LIBERORunner.__main__ import main; "
        f"sys.exit(main(['run','--run-dir',{str(run_dir)!r}]))"
    )
    quote = shlex.quote
    render_env, _ = render_environment(task_root, settings)
    command = (
        f"env LIBERO_IPV4_ONLY=1 {render_env} OMP_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 "
        f"PYTHONPATH={quote(str(task_root / 'source'))} "
        f"{quote(str(task_root / 'runtime/venv/bin/python'))} -c {quote(code)}"
    )
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if settings.get("ssh_config"):
        ssh += ["-F", str(Path(settings["ssh_config"]).expanduser().resolve())]
    ssh += [settings["host"], command]
    key = resolve_api_key(Config().api_profile(), ROOT.parent / "xcodeapi/credentials.local.env")
    return subprocess.run(ssh, input=key.encode()).returncode


if __name__ == "__main__":
    raise SystemExit(main())
